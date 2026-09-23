"""`run_item`/`run_arm` end to end, entirely offline: fake routes, a fake judge, no network.

This is what proves the harness's wiring — usage capture, the agent tool round trip, the
embedding-cost estimate, per-item error isolation — works before it is ever pointed at a real,
metered provider (T-140: build and verify the harness without spending further Gemini quota).
"""

from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall

from benchmark.arms import Arm
from benchmark.dataset import WorkloadItem
from benchmark.judge import JudgeVerdict
from benchmark.runner import run_arm, run_item
from benchmark.tests.fakes import FakeJudge, RaisingModel
from benchmark.tools import ALL_TOOLS
from langchain_llm_router import ChatRouter
from langchain_llm_router.strategies import (
    ClassifierStrategy,
    EmbeddingStrategy,
    HeuristicStrategy,
    KeywordStrategy,
)
from tests.fakes import NativeStructuredFakeChatModel


def _small(model_name: str = "ollama:qwen3:8b") -> NativeStructuredFakeChatModel:
    return NativeStructuredFakeChatModel(
        model_name=model_name, reply="a small answer", input_tokens=5, output_tokens=8
    )


def _frontier() -> NativeStructuredFakeChatModel:
    return NativeStructuredFakeChatModel(
        model_name="google_genai:gemini-3-flash-preview",
        reply="a frontier answer",
        input_tokens=20,
        output_tokens=40,
    )


SINGLE_TURN_ITEM = WorkloadItem(
    id="st-1",
    domain="math",
    difficulty="easy",
    kind="single_turn",
    prompt="what is 2 + 2?",
    rubric="answers 4",
    provenance="test fixture",
)

AGENT_ITEM = WorkloadItem(
    id="agent-1",
    domain="math",
    difficulty="easy",
    kind="agent",
    prompt="what is 2 + 2, using the calculator?",
    rubric="calls calculator and reports 4",
    provenance="test fixture",
    expected_tool="calculator",
)


def test_a_single_turn_item_is_run_and_graded() -> None:
    router = ChatRouter(routes={"frontier": _frontier()}, default_route="frontier")
    arm = Arm("baseline", router)

    result = run_item(arm, SINGLE_TURN_ITEM, FakeJudge(), embed=False)

    assert result.error is None
    assert result.decision is not None
    assert result.decision.route == "frontier"
    assert result.answer == "a frontier answer"
    assert result.tool_called is None
    assert result.tool_correct is None
    assert result.judge == JudgeVerdict(score=8, rationale="fake judge, fixed verdict")
    assert result.cost.total_usd > 0.0
    assert result.cost.embedding_usd == 0.0


def test_an_agent_item_calls_the_right_tool_and_sums_usage_across_both_calls() -> None:
    model = NativeStructuredFakeChatModel(
        model_name="google_genai:gemini-3-flash-preview",
        script=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(name="calculator", args={"expression": "2 + 2"}, id="c1")],
            ),
            AIMessage(content="4"),
        ],
    )
    router = ChatRouter(routes={"frontier": model}, default_route="frontier")
    arm = Arm("baseline", router)

    result = run_item(arm, AGENT_ITEM, FakeJudge(), embed=False)

    assert result.error is None
    assert result.tool_called == "calculator"
    assert result.tool_correct is True
    assert result.answer == "4"
    # Two calls to the same model: both invocations' usage is summed under one key.
    assert result.usage_by_model["google_genai:gemini-3-flash-preview"]["input_tokens"] == 6
    assert result.usage_by_model["google_genai:gemini-3-flash-preview"]["output_tokens"] == 10


def test_an_agent_item_that_calls_the_wrong_tool_is_marked_incorrect() -> None:
    model = NativeStructuredFakeChatModel(
        model_name="google_genai:gemini-3-flash-preview",
        script=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(name="get_weather", args={"city": "Paris"}, id="c1")],
            ),
            AIMessage(content="I checked the weather instead"),
        ],
    )
    router = ChatRouter(routes={"frontier": model}, default_route="frontier")
    arm = Arm("baseline", router)

    result = run_item(arm, AGENT_ITEM, FakeJudge(), embed=False)

    assert result.tool_called == "get_weather"
    assert result.tool_correct is False


def test_an_agent_item_that_answers_without_a_tool_call_is_marked_incorrect() -> None:
    model = NativeStructuredFakeChatModel(
        model_name="google_genai:gemini-3-flash-preview", reply="the answer is 4"
    )
    router = ChatRouter(routes={"frontier": model}, default_route="frontier")
    arm = Arm("baseline", router)

    result = run_item(arm, AGENT_ITEM, FakeJudge(), embed=False)

    assert result.tool_called is None
    assert result.tool_correct is False


def test_a_heuristic_arm_runs_and_reports_a_strategy_name() -> None:
    router = ChatRouter(
        routes={"small": _small(), "frontier": _frontier()},
        default_route="small",
        strategy=HeuristicStrategy("small", "frontier"),
    )
    arm = Arm("heuristic", router)

    result = run_item(arm, SINGLE_TURN_ITEM, FakeJudge(), embed=False)

    assert result.error is None
    assert result.decision is not None
    assert result.decision.strategy == "HeuristicStrategy"


def test_a_classifier_arm_s_own_call_is_folded_into_the_recorded_usage() -> None:
    # Same model id as the "small" route: a real overhead in a live run, $0 either way here.
    classifier_model = _small()
    router = ChatRouter(
        routes={"small": _small(), "frontier": _frontier()},
        default_route="small",
        strategy=ClassifierStrategy(
            classifier_model, {"small": "simple requests", "frontier": "hard requests"}
        ),
    )
    arm = Arm("classifier", router)

    result = run_item(arm, SINGLE_TURN_ITEM, FakeJudge(), embed=False)

    assert result.error is None
    # The classifier's own call and the answering route both report under "ollama:qwen3:8b";
    # more than the answering route's own tokens proves the classifier's call was captured too.
    assert result.usage_by_model["ollama:qwen3:8b"]["input_tokens"] > 5


def test_an_embedding_arm_adds_the_estimated_embedding_cost() -> None:
    router = ChatRouter(
        routes={"small": _small(), "frontier": _frontier()},
        default_route="small",
        strategy=EmbeddingStrategy(
            DeterministicFakeEmbedding(size=8),
            {"small": ["a simple question"], "frontier": ["a hard analytical question"]},
            threshold=0.0,  # always decides, for a deterministic offline test
        ),
    )
    arm = Arm("embedding", router)

    result = run_item(arm, SINGLE_TURN_ITEM, FakeJudge(), embed=True)

    assert result.error is None
    assert result.cost.embedding_usd > 0.0


def test_a_failing_route_is_recorded_as_an_error_not_raised() -> None:
    router = ChatRouter(routes={"frontier": RaisingModel()}, default_route="frontier")
    arm = Arm("baseline", router)

    result = run_item(arm, SINGLE_TURN_ITEM, FakeJudge(), embed=False)

    assert result.error is not None
    assert "always fails" in result.error
    assert result.judge is None
    assert result.cost.total_usd == 0.0


def test_run_arm_runs_every_item_and_keeps_one_failure_from_losing_the_rest() -> None:
    router = ChatRouter(
        routes={"small": _small(), "frontier": _frontier()},
        default_route="small",
        strategy=HeuristicStrategy("small", "frontier"),
    )
    arm = Arm("heuristic", router)
    items = [SINGLE_TURN_ITEM, AGENT_ITEM]

    results = run_arm(arm, items, FakeJudge(), embed=False, pause_seconds=0.0)

    assert [r.item_id for r in results] == ["st-1", "agent-1"]
    assert all(r.error is None for r in results)


def test_all_five_arm_shapes_can_be_pointed_at_the_full_toolkit_without_error() -> None:
    """Not `build_arms()` itself (that hard-codes real Gemini/Ollama model ids) — the same five
    shapes, on fakes, proving the arm/runner wiring generalises rather than having only been
    exercised on the two or three shapes the tests above happen to cover."""
    small, frontier = _small(), _frontier()
    routes: dict[str, BaseChatModel] = {"small": small, "frontier": frontier}
    shapes: list[tuple[str, ChatRouter, bool]] = [
        ("baseline", ChatRouter(routes={"frontier": frontier}, default_route="frontier"), False),
        (
            "keyword",
            ChatRouter(
                routes=routes,
                default_route="small",
                strategy=KeywordStrategy({"frontier": ["derive", "prove"]}),
            ),
            False,
        ),
        (
            "heuristic",
            ChatRouter(
                routes=routes,
                default_route="small",
                strategy=HeuristicStrategy("small", "frontier"),
            ),
            False,
        ),
        (
            "embedding",
            ChatRouter(
                routes=routes,
                default_route="small",
                strategy=EmbeddingStrategy(
                    DeterministicFakeEmbedding(size=8),
                    {"small": ["simple"], "frontier": ["complex analysis"]},
                    threshold=0.0,
                ),
            ),
            True,
        ),
        (
            "classifier",
            ChatRouter(
                routes=routes,
                default_route="small",
                strategy=ClassifierStrategy(small, {"small": "easy", "frontier": "hard"}),
            ),
            False,
        ),
    ]
    for name, router, embed in shapes:
        arm = Arm(name, router)
        result = run_item(arm, SINGLE_TURN_ITEM, FakeJudge(), embed=embed)
        assert result.error is None, f"{name}: {result.error}"


def test_the_full_toolkit_is_bound_for_every_agent_item() -> None:
    """Sanity check on the fixture set itself: `ALL_TOOLS` really does include the tool every
    agent item in the shipped dataset names as `expected_tool` (caught here, cheaply, rather
    than as a `KeyError` deep in a live run)."""
    from benchmark.dataset import load_dataset

    names = {tool.name for tool in ALL_TOOLS}
    for item in load_dataset():
        if item.expected_tool:
            assert item.expected_tool in names
