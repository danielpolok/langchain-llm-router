"""One real call against a local Ollama server — the one thing offline fakes can't prove:
that `run_item` works against an actual provider's real `usage_metadata` shape, and that
`bind_tools` + the two-step round trip work against a real (if small) model, not a scripted one.

Deliberately avoids Gemini entirely (`arms.ollama_smoke_model()` on both tiers, not the
benchmark's own `small_model()`/`frontier_model()`): free, local and unlimited, so it proves the
harness works against a *real* provider independent of any API budget. Skips unless a local
Ollama server answers (`requires_ollama`, root `conftest.py`).
"""

from __future__ import annotations

import pytest

from benchmark.arms import Arm, ollama_smoke_model
from benchmark.dataset import WorkloadItem, load_dataset
from benchmark.judge import JudgeVerdict
from benchmark.runner import run_item
from benchmark.tests.fakes import FakeJudge
from langchain_llm_router import ChatRouter
from langchain_llm_router.strategies import HeuristicStrategy

pytestmark = pytest.mark.requires_ollama


def _real_dataset_item(item_id: str) -> WorkloadItem:
    (item,) = [i for i in load_dataset() if i.id == item_id]
    return item


def test_a_real_single_turn_call_against_ollama_is_run_and_graded() -> None:
    router = ChatRouter(
        routes={"small": ollama_smoke_model(), "frontier": ollama_smoke_model()},
        default_route="small",
        strategy=HeuristicStrategy("small", "frontier"),
    )
    arm = Arm("ollama-only-smoke", router)
    item = _real_dataset_item("math-easy-st-1")  # "What is 17 + 26?"

    result = run_item(arm, item, FakeJudge(), embed=False)

    assert result.error is None
    assert result.decision is not None
    assert result.answer  # a real model answered something
    assert result.judge == JudgeVerdict(score=8, rationale="fake judge, fixed verdict")
    assert any(model.endswith("qwen3:8b") for model in result.usage_by_model)


def test_a_real_agent_call_against_ollama_completes_the_tool_round_trip() -> None:
    router = ChatRouter(
        routes={"small": ollama_smoke_model(), "frontier": ollama_smoke_model()},
        default_route="small",
        strategy=HeuristicStrategy("small", "frontier"),
    )
    arm = Arm("ollama-only-smoke", router)
    item = _real_dataset_item("math-easy-agent-1")  # "What's 234 * 18?"

    result = run_item(arm, item, FakeJudge(), embed=False)

    assert result.error is None
    assert result.answer
    # A real, non-scripted model: it may or may not reach for the tool, but if it does, the
    # round trip has to have actually completed rather than raised.
    if result.tool_called is not None:
        assert result.tool_correct is not None
