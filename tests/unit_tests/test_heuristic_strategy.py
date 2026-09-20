"""T-131: the ready-made heuristic strategy — REQ-R7-1 and REQ-R6-1.

Cost tiering without extra calls (PRD §4): each signal is checked on its own, then the score's
tier boundaries, then a corpus of realistic requests, then the whole thing through `ChatRouter`.

Two threads run through the module:

- **REQ-R7-1** — `model_calls` is autouse, so *every* test here runs with a counter on every
  LLM start in the process. A call the strategy made would sit under the strategy's run, or
  under no run at all; only a call the router made on a route's behalf is allowed, and the
  teardown of each test asserts as much. `test_the_no_call_check_catches_a_call` shows the
  check is not vacuous.
- **REQ-R6-1** — nothing below reaches into the package: the strategy is built, configured and
  routed with names `langchain_llm_router` exports. (`_extraction.build_request` is the one
  exception, and it is the *router's* work, done here so the requests under test are the ones a
  router would really hand over.)

The corpus in `REALISTIC` is what T-140 will argue with: it names, per request, the signals that
fired and the tier they add up to.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextvars import ContextVar
from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.context import register_configure_hook
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

import langchain_llm_router
from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    RoutingChoice,
    RoutingError,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    routing_decision,
)
from langchain_llm_router._extraction import build_request
from langchain_llm_router.strategies.heuristic import (
    DEFAULT_LENGTH_RANGE,
    DEFAULT_SIGNALS,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    HeuristicStrategy,
    Signal,
    analysis_signal,
    code_signal,
    length_signal,
    modality_signal,
    parts_signal,
)
from tests.conventions import CONVENTIONS, Convention, respond
from tests.fakes import FakeChatModel, call_log
from tests.tracing import model_runs, priced_calls, walk

TIERS = ("small", "frontier")
IMAGE: dict[str, Any] = {"type": "image", "url": "https://example.com/diagram.png"}
MODALITIES = frozenset({"text", "image", "audio", "video", "file", "other"})
"""`RoutingRequest.modalities`' closed vocabulary (C7), which `modality_signal` reads."""


def make_request(
    text: str = "",
    *,
    blocks: Sequence[str | dict[Any, Any]] | None = None,
    routes: tuple[str, ...] = TIERS,
) -> RoutingRequest:
    """The request the router would build from a one-turn conversation (R4)."""
    message = HumanMessage(content=list(blocks)) if blocks is not None else HumanMessage(text)
    request = build_request(
        [message],
        routes=routes,
        tools_bound=False,
        wants_full_context=False,
        config=RunnableConfig(),
    )
    assert request is not None
    return request


def carrying(*modalities: str) -> RoutingRequest:
    """A request that carries these modalities, without minting content to match.

    `modality_signal` reads `modalities` and nothing else, so the vocabulary can be covered
    exhaustively here; `test_a_real_image_request_scores_the_image` checks a real one.
    """
    return RoutingRequest(
        text="Look:" if "text" in modalities else "",
        content_blocks=[],
        modalities=frozenset(modalities),
        routes=TIERS,
        tools_bound=False,
    )


def fixed(strength: float) -> Signal:
    """A signal of a fixed strength, so a test can put the score exactly on a boundary."""
    return lambda request: strength


# --- REQ-R7-1 · no model, embedding or API call, in any test in this module ---


class ModelCallCounter(BaseCallbackHandler):
    """Every LLM start in the process, and the chain runs that surround it.

    A route's call is the router's own: its parent run is the router's chain run (D9). Anything
    else — a call under the strategy's child run, or one with no run around it at all, as a
    direct `decide` in a test would make — is a call the *strategy* made, which R7 forbids.
    """

    def __init__(self) -> None:
        self.chains: dict[UUID, tuple[str, UUID | None]] = {}
        self.starts: list[UUID | None] = []

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.chains[run_id] = (str(kwargs.get("name") or ""), parent_run_id)

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.starts.append(parent_run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.starts.append(parent_run_id)

    def opened_by(self, parent: UUID | None) -> list[str]:
        """The names of the chain runs around a call, innermost first."""
        names: list[str] = []
        while parent is not None and parent in self.chains:
            name, parent = self.chains[parent]
            names.append(name)
        return names

    @property
    def outside_a_route(self) -> list[str]:
        """Where every call that isn't a route's was made — empty is REQ-R7-1 holding."""
        return [
            "/".join(reversed(self.opened_by(parent))) or "(no run)"
            for parent in self.starts
            if self.opened_by(parent)[:1] != ["ChatRouter"]
        ]

    def forget(self) -> None:
        """Drop what was recorded — for the one test that makes a call on purpose."""
        self.starts.clear()


_counter: ContextVar[ModelCallCounter | None] = ContextVar("heuristic_model_calls", default=None)
# LangChain's own hook for a handler that should see every run in this context, whoever
# configured it — how `get_usage_metadata_callback` counts tokens across unrelated models.
register_configure_hook(_counter, inheritable=True)


@pytest.fixture(autouse=True)
def model_calls() -> Iterator[ModelCallCounter]:
    """REQ-R7-1: count every LLM start any test in this module causes, and require that the
    only ones are the route calls the router itself made."""
    counter = ModelCallCounter()
    token = _counter.set(counter)
    try:
        yield counter
    finally:
        _counter.reset(token)
    assert counter.outside_a_route == []


# --- The signals, one at a time ---


@pytest.mark.parametrize(
    ("words", "strength"),
    [(0, 0.0), (1, 0.0), (20, 0.0), (65, 0.25), (110, 0.5), (200, 1.0), (600, 1.0)],
)
def test_the_length_signal_ramps_between_its_bounds(words: int, strength: float) -> None:
    """REQ-R7-1: length is a word count — no call — rising from `DEFAULT_LENGTH_RANGE[0]`
    words to `[1]`, silent below and capped above."""
    request = make_request(" ".join(["word"] * words))

    assert DEFAULT_LENGTH_RANGE == (20, 200)
    assert length_signal()(request) == strength


def test_the_length_bounds_are_overridable() -> None:
    """REQ-R7-1: the ramp is a factory, so T-140 retunes it with two numbers through
    `signals=` rather than by editing the strategy."""
    assert length_signal(0, 10)(make_request("one two three four five")) == 0.5


@pytest.mark.parametrize(("floor", "ceiling"), [(-1, 10), (10, 10), (10, 5)])
def test_impossible_length_bounds_are_refused(floor: int, ceiling: int) -> None:
    """A ramp that can't rise is a configuration error, raised where it is written."""
    with pytest.raises(RoutingError, match=r"^length_signal needs 0 <= floor < ceiling"):
        length_signal(floor, ceiling)


FENCE = "```\nhello\n```"
SYNTAX = "def parse(payload):"
TRACEBACK = "ValueError: Expecting value"


@pytest.mark.parametrize(
    ("text", "strength"),
    [
        pytest.param("Rewrite this paragraph so it reads more warmly.", 0.0, id="prose"),
        pytest.param("A whole class of problems, and the import duties on them.", 0.0, id="near"),
        pytest.param(FENCE, 0.5, id="fence"),
        pytest.param(SYNTAX, 0.5, id="syntax"),
        pytest.param(TRACEBACK, 0.5, id="traceback"),
        pytest.param('  File "parse.py", line 42, in parse', 0.5, id="traceback-frame"),
        pytest.param("SELECT name FROM users WHERE id = 1", 0.5, id="sql"),
        pytest.param(f"{FENCE}\n{SYNTAX}", 1.0, id="fence-and-syntax"),
        pytest.param(f"{FENCE}\n{SYNTAX}\n{TRACEBACK}", 1.0, id="all-three-capped"),
    ],
)
def test_the_code_signal_counts_marker_families(text: str, strength: float) -> None:
    """REQ-R7-1: code is spotted by family — a fence, code syntax, a traceback — so one long
    paste doesn't outweigh a short snippet with an error. Prose that merely says "class" or
    "import" is not code."""
    assert code_signal(make_request(text)) == strength


@pytest.mark.parametrize(
    ("text", "strength"),
    [
        pytest.param("Summarise this.", 0.0, id="statement"),
        pytest.param("What time is it in Tokyo?", 0.0, id="one-question"),
        pytest.param("Where is it? And what does it cost?", 0.5, id="two-questions"),
        pytest.param("Where is it and what does it cost?", 0.0, id="two-in-one-sentence"),
        pytest.param("Who? What? Where? When?", 1.0, id="four-questions"),
        pytest.param("Do these:\n1. back it up\n2. restore it", 0.5, id="numbered"),
        pytest.param("Please:\n- one\n- two\n- three", 1.0, id="bulleted"),
        pytest.param("1. Why is it slow?\n2. How do I fix it?", 0.5, id="numbered-questions"),
    ],
)
def test_the_parts_signal_counts_questions_or_enumerated_items(text: str, strength: float) -> None:
    """REQ-R7-1: the larger of the two counts, not their sum — a numbered list of questions is
    one list of two parts, not four — and the first part is free. Punctuation is all it has:
    two questions sharing one question mark read as one part (a limit T-140 may weigh)."""
    assert parts_signal(make_request(text)) == strength


@pytest.mark.parametrize(
    ("text", "strength"),
    [
        pytest.param("What time is it in Tokyo?", 0.0, id="recall"),
        pytest.param("Explain why this happens.", 0.5, id="one-term"),
        pytest.param("Compare and then compare again.", 0.5, id="repetition-counts-once"),
        pytest.param("Compare the trade-offs.", 1.0, id="two-terms"),
        pytest.param("Analyse the design and justify the optimisation.", 1.0, id="capped"),
    ],
)
def test_the_analysis_signal_counts_distinct_terms(text: str, strength: float) -> None:
    """REQ-R7-1: words that ask for reasoning rather than recall, counted once each —
    repetition is emphasis, not a second thing to reason about."""
    assert analysis_signal(make_request(text)) == strength


@pytest.mark.parametrize("modality", sorted(MODALITIES))
def test_the_modality_signal_fires_on_anything_but_text(modality: str) -> None:
    """REQ-R7-1: every modality but `"text"` needs a model that can read it, so the signal is
    binary over the closed vocabulary (C7)."""
    assert modality_signal(carrying(modality)) == (0.0 if modality == "text" else 1.0)
    assert modality_signal(carrying("text", modality)) == (0.0 if modality == "text" else 1.0)


def test_a_real_image_request_scores_the_image() -> None:
    """REQ-R7-1: the signal reads what extraction really produces — an image beside no text is
    an image-only request, which on the default threshold is exactly the frontier tier's bar."""
    request = make_request(blocks=[IMAGE])

    assert request.modalities == frozenset({"image"})
    assert HeuristicStrategy(*TIERS).decide(request) == RoutingChoice(
        "frontier", "difficulty 1.00 >= 1.00 (modalities 1.00)"
    )


def test_the_default_signals_are_the_five_documented_ones() -> None:
    """REQ-R7-1: the set T-140 retunes — and every one of them weighs the same until it has
    evidence for anything else."""
    assert list(DEFAULT_SIGNALS) == ["length", "code", "parts", "analysis", "modalities"]
    assert dict(DEFAULT_WEIGHTS) == dict.fromkeys(DEFAULT_SIGNALS, 1.0)
    assert DEFAULT_THRESHOLD == 1.0


# --- The score's tiers ---


@pytest.mark.parametrize(
    ("strength", "route", "reason"),
    [
        pytest.param(0.0, "small", "difficulty 0.00 < 1.00 (no signal fired)", id="nothing"),
        pytest.param(0.99, "small", "difficulty 0.99 < 1.00 (stub 0.99)", id="just-below"),
        pytest.param(1.0, "frontier", "difficulty 1.00 >= 1.00 (stub 1.00)", id="exactly-on"),
        pytest.param(1.01, "frontier", "difficulty 1.01 >= 1.00 (stub 1.01)", id="just-above"),
    ],
)
def test_a_score_on_the_threshold_takes_the_upper_tier(
    strength: float, route: str, reason: str
) -> None:
    """REQ-R7-1, R2: the threshold is the price of admission to the tier above it, and the
    reason says which side of it the score fell on."""
    strategy = HeuristicStrategy(*TIERS, signals={"stub": fixed(strength)})

    assert strategy.decide(make_request("anything")) == RoutingChoice(route, reason)


@pytest.mark.parametrize(
    ("strength", "route"), [(1.99, "small"), (2.0, "frontier"), (3.0, "frontier")]
)
def test_an_overridden_threshold_moves_the_boundary(strength: float, route: str) -> None:
    """REQ-R7-1: `thresholds=` is how an application raises the bar for its expensive tier —
    and how T-140's number reaches a live router."""
    strategy = HeuristicStrategy(*TIERS, thresholds=[2.0], signals={"stub": fixed(strength)})

    assert strategy.decide(make_request("anything")).route == route  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("strength", "route", "reason"),
    [
        pytest.param(0.5, "small", "difficulty 0.50 < 1.00 (stub 0.50)", id="bottom"),
        pytest.param(1.0, "mid", "difficulty 1.00 in [1.00, 3.00) (stub 1.00)", id="middle-low"),
        pytest.param(2.9, "mid", "difficulty 2.90 in [1.00, 3.00) (stub 2.90)", id="middle-high"),
        pytest.param(3.0, "frontier", "difficulty 3.00 >= 3.00 (stub 3.00)", id="top"),
    ],
)
def test_more_than_two_tiers_use_ascending_bands(strength: float, route: str, reason: str) -> None:
    """REQ-R7-1: the two-tier case generalises — n tiers, n-1 thresholds, each tier taking the
    band below the next, and the reason naming the band."""
    strategy = HeuristicStrategy(
        "small", "mid", "frontier", thresholds=[1.0, 3.0], signals={"stub": fixed(strength)}
    )

    assert strategy.decide(make_request("anything")) == RoutingChoice(route, reason)


def test_a_weight_scales_one_signals_contribution() -> None:
    """REQ-R7-1: `weights=` retunes a signal without replacing any — here analysis alone,
    half on, is tripled past the bar a request it half-fires would not reach."""
    request = make_request("Explain why this happens.")
    weighted = HeuristicStrategy(*TIERS, weights={"analysis": 3.0})

    assert HeuristicStrategy(*TIERS).decide(request) == RoutingChoice(
        "small", "difficulty 0.50 < 1.00 (analysis 0.50)"
    )
    assert weighted.decide(request) == RoutingChoice(
        "frontier", "difficulty 1.50 >= 1.00 (analysis 1.50)"
    )


def test_a_weight_of_zero_silences_a_signal() -> None:
    """REQ-R7-1: an application that routes images itself can turn that signal off without
    touching the others."""
    request = make_request(blocks=[IMAGE])
    strategy = HeuristicStrategy(*TIERS, weights={"modalities": 0.0})

    assert strategy.decide(request) == RoutingChoice(
        "small", "difficulty 0.00 < 1.00 (no signal fired)"
    )


def test_a_custom_signal_set_replaces_the_defaults() -> None:
    """REQ-R7-1: `signals=` is how T-140 tries a set that isn't in the module; a signal with no
    entry in `DEFAULT_WEIGHTS` weighs 1.0 unless `weights=` says otherwise."""
    strategy = HeuristicStrategy(*TIERS, thresholds=[0.5], signals={"words": length_signal(0, 10)})

    assert strategy.decide(make_request("one two three four")) == RoutingChoice(
        "small", "difficulty 0.40 < 0.50 (words 0.40)"
    )
    assert strategy.decide(make_request(" ".join(["word"] * 8))) == RoutingChoice(
        "frontier", "difficulty 0.80 >= 0.50 (words 0.80)"
    )


def test_the_reason_leads_with_the_signal_that_drove_the_score() -> None:
    """R2: a human reading a trace sees the score, the band and what made it — largest
    contribution first, and only the signals that fired."""
    strategy = HeuristicStrategy(
        *TIERS, signals={"minor": fixed(0.25), "major": fixed(1.0), "silent": fixed(0.0)}
    )

    choice = strategy.decide(make_request("anything"))

    assert choice == RoutingChoice("frontier", "difficulty 1.25 >= 1.00 (major 1.00, minor 0.25)")


# --- Realistic requests: the corpus T-140 argues with ---

ONE_LINE_QUESTION = "What's the capital of France?"
SHORT_CREATIVE = "Write a haiku about autumn rain."
SHORT_EXPLANATION = "Explain why the sky is blue."
MULTI_PART_ANALYSIS = (
    "Compare Postgres and DynamoDB for our event store.\n"
    "1. Which handles 50k writes a second more cheaply?\n"
    "2. What are the trade-offs on query flexibility?\n"
    "3. How would you migrate the existing table?"
)
DEBUGGING_PASTE = (
    "My parser blows up on this file and I cannot see why. Here is the traceback:\n\n"
    "Traceback (most recent call last):\n"
    '  File "parse.py", line 42, in parse\n'
    "    return json.loads(payload)\n"
    "ValueError: Expecting value: line 1 column 1 (char 0)\n\n"
    "```python\n"
    "def parse(payload):\n"
    "    return json.loads(payload)\n"
    "```\n" + "I have tried a few things already and none of them helped at all today. " * 6
)

REALISTIC = [
    pytest.param(ONE_LINE_QUESTION, {}, "small", id="one-line-question"),
    pytest.param(SHORT_CREATIVE, {}, "small", id="short-creative"),
    pytest.param(SHORT_EXPLANATION, {"analysis": 0.5}, "small", id="short-explanation"),
    pytest.param(
        MULTI_PART_ANALYSIS,
        {"length": 0.07, "parts": 1.0, "analysis": 1.0},
        "frontier",
        id="multi-part-analysis",
    ),
    pytest.param(DEBUGGING_PASTE, {"length": 0.63, "code": 1.0}, "frontier", id="debugging-paste"),
]


@pytest.mark.parametrize(("text", "fired", "tier"), REALISTIC)
def test_realistic_requests_land_in_the_tier_they_look_like(
    text: str, fired: dict[str, float], tier: str
) -> None:
    """REQ-R7-1: the defaults, on requests of the kind PRD §4 is about. The signals each
    request fires are spelled out so T-140 can see exactly what it is retuning."""
    request = make_request(text)

    scored = {
        name: round(signal(request), 2)
        for name, signal in DEFAULT_SIGNALS.items()
        if signal(request)
    }

    assert scored == fired
    assert HeuristicStrategy(*TIERS).decide(request).route == tier  # type: ignore[union-attr]


# --- Not deciding, and not swapping (R9) ---


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_a_request_with_nothing_to_score_abstains(text: str) -> None:
    """REQ-R7-1, R9: no text and no other modality is nothing to judge, so the strategy returns
    `None` and leaves the default route — and the one warning — to the router."""
    assert HeuristicStrategy(*TIERS).decide(make_request(text)) is None


def test_a_tier_that_is_not_a_route_is_named_not_swapped() -> None:
    """R9, REQ-R9-2: the strategy names the tier its policy chose even when the router has no
    such route. Quietly falling to a neighbouring tier would route on a misconfiguration; the
    router reports it instead, once, with the missing name in the record."""
    strategy = HeuristicStrategy(*TIERS)
    request = make_request(MULTI_PART_ANALYSIS, routes=("small", "huge"))

    assert strategy.decide(request).route == "frontier"  # type: ignore[union-attr]


def test_the_router_reports_a_tier_it_has_no_route_for() -> None:
    """R9, REQ-R9-2: end to end — one `FallbackWarning`, the default route, and a record
    naming the tier that doesn't exist."""
    router = ChatRouter(
        routes={"small": FakeChatModel(reply="small"), "huge": FakeChatModel(reply="huge")},
        default_route="small",
        strategy=HeuristicStrategy(*TIERS),
    )

    with pytest.warns(RoutingWarning) as caught:
        answer = router.invoke(MULTI_PART_ANALYSIS)

    decision = routing_decision(answer)
    assert decision is not None
    assert (decision.route, decision.fallback, decision.strategy) == (
        "small",
        True,
        "HeuristicStrategy",
    )
    assert decision.reason == (
        "HeuristicStrategy chose 'frontier', which is not one of the routes; "
        "fell back to the default route"
    )
    assert [type(warning.message) for warning in caught] == [FallbackWarning]


# --- Bad configuration fails at construction, not per request ---


@pytest.mark.parametrize(
    ("kwargs", "tiers", "message"),
    [
        pytest.param({}, ("small",), r"needs at least two tiers", id="one-tier"),
        pytest.param({}, (), r"needs at least two tiers", id="no-tiers"),
        pytest.param({}, ("small", " "), r"^tier ' ' is blank", id="blank-tier"),
        pytest.param(
            {},
            ("small", "mid", "frontier"),
            r"^3 tiers need 2 thresholds: there is a default",
            id="no-default-past-two-tiers",
        ),
        pytest.param(
            {"thresholds": [1.0, 2.0]}, TIERS, r"^2 tiers need 1 threshold, got 2", id="too-many"
        ),
        pytest.param(
            {"thresholds": [3.0, 1.0]},
            ("small", "mid", "frontier"),
            r"^thresholds must ascend",
            id="descending",
        ),
        pytest.param(
            {"thresholds": [1.0, 1.0]},
            ("small", "mid", "frontier"),
            r"^thresholds must ascend",
            id="equal",
        ),
        pytest.param(
            {"weights": {"lenght": 2.0}},
            TIERS,
            r"^weights names signals that don't exist: 'lenght'",
            id="misspelled-weight",
        ),
        pytest.param(
            {"weights": {"length": -1.0}},
            TIERS,
            r"^weights must not be negative: 'length'",
            id="negative-weight",
        ),
        pytest.param({"signals": {}}, TIERS, r"^signals is empty", id="no-signals"),
    ],
)
def test_configuration_that_could_never_route_fails_at_construction(
    kwargs: dict[str, Any], tiers: tuple[str, ...], message: str
) -> None:
    """REQ-R6-1: a built-in fails like the rest of the package — a `RoutingError` when it is
    built, not a surprise on the first request."""
    with pytest.raises(RoutingError, match=message):
        HeuristicStrategy(*tiers, **kwargs)


# --- REQ-R6-1 · an ordinary public strategy ---


def test_the_builtin_is_an_ordinary_public_strategy() -> None:
    """REQ-R6-1: one interface serves all three levels — the built-in is a concrete
    `RoutingStrategy` exported from the package, on the interface's defaults (R4, C2)."""
    strategy = HeuristicStrategy(*TIERS)

    assert isinstance(strategy, RoutingStrategy)
    assert langchain_llm_router.HeuristicStrategy is HeuristicStrategy
    assert "HeuristicStrategy" in langchain_llm_router.__all__
    assert strategy.wants_full_context is False


async def test_the_async_path_decides_the_same(model_calls: ModelCallCounter) -> None:
    """REQ-R6-1, C2: `adecide` is the interface's default — `decide` in a worker thread — and
    the strategy is thread-safe, so both paths give the same choice."""
    strategy = HeuristicStrategy(*TIERS)
    request = make_request(MULTI_PART_ANALYSIS)

    assert await strategy.adecide(request) == strategy.decide(request)
    assert model_calls.outside_a_route == []


def make_router(strategy: RoutingStrategy | None = None) -> ChatRouter:
    """A two-tier router, each route answering with its own name."""
    routes: dict[str, BaseChatModel] = {
        "small": FakeChatModel(model_name="small", reply="small answered"),
        "frontier": FakeChatModel(model_name="frontier", reply="frontier answered"),
    }
    return ChatRouter(
        routes=routes,
        default_route="small",
        strategy=strategy if strategy is not None else HeuristicStrategy(*TIERS),
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize(
    ("text", "route", "reason"),
    [
        pytest.param(
            ONE_LINE_QUESTION,
            "small",
            "difficulty 0.00 < 1.00 (no signal fired)",
            id="easy",
        ),
        pytest.param(
            MULTI_PART_ANALYSIS,
            "frontier",
            "difficulty 2.07 >= 1.00 (parts 1.00, analysis 1.00, length 0.07)",
            id="hard",
        ),
    ],
)
async def test_the_router_answers_from_the_tier_the_score_picked(
    convention: Convention, text: str, route: str, reason: str
) -> None:
    """REQ-R6-1, R2: the strategy plugged into `ChatRouter` through `strategy=` alone — the
    scored tier answers, on every entry point, and its reason is what the record carries."""
    router = make_router()
    other = "frontier" if route == "small" else "small"

    answer = await respond(router, convention, text)

    decision = routing_decision(answer)
    assert decision is not None
    assert (decision.route, decision.reason, decision.strategy, decision.fallback) == (
        route,
        reason,
        "HeuristicStrategy",
        False,
    )
    assert answer.text == f"{route} answered"
    assert len(call_log(router.routes[route])) == 1
    assert call_log(router.routes[other]) == []


def test_an_empty_request_falls_back_through_the_router() -> None:
    """R9: the strategy's `None` becomes the router's one `FallbackWarning` and a record
    saying it could not decide — the strategy neither warns nor raises itself."""
    router = make_router()

    with pytest.warns(RoutingWarning) as caught:
        answer = router.invoke([HumanMessage("")])

    decision = routing_decision(answer)
    assert decision is not None
    assert (decision.route, decision.fallback) == ("small", True)
    assert decision.reason == "HeuristicStrategy could not decide; fell back to the default route"
    assert [type(warning.message) for warning in caught] == [FallbackWarning]


# --- REQ-R7-1 · the check itself ---


def test_the_strategy_adds_no_run_of_its_own_to_the_trace(
    model_calls: ModelCallCounter,
) -> None:
    """REQ-R7-1: the trace is the visible half of the counter — the strategy's run has no
    children, and the only priced call is the route's (R3)."""
    collector = RunCollectorCallbackHandler()

    make_router().invoke(MULTI_PART_ANALYSIS, {"callbacks": [collector]})

    (root,) = collector.traced_runs
    strategy_runs = [run for run in walk([root]) if run.name == "HeuristicStrategy"]
    assert [run.child_runs for run in strategy_runs] == [[]]
    assert [run.parent_run_id for run in model_runs([root])] == [root.id]
    assert len(priced_calls([root])) == 1
    assert model_calls.outside_a_route == []


class CallsAModel(RoutingStrategy):
    """What REQ-R7-1 forbids: a strategy that asks a model, as T-134's will (REQ-R7-2)."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        self.model.invoke(request.text, config=request.config)
        return RoutingChoice("small", "the model said so")


def test_the_no_call_check_catches_a_call(model_calls: ModelCallCounter) -> None:
    """REQ-R7-1: the autouse counter is not vacuous — a strategy that does call a model is
    caught, under its own run and with no run at all."""
    strategy = CallsAModel(FakeChatModel())

    make_router(strategy).invoke(ONE_LINE_QUESTION)
    strategy.decide(make_request(ONE_LINE_QUESTION))

    assert model_calls.outside_a_route == ["ChatRouter/CallsAModel", "(no run)"]
    # Deliberate: forgotten so this test's own calls don't fail the module-wide check.
    model_calls.forget()


def test_the_defaults_blind_spots_are_the_ones_we_know_of() -> None:
    """R7, T-140: what local signals cannot see, pinned so the benchmark argues with numbers.

    Not a specification of good routing — the opposite. Each case is a request the defaults
    score wrongly, kept here so T-140 can measure whether retuning fixes it.
    """
    strategy = HeuristicStrategy("small", "frontier")

    def route_for(text: str) -> str:
        request = build_request(
            [HumanMessage(text)],
            routes=("small", "frontier"),
            tools_bound=False,
            wants_full_context=False,
            config=RunnableConfig(),
        )
        assert request is not None
        choice = strategy.decide(request)
        assert choice is not None
        return choice.route

    # A script without spaces between words: "compare Transformers with RNNs on long
    # dependencies, and analyse the computational cost" — hard, and scored 0.00.
    chinese = "请比较 Transformer 与 RNN 在长依赖建模上的差异，并分析计算成本。"  # noqa: RUF001
    assert route_for(chinese) == "small"
    # Short and genuinely hard: one analysis term where a full signal needs two.
    assert route_for("Prove that if P != NP then one-way functions exist.") == "small"
    # Long and trivial: length alone clears the bar.
    pasted_log = "Here is my log:\n" + "INFO served in 12ms\n" * 60 + "what does this mean?"
    assert route_for(pasted_log) == "frontier"
