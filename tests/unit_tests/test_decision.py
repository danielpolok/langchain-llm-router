"""T-114: the routing decision record (R1, R2, D3, D8).

What the record holds, the three places a caller finds it — on the response, on the trace, and
through `last_routing_decision()` for the one path that can carry nothing — and the promise it
is the *only* thing the router adds to the route's answer.

Neighbours, not repeated here: the wording of each fallback reason is `test_fallback.py`'s, the
shape of the run tree is `test_routes.py`'s, and "exactly one streamed chunk carries the
record" (REQ-R2-3) is T-113's.

Two of REQ-R2-1's paths cannot be reached yet, and are gaps rather than omissions: a request
diverted off a tool-incapable route (R10) needs `bind_tools`, which is T-115's, and a forced
route (R11) needs the configurable key, which is T-116's — each owns the record's
`diverted_from` / `forced` field on its own path. `generate()` / `agenerate()` are T-117's and
raise today.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnableMap
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    RoutingCallable,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
    last_routing_decision,
    routing_decision,
)
from langchain_llm_router import decision as decision_module
from langchain_llm_router.decision import ROUTING_KEY
from tests.conventions import CONVENTIONS, Convention, respond
from tests.fakes import FakeChatModel

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.tracers.schemas import Run


# --- The router under test, and the records its three decision paths produce ---


class Chooses(RoutingStrategy):
    """A strategy that decides, so the record is a plain choice."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route="frontier", reason="frontier is better at this")


class Abstains(RoutingStrategy):
    """A strategy with no opinion, so R9's default route answers and the record says so.

    `test_fallback.py` owns what falling back *does*; here it is one more record to find.
    """

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        return None


def by_name(request: RoutingRequest) -> str:
    """A policy in one line (REQ-R6-2): the request's text names the route."""
    return request.text


DECIDED = RoutingDecision(route="frontier", reason="frontier is better at this", strategy="Chooses")
FELL_BACK = RoutingDecision(
    route="cheap",
    reason="Abstains could not decide; fell back to the default route",
    strategy="Abstains",
    fallback=True,
)
NO_STRATEGY = RoutingDecision(route="cheap", reason="no strategy configured")

DECISIONS: list[tuple[RoutingStrategy | None, RoutingDecision]] = [
    (None, NO_STRATEGY),
    (Chooses(), DECIDED),
    (Abstains(), FELL_BACK),
]
"""Every decision the router can reach today, and the record each one writes."""

DECISION_IDS = ["no strategy", "chose a route", "fell back"]

TOOL_CALL = ToolCall(name="search", args={"q": "kettles"}, id="call-1", type="tool_call")


def two_routes() -> dict[str, BaseChatModel]:
    """Two fakes, each answering with its own name; `cheap` is the default route."""
    return {
        "cheap": FakeChatModel(model_name="model-cheap", reply="cheap answer"),
        "frontier": FakeChatModel(model_name="model-frontier", reply="frontier answer"),
    }


def router_with(strategy: RoutingStrategy | RoutingCallable | None) -> ChatRouter:
    return ChatRouter(routes=two_routes(), default_route="cheap", strategy=strategy)


Path = Literal["invoke", "ainvoke", "stream", "astream", "batch", "abatch"]

PATHS: tuple[Path, ...] = (*CONVENTIONS, "batch", "abatch")
"""Every entry point the router routes today (`generate` is T-117's)."""


async def answer(
    router: ChatRouter, path: Path, text: str, config: RunnableConfig | None = None
) -> BaseMessage:
    """The router's answer through one entry point; `batch` and `abatch` take one input."""
    if path == "batch":
        return router.batch([text], config)[0]
    if path == "abatch":
        return (await router.abatch([text], config))[0]
    return await respond(router, path, text, config)


def record_of(message: BaseMessage) -> RoutingDecision:
    """The record the response carries — which, on every path tested here, it does."""
    record = routing_decision(message)
    assert record is not None
    return record


def metadata_of(run: Run) -> dict[str, Any]:
    return (run.extra or {}).get("metadata") or {}


def forget_published_record() -> None:
    """Forget what this thread last routed, so a test starts from nothing."""
    decision_module._published.decision = None


@pytest.fixture(autouse=True)
def _forget_published_records() -> Iterator[None]:
    """`last_routing_decision()` outlives the call it answers for, by design (D3) — and would
    outlive the test that made it. Cleared around each one, so what a test reads back is what
    that test routed."""
    forget_published_record()
    yield
    forget_published_record()


# --- R1: the response is the route's own (REQ-R1-1) ---


class Pinned(FakeChatModel):
    """A route that sets its own message ids, and fills in the extras a provider fills in.

    LangChain stamps a message or chunk that arrives without an id with the *run's* id
    (`chat_models.py:2037`, `:795`), which differs between any two calls — including this route
    called directly and the same route called through the router. Real providers send their own
    ids; so does this one, so REQ-R1-1's comparison can cover `id` along with everything else.
    """

    message_id: str = "route-message-1"

    def _message(self) -> AIMessage:
        message = super()._message()
        message.id = self.message_id
        message.name = "assistant"
        message.additional_kwargs = {"refusal": None, "logprobs": {"content": []}}
        message.response_metadata["finish_reason"] = "tool_calls"
        return message

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in super()._stream(messages, stop, run_manager, **kwargs):
            chunk.message.id = self.message_id
            yield chunk


def fields_of(message: BaseMessage) -> dict[str, Any]:
    """Every field the message's own class declares.

    Structural on purpose (REQ-R1-1): a field LangChain adds to `AIMessage` in a later release
    is compared too, without this test being taught about it first.
    """
    return {name: getattr(message, name) for name in type(message).model_fields}


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_response_is_the_route_s_own_message_plus_the_record(
    convention: Convention,
) -> None:
    """REQ-R1-1: content, tool calls, usage, response metadata, id, name, additional kwargs —
    every field of the route's answer is what that route returns when it is called directly,
    and `response_metadata["routing"]` is the one and only difference."""
    route = Pinned(model_name="model-only", reply="an answer", tool_calls=[TOOL_CALL])
    router = ChatRouter(routes={"only": route}, default_route="only")

    direct = await respond(route, convention, "hello")
    routed = await respond(router, convention, "hello")

    expected = fields_of(direct)
    expected["response_metadata"] = {
        **direct.response_metadata,
        ROUTING_KEY: RoutingDecision(route="only", reason="no strategy configured").as_dict(),
    }
    assert fields_of(routed) == expected
    assert type(routed) is type(direct)  # a chunk stays a chunk; nothing is re-wrapped


async def test_nothing_is_added_to_the_route_s_answer_when_the_router_falls_back() -> None:
    """REQ-R1-1: the promise holds on the R9 path too — a fallback changes which route
    answers, not what its answer looks like."""
    route = Pinned(model_name="model-cheap", reply="cheap answer")
    router = ChatRouter(
        routes={"cheap": route, "frontier": FakeChatModel()},
        default_route="cheap",
        strategy=Abstains(),
    )

    direct = route.invoke("hello")
    with pytest.warns(FallbackWarning):
        routed = router.invoke("hello")

    expected = fields_of(direct)
    expected["response_metadata"] = {
        **direct.response_metadata,
        ROUTING_KEY: FELL_BACK.as_dict(),
    }
    assert fields_of(routed) == expected


# --- R2: every response carries the record (REQ-R2-1) ---


@pytest.mark.parametrize(("strategy", "expected"), DECISIONS, ids=DECISION_IDS)
@pytest.mark.parametrize("path", PATHS)
async def test_every_entry_point_answers_with_the_record(
    path: Path, strategy: RoutingStrategy | None, expected: RoutingDecision
) -> None:
    """REQ-R2-1: every response carries the D8 record — through each of the six entry points
    the router routes, and whether the strategy chose, could not decide (R9), or there was none
    to consult."""
    router = router_with(strategy)

    with warnings.catch_warnings():
        # The R9 warning itself is `test_fallback.py`'s; here only the record is under test.
        warnings.simplefilter("ignore", FallbackWarning)
        message = await answer(router, path, "hello")

    assert routing_decision(message) == expected


def test_the_record_is_the_six_field_schema() -> None:
    """REQ-R2-1, D8: what rides on the message is the pinned schema — those six fields, in
    that order, and nothing else."""
    record = RoutingDecision(route="cheap", reason="why").as_dict()

    assert record == {
        "route": "cheap",
        "reason": "why",
        "strategy": None,
        "fallback": False,
        "forced": False,
        "diverted_from": None,
    }
    assert tuple(record) == ("route", "reason", "strategy", "fallback", "forced", "diverted_from")


# --- R2: the same record is on the trace, where D9 puts it (REQ-R2-2) ---


@pytest.mark.parametrize(("strategy", "expected"), DECISIONS, ids=DECISION_IDS)
@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_trace_carries_the_whole_record_in_each_of_d9_s_places(
    convention: Convention, strategy: RoutingStrategy | None, expected: RoutingDecision
) -> None:
    """REQ-R2-2, D9: the strategy run's output, the router run's outputs and the selected
    route's run metadata each hold the same record, all six fields of it — and it is on no
    run's *start* metadata, because the router's run opens before the decision is made."""
    router = router_with(strategy)
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FallbackWarning)
        message = await respond(router, convention, "hello", {"callbacks": [collector]})

    record = expected.as_dict()
    assert routing_decision(message) == expected
    (router_run,) = collector.traced_runs
    *strategy_runs, route_run = router_run.child_runs
    assert (router_run.outputs or {})[ROUTING_KEY] == record
    assert metadata_of(route_run)[ROUTING_KEY] == record
    assert [run.outputs for run in strategy_runs] == ([] if strategy is None else [record])
    carry_it_at_the_start = [
        run.name for run in (router_run, *strategy_runs) if ROUTING_KEY in metadata_of(run)
    ]
    assert carry_it_at_the_start == []


# --- R2: the structured-output answer (REQ-R2-4, D3) ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_last_decision_is_the_one_the_response_carries(
    convention: Convention,
) -> None:
    """REQ-R2-4, D3: nothing routed, nothing to report; after a routed call in the caller's own
    context, `last_routing_decision()` is that call's record — the same one the response
    carries, not a summary of it."""
    router = router_with(Chooses())

    assert last_routing_decision() is None
    message = await respond(router, convention, "hello")

    assert last_routing_decision() == record_of(message) == DECIDED


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_parsed_only_call_is_covered_by_the_last_decision(
    convention: Convention,
) -> None:
    """REQ-R2-4, D3: `with_structured_output` without `include_raw=True` hands the caller a
    parsed object with nowhere to carry a record, so `last_routing_decision()` is the answer.

    That path is `llm | output_parser` (`chat_models.py:2565`), and a `RunnableSequence` runs
    each step in a *copy* of the caller's context — `context.run(step.invoke, …)`
    (`runnables/base.py:3454`), and a task carrying the copy for `ainvoke` (`:3496`). A record
    published to a context variable inside the router is discarded there, so this is the shape
    that proves the record still reaches the caller. `ChatRouter.with_structured_output` is
    T-115's; it inherits the mechanism by going through the same entry points.

    The other half of REQ-R2-4's answer is the trace, which is the one placement always
    available (D3): the router's own run carries the record here as it does anywhere else.
    """
    parsing = router_with(Chooses()) | StrOutputParser()
    collector = RunCollectorCallbackHandler()
    config: RunnableConfig = {"callbacks": [collector]}

    if convention == "invoke":
        parsed = parsing.invoke("hello", config)
    elif convention == "ainvoke":
        parsed = await parsing.ainvoke("hello", config)
    elif convention == "stream":
        parsed = "".join(parsing.stream("hello", config))
    else:
        parsed = "".join([chunk async for chunk in parsing.astream("hello", config)])

    assert parsed == "frontier answer"
    assert last_routing_decision() == DECIDED
    (sequence_run,) = collector.traced_runs
    router_run, _parser_run = sequence_run.child_runs
    assert (router_run.outputs or {})[ROUTING_KEY] == DECIDED.as_dict()


def test_a_deeper_call_supersedes_the_record_a_direct_one_left() -> None:
    """REQ-R2-4, D3: a call made a step deeper publishes its record in the same place a direct
    call does, so the answer is always the most recent routed call — never an older one that
    happened to be made closer to the caller."""
    router = router_with(by_name)

    router.invoke("cheap")
    (router | StrOutputParser()).invoke("frontier")

    assert last_routing_decision() == RoutingDecision(
        route="frontier", reason="by_name chose 'frontier'", strategy="by_name"
    )


def test_the_raw_message_carries_the_record_when_structured_output_asks_for_it() -> None:
    """REQ-R2-4, D3: `include_raw=True` needs no escape hatch. LangChain builds
    `RunnableMap(raw=llm) | …` (`chat_models.py:2564`), so what comes back under `"raw"` is the
    router's own message, record and all — this is the shape T-115's override will produce."""
    raw = RunnableMap(raw=router_with(Chooses())).invoke("hello")["raw"]

    assert record_of(raw) == DECIDED


def test_a_record_published_on_a_worker_thread_does_not_reach_the_caller() -> None:
    """D3's documented limit: `batch` over several inputs runs each in a worker thread, and a
    record survives a copied context only as far as the thread it was routed on. So the caller
    reads back the last call it made *itself* — here an older one, which is the sharp edge the
    docstring warns about. Each response carries its own record: after a batch, that is what to
    read."""
    router = router_with(by_name)

    router.invoke("cheap")
    messages = router.batch(["frontier", "frontier"])

    assert [record_of(message).route for message in messages] == ["frontier", "frontier"]
    assert last_routing_decision() == RoutingDecision(
        route="cheap", reason="by_name chose 'cheap'", strategy="by_name"
    )


async def test_calls_that_overlap_on_one_thread_answer_with_the_last_to_return() -> None:
    """D3's other documented limit: routed calls that overlap on a single thread overwrite each
    other's published record, so `last_routing_decision()` names one of them rather than
    yours. The record on each response is the one that stays right."""
    router = router_with(by_name)

    frontier, cheap = await asyncio.gather(router.ainvoke("frontier"), router.ainvoke("cheap"))

    assert (record_of(frontier).route, record_of(cheap).route) == ("frontier", "cheap")
    last = last_routing_decision()
    assert last is not None
    assert last.route in {"frontier", "cheap"}


# --- Reading a record back off a message (R2) ---


def test_a_message_that_carries_no_record_reads_back_as_nothing() -> None:
    """R2: `routing_decision` answers for any message, including one no router ever saw."""
    assert routing_decision(AIMessage(content="hello")) is None


@pytest.mark.parametrize(
    "foreign",
    ["frontier", ["frontier"], 7, None],
    ids=["a string", "a list", "a number", "null"],
)
def test_a_foreign_routing_value_is_not_read_as_a_record(foreign: object) -> None:
    """R2: ours is a mapping. Another library's `"routing"` key is left alone, not guessed
    at — a router's record is the only thing `routing_decision` claims to know."""
    message = AIMessage(content="hello", response_metadata={ROUTING_KEY: foreign})

    assert routing_decision(message) is None


def test_a_record_with_keys_this_version_does_not_know_reads_back_anyway() -> None:
    """R2: a record written by a later release — one with a seventh field — still reads back
    as the six fields this release has, rather than raising in the caller's code."""
    message = AIMessage(
        content="hello",
        response_metadata={ROUTING_KEY: {**DECIDED.as_dict(), "confidence": 0.91}},
    )

    assert routing_decision(message) == DECIDED


def test_the_record_round_trips_through_the_dict_it_rides_as() -> None:
    """D8: `as_dict()` is what rides on the message and the trace, and `from_dict` is its
    inverse — including the two fields no path sets yet (T-115's and T-116's)."""
    decided = RoutingDecision(
        route="cheap",
        reason="frontier cannot use the bound tools",
        strategy="Chooses",
        fallback=True,
        forced=True,
        diverted_from="frontier",
    )

    assert RoutingDecision.from_dict(decided.as_dict()) == decided
