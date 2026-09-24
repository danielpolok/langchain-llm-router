"""Every way of calling a chat model works on the router, and only those.

`invoke`, `ainvoke`, `stream` and `astream` are the router's own; `batch`, `abatch` and
`astream_events` come from `Runnable` and reach the route through them. The streaming protocol
`langchain-core` 1.4 added — `stream_events(version="v3")` — does not, and is refused here
rather than silently bypassing the router; `router._V3_UNSUPPORTED` says why.
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ConfigDict

from langchain_llm_router import (
    ChatRouter,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    routing_decision,
)
from langchain_llm_router.decision import ROUTING_KEY
from tests.conventions import ALL_CONVENTIONS, AnyConvention, respond
from tests.fakes import FakeChatModel, GenerateOnlyFakeChatModel, call_log

STREAMING: tuple[AnyConvention, ...] = ("stream", "astream", "events")
"""The conventions that hand back chunks rather than a finished message."""

_TIMEOUT = 10.0
"""Seconds to wait for something that should already have happened. Generous: it only bounds
how long a broken test hangs before failing, and never how fast a passing one runs."""

_TICKS = 50
"""Loop iterations that count as "the event loop has run everything it can". Nothing waits on
wall-clock time here — a task that needs more than 50 turns to reach its first real await is
not slow, it is stuck."""

_BATCH = 4
"""Requests per batch, and the width of the gates below."""

_ALONE = 2.0
"""Seconds a thread waits for company before concluding it is running alone. The one
wall-clock number here, and only in the direction of declaring a *failure*: threads that do run
together meet at once, however slow the machine."""


class ByText(RoutingStrategy):
    """Routes to the route the request's text names."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route=request.text, reason=f"the request named {request.text!r}")


def fake_routes(*names: str) -> dict[str, BaseChatModel]:
    """One fake route per name, each answering with its own name."""
    return {
        name: FakeChatModel(model_name=f"model-{name}", reply=f"{name} answer here")
        for name in names
    }


def by_text(*names: str) -> ChatRouter:
    """A router over `names` whose strategy routes on the request's text."""
    return ChatRouter(routes=fake_routes(*names), default_route=names[0], strategy=ByText())


def record_for(route: str) -> dict[str, Any]:
    """The decision record `ByText` produces for `route`."""
    return RoutingDecision(
        route=route, reason=f"the request named {route!r}", strategy="ByText"
    ).as_dict()


def fields_of(message: AIMessage) -> dict[str, Any]:
    """What a caller reads off an answer, leaving out what streaming necessarily changes.

    `id` and `type` are the two: LangChain gives a streamed message its own run id and the
    `AIMessageChunk` type. Both differ for a bare chat model exactly as they do for a router,
    which `test_streaming_differs_from_invoke_only_where_langchain_makes_it_differ` pins.
    """
    return {
        "content": message.content,
        "response_metadata": dict(message.response_metadata),
        "usage_metadata": message.usage_metadata,
        "additional_kwargs": dict(message.additional_kwargs),
        "tool_calls": list(message.tool_calls),
        "invalid_tool_calls": list(message.invalid_tool_calls),
        "name": message.name,
    }


# --- Every convention returns the route's output ---


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_every_calling_convention_returns_the_selected_route_s_output(
    convention: AnyConvention,
) -> None:
    """`invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch` and
    `astream_events` all answer with the selected route's own message and its record — there is
    no extra call form to learn, and no convention that routes differently."""
    routes = fake_routes("cheap", "frontier")
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())

    message = await respond(router, convention, "frontier")

    assert message.content == "frontier answer here"
    assert message.response_metadata[ROUTING_KEY] == record_for("frontier")
    assert message.usage_metadata == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
    assert (len(call_log(routes["cheap"])), len(call_log(routes["frontier"]))) == (0, 1)


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_the_router_answers_exactly_as_the_route_would(convention: AnyConvention) -> None:
    """ "no router-specific call form" cuts both ways — the same call on the route
    itself gives the same answer, bar the record the router adds."""
    router = by_text("frontier")
    direct = FakeChatModel(model_name="model-frontier", reply="frontier answer here")

    through_router = await respond(router, convention, "frontier")
    straight = await respond(direct, convention, "frontier")

    assert fields_of(through_router) == {
        **fields_of(straight),
        "response_metadata": {
            **straight.response_metadata,
            ROUTING_KEY: record_for("frontier"),
        },
    }


@pytest.mark.parametrize("convention", STREAMING)
async def test_merged_stream_chunks_equal_the_invoke_output(convention: AnyConvention) -> None:
    """Merging what a deterministic route streams rebuilds what `invoke` returned —
    every field of it, not only the text."""
    router = by_text("frontier")

    invoked = await respond(router, "invoke", "frontier")
    merged = await respond(router, convention, "frontier")

    assert fields_of(merged) == fields_of(invoked)


@pytest.mark.parametrize("convention", STREAMING)
async def test_streaming_differs_from_invoke_only_where_langchain_makes_it_differ(
    convention: AnyConvention,
) -> None:
    """`id` and `type` are the two fields `fields_of` leaves out. They differ for a
    bare chat model too, so the router adds no difference of its own."""
    direct = FakeChatModel(model_name="model-frontier", reply="frontier answer here")
    router = by_text("frontier")

    streamed = await respond(router, convention, "frontier")
    straight = await respond(direct, convention, "frontier")

    assert (streamed.type, straight.type) == ("AIMessageChunk", "AIMessageChunk")
    invoked = await respond(router, "invoke", "frontier")
    assert invoked.type == "ai"
    assert streamed.id != invoked.id  # each run names its own message


async def test_v2_events_show_the_router_s_chain_run_around_one_model_run() -> None:
    """An event stream reports what the trace does — the router's chain run
    around the strategy's, and the route's call as the only model run in the call."""
    router = by_text("cheap", "frontier")

    events = [
        (event["event"], event["name"])
        async for event in router.astream_events("frontier", version="v2")
    ]

    assert events[0] == ("on_chain_start", "ChatRouter")
    assert events[-1] == ("on_chain_end", "ChatRouter")
    assert [name for kind, name in events if kind == "on_chat_model_start"] == ["FakeChatModel"]
    assert [name for kind, name in events if kind == "on_chain_start"] == [
        "ChatRouter",
        "ByText",
    ]


# --- A route that cannot stream ---


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_a_route_without_native_streaming_still_streams(convention: AnyConvention) -> None:
    """A route that implements only `_generate` yields one chunk, equal to its whole
    message — LangChain's own fallback, which the router passes through rather than replaces."""
    route = GenerateOnlyFakeChatModel(model_name="model-only", reply="one shot answer")
    router = ChatRouter(routes={"only": route}, default_route="only")

    if convention == "stream":
        chunks = list(router.stream("hello"))
    else:
        chunks = [chunk async for chunk in router.astream("hello")]

    [only] = chunks
    assert only.content == "one shot answer"
    assert len(call_log(route)) == 1  # one call: `invoke` under LangChain's fallback
    assert fields_of(only) == fields_of(router.invoke("hello"))


async def test_a_route_without_native_streaming_streams_what_it_would_have_returned() -> None:
    """The single chunk is the route's message with the record added, and nothing
    else about it changed."""
    answered = AIMessage(content="one shot answer", id="route-message")
    route = GenerateOnlyFakeChatModel(model_name="model-only", script=[answered])
    router = ChatRouter(routes={"only": route}, default_route="only")
    record = RoutingDecision(route="only", reason="no strategy configured").as_dict()

    [only] = list(router.stream("hello"))

    assert (only.content, only.id) == ("one shot answer", "route-message")
    assert only.response_metadata == {"model_name": "model-only", ROUTING_KEY: record}
    assert ROUTING_KEY not in answered.response_metadata  # the record went on a copy


# --- One record, however many chunks ---


@pytest.mark.parametrize("convention", STREAMING)
async def test_merging_every_chunk_yields_one_decision_record(convention: AnyConvention) -> None:
    """Exactly one chunk carries the record, so merging the stream leaves the
    record itself — `merge_dicts` concatenates a string repeated across chunks, and a record on
    every chunk would come out as `'frontierfrontier…'`."""
    router = by_text("cheap", "frontier")
    record = record_for("frontier")

    merged = await respond(router, convention, "frontier")

    assert merged.response_metadata[ROUTING_KEY] == record
    assert merged.response_metadata[ROUTING_KEY]["route"] == "frontier"
    assert merged.response_metadata[ROUTING_KEY]["reason"] == record["reason"]


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_only_one_streamed_chunk_carries_the_record(convention: AnyConvention) -> None:
    """The first chunk carries it and the rest carry nothing — what makes merging
    above come out as one record rather than a concatenation."""
    router = by_text("cheap", "frontier")

    if convention == "stream":
        chunks = list(router.stream("frontier"))
    else:
        chunks = [chunk async for chunk in router.astream("frontier")]

    carried = [chunk.response_metadata.get(ROUTING_KEY) for chunk in chunks]
    assert carried[0] == record_for("frontier")
    assert carried[1:] == [None] * (len(chunks) - 1)
    assert len(chunks) > 1  # otherwise there is nothing for merging to concatenate


# --- Nothing blocks the event loop ---


class Parked(RoutingStrategy):
    """A strategy that waits in `adecide` until the test releases it, without blocking."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        msg = "the async path must await adecide, not fall back to decide"
        raise AssertionError(msg)

    async def adecide(self, request: RoutingRequest) -> RoutingChoice:
        self.entered.set()
        await self.release.wait()
        return RoutingChoice(route="frontier", reason="released")


class Blocking(RoutingStrategy):
    """A strategy with only a synchronous `decide`, which blocks the thread it runs on.

    The router must keep it off the event loop — `RoutingStrategy.adecide` runs `decide` in an
    executor, and an async entry point that called `decide` directly would stall everything.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        self.entered.set()
        if not self.release.wait(timeout=_TIMEOUT):  # pragma: no cover - only on failure
            msg = "the test never released the strategy"
            raise AssertionError(msg)
        return RoutingChoice(route="frontier", reason="released")


async def ticks(count: int = _TICKS) -> int:
    """Run `count` turns of the event loop, counting the ones that actually happened.

    A task that had the loop to itself finishes this; one waiting behind blocked work does not.
    """
    done = 0
    for _ in range(count):
        done += 1
        await asyncio.sleep(0)
    return done


async def test_an_async_strategy_that_waits_does_not_stall_a_concurrent_task() -> None:
    """While `adecide` is parked, another task runs to completion and the routed
    call is still pending — the router awaited the strategy rather than blocking on it."""
    strategy = Parked()
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=strategy
    )

    call = asyncio.create_task(router.ainvoke("hello"))
    await asyncio.wait_for(strategy.entered.wait(), _TIMEOUT)
    bystander = await ticks()

    assert (bystander, call.done()) == (_TICKS, False)
    strategy.release.set()
    message = await asyncio.wait_for(call, _TIMEOUT)
    assert routing_decision(message) == RoutingDecision(
        route="frontier", reason="released", strategy="Parked"
    )


async def test_a_blocking_strategy_runs_off_the_event_loop() -> None:
    """A strategy with only a synchronous `decide` blocks a worker thread, not the
    loop — the default `adecide` puts it in an executor, and a concurrent task keeps
    running while it sits there."""
    strategy = Blocking()
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=strategy
    )

    call = asyncio.create_task(router.ainvoke("hello"))
    loop = asyncio.get_running_loop()
    entered = await loop.run_in_executor(None, strategy.entered.wait, _TIMEOUT)
    bystander = await ticks()

    assert (entered, bystander, call.done()) == (True, _TICKS, False)
    strategy.release.set()
    message = await asyncio.wait_for(call, _TIMEOUT)
    assert routing_decision(message) == RoutingDecision(
        route="frontier", reason="released", strategy="Blocking"
    )


class Gate:
    """Lets calls through only once `width` of them are waiting at the same time.

    A batch that genuinely overlaps opens it; one that runs its requests in turn never does, so
    the test fails on its timeout rather than passing on a guess about how long things take.
    """

    def __init__(self, width: int) -> None:
        self.width = width
        self.waiting = 0
        self.peak = 0
        self._opened: asyncio.Event | None = None

    def _event(self) -> asyncio.Event:
        # Built on the running loop rather than at construction, so a gate can be made
        # before the test's loop exists and still belong to it.
        if self._opened is None:
            self._opened = asyncio.Event()
        return self._opened

    def open(self) -> None:
        """Release whoever is waiting, however few of them there are."""
        self._event().set()

    async def arrive(self) -> None:
        opened = self._event()
        self.waiting += 1
        self.peak = max(self.peak, self.waiting)
        if self.waiting >= self.width:
            opened.set()
        await opened.wait()
        self.waiting -= 1


class GatedRoute(FakeChatModel):
    """A route whose async answer waits at a `Gate`, so a test can see what overlaps."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    gate: Gate

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        await self.gate.arrive()
        return self._generate(messages, stop, None, **kwargs)


async def test_abatch_runs_its_requests_at_the_same_time() -> None:
    """`abatch` of N overlaps — the gate opens only when all N are in flight, so a
    batch that awaited one request after another would never get past it."""
    gate = Gate(width=_BATCH)
    route = GatedRoute(model_name="gated", reply="an answer", gate=gate)
    router = ChatRouter(routes={"only": route}, default_route="only")

    messages = await asyncio.wait_for(router.abatch(["hello"] * _BATCH), _TIMEOUT)

    assert gate.peak == _BATCH
    assert [message.content for message in messages] == ["an answer"] * _BATCH


async def test_abatch_honours_max_concurrency() -> None:
    """The caller's `max_concurrency` still caps a routed batch. The gate would open at
    four, so it stays shut, and exactly two requests sit against it."""
    gate = Gate(width=_BATCH)
    route = GatedRoute(model_name="gated", reply="an answer", gate=gate)
    router = ChatRouter(routes={"only": route}, default_route="only")

    call = asyncio.create_task(router.abatch(["hello"] * _BATCH, {"max_concurrency": 2}))
    await ticks()

    assert (gate.waiting, gate.peak) == (2, 2)
    gate.open()
    messages = await asyncio.wait_for(call, _TIMEOUT)
    assert [message.content for message in messages] == ["an answer"] * _BATCH


# --- batch and abatch ---


@pytest.mark.parametrize("convention", ["batch", "abatch"])
async def test_every_input_in_a_batch_gets_its_own_decision(convention: AnyConvention) -> None:
    """A batch is N routed calls, not one — each input is decided on its own and
    answered by the route its own text names."""
    routes = fake_routes("cheap", "frontier", "local")
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())
    names = ["frontier", "local", "frontier"]
    inputs: list[LanguageModelInput] = [*names]

    if convention == "batch":
        messages = router.batch(inputs)
    else:
        messages = await router.abatch(inputs)

    assert [message.content for message in messages] == [f"{name} answer here" for name in names]
    assert [routing_decision(message) for message in messages] == [
        RoutingDecision.from_dict(record_for(name)) for name in names
    ]
    assert [len(call_log(route)) for route in routes.values()] == [0, 2, 1]


class FailingRoute(FakeChatModel):
    """A route whose provider is down (its error surfaces unchanged)."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = "the provider is down"
        raise RuntimeError(msg)


@pytest.mark.parametrize("convention", ["batch", "abatch"])
async def test_a_batch_returns_exceptions_when_asked_and_raises_when_not(
    convention: AnyConvention,
) -> None:
    """`return_exceptions` behaves as it does on any Runnable — the failed input's
    error takes its place in the results, and the ones that worked keep their records."""
    router = ChatRouter(
        routes={"good": FakeChatModel(reply="good answer"), "bad": FailingRoute()},
        default_route="good",
        strategy=ByText(),
    )
    inputs: list[LanguageModelInput] = ["good", "bad", "good"]

    async def batched(*, return_exceptions: bool) -> list[Any]:
        if convention == "batch":
            return router.batch(inputs, return_exceptions=return_exceptions)
        return await router.abatch(inputs, return_exceptions=return_exceptions)

    good, failed, also_good = await batched(return_exceptions=True)

    assert isinstance(failed, RuntimeError)
    assert str(failed) == "the provider is down"
    for message in (good, also_good):
        assert message.content == "good answer"
        assert routing_decision(message) == RoutingDecision.from_dict(record_for("good"))

    with pytest.raises(RuntimeError, match="the provider is down"):
        await batched(return_exceptions=False)


class Meeting(FakeChatModel):
    """A route that waits at a `threading.Barrier`, to see what a sync batch runs at once."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    barrier: threading.Barrier

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.barrier.wait(timeout=_ALONE)
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_batch_runs_its_inputs_in_parallel_and_honours_max_concurrency() -> None:
    """A routed `batch` parallelises like any Runnable's — four inputs meet at a barrier of
    four — and `max_concurrency=1` serialises them, so a barrier of two is never met."""
    barrier = threading.Barrier(_BATCH)
    router = ChatRouter(
        routes={"only": Meeting(reply="an answer", barrier=barrier)}, default_route="only"
    )

    messages = router.batch(["hello"] * _BATCH)
    assert [message.content for message in messages] == ["an answer"] * _BATCH

    alone = ChatRouter(
        routes={"only": Meeting(reply="an answer", barrier=threading.Barrier(2))},
        default_route="only",
    )
    with pytest.raises(threading.BrokenBarrierError):
        alone.batch(["hello"] * _BATCH, {"max_concurrency": 1})


# --- LangGraph's messages stream ---


class GraphState(TypedDict):
    """The one-key state the graphs below carry."""

    messages: Annotated[list[AnyMessage], add_messages]


def graph_with(model: BaseChatModel) -> Any:
    """A real graph with one node that answers with `model`."""

    def answer(state: GraphState) -> dict[str, list[AnyMessage]]:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(GraphState)
    builder.add_node("model", answer)
    builder.add_edge(START, "model")
    return builder.compile()


def test_a_langgraph_node_streams_the_route_s_tokens_exactly_once() -> None:
    """With the router as a graph node, `stream_mode="messages"` yields the route's
    tokens once — not twice, not zero.

    Because the router's run is a chain run, `on_chat_model_start` registers
    the *route*, and LangGraph's handler emits that one token stream. The finished message the
    node returns is deduplicated against it, because the record goes on a copy that keeps the
    route's message id.
    """
    routes = fake_routes("cheap", "frontier")
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())
    question = {"messages": [("human", "frontier")]}

    parts = list(graph_with(router).stream(question, stream_mode="messages"))

    tokens = [message for message, _ in parts]
    assert "".join(str(token.content) for token in tokens) == "frontier answer here"
    assert len({token.id for token in tokens}) == 1  # one token stream, not two
    assert {metadata["langgraph_node"] for _, metadata in parts} == {"model"}
    assert (len(call_log(routes["cheap"])), len(call_log(routes["frontier"]))) == (0, 1)

    bare = graph_with(FakeChatModel(model_name="model-frontier", reply="frontier answer here"))
    unrouted = list(bare.stream(question, stream_mode="messages"))
    assert [str(token.content) for token in tokens] == [
        str(message.content) for message, _ in unrouted
    ]


def test_a_langgraph_node_puts_the_record_on_the_message_it_returns() -> None:
    """Streaming tokens is not the only thing a graph run yields — the message
    that lands in the state still carries the decision."""
    router = by_text("cheap", "frontier")

    state = graph_with(router).invoke({"messages": [("human", "frontier")]})

    answer = state["messages"][-1]
    assert routing_decision(answer) == RoutingDecision.from_dict(record_for("frontier"))


# --- The v3 streaming protocol is refused, not bypassed ---


@pytest.mark.parametrize("entry", ["stream_events", "astream_events"])
def test_the_v3_streaming_protocol_is_refused(entry: str) -> None:
    """`stream_events(version="v3")` drives a model through `_stream` / `_generate`
    directly (`language_models/chat_models.py:995`), which a delegating router does not
    implement — so routing, the record and the run shape would all be skipped. The router says
    so at the call, before any run opens or any route is touched."""
    routes = fake_routes("cheap", "frontier")
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())
    collector = RunCollectorCallbackHandler()

    with pytest.raises(NotImplementedError, match="v3") as caught:
        getattr(router, entry)("frontier", {"callbacks": [collector]}, version="v3")

    assert "stream(), astream()" in str(caught.value)
    assert collector.traced_runs == []
    assert [len(call_log(route)) for route in routes.values()] == [0, 0]


async def test_refusing_v3_leaves_the_v2_events_routed() -> None:
    """The override forwards every version it does support, so v2 events still come
    from the routed `astream`."""
    routes = fake_routes("cheap", "frontier")
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())

    events = [event async for event in router.astream_events("frontier", version="v2")]

    assert [event["name"] for event in events].count("ChatRouter") >= 2
    assert (len(call_log(routes["cheap"])), len(call_log(routes["frontier"]))) == (0, 1)


# --- No stray warnings on any of it ---


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_no_convention_warns_about_routing_on_its_own(convention: AnyConvention) -> None:
    """A strategy that decides leaves nothing to fall back to — no convention may raise a
    routing warning of its own just by being the one used."""
    router = by_text("cheap", "frontier")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await respond(router, convention, "frontier")

    assert [warning for warning in caught if issubclass(warning.category, RoutingWarning)] == []


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_every_convention_opens_one_router_run_around_one_model_run(
    convention: AnyConvention,
) -> None:
    """The conventions `Runnable` gives the router for free go through the ones
    it overrides, so each of them puts the same shape on the trace — the router's chain run,
    the strategy's, and the route's call as the only model run."""
    router = by_text("cheap", "frontier")
    collector = RunCollectorCallbackHandler()

    await respond(router, convention, "frontier", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    assert (router_run.run_type, router_run.name) == ("chain", "ChatRouter")
    strategy_run, route_run = router_run.child_runs
    assert (strategy_run.run_type, strategy_run.name) == ("chain", "ByText")
    assert (route_run.run_type, route_run.name) == ("llm", "FakeChatModel")
    assert (router_run.outputs or {})[ROUTING_KEY] == record_for("frontier")
