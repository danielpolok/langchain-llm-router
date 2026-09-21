"""T-118: C6 — errors, retries and fallbacks stay LangChain's job.

The router adds no error handling of its own. A route's exception is the caller's exception:
the same object, with its type and args intact, raised after exactly one attempt and with no
`FallbackWarning` — R9's fallback absorbs a *strategy's* failure and nothing else (REQ-R9-3).
What a caller wants instead is what LangChain already gives every runnable:
`router.with_retry(...)` and `router.with_fallbacks([...])` wrap the router as they wrap a
model (REQ-C6-2).

However a call fails, the router's chain run closes through `on_chain_error` and no run is left
open (REQ-C6-3) — including on the paths that are not ordinary exceptions, where absorbing the
error would be the real bug: a `KeyboardInterrupt` or a cancelled task must reach the caller
untouched rather than be recorded as a strategy that "could not decide".
"""

from __future__ import annotations

import asyncio
import gc
import warnings
from collections.abc import Iterator
from typing import Any, Literal

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.tracers.schemas import Run

from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    routing_decision,
)
from tests.conventions import CONVENTIONS, Convention, respond
from tests.fakes import FakeChatModel, call_log

Blocking = Literal["invoke", "ainvoke"]
"""The two conventions that return one message, for the wrappers that only override those."""

Streaming = Literal["stream", "astream"]


class RouteDown(Exception):
    """What a route's provider raises when it can't answer — with details worth preserving."""


FAILURE = RouteDown("the provider is down", 503)
"""One instance, so a test can assert the caller got *this* object and not a copy of it."""


class FailingChatModel(FakeChatModel):
    """A route whose provider is down.

    It fails where `FakeChatModel` builds its answer, which `_generate` and `_stream` both
    reach after logging the attempt — so every convention fails the same way and `calls` still
    counts the attempts.
    """

    def _message(self) -> AIMessage:
        raise FAILURE


class InterruptingChatModel(FakeChatModel):
    """A route stopped by something that is not an `Exception` at all: an interrupt at the
    keyboard when called synchronously, a cancelled task when awaited."""

    def _message(self) -> AIMessage:
        raise KeyboardInterrupt

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Raised in the loop rather than in `run_in_executor`'s worker thread, which is where
        # a cancelled provider call would really come from.
        self.calls.append(dict(kwargs))
        raise asyncio.CancelledError


class FlakyChatModel(FakeChatModel):
    """A route that fails its first `failures` attempts and answers after that."""

    failures: int = 2

    def _message(self) -> AIMessage:
        if self.failures > 0:
            self.failures -= 1
            raise FAILURE
        return super()._message()


class HalfStreamingChatModel(FakeChatModel):
    """A route that dies part way through its answer, as a dropped connection would."""

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.calls.append(dict(kwargs))
        yield ChatGenerationChunk(message=AIMessageChunk(content="half an "))
        yield ChatGenerationChunk(message=AIMessageChunk(content="answer"))
        raise FAILURE


class PicksFrontier(RoutingStrategy):
    """A strategy that always chooses the route that is about to fail, counting its turns."""

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        self.calls += 1
        return RoutingChoice(route="frontier", reason="always frontier")


class Abstains(RoutingStrategy):
    """A strategy with no opinion, so the R9 fallback runs and can be escalated to an error."""

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        return None


class Interrupts(RoutingStrategy):
    """A strategy interrupted mid-decision: at the keyboard when sync, cancelled when async."""

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        raise KeyboardInterrupt

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        raise asyncio.CancelledError


def failing_router(
    strategy: RoutingStrategy | None = None,
    frontier: BaseChatModel | None = None,
) -> tuple[ChatRouter, dict[str, BaseChatModel]]:
    """A router with a broken `frontier` route and a working `cheap` one.

    With no strategy, `frontier` is the default route, so it is the one that runs. With a
    strategy — which always picks `frontier` — `cheap` is the default instead, so a failure the
    router wrongly absorbed could not hide: it would come back as `"cheap answer"` (REQ-R9-3).
    """
    routes: dict[str, BaseChatModel] = {
        "frontier": frontier if frontier is not None else FailingChatModel(),
        "cheap": FakeChatModel(reply="cheap answer"),
    }
    return (
        ChatRouter(
            routes=routes,
            default_route="frontier" if strategy is None else "cheap",
            strategy=strategy,
        ),
        routes,
    )


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[type[Warning]]:
    """The categories of the routing warnings among everything caught — nothing else."""
    return [warning.category for warning in caught if issubclass(warning.category, RoutingWarning)]


async def answer(
    model: Runnable[LanguageModelInput, AIMessage],
    convention: Blocking,
    input_: str,
    config: RunnableConfig | None = None,
) -> AIMessage:
    """`respond` for what `with_retry` and `with_fallbacks` hand back — a plain `Runnable`,
    not a chat model, and one that only overrides the blocking conventions."""
    if convention == "invoke":
        return model.invoke(input_, config)
    return await model.ainvoke(input_, config)


async def stream_into(
    chunks: list[AIMessage],
    model: Runnable[LanguageModelInput, AIMessage],
    convention: Streaming,
    input_: str,
    config: RunnableConfig | None = None,
) -> None:
    """Stream into `chunks`, so a caller expecting a failure can still read what arrived.

    A stream that dies has handed its caller real chunks already, and C6 says those are the
    route's own; `pytest.raises` returns the exception, not the chunks, so they go in a list
    the test holds.
    """
    if convention == "stream":
        for chunk in model.stream(input_, config):
            chunks.append(chunk)
    else:
        async for chunk in model.astream(input_, config):
            chunks.append(chunk)


async def failure_of(model: BaseChatModel, convention: Convention) -> BaseException:
    """Whatever the call raised, `Exception` or not — for comparing two failures."""
    try:
        await respond(model, convention, "hello")
    except BaseException as error:
        return error
    msg = f"{type(model).__name__}.{convention} was expected to fail"
    raise AssertionError(msg)


def outline(collector: RunCollectorCallbackHandler) -> list[tuple[str, bool]]:
    """Each root run the collector saw, as `(run_type, failed)`."""
    return [(run.run_type, run.error is not None) for run in collector.traced_runs]


def open_runs(collector: RunCollectorCallbackHandler) -> list[str]:
    """Runs the tracer started and never saw end — REQ-C6-3's "no dangling open run"."""
    return [run.name for run in collector.run_map.values()]


def one_tree(collector: RunCollectorCallbackHandler) -> tuple[Run, list[Run]]:
    """The single root run the collector saw, and its children."""
    (root,) = collector.traced_runs
    return root, list(root.child_runs or [])


# --- REQ-C6-1 / REQ-R9-3: a route's exception is the caller's exception ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_route_s_exception_reaches_the_caller_unchanged(convention: Convention) -> None:
    """REQ-C6-1: the selected route's exception arrives as the same object, with its type and
    args intact. The route ran exactly once, so the router retried nothing, and it warned
    about nothing."""
    router, routes = failing_router()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RouteDown) as raised:
            await respond(router, convention, "hello")

    assert raised.value is FAILURE
    assert type(raised.value) is RouteDown
    assert raised.value.args == ("the provider is down", 503)
    assert len(call_log(routes["frontier"])) == 1
    assert routing_warnings(caught) == []


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_route_failure_never_reaches_another_route(convention: Convention) -> None:
    """REQ-R9-3: R9's fallback absorbs a *strategy's* failure only. Here the strategy decided
    perfectly well, so the chosen route's failure is the answer — the default route is not
    tried instead, and nothing warns about a fallback that never happened."""
    strategy = PicksFrontier()
    router, routes = failing_router(strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RouteDown) as raised:
            await respond(router, convention, "hello")

    assert raised.value is FAILURE
    assert strategy.calls == 1
    assert len(call_log(routes["frontier"])) == 1
    assert call_log(routes["cheap"]) == []
    assert routing_warnings(caught) == []


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_a_route_that_dies_mid_stream_keeps_the_chunks_it_yielded(
    convention: Streaming,
) -> None:
    """REQ-C6-1 while streaming: the chunks already handed over are the route's own — the
    first of them carrying the record (D8) — and the failure that follows them is unchanged."""
    router, routes = failing_router(frontier=HalfStreamingChatModel())
    chunks: list[AIMessage] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RouteDown) as raised:
            await stream_into(chunks, router, convention, "hello")

    assert raised.value is FAILURE
    assert [chunk.content for chunk in chunks] == ["half an ", "answer"]
    assert routing_decision(chunks[0]) == RoutingDecision(
        route="frontier", reason="no strategy configured"
    )
    assert routing_decision(chunks[1]) is None
    assert len(call_log(routes["frontier"])) == 1
    assert routing_warnings(caught) == []


def test_a_route_s_keyboard_interrupt_is_not_an_error_to_absorb() -> None:
    """REQ-C6-1 for what is not an `Exception`: an interrupted route stops the caller.
    Answering from another route here would answer a request the user gave up on."""
    router, routes = failing_router(frontier=InterruptingChatModel())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(KeyboardInterrupt):
            router.invoke("hello")

    assert len(call_log(routes["frontier"])) == 1
    assert call_log(routes["cheap"]) == []
    assert routing_warnings(caught) == []


async def test_a_cancelled_route_ends_as_it_does_without_the_router() -> None:
    """REQ-C6-1 on the async path, where `BaseChatModel` itself does not pass a cancelled call
    through unchanged: `agenerate` collects any `BaseException` but filters its `on_llm_end`
    cleanup on `Exception` (`chat_models.py:1837`), so a `CancelledError` becomes an
    `AttributeError` before any caller sees it.

    That is LangChain's behaviour, and C6 says the router neither hides it nor adds to it: the
    same route called directly and called through the router must fail identically — and the
    router must still not treat it as a reason to answer from somewhere else.
    """
    bare = InterruptingChatModel()
    router, routes = failing_router(frontier=InterruptingChatModel())

    direct = await failure_of(bare, "ainvoke")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        routed = await failure_of(router, "ainvoke")

    assert (type(routed), routed.args) == (type(direct), direct.args)
    assert len(call_log(routes["frontier"])) == 1
    assert call_log(routes["cheap"]) == []
    assert routing_warnings(caught) == []


# --- REQ-C6-2: LangChain's own wrappers, on the router as on a model ---


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_with_retry_retries_the_whole_routed_call(convention: Blocking) -> None:
    """REQ-C6-2: `router.with_retry()` is the supported way to retry a routed request. Each
    attempt re-runs the pipeline under a nested run of its own, and the answer that finally
    arrives still carries its record — the wrapper sits outside the router, so nothing about
    routing changes."""
    flaky = FlakyChatModel(failures=2, reply="eventually")
    router = ChatRouter(routes={"flaky": flaky}, default_route="flaky")
    retrying = router.with_retry(stop_after_attempt=3, wait_exponential_jitter=False)
    collector = RunCollectorCallbackHandler()

    message = await answer(retrying, convention, "hello", {"callbacks": [collector]})

    assert message.content == "eventually"
    assert len(flaky.calls) == 3
    assert routing_decision(message) == RoutingDecision(
        route="flaky", reason="no strategy configured"
    )
    root, attempts = one_tree(collector)
    assert root.error is None
    assert [(run.run_type, run.error is not None) for run in attempts] == [
        ("chain", True),
        ("chain", True),
        ("chain", False),
    ]
    assert open_runs(collector) == []


def test_with_retry_gives_up_with_the_route_s_own_exception() -> None:
    """REQ-C6-2 with REQ-C6-1: when the retries run out, what surfaces is still the route's
    exception — the wrapper re-raises it rather than one of its own."""
    router, routes = failing_router()
    retrying = router.with_retry(stop_after_attempt=2, wait_exponential_jitter=False)

    with pytest.raises(RouteDown) as raised:
        retrying.invoke("hello")

    assert raised.value is FAILURE
    assert len(call_log(routes["frontier"])) == 2


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_with_retry_streams_exactly_as_it_does_on_a_bare_model(
    convention: Streaming,
) -> None:
    """REQ-C6-2's "as on a model", including where LangChain's wrapper stops short:
    `RunnableBindingBase.stream` hands straight to `self.bound.stream`, so `with_retry` does
    not retry a *streamed* call at all. The router must neither paper over that nor make it
    worse — the same scenario, through a bare chat model and through a router over that model,
    has to end identically."""
    bare = FailingChatModel()
    routed = FailingChatModel()
    router = ChatRouter(routes={"frontier": routed}, default_route="frontier")

    with pytest.raises(RouteDown) as on_model:
        await stream_into(
            [],
            bare.with_retry(stop_after_attempt=3, wait_exponential_jitter=False),
            convention,
            "hello",
        )
    with pytest.raises(RouteDown) as on_router:
        await stream_into(
            [],
            router.with_retry(stop_after_attempt=3, wait_exponential_jitter=False),
            convention,
            "hello",
        )

    assert on_model.value is FAILURE
    assert on_router.value is FAILURE
    assert len(routed.calls) == len(bare.calls) == 1


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_with_fallbacks_hands_a_failing_router_over_to_another_model(
    convention: Blocking,
) -> None:
    """REQ-C6-2: because the router lets the route's failure through, `with_fallbacks` sees it
    and the caller gets the other model's answer. That answer is not the router's, so it
    carries no routing record, and the router warned about nothing."""
    router, routes = failing_router()
    other = FakeChatModel(reply="other answer")
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await answer(
            router.with_fallbacks([other]), convention, "hello", {"callbacks": [collector]}
        )

    assert message.content == "other answer"
    assert len(call_log(routes["frontier"])) == 1
    assert len(other.calls) == 1
    assert routing_decision(message) is None
    assert routing_warnings(caught) == []
    assert open_runs(collector) == []


def test_with_fallbacks_leaves_the_same_trace_a_model_would() -> None:
    """REQ-C6-2, REQ-C6-3: the router's run closes as an error before the fallback model runs,
    the fallback's own run is clean, and nothing is left open. The router's run is a *chain*
    run where a model's would be an LLM run (PRD §11); which runs failed, and in what order,
    is what a bare model does too — that is what "behaves as on a model" means here."""
    router, _ = failing_router()
    routed = RunCollectorCallbackHandler()
    bare = RunCollectorCallbackHandler()

    router.with_fallbacks([FakeChatModel(reply="other answer")]).invoke(
        "hello", config={"callbacks": [routed]}
    )
    FailingChatModel().with_fallbacks([FakeChatModel(reply="other answer")]).invoke(
        "hello", config={"callbacks": [bare]}
    )

    assert outline(routed) == [("chain", True), ("llm", False), ("chain", False)]
    assert outline(bare) == [("llm", True), ("llm", False), ("chain", False)]
    assert [failed for _, failed in outline(routed)] == [failed for _, failed in outline(bare)]
    assert open_runs(routed) == open_runs(bare) == []


# --- REQ-C6-3: every failure closes the router's run and leaves nothing open ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_route_failure_closes_the_router_s_run_as_an_error(
    convention: Convention,
) -> None:
    """REQ-C6-3: the tracer records `on_chain_error` on the router's chain run, the route's
    own run is closed as an error beneath it, the router's run publishes no outputs, and no
    run is left open."""
    router, _ = failing_router()
    collector = RunCollectorCallbackHandler()

    with pytest.raises(RouteDown):
        await respond(router, convention, "hello", {"callbacks": [collector]})

    router_run, (route_run,) = one_tree(collector)
    assert router_run.run_type == "chain"
    assert repr(FAILURE) in (router_run.error or "")
    assert repr(FAILURE) in (route_run.error or "")
    assert router_run.outputs == {}
    assert open_runs(collector) == []


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_a_stream_that_dies_part_way_still_closes_the_router_s_run(
    convention: Streaming,
) -> None:
    """REQ-C6-3 on the streaming paths, where the failure arrives after the router has already
    handed chunks to its caller: the run still closes as an error, not as a success."""
    router, _ = failing_router(frontier=HalfStreamingChatModel())
    collector = RunCollectorCallbackHandler()

    with pytest.raises(RouteDown):
        await stream_into([], router, convention, "hello", {"callbacks": [collector]})

    router_run, (route_run,) = one_tree(collector)
    assert repr(FAILURE) in (router_run.error or "")
    assert repr(FAILURE) in (route_run.error or "")
    assert router_run.outputs == {}
    assert open_runs(collector) == []


def test_a_stream_the_caller_walks_away_from_leaves_no_run_open() -> None:
    """REQ-C6-3's other half — "no dangling open run" — for the one path that is not a
    failure: a caller that `break`s out of `stream` and drops the generator. `GeneratorExit`
    reaches the router when the generator is finalized, and the run has to close there or the
    trace keeps an open run for a request that is long over.

    What this does *not* do is withdraw the record; that is T-114's, in `test_decision.py`.
    """
    router = ChatRouter(
        routes={"frontier": FakeChatModel(reply="one two three four")},
        default_route="frontier",
    )
    collector = RunCollectorCallbackHandler()

    for _ in router.stream("hello", config={"callbacks": [collector]}):
        break
    gc.collect()  # the generator the loop dropped is finalized here at the latest

    router_run, (route_run,) = one_tree(collector)
    assert "GeneratorExit" in (router_run.error or "")
    assert "GeneratorExit" in (route_run.error or "")
    assert open_runs(collector) == []


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_an_interrupted_strategy_is_never_absorbed_as_a_fallback(
    convention: Convention,
) -> None:
    """REQ-C6-3, and R9's limit: `KeyboardInterrupt` and a cancelled task are not "the
    strategy could not decide". They propagate, no route is called, nothing warns, and both
    the strategy's run and the router's close as errors with nothing left open."""
    expected = KeyboardInterrupt if convention == "invoke" else asyncio.CancelledError
    router, routes = failing_router(Interrupts())
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(expected):
            await respond(router, convention, "hello", {"callbacks": [collector]})

    assert routing_warnings(caught) == []
    assert call_log(routes["frontier"]) == []
    assert call_log(routes["cheap"]) == []
    router_run, (strategy_run,) = one_tree(collector)
    assert strategy_run.name == "Interrupts"
    assert expected.__name__ in (strategy_run.error or "")
    assert expected.__name__ in (router_run.error or "")
    assert open_runs(collector) == []


def test_a_fallback_warning_escalated_to_an_error_closes_both_runs() -> None:
    """REQ-C6-3 for an application running under `-W error`: turning `FallbackWarning` into an
    exception must not strand the runs the router had already opened. It closes the strategy's
    run and its own, calls no route, and the warning reaches the caller."""
    router, routes = failing_router(Abstains())
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        with pytest.raises(FallbackWarning):
            router.invoke("hello", config={"callbacks": [collector]})

    assert call_log(routes["frontier"]) == []
    assert call_log(routes["cheap"]) == []
    router_run, (strategy_run,) = one_tree(collector)
    assert "FallbackWarning" in (strategy_run.error or "")
    assert "FallbackWarning" in (router_run.error or "")
    assert open_runs(collector) == []


def test_the_router_has_nothing_of_its_own_to_retry_or_fall_back_with() -> None:
    """REQ-C6-1's "no router-level retry or model fallback", as configuration: retries and
    model fallbacks are `with_retry` / `with_fallbacks`, so the router grows no field for
    them. `default_route` is R9's — a strategy's fallback, never a failed model's."""
    assert not {"retry", "retries", "max_retries", "fallbacks", "fallback_model"} & set(
        ChatRouter.model_fields
    )
