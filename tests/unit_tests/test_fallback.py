"""The router always decides.

Whenever the strategy can't: the default route answers, one `FallbackWarning` fires, and the
record says it fell back and why. A *route's* failure is not the strategy's and is never
absorbed (covered in `test_errors.py`).
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

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
from tests import conventions
from tests.conventions import CONVENTIONS, Convention, respond
from tests.fakes import FakeChatModel, call_log


class Abstains(RoutingStrategy):
    """A strategy with no opinion on this request (`None` means "can't decide")."""

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        return None


class Raises(RoutingStrategy):
    """A strategy that fails — a bug, a missing rule, a provider that didn't answer."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        msg = "no rule matched"
        raise LookupError(msg)


class NamesUnknownRoute(RoutingStrategy):
    """A strategy naming a route this router doesn't have — a stale policy, say."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route="gpt-9", reason="it is the best")


class ReturnsGarbage(RoutingStrategy):
    """A strategy that breaks the interface's return contract."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return cast("RoutingChoice", "frontier")


class Counting(RoutingStrategy):
    """Counts how often it was consulted."""

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        self.calls += 1
        return RoutingChoice(route="frontier", reason="always frontier")


class RouteDown(Exception):
    """What a route's provider raises when it can't answer."""


FAILURE = RouteDown("the provider is down")


class FailingChatModel(BaseChatModel):
    """A route that raises whatever its provider raised."""

    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise FAILURE


CANNOT_DECIDE = [
    (Abstains(), "Abstains could not decide"),
    (Raises(), "Raises raised LookupError: no rule matched"),
    (NamesUnknownRoute(), "NamesUnknownRoute chose 'gpt-9', which is not one of the routes"),
    (ReturnsGarbage(), "ReturnsGarbage returned str, not a RoutingChoice"),
]
"""Every way a strategy can fail to decide, and the cause the router reports for it."""

CANNOT_DECIDE_IDS = ["abstained", "raised", "unknown route", "not a choice"]


def router_with(strategy: RoutingStrategy | None) -> tuple[ChatRouter, dict[str, BaseChatModel]]:
    """A router whose default route is *not* its first route, so fallbacks are unambiguous."""
    routes: dict[str, BaseChatModel] = {
        "frontier": FakeChatModel(model_name="model-frontier", reply="frontier answer"),
        "cheap": FakeChatModel(model_name="model-cheap", reply="cheap answer"),
    }
    return (
        ChatRouter(routes=routes, default_route="cheap", strategy=strategy),
        routes,
    )


def routing_warnings(
    caught: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


@pytest.mark.parametrize(("strategy", "cause"), CANNOT_DECIDE, ids=CANNOT_DECIDE_IDS)
@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_strategy_that_cannot_decide_falls_back_to_the_default_route(
    strategy: RoutingStrategy, cause: str, convention: Convention
) -> None:
    """Abstaining, raising, naming an unknown route or returning something that
    isn't a choice — each takes the default route, warns exactly once, and is recorded as a
    fallback with a reason naming the cause."""
    router, routes = router_with(strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "hello")

    assert message.content == "cheap answer"
    assert (len(call_log(routes["cheap"])), len(call_log(routes["frontier"]))) == (1, 0)
    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]
    assert str(routing_warnings(caught)[0].message) == (
        f"{cause}; falling back to the default route 'cheap'"
    )
    assert routing_decision(message) == RoutingDecision(
        route="cheap",
        reason=f"{cause}; fell back to the default route",
        strategy=type(strategy).__name__,
        fallback=True,
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_fallback_warning_points_at_the_caller(convention: Convention) -> None:
    """The warning is for the application to act on, so it names the line that called the
    router rather than a frame inside it."""
    router, _ = router_with(Abstains())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await respond(router, convention, "hello")

    (warning,) = routing_warnings(caught)
    assert warning.filename == conventions.__file__


@pytest.mark.parametrize(
    ("strategy", "cause"),
    [CANNOT_DECIDE[0], CANNOT_DECIDE[2]],
    ids=["abstained", "unknown route"],
)
async def test_the_strategy_run_records_the_fallback_it_led_to(
    strategy: RoutingStrategy, cause: str
) -> None:
    """The strategy's run is where a trace shows what it decided — including that it
    couldn't, and what the router did instead."""
    router, _ = router_with(strategy)
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        message = router.invoke("hello", config={"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    record = routing_decision(message)
    assert record is not None
    assert record.reason.startswith(cause)
    assert strategy_run.outputs == record.as_dict()
    assert strategy_run.error is None
    assert (route_run.extra or {})["metadata"]["routing"] == record.as_dict()


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_a_strategy_that_raises_ends_its_own_run_with_the_error(
    convention: Convention,
) -> None:
    """The strategy's failure closes the strategy's run — where its traceback stays for
    whoever debugs it — while the router's run and the route's call complete normally."""
    router, _ = router_with(Raises())
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        await respond(router, convention, "hello", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    assert strategy_run.outputs == {}
    assert "LookupError: no rule matched" in (strategy_run.error or "")
    assert (router_run.error, route_run.error) == (None, None)


async def test_a_request_with_no_user_message_never_reaches_the_strategy() -> None:
    """The fallback, and the router's side of extraction: with no user message there is nothing to
    decide on, so the strategy is not consulted — no strategy run — and the default route answers
    with one warning and the reason recorded."""
    strategy = Counting()
    router, routes = router_with(strategy)
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke(
            [SystemMessage("You are terse.")], config={"callbacks": [collector]}
        )

    assert strategy.calls == 0
    assert message.content == "cheap answer"
    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]
    assert routing_decision(message) == RoutingDecision(
        route="cheap",
        reason="the request has no user message to route on; fell back to the default route",
        strategy=None,
        fallback=True,
    )
    (router_run,) = collector.traced_runs
    assert [run.run_type for run in router_run.child_runs] == ["llm"]
    assert call_log(routes["frontier"]) == []


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_route_s_failure_is_never_absorbed(convention: Convention) -> None:
    """The fallback's boundary: only a strategy's failure falls back. The selected
    route's exception reaches the caller as itself, no other route is tried, nothing warns,
    and the router's run closes as an error."""
    cheap = FakeChatModel(reply="cheap answer")
    router = ChatRouter(
        routes={"frontier": FailingChatModel(), "cheap": cheap},
        default_route="cheap",
        strategy=Counting(),
    )
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RouteDown) as raised:
            await respond(router, convention, "hello", {"callbacks": [collector]})

    assert raised.value is FAILURE
    assert cheap.calls == []
    assert routing_warnings(caught) == []
    (router_run,) = collector.traced_runs
    assert "RouteDown: the provider is down" in (router_run.error or "")
