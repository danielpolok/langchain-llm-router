"""Tools and structured output through the router (tool binding, tool-aware routing, and the
structured half of the record).

What is here, in order: capability detection, one signal at a time; the bind-time
cases and the per-request diversion, each run through `bind_tools` *and*
through `with_structured_output`; the binding replayed on the route the strategy
chose and the raw `tools` copy kept for LangChain's own checks;
`with_structured_output` forwarding what the base default drops; the record on
structured output; and where each warning points — a count of frames, so asserted
rather than assumed.

Neighbours, not repeated here: what falling back does is `test_fallback.py`'s, and the record's
schema and the limits of `last_routing_decision()` are `test_decision.py`'s. A *forced* route
that can't use the bound tools is `test_forced_routes.py`'s.
"""

from __future__ import annotations

import gc
import inspect
import warnings
from collections.abc import Callable, Iterator
from typing import Any, Literal, NamedTuple, cast

import pytest
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.outputs import ChatGeneration
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig, RunnableLambda
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.tools import tool
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.tracers.schemas import Run
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    NoToolCapableRouteError,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    ToolSupportWarning,
    last_routing_decision,
    routing_decision,
)
from langchain_llm_router._tools import BINDING_KEY, supports_tools
from langchain_llm_router.decision import ROUTING_KEY
from tests.conventions import ALL_CONVENTIONS, CONVENTIONS, Convention, generated, respond
from tests.fakes import (
    FakeChatModel,
    NativeStructuredFakeChatModel,
    StreamingStructuredFakeChatModel,
    ToolCallingFakeChatModel,
    call_log,
)
from tests.tracing import model_runs

# --- What the tests bind, and the routes and strategies they bind it to ---


@tool
def get_weather(city: str) -> str:
    """Look up the weather in a city."""
    return f"sunny in {city}"


class Answer(BaseModel):
    """An answer."""

    answer: str


ANSWER_CALL = ToolCall(name="Answer", args={"answer": "42"}, id="call_1", type="tool_call")
"""What a tool-capable route replies with: a call to `Answer`, which structured output parses."""


def capable(name: str, **fields: Any) -> NativeStructuredFakeChatModel:
    """A route that can use tools, with a structured output of its own."""
    return NativeStructuredFakeChatModel(
        model_name=f"model-{name}", reply=f"{name} answer", tool_calls=[ANSWER_CALL], **fields
    )


def incapable(name: str) -> FakeChatModel:
    """A route that can't: it keeps `BaseChatModel.bind_tools`, which raises at call time."""
    return FakeChatModel(model_name=f"model-{name}", reply=f"{name} answer")


class ByText(RoutingStrategy):
    """The request's text names the route, so each test picks a route with its input.

    Counts how often it was asked: a diversion happens after the strategy has decided, and must
    not ask again.
    """

    def __init__(self) -> None:
        self.asked = 0

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        self.asked += 1
        return RoutingChoice(route=request.text, reason=f"asked for {request.text!r}")


class Abstains(RoutingStrategy):
    """A strategy with no opinion, so the default route answers."""

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        return None


def router_of(
    capability: dict[str, bool],
    *,
    default: str,
    strategy: RoutingStrategy | None = None,
    **fields: Any,
) -> tuple[ChatRouter, dict[str, BaseChatModel]]:
    """One route per entry, in order: one that can use tools when the value is `True`."""
    routes: dict[str, BaseChatModel] = {
        name: capable(name) if can_use_tools else incapable(name)
        for name, can_use_tools in capability.items()
    }
    router = ChatRouter(routes=routes, default_route=default, strategy=strategy, **fields)
    return router, routes


def mixed(strategy: RoutingStrategy | None = None) -> tuple[ChatRouter, dict[str, BaseChatModel]]:
    """`first` and `frontier` (the default) can use tools; `cheap` can't.

    In that order, so the default is *not* the first route able to take a diversion — the diversion
    has a choice to make, and a test can tell which way it went.
    """
    return router_of(
        {"first": True, "cheap": False, "frontier": True},
        default="frontier",
        strategy=strategy or ByText(),
    )


class Binder(NamedTuple):
    """One way to put tools on a router, so a test can run through each."""

    bind: Callable[[ChatRouter], Runnable[LanguageModelInput, Any]]
    message: Callable[[Any], AIMessage]
    """The route's message out of what the bound runnable returned: with `include_raw=True`,
    structured output answers `{"raw", "parsed", "parsing_error"}` and the record is on `raw`."""


# Each is one statement with no docstring: a test finds the line a warning should point at by
# `co_firstlineno + 1`.
def bind_tools(router: ChatRouter) -> Runnable[LanguageModelInput, Any]:
    return router.bind_tools([get_weather])


def bind_structured_output(router: ChatRouter) -> Runnable[LanguageModelInput, Any]:
    return router.with_structured_output(Answer, include_raw=True)


TOOLS = Binder(bind_tools, lambda answer: cast("AIMessage", answer))
STRUCTURED = Binder(bind_structured_output, lambda answer: cast("AIMessage", answer["raw"]))
BINDERS = [pytest.param(TOOLS, id="bind_tools"), pytest.param(STRUCTURED, id="with_structured")]

AskConvention = Literal["invoke", "ainvoke"]


async def ask(
    bound: Runnable[LanguageModelInput, Any], convention: AskConvention, text: str
) -> Any:
    """What `bound` answers through `convention`."""
    if convention == "invoke":
        return bound.invoke(text)
    return await bound.ainvoke(text)


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


def bind(binder: Binder, router: ChatRouter) -> Runnable[LanguageModelInput, Any]:
    """Bind on a router that has a tool-incapable route, hearing the warning tool-aware routing
    gives for it.

    Tests of what a *request* does start here, so the bind-time notice neither goes unasserted
    nor reaches the request's own warning count.
    """
    with pytest.warns(ToolSupportWarning):
        return binder.bind(router)


async def routed(
    bound: Runnable[LanguageModelInput, Any], convention: AskConvention, text: str
) -> tuple[Any, list[warnings.WarningMessage]]:
    """One request, and every routing warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answer = await ask(bound, convention, text)
    return answer, routing_warnings(caught)


def rebind(bound: Runnable[LanguageModelInput, Any], tools: list[Any]) -> Runnable[Any, Any]:
    """`bound.bind_tools(tools)`: a binding over a chat model hands the call to the model's own
    `bind_tools` (`RunnableBinding.__getattr__`), which `Runnable`'s type doesn't show."""
    return cast("Runnable[Any, Any]", cast("Any", bound).bind_tools(tools))


def record_of(message: AIMessage) -> RoutingDecision:
    record = routing_decision(message)
    assert record is not None
    return record


def metadata_of(run: Run) -> dict[str, Any]:
    return (run.extra or {}).get("metadata") or {}


def diverted(route: str, *, to: str) -> RoutingDecision:
    """The record for a request `ByText` sent to `route` and the diversion sent to `to`."""
    return RoutingDecision(
        route=to,
        reason=f"asked for {route!r}; {route!r} can't use the bound tools, so it was diverted "
        f"to {to!r}",
        strategy="ByText",
        diverted_from=route,
    )


# --- capability is detected from each signal in turn, and an override wins ---

CAPABILITY = [
    pytest.param(
        ToolCallingFakeChatModel(profile={"tool_calling": False}),
        None,
        False,
        id="profile says no, though the class overrides bind_tools",
    ),
    pytest.param(
        FakeChatModel(profile={"tool_calling": True}),
        None,
        True,
        id="profile says yes, though the class does not override bind_tools",
    ),
    pytest.param(ToolCallingFakeChatModel(), None, True, id="no profile: class overrides"),
    pytest.param(FakeChatModel(), None, False, id="no profile: class does not override"),
    pytest.param(
        ToolCallingFakeChatModel(profile={"image_inputs": True}),
        None,
        True,
        id="profile silent on tool_calling defers to the class: overrides",
    ),
    pytest.param(
        FakeChatModel(profile={"image_inputs": True}),
        None,
        False,
        id="profile silent on tool_calling defers to the class: does not",
    ),
    pytest.param(
        ToolCallingFakeChatModel(profile={"tool_calling": False}),
        True,
        True,
        id="override yes beats profile no",
    ),
    pytest.param(
        ToolCallingFakeChatModel(profile={"tool_calling": True}),
        False,
        False,
        id="override no beats profile yes",
    ),
    pytest.param(FakeChatModel(), True, True, id="override yes beats a class without bind_tools"),
    pytest.param(
        ToolCallingFakeChatModel(), False, False, id="override no beats a class with bind_tools"
    ),
]


@pytest.mark.parametrize(("route", "override", "expected"), CAPABILITY)
def test_tool_capability_is_read_from_each_signal_in_d5_s_order(
    route: BaseChatModel, override: bool | None, expected: bool
) -> None:
    """`profile["tool_calling"]` when the route reports one, else whether its
    class overrides `bind_tools` — and an override always wins. Each signal is exercised on its
    own, and against the ones that come after it."""
    assert supports_tools(route, override=override) is expected


async def test_tool_support_overrides_change_what_the_router_does_with_a_route() -> None:
    """Through the router, `tool_support_overrides` is what the warning names and
    what a request is diverted on — a route that could use tools but is pinned off is skipped,
    and one the profile rules out but the application vouches for is kept."""
    routes: dict[str, BaseChatModel] = {
        "frontier": capable("frontier"),
        "pinned": capable("pinned"),
        "revived": capable("revived", profile={"tool_calling": False}),
    }
    router = ChatRouter(
        routes=routes,
        default_route="frontier",
        strategy=ByText(),
        tool_support_overrides={"pinned": False, "revived": True},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bound = router.bind_tools([get_weather])
    (warning,) = routing_warnings(caught)
    assert str(warning.message) == (
        "routes that can't use tools: 'pinned'; a request routed to one of them goes to "
        "'frontier' instead"
    )

    pinned, pinned_warnings = await routed(bound, "invoke", "pinned")
    revived, revived_warnings = await routed(bound, "invoke", "revived")

    assert routing_decision(pinned) == diverted("pinned", to="frontier")
    assert [w.category for w in pinned_warnings] == [ToolSupportWarning]
    assert routing_decision(revived) == RoutingDecision(
        route="revived", reason="asked for 'revived'", strategy="ByText"
    )
    assert revived_warnings == []


# --- binding warns up front, or refuses when nothing could serve ---


@pytest.mark.parametrize("binder", BINDERS)
def test_binding_to_routes_that_can_all_use_tools_says_nothing(binder: Binder) -> None:
    """The case where nothing is wrong is silent — no warning, no error."""
    router, _ = router_of({"a": True, "b": True}, default="a")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bound = binder.bind(router)

    assert routing_warnings(caught) == []
    assert isinstance(bound, Runnable)


@pytest.mark.parametrize("binder", BINDERS)
def test_binding_warns_once_naming_the_routes_that_cannot_use_tools(binder: Binder) -> None:
    """One `ToolSupportWarning`, not one per route, listing exactly the
    routes that can't — in declaration order — and saying where their requests will go."""
    router, _ = router_of({"local": False, "frontier": True, "cheap": False}, default="frontier")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        binder.bind(router)

    (warning,) = routing_warnings(caught)
    assert warning.category is ToolSupportWarning
    assert str(warning.message) == (
        "routes that can't use tools: 'local', 'cheap'; a request routed to one of them goes to "
        "'frontier' instead"
    )


@pytest.mark.parametrize("binder", BINDERS)
def test_the_bind_time_warning_names_the_route_d1_will_divert_to(binder: Binder) -> None:
    """When the default route is one of those that can't use tools, requests go
    to the first route that can — and the warning says so rather than naming the default."""
    router, _ = router_of({"cheap": False, "first": True, "second": True}, default="cheap")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        binder.bind(router)

    (warning,) = routing_warnings(caught)
    assert str(warning.message) == (
        "routes that can't use tools: 'cheap'; a request routed to one of them goes to "
        "'first' instead"
    )


@pytest.mark.parametrize("binder", BINDERS)
def test_binding_when_no_route_can_use_tools_is_an_error(binder: Binder) -> None:
    """`NoToolCapableRouteError` at bind time, naming the routes, and no
    warning — there is nothing to divert to, so the notice would only be noise."""
    router, routes = router_of({"a": False, "b": False}, default="a")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(NoToolCapableRouteError) as raised:
            binder.bind(router)

    assert str(raised.value) == (
        "no route can use tools: 'a', 'b'; binding tools or structured output needs at least "
        "one tool-capable route — tool_support_overrides can name one"
    )
    assert routing_warnings(caught) == []
    assert [call_log(route) for route in routes.values()] == [[], []]


@pytest.mark.parametrize("binder", BINDERS)
def test_an_override_can_make_a_route_the_one_that_can_use_tools(binder: Binder) -> None:
    """The error is about what the router believes, and the override is
    how an application corrects that belief — with it, binding succeeds and warns about the
    other route."""
    router, _ = router_of({"a": False, "b": False}, default="a", tool_support_overrides={"b": True})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        binder.bind(router)

    (warning,) = routing_warnings(caught)
    assert str(warning.message) == (
        "routes that can't use tools: 'a'; a request routed to one of them goes to 'b' instead"
    )


def test_each_binding_gets_its_own_bind_time_warning() -> None:
    """The notice belongs to the act of binding, so binding again says it again."""
    router, _ = mixed()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rebind(router.bind_tools([get_weather]), [Answer])

    assert [w.category for w in routing_warnings(caught)] == [ToolSupportWarning] * 2


# --- a request sent to a route that can't use the tools is diverted ---


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
@pytest.mark.parametrize("binder", BINDERS)
async def test_a_request_sent_to_a_tool_incapable_route_goes_to_the_default_route(
    binder: Binder, convention: AskConvention
) -> None:
    """The default route is the first choice for a diversion — even
    when a tool-capable route comes before it in declaration order. The request runs there, the
    warning says so once, `diverted_from` names where it was headed, and the route it left is
    never called."""
    strategy = ByText()
    router, routes = mixed(strategy)
    bound = bind(binder, router)

    answer, heard = await routed(bound, convention, "cheap")

    message = binder.message(answer)
    assert message.content == "frontier answer"
    assert routing_decision(message) == diverted("cheap", to="frontier")
    assert [(w.category, str(w.message)) for w in heard] == [
        (ToolSupportWarning, "'cheap' can't use the bound tools; diverted to 'frontier'")
    ]
    assert [len(call_log(routes[name])) for name in ("first", "cheap", "frontier")] == [0, 0, 1]
    assert strategy.asked == 1


@pytest.mark.parametrize("chosen", ["local", "cheap"])
@pytest.mark.parametrize("binder", BINDERS)
async def test_when_the_default_cannot_use_tools_a_diverted_request_goes_to_the_first_that_can(
    binder: Binder, chosen: str
) -> None:
    """With the default route unable to take a diversion, it goes to the first
    tool-capable route in declaration order — not the last, and not one the strategy would
    have liked better. `chosen` covers both a route that isn't the default and the default
    itself."""
    router, routes = router_of(
        {"cheap": False, "local": False, "first": True, "second": True},
        default="cheap",
        strategy=ByText(),
    )
    bound = bind(binder, router)

    answer, heard = await routed(bound, "invoke", chosen)

    assert routing_decision(binder.message(answer)) == diverted(chosen, to="first")
    assert [w.category for w in heard] == [ToolSupportWarning]
    assert [len(call_log(routes[name])) for name in ("first", "second")] == [1, 0]


BOUND_CONVENTIONS = tuple(c for c in ALL_CONVENTIONS if c not in ("generate", "agenerate"))
"""The conventions a *bound* router answers: `generate` reaches a binding only through
`__getattr__`, which drops the bound arguments on any chat model (see `respond`)."""


@pytest.mark.parametrize("convention", BOUND_CONVENTIONS)
async def test_every_calling_convention_diverts_and_records_it(convention: Convention) -> None:
    """A diverted request carries the record with `diverted_from` however
    it was called — including `batch` and event streams — and warns exactly once."""
    router, routes = mixed()
    bound = bind(TOOLS, router)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(bound, convention, "cheap")

    assert message.content == "frontier answer"
    assert routing_decision(message) == diverted("cheap", to="frontier")
    assert [w.category for w in routing_warnings(caught)] == [ToolSupportWarning]
    assert [len(call_log(routes[name])) for name in ("first", "cheap", "frontier")] == [0, 0, 1]


@pytest.mark.parametrize("convention", ["generate", "agenerate"])
async def test_generate_diverts_when_the_tools_are_passed_as_a_call_argument(
    convention: Literal["generate", "agenerate"],
) -> None:
    """`generate` takes its tools as a call argument, as on any
    chat model, and diverts a request off a route that can't use them — the diversion recorded,
    one warning, and no second model run."""
    router, routes = mixed()
    raw = [convert_to_openai_tool(get_weather)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await generated(router, convention, [[HumanMessage("cheap")]], tools=raw)

    (generation,) = result.generations[0]
    assert isinstance(generation, ChatGeneration)
    assert generation.message.content == "frontier answer"
    assert routing_decision(generation.message) == diverted("cheap", to="frontier")
    assert [w.category for w in routing_warnings(caught)] == [ToolSupportWarning]
    assert [len(call_log(routes[name])) for name in ("first", "cheap", "frontier")] == [0, 0, 1]


async def test_one_warning_is_raised_for_each_diverted_request() -> None:
    """Per request, and only for the requests that were diverted."""
    router, _ = mixed()
    bound = bind(TOOLS, router)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answers = [bound.invoke(text) for text in ("cheap", "first", "cheap", "frontier")]

    assert [record_of(answer).route for answer in answers] == [
        "frontier",
        "first",
        "frontier",
        "frontier",
    ]
    assert [str(w.message) for w in routing_warnings(caught)] == [
        "'cheap' can't use the bound tools; diverted to 'frontier'"
    ] * 2


class _DiversionCase(NamedTuple):
    """One way a request ends up diverted, and what the trace should show for it."""

    build: Callable[[], ChatRouter]
    text: str
    record: RoutingDecision
    has_strategy_run: bool
    """Whether a strategy ran at all — no strategy means no strategy run, so there are
    only two placements to compare, not three."""


def _with_a_strategy() -> ChatRouter:
    router, _ = mixed(ByText())
    return router


def _with_no_strategy() -> ChatRouter:
    router, _ = router_of({"cheap": False, "first": True}, default="cheap")
    return router


def _falling_back_onto_an_incapable_default() -> ChatRouter:
    router, _ = router_of({"cheap": False, "first": True}, default="cheap", strategy=Abstains())
    return router


DIVERSION_CASES = [
    pytest.param(
        _DiversionCase(
            _with_a_strategy,
            "cheap",
            diverted("cheap", to="frontier"),
            has_strategy_run=True,
        ),
        id="with a strategy",
    ),
    pytest.param(
        _DiversionCase(
            _with_no_strategy,
            "hello",
            RoutingDecision(
                route="first",
                reason="no strategy configured; 'cheap' can't use the bound tools, so it "
                "was diverted to 'first'",
                diverted_from="cheap",
            ),
            has_strategy_run=False,
        ),
        id="no strategy",
    ),
    pytest.param(
        _DiversionCase(
            _falling_back_onto_an_incapable_default,
            "hello",
            RoutingDecision(
                route="first",
                reason="Abstains could not decide; fell back to the default route; 'cheap' "
                "can't use the bound tools, so it was diverted to 'first'",
                strategy="Abstains",
                fallback=True,
                diverted_from="cheap",
            ),
            has_strategy_run=True,
        ),
        id="fallback onto an incapable default",
    ),
]


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
@pytest.mark.parametrize("case", DIVERSION_CASES)
async def test_the_diverted_record_is_the_same_in_all_of_d9_s_places(
    case: _DiversionCase, convention: AskConvention
) -> None:
    """A diversion is settled inside the decision step (`_decide`), before
    the strategy run closes — so the strategy run's output, the router run's outputs, the route
    run's metadata and the response all carry the *same* diverted record, not the strategy's
    undiverted choice. The strategy still ran once (no second consultation) and its run is
    still a success — diverting is not a failure of the strategy."""
    router = case.build()
    bound = bind(TOOLS, router)
    collector = RunCollectorCallbackHandler()
    config: RunnableConfig = {"callbacks": [collector]}

    async def call() -> Any:
        if convention == "invoke":
            return bound.invoke(case.text, config)
        return await bound.ainvoke(case.text, config)

    with pytest.warns(ToolSupportWarning):
        message = await call()

    (router_run,) = collector.traced_runs
    if case.has_strategy_run:
        strategy_run, route_run = router_run.child_runs
        assert strategy_run.run_type == "chain"
        assert strategy_run.error is None  # diverting is not a strategy failure
        assert strategy_run.outputs == case.record.as_dict()
    else:
        (route_run,) = router_run.child_runs
    assert route_run.run_type == "llm"
    assert routing_decision(message) == case.record
    assert (router_run.outputs or {})[ROUTING_KEY] == case.record.as_dict()
    assert metadata_of(route_run)[ROUTING_KEY] == case.record.as_dict()
    # Exactly one child run of each kind proves the strategy was not consulted a second time
    # for the diversion — a re-run would show as a second chain run among the children.
    assert [run.run_type for run in router_run.child_runs].count("chain") == (
        1 if case.has_strategy_run else 0
    )


async def test_an_escalated_diversion_warning_closes_both_runs() -> None:
    """`_divert`'s warning can be escalated to an error like any other; the
    `try`/`except` in `_decide` that already closes the strategy run when `_conclude` raises
    covers `_divert` too (it runs inside the same block), so the strategy run closes as an
    error and the router run closes as an error behind it — neither is left open."""
    strategy = ByText()
    router, _ = mixed(strategy)
    bound = bind(TOOLS, router)
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ToolSupportWarning)
        with pytest.raises(ToolSupportWarning):
            bound.invoke("cheap", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, *rest = router_run.child_runs
    assert router_run.error is not None
    assert strategy_run.error is not None
    assert all(run.end_time is not None for run in (router_run, strategy_run, *rest))


async def test_a_request_with_nothing_bound_is_never_diverted() -> None:
    """Tool-aware routing applies to tools and structured output, and only when they are bound: the
    same router, called plainly, sends a request to the route the strategy chose."""
    router, routes = mixed()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke("cheap")

    assert routing_decision(message) == RoutingDecision(
        route="cheap", reason="asked for 'cheap'", strategy="ByText"
    )
    assert message.content == "cheap answer"
    assert routing_warnings(caught) == []
    assert len(call_log(routes["cheap"])) == 1


async def test_a_fallback_onto_a_default_that_cannot_use_tools_is_diverted_too() -> None:
    """The default route is where a strategy that can't decide sends the
    request, and it may itself be a route that can't use the tools. The request is diverted
    like any other, and the record keeps both facts — it fell back, and it was diverted — with
    a warning each."""
    router, _ = router_of({"cheap": False, "first": True}, default="cheap", strategy=Abstains())
    bound = bind(TOOLS, router)

    answer, heard = await routed(bound, "invoke", "hello")

    assert routing_decision(answer) == RoutingDecision(
        route="first",
        reason=(
            "Abstains could not decide; fell back to the default route; "
            "'cheap' can't use the bound tools, so it was diverted to 'first'"
        ),
        strategy="Abstains",
        fallback=True,
        diverted_from="cheap",
    )
    assert [w.category for w in heard] == [FallbackWarning, ToolSupportWarning]


async def test_with_no_strategy_a_default_that_cannot_use_tools_is_diverted() -> None:
    """No strategy means the default route, and the rule is the same — the request
    goes to the first route that can use the tools, and no strategy run is opened for it."""
    router, _ = router_of({"cheap": False, "first": True}, default="cheap")
    bound = bind(TOOLS, router)
    collector = RunCollectorCallbackHandler()

    answer, heard = await routed(bound, "invoke", "hello")
    with pytest.warns(ToolSupportWarning):
        bound.invoke("hello", {"callbacks": [collector]})

    assert routing_decision(answer) == RoutingDecision(
        route="first",
        reason="no strategy configured; 'cheap' can't use the bound tools, so it was diverted "
        "to 'first'",
        diverted_from="cheap",
    )
    assert [w.category for w in heard] == [ToolSupportWarning]
    (router_run,) = collector.traced_runs
    assert [run.run_type for run in router_run.child_runs] == ["llm"]


async def test_tools_bound_past_the_bind_time_check_still_cannot_reach_an_incapable_route() -> None:
    """`bind_tools` refuses to bind when no route can use tools, but
    `bind(tools=...)` puts them in the call kwargs directly and never asks. The request is then
    the last chance, and it fails as a `NoToolCapableRouteError` — with the router's run closed
    as an error — rather than as the route's own `NotImplementedError`."""
    router, routes = router_of({"a": False, "b": False}, default="a", strategy=ByText())
    bound = router.bind(tools=[convert_to_openai_tool(get_weather)])
    collector = RunCollectorCallbackHandler()

    with pytest.raises(NoToolCapableRouteError):
        bound.invoke("a", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    assert "NoToolCapableRouteError" in (router_run.error or "")
    assert [call_log(route) for route in routes.values()] == [[], []]


# --- What a strategy is told: `RoutingRequest.tools_bound` ---

TOOLS_BOUND = [
    pytest.param(lambda router: router, False, id="nothing bound"),
    pytest.param(lambda router: router.bind_tools([get_weather]), True, id="bind_tools"),
    pytest.param(
        lambda router: router.with_structured_output(Answer), True, id="with_structured_output"
    ),
    pytest.param(
        lambda router: router.bind(tools=[convert_to_openai_tool(get_weather)]),
        True,
        id="bind(tools=...)",
    ),
]


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
@pytest.mark.parametrize(("bind_it", "expected"), TOOLS_BOUND)
async def test_a_strategy_is_told_whether_tools_or_structured_output_are_bound(
    bind_it: Callable[[ChatRouter], Runnable[LanguageModelInput, Any]],
    expected: bool,
    convention: AskConvention,
) -> None:
    """`RoutingRequest.tools_bound` is true whenever tools *or structured output* are
    bound — the latter leaves nothing in the call kwargs to read, so the router has to know —
    and false otherwise. The strategy's run in the trace records the same."""
    seen: list[bool] = []

    def records(request: RoutingRequest) -> str:
        seen.append(request.tools_bound)
        return "a"

    router = ChatRouter(routes={"a": capable("a")}, default_route="a", strategy=records)
    collector = RunCollectorCallbackHandler()

    await ask(bind_it(router), convention, "hello")
    bind_it(router).invoke("hello", {"callbacks": [collector]})

    assert seen == [expected, expected]
    (router_run,) = collector.traced_runs
    strategy_run, _route_run = router_run.child_runs
    assert (strategy_run.inputs or {})["tools_bound"] is expected


# --- the binding is replayed on the route that was chosen ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_each_route_receives_the_tools_in_its_own_form(convention: Convention) -> None:
    """The router keeps the tools as given, and the route the strategy picks converts
    them — two routes with different conversions each get their own, from the same binding.

    Nothing is converted at bind time (no route's `bind_tools` has run), the route's binder
    receives the caller's own objects, and what reaches the provider is the route's form of
    them — not the raw list the router also holds, and not the router's private
    binding kwarg."""
    routes: dict[str, BaseChatModel] = {
        "openai": capable("openai"),
        "anthropic": capable("anthropic", tool_format="anthropic"),
    }
    router = ChatRouter(routes=routes, default_route="openai", strategy=ByText())

    bound = router.bind_tools([get_weather, Answer])
    assert [cast("NativeStructuredFakeChatModel", r).bind_calls for r in routes.values()] == [
        [],
        [],
    ]

    await respond(bound, convention, "openai")
    await respond(bound, convention, "anthropic")

    openai, anthropic = (cast("NativeStructuredFakeChatModel", r) for r in routes.values())
    assert openai.bind_calls == [{"tools": [get_weather, Answer], "tool_choice": None}]
    assert anthropic.bind_calls == [{"tools": [get_weather, Answer], "tool_choice": None}]
    assert openai.bind_calls[0]["tools"][0] is get_weather  # the caller's own object
    (openai_call,) = openai.calls
    (anthropic_call,) = anthropic.calls
    assert openai_call == {
        "tools": [convert_to_openai_tool(get_weather), convert_to_openai_tool(Answer)]
    }
    assert [spec["function"]["name"] for spec in openai_call["tools"]] == ["get_weather", "Answer"]
    assert list(anthropic_call) == ["tools"]
    assert [set(spec) for spec in anthropic_call["tools"]] == [
        {"name", "description", "input_schema"}
    ] * 2
    assert [spec["name"] for spec in anthropic_call["tools"]] == ["get_weather", "Answer"]


BINDING_KWARGS = [
    pytest.param(
        {"tool_choice": "any", "strict": True, "parallel_tool_calls": False},
        id="tool_choice and provider kwargs",
    ),
    pytest.param({"strict": True}, id="a provider kwarg alone"),
    pytest.param({"tool_choice": "get_weather"}, id="tool_choice alone"),
]


@pytest.mark.parametrize("kwargs", BINDING_KWARGS)
@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_binding_kwargs_reach_the_route_s_binder_and_not_its_call(
    convention: Convention, kwargs: dict[str, Any]
) -> None:
    """`tool_choice` and binding kwargs such as `strict=` are replayed on the route's
    own `bind_tools` — the same call, with the same arguments, as binding the route directly,
    whether or not a `tool_choice` comes with them.

    `strict=` is the one that matters: it changes how the schema is *built*, so passed as a call
    kwarg it would be too late and never read. It has done its work at bind time
    (`function.strict`) and is not in the call; the other kwargs are the route's own to bind."""
    through_router, direct = capable("a"), capable("a")
    router = ChatRouter(routes={"a": through_router}, default_route="a")

    await respond(router.bind_tools([get_weather], **kwargs), convention, "hello")
    await respond(direct.bind_tools([get_weather], **kwargs), convention, "hello")

    expected = {"tools": [get_weather], "tool_choice": None, **kwargs}
    assert through_router.bind_calls == direct.bind_calls == [expected]
    assert through_router.calls == direct.calls
    (call,) = through_router.calls
    assert "strict" not in call
    assert call["tools"][0]["function"].get("strict") is kwargs.get("strict")
    bound_by_the_route = {key: value for key, value in call.items() if key != "tools"}
    assert bound_by_the_route == {k: v for k, v in kwargs.items() if k != "strict"}


def test_bind_tools_puts_nothing_but_the_private_binding_in_the_kwargs() -> None:
    """The router binds no `tools` kwarg of its own — a consumer that reads
    one off a chat model's binding would misread the raw, unconverted objects as the converted
    form a plain model puts there. Only the private binding key rides along."""
    router, _ = router_of({"a": True}, default="a")

    bound = router.bind_tools([get_weather, Answer], tool_choice="any")

    assert isinstance(bound, RunnableBinding)
    assert list(bound.kwargs) == [BINDING_KEY]
    assert "tools" not in bound.kwargs
    assert "tool_choice" not in bound.kwargs  # a binding kwarg, not a call kwarg


def _agent_script() -> list[AIMessage]:
    """A tool call, then a final answer — what a `create_react_agent` / `create_agent` loop
    needs from the route it eventually reaches."""
    return [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="get_weather", args={"city": "Paris"}, id="c1", type="tool_call")
            ],
        ),
        AIMessage(content="It is sunny in Paris"),
    ]


def test_create_react_agent_builds_and_answers_with_a_pre_bound_router() -> None:
    """`create_react_agent` reads a bound model's own `bind_tools` output to decide
    whether to bind tools again (`_should_bind_tools`, `langgraph/prebuilt/chat_agent_executor.py`)
    — before the fix this read the raw `tools` slot and crashed with `AttributeError:
    'StructuredTool' object has no attribute 'get'`. With no slot to misread, the agent builds
    and completes its tool loop through the router exactly as it would through a plain model."""
    from langgraph.prebuilt import create_react_agent

    route = ToolCallingFakeChatModel(model_name="a", script=_agent_script())
    router = ChatRouter(routes={"a": route}, default_route="a")

    agent = create_react_agent(router.bind_tools([get_weather]), [get_weather])
    out = agent.invoke({"messages": [{"role": "user", "content": "weather in Paris?"}]})

    assert out["messages"][-1].content == "It is sunny in Paris"
    # Two turns of the loop, each a request through the router: the binding is replayed on the
    # route at call time, so it is replayed once per turn, not once for the agent.
    assert route.bind_calls == [{"tools": [get_weather], "tool_choice": None}] * 2


def test_create_agent_builds_and_answers_with_a_pre_bound_router() -> None:
    """`langchain.agents.create_agent` is the other consumer the reviewer named; the
    same pre-bound router builds and answers through it."""
    from langchain.agents import create_agent

    route = ToolCallingFakeChatModel(model_name="a", script=_agent_script())
    router = ChatRouter(routes={"a": route}, default_route="a")
    # `create_agent` types `model` as `str | BaseChatModel`; a bound model is a `Runnable`, not
    # a `BaseChatModel`, on the router as on any chat model (the whole point — LangChain
    # accepts it at runtime; the annotation just doesn't say so).
    bound = cast("BaseChatModel", router.bind_tools([get_weather]))

    agent = create_agent(model=bound, tools=[get_weather])
    out = agent.invoke({"messages": [{"role": "user", "content": "weather in Paris?"}]})

    assert out["messages"][-1].content == "It is sunny in Paris"


def test_a_later_bare_bind_tools_is_honoured() -> None:
    """With no slot for `bound_route` to discard, a bare `bind(tools=...)` layered on
    top of `bind_tools` reaches the route as a call kwarg — exactly as it does on a plain chat
    model, where the later binding wins over the earlier one's converted slot."""
    explicit = [
        {
            "type": "function",
            "function": {
                "name": "explicit",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    plain, routed_route = capable("a"), capable("a")
    router = ChatRouter(routes={"a": routed_route}, default_route="a")

    plain.bind_tools([get_weather]).bind(tools=explicit).invoke("hi")
    router.bind_tools([get_weather]).bind(tools=explicit).invoke("hi")

    assert routed_route.calls == plain.calls
    assert [spec["function"]["name"] for spec in routed_route.calls[-1]["tools"]] == ["explicit"]


async def chunks_of(
    model: Runnable[LanguageModelInput, AIMessage], convention: Literal["stream", "astream"]
) -> list[AIMessage]:
    if convention == "stream":
        return list(model.stream("hello"))
    return [chunk async for chunk in model.astream("hello")]


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_disable_streaming_for_tool_calling_behaves_as_on_the_route(
    convention: Literal["stream", "astream"],
) -> None:
    """A route with `disable_streaming="tool_calling"` does not stream while tools
    are bound and does otherwise — and bound through the router it does exactly the same. The
    unbound calls are the control: streaming is on, so the difference is the tools."""
    route = capable("a", disable_streaming="tool_calling")
    router = ChatRouter(routes={"a": route}, default_route="a")

    direct_bound = await chunks_of(route.bind_tools([get_weather]), convention)
    routed_bound = await chunks_of(router.bind_tools([get_weather]), convention)
    direct_plain = await chunks_of(route, convention)
    routed_plain = await chunks_of(router, convention)

    assert [chunk.content for chunk in routed_bound] == [chunk.content for chunk in direct_bound]
    assert [chunk.content for chunk in routed_bound] == ["a answer"]
    assert [chunk.content for chunk in routed_plain] == [chunk.content for chunk in direct_plain]
    assert len(routed_plain) > 1


# --- `with_structured_output` reaches the route's own implementation ---


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_method_and_strict_reach_the_route_s_structured_output(
    convention: AskConvention,
) -> None:
    """The base default pops `method=` and `strict=` and drops them
    (`chat_models.py:2530`), so every route would be forced through function calling. The
    override forwards them, and the answer is the route's own parse of its own reply."""
    route = capable("a")
    router = ChatRouter(routes={"a": route}, default_route="a")

    structured = router.with_structured_output(Answer, method="json_schema", strict=True)
    result = await ask(structured, convention, "hello")

    assert result == Answer(answer="42")
    assert route.structured_output_calls == [
        {"method": "json_schema", "include_raw": False, "strict": True}
    ]


def test_a_route_keeps_its_own_structured_output_defaults() -> None:
    """What the caller didn't pass isn't passed — no `strict`, and the route picks
    its own `method` — and the schema reaches the route as it was given."""
    route = capable("a")
    router = ChatRouter(routes={"a": route}, default_route="a")

    router.with_structured_output(Answer).invoke("hello")

    assert route.structured_output_calls == [{"method": "function_calling", "include_raw": False}]
    assert route.bind_calls[0]["tools"] == [Answer]


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_include_raw_returns_the_usual_triple(convention: AskConvention) -> None:
    """`include_raw=True` answers `{"raw", "parsed", "parsing_error"}` as it does on
    any chat model, with the route's own message under `raw`."""
    router, _ = router_of({"a": True}, default="a")

    result = await ask(router.with_structured_output(Answer, include_raw=True), convention, "hi")

    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert result["parsed"] == Answer(answer="42")
    assert result["parsing_error"] is None
    assert isinstance(result["raw"], AIMessage)
    assert result["raw"].tool_calls == [ANSWER_CALL]


def test_a_parse_failure_is_reported_under_parsing_error() -> None:
    """The route's parser is the one that runs, so its failure is reported the way
    LangChain reports it — `parsed` is `None`, `parsing_error` says why — and the raw message
    is still there, record included."""
    bad_call = ToolCall(name="Answer", args={}, id="call_1", type="tool_call")
    route = capable("a", script=[AIMessage(content="", tool_calls=[bad_call])])
    router = ChatRouter(routes={"a": route}, default_route="a")

    result = cast(
        "dict[str, Any]", router.with_structured_output(Answer, include_raw=True).invoke("hello")
    )

    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], Exception)
    assert result["raw"].tool_calls == [bad_call]
    assert routing_decision(result["raw"]) is not None


def test_a_dict_schema_answers_a_dict() -> None:
    """The schema is given to the route as it came, so a JSON-schema dict gets the
    route's dict-shaped parse, not a Pydantic object."""
    router, _ = router_of({"a": True}, default="a")

    result = router.with_structured_output(convert_to_openai_tool(Answer)).invoke("hello")

    assert result == {"answer": "42"}


async def test_structured_output_streams_whatever_the_route_itself_gives() -> None:
    """`capable()`'s fake hands its tool call over in one final chunk (as some real
    routes do too), so `stream`/`astream` answer with the one item that produces — not because
    the router holds a stream back to one item, which `test_structured_streaming_*` next to
    this disproves for a route that streams progressively."""
    router, _ = router_of({"a": True}, default="a")
    structured = router.with_structured_output(Answer)

    assert list(structured.stream("hello")) == [Answer(answer="42")]
    assert [item async for item in structured.astream("hello")] == [Answer(answer="42")]


# --- Progressive structured streaming: the router matches a plain route exactly ---

StreamConvention = Literal["stream", "astream"]


class Person(BaseModel):
    """A person mentioned in the text."""

    name: str
    age: int
    city: str


PERSON_CALL = ToolCall(
    name="Person", args={"name": "Ada", "age": 36, "city": "London"}, id="call_1", type="tool_call"
)


def streaming_route(**fields: Any) -> StreamingStructuredFakeChatModel:
    """A route whose tool call streams in several pieces, as a real provider's does."""
    return StreamingStructuredFakeChatModel(model_name="p", tool_calls=[PERSON_CALL], **fields)


async def stream_all(
    model: Runnable[LanguageModelInput, Any], convention: StreamConvention, text: str = "hello"
) -> list[Any]:
    if convention == "stream":
        return list(model.stream(text))
    return [item async for item in model.astream(text)]


def reduced(items: list[Any]) -> Any:
    """`items` folded together the way `ChatRouter.stream` folds them for its own run output:
    `+` for a delta, replaced by the newest when that raises — a parsed partial is
    cumulative, not a delta. What a caller gets is each `item` on its own, unfolded; this is
    only for asserting on the *final* state the whole sequence adds up to."""
    total = items[0]
    for item in items[1:]:
        try:
            total = total + item
        except TypeError:
            total = item
    return total


PERSON_DICT: dict[str, Any] = {"name": "Ada", "age": 36, "city": "London"}
PERSON = Person(name="Ada", age=36, city="London")

STRUCTURED_SCHEMAS = [
    pytest.param({"schema": Person}, PERSON, id="pydantic"),
    pytest.param({"schema": Person.model_json_schema()}, PERSON_DICT, id="dict schema"),
    pytest.param({"schema": Person, "method": "json_mode"}, PERSON_DICT, id="json_mode"),
]


@pytest.mark.parametrize("include_raw", [False, True], ids=["", "include_raw"])
@pytest.mark.parametrize("convention", ["stream", "astream"])
@pytest.mark.parametrize(("case", "final"), STRUCTURED_SCHEMAS)
async def test_structured_streaming_matches_the_plain_route_exactly(
    case: dict[str, Any], final: object, convention: StreamConvention, include_raw: bool
) -> None:
    """The reviewer's measurement, pinned — a route that streams progressive
    tool-call or JSON-mode chunks answers `with_structured_output(...).stream()`/`.astream()`
    through the router exactly as it does when called directly: the same number of items
    (genuinely more than one, the whole point of the fix), the same `parsed` values — up to
    ordering, which is `RunnablePassthrough.assign`'s own race between its `parsed` and
    `parsing_error` steps (`runnables/passthrough.py`), true of the plain route too and not
    anything routing adds — and the same final state. `StructuredRouter.stream`/`astream`
    delegate to the router's own, which passes each of the route's own items through unchanged
    bar the record — which is why `raw` is not compared here: it is the one item that legitimately
    differs, by carrying it.
    """
    direct = streaming_route().with_structured_output(include_raw=include_raw, **case)
    through_router = ChatRouter(routes={"a": streaming_route()}, default_route="a")
    routed = through_router.with_structured_output(include_raw=include_raw, **case)

    direct_items = await stream_all(direct, convention)
    routed_items = await stream_all(routed, convention)

    assert len(routed_items) == len(direct_items)
    assert len(routed_items) > 1  # genuinely progressive, not the finished object once
    if include_raw:
        parsed = [cast("dict[str, Any]", item).get("parsed") for item in routed_items]
        direct_parsed = [cast("dict[str, Any]", item).get("parsed") for item in direct_items]
        assert sorted(map(repr, parsed)) == sorted(map(repr, direct_parsed))
        assert reduced(routed_items)["parsed"] == final
    else:
        assert routed_items == direct_items
        assert reduced(routed_items) == final


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_exactly_one_structured_item_carries_the_record(convention: StreamConvention) -> None:
    """With `include_raw=True`, exactly one item's `raw` message carries the
    routing record — the first, which the reviewer's measurement shows always has a `"raw"`
    key before any `"parsed"` key can (the parser needs raw content first)."""
    router = ChatRouter(routes={"a": streaming_route()}, default_route="a")
    structured = router.with_structured_output(Person, include_raw=True)

    items = cast("list[dict[str, Any]]", await stream_all(structured, convention))

    carriers = [
        index
        for index, item in enumerate(items)
        if isinstance(item.get("raw"), AIMessage) and routing_decision(item["raw"]) is not None
    ]
    assert carriers == [0]
    assert routing_decision(items[0]["raw"]) == RoutingDecision(
        route="a", reason="no strategy configured"
    )


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_a_parsed_only_structured_stream_still_sets_the_last_decision(
    convention: StreamConvention,
) -> None:
    """Nothing in a parsed-only progressive stream can carry the record (no item is a
    message, or a mapping with one under `"raw"`), so `last_routing_decision()` is what a
    caller reads — set as soon as the first partial arrives, same as for a whole answer."""
    router = ChatRouter(
        routes={"a": streaming_route(), "b": streaming_route()},
        default_route="a",
        strategy=ByText(),
    )
    structured = router.with_structured_output(Person)

    items = await stream_all(structured, convention, "a")

    assert len(items) > 1
    assert items[-1] == Person(name="Ada", age=36, city="London")
    assert last_routing_decision() == RoutingDecision(
        route="a", reason="asked for 'a'", strategy="ByText"
    )


def test_an_abandoned_structured_stream_keeps_its_record() -> None:
    """The `GeneratorExit` handling covers a structured stream too — stopping early
    is not a failure, and the record the first item published stays readable."""
    router = ChatRouter(routes={"a": streaming_route()}, default_route="a")
    structured = router.with_structured_output(Person)

    for _item in structured.stream("hello"):
        break
    gc.collect()  # the generator the loop dropped is finalized here at the latest

    assert last_routing_decision() == RoutingDecision(route="a", reason="no strategy configured")


def test_a_structured_stream_that_fails_withdraws_the_record() -> None:
    """A route that raises partway through a structured stream closes the router's run
    as an error and withdraws the record, exactly as a plain message stream does."""

    class FailsPartway(StreamingStructuredFakeChatModel):
        def _stream(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Iterator[Any]:
            iterator = super()._stream(messages, stop, run_manager, **kwargs)
            yield next(iterator)
            msg = "the provider is down"
            raise RuntimeError(msg)

    route = FailsPartway(model_name="p", tool_calls=[PERSON_CALL])
    router = ChatRouter(routes={"a": route}, default_route="a")
    structured = router.with_structured_output(Person)
    collector = RunCollectorCallbackHandler()

    with pytest.raises(RuntimeError, match="the provider is down"):
        list(structured.stream("hello", {"callbacks": [collector]}))

    assert last_routing_decision() is None
    (router_run,) = collector.traced_runs
    assert "the provider is down" in (router_run.error or "")


async def test_structured_output_batches_each_request_to_its_own_route() -> None:
    """`batch` and `abatch` are `Runnable`'s, built on the calling convention
    the structured output has — each input is routed on its own, and each raw message carries
    its own record."""
    router, _ = router_of({"a": True, "b": True}, default="a", strategy=ByText())
    structured = router.with_structured_output(Answer, include_raw=True)

    batched = cast("list[dict[str, Any]]", structured.batch(["a", "b"]))
    abatched = cast("list[dict[str, Any]]", await structured.abatch(["b", "a"]))

    assert [record_of(result["raw"]).route for result in batched] == ["a", "b"]
    assert [record_of(result["raw"]).route for result in abatched] == ["b", "a"]
    assert [result["parsed"] for result in (*batched, *abatched)] == [Answer(answer="42")] * 4


def test_structured_output_forwards_the_router_s_config_specs() -> None:
    """`StructuredRouter.config_specs` is the router's own, not `Runnable`'s default `[]`
    — so a spec the router declares is visible through `with_structured_output(...)` too,
    for `with_config`, `config={"configurable": …}` and config-schema introspection run on
    the structured runnable rather than the router itself. The router's real `"route"` key
    is covered end to end in `test_forced_routes.py`; this checks the forwarding
    itself, generically, with a throwaway spec added through a subclass."""
    spec = ConfigurableFieldSpec(id="probe", annotation=str, default="x")

    class WithASpec(ChatRouter):
        @property
        def config_specs(self) -> list[ConfigurableFieldSpec]:
            return [spec]

    router = WithASpec(routes={"a": capable("a")}, default_route="a")

    assert router.config_specs == [spec]
    assert router.with_structured_output(Answer).config_specs == [spec]


# --- structured output has an answer for the decision record ---


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_with_include_raw_the_raw_message_carries_the_record(
    convention: AskConvention,
) -> None:
    """The record rides on the route's own message under `raw`, and the parsed
    object beside it is the route's parse, untouched."""
    router, _ = router_of({"a": True, "b": True}, default="a", strategy=ByText())

    result = await ask(router.with_structured_output(Answer, include_raw=True), convention, "b")

    assert routing_decision(result["raw"]) == RoutingDecision(
        route="b", reason="asked for 'b'", strategy="ByText"
    )
    assert result["parsed"] == Answer(answer="42")


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_a_parsed_only_call_is_answered_by_the_last_decision(
    convention: AskConvention,
) -> None:
    """Without `include_raw=True` the caller holds a parsed object with nowhere
    to carry a record, and `last_routing_decision()` is the answer.

    An earlier call, routed elsewhere, comes first: whatever the previous test — or this
    thread's last request — left behind is superseded by *this* call's record, or the
    assertion could pass on a stale one."""
    router, _ = router_of({"a": True, "b": True}, default="a", strategy=ByText())
    await ask(router, convention, "a")

    parsed = await ask(router.with_structured_output(Answer), convention, "b")

    assert parsed == Answer(answer="42")
    assert last_routing_decision() == RoutingDecision(
        route="b", reason="asked for 'b'", strategy="ByText"
    )


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_the_last_decision_survives_a_structured_call_made_as_a_sequence_step(
    convention: AskConvention,
) -> None:
    """`prompt | structured` runs the structured call in a *copy* of the caller's
    context, which discards what it sets — the shape the thread-level record exists for."""
    router, _ = router_of({"a": True, "b": True}, default="a", strategy=ByText())
    await ask(router, convention, "a")
    chain = RunnableLambda(lambda text: text) | router.with_structured_output(Answer)

    await ask(chain, convention, "b")

    assert last_routing_decision() == RoutingDecision(
        route="b", reason="asked for 'b'", strategy="ByText"
    )


@pytest.mark.parametrize("binder", BINDERS)
async def test_a_bound_call_is_one_router_run_over_one_model_run(binder: Binder) -> None:
    """Tools or structured output, the router's own run is the root and
    carries the record, and the route's call is the only model run below it — the strategy's
    run opens none — so cost is counted once."""
    router, _ = router_of({"a": True, "b": True}, default="a", strategy=ByText())
    collector = RunCollectorCallbackHandler()

    answer = binder.bind(router).invoke("b", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    record = RoutingDecision(route="b", reason="asked for 'b'", strategy="ByText")
    assert routing_decision(binder.message(answer)) == record
    assert (router_run.outputs or {})[ROUTING_KEY] == record.as_dict()
    (model_run,) = model_runs([router_run])
    assert metadata_of(model_run)[ROUTING_KEY] == record.as_dict()


class DecideFailsAdecideWorks(RoutingStrategy):
    """`decide` (sync) always raises; `adecide` overrides with its own working implementation.

    Tells apart `StructuredRouter.ainvoke` calling `router.ainvoke` from calling the *sync*
    `router.invoke` by mistake (mutation M25, previously uncaught): the wrong call would reach
    `decide`, not `adecide`, and surface as `Raises raised AssertionError` absorbed into a
    fallback instead of the strategy's own choice.
    """

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        msg = "decide must not run on the async path"
        raise AssertionError(msg)

    async def adecide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route=request.text, reason="adecide chose it")


async def test_structured_ainvoke_awaits_the_router_s_own_ainvoke() -> None:
    """Mutation M25: `StructuredRouter.ainvoke` has to await `router.ainvoke`, and not fall back
    to calling the sync `router.invoke` — a strategy whose sync half fails and whose async half
    works is the one case that tells the two apart, since anything symmetric between them
    can't."""
    router = ChatRouter(
        routes={"a": capable("a")}, default_route="a", strategy=DecideFailsAdecideWorks()
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await router.with_structured_output(Answer).ainvoke("a")

    assert result == Answer(answer="42")
    assert routing_warnings(caught) == []


# --- A route bound more than once ---


def test_binding_twice_keeps_the_last_binding_as_on_any_chat_model() -> None:
    """A binding over a binding reaches the model's own `bind_tools` again
    (`RunnableBinding.__getattr__`), so the second call replaces the first rather than adding
    to it. The router does what a plain tool-capable chat model does, and the check is
    against one: what the route's last binder saw, and what its call got, are identical.

    A plain model converts eagerly, so it also ran the *first* binder; the router converts only
    what is replayed, so the route sees one."""
    plain, routed_route = capable("a"), capable("a")
    router = ChatRouter(routes={"a": routed_route}, default_route="a")

    rebind(plain.bind_tools([get_weather]), [Answer]).invoke("hello")
    rebind(router.bind_tools([get_weather]), [Answer]).invoke("hello")

    assert routed_route.bind_calls == plain.bind_calls[-1:]
    assert routed_route.bind_calls == [{"tools": [Answer], "tool_choice": None}]
    assert routed_route.calls == plain.calls
    assert [spec["function"]["name"] for spec in routed_route.calls[0]["tools"]] == ["Answer"]


def test_structured_output_after_tools_replaces_them_as_on_any_chat_model() -> None:
    """The same rule across the two kinds of binding: `with_structured_output` on a
    runnable already bound with tools is the model's own `with_structured_output`, and the
    tools the first binding held are not part of it — on a plain chat model and through the
    router."""
    plain, routed_route = capable("a"), capable("a")
    router = ChatRouter(routes={"a": routed_route}, default_route="a")

    cast("Any", plain.bind_tools([get_weather])).with_structured_output(Answer).invoke("hello")
    cast("Any", router.bind_tools([get_weather])).with_structured_output(Answer).invoke("hello")

    assert routed_route.bind_calls == plain.bind_calls[-1:]
    assert [call["tools"] for call in routed_route.bind_calls] == [[Answer]]
    assert routed_route.structured_output_calls == plain.structured_output_calls
    assert routed_route.calls == plain.calls


# --- Where each warning points ---
#
# `warnings.warn(stacklevel=n)` counts frames, and the two ToolSupportWarnings are raised at
# different depths from `FallbackWarning`, so they have counts of their own (`_BINDING_CALLER`,
# `_BOUND_CALLER` in `router.py`). A miscount is silent: the warning still fires, and names a
# frame inside LangChain instead of the line an application can act on.


def next_line() -> int:
    """The line after the one that called this — where the caller's next statement sits."""
    frame = inspect.currentframe()
    assert frame is not None
    assert frame.f_back is not None
    return frame.f_back.f_lineno + 1


async def call_from_here(
    bound: Runnable[LanguageModelInput, Any], convention: Convention, text: str
) -> int:
    """Call `bound` through `convention`; return the line it was called from."""
    if convention == "invoke":
        line = next_line()
        bound.invoke(text)
    elif convention == "ainvoke":
        line = next_line()
        await bound.ainvoke(text)
    elif convention == "stream":
        line = next_line()
        list(bound.stream(text))
    else:
        line = next_line()
        _ = [chunk async for chunk in bound.astream(text)]
    return line


@pytest.mark.parametrize("binder", BINDERS)
def test_the_bind_time_warning_points_at_the_line_that_bound(binder: Binder) -> None:
    """The warning is for the application to act on, so it names the line that
    called `bind_tools` or `with_structured_output`, not a frame inside the router."""
    router, _ = mixed()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        binder.bind(router)

    (warning,) = routing_warnings(caught)
    assert (warning.filename, warning.lineno) == (
        __file__,
        binder.bind.__code__.co_firstlineno + 1,
    )


CALLED_THROUGH = [
    *(pytest.param(TOOLS, convention, id=f"bind_tools-{convention}") for convention in CONVENTIONS),
    *(
        pytest.param(STRUCTURED, convention, id=f"with_structured-{convention}")
        for convention in CONVENTIONS
    ),
]


@pytest.mark.parametrize(("binder", "convention"), CALLED_THROUGH)
async def test_the_per_request_warning_points_at_the_line_that_called(
    binder: Binder, convention: Convention
) -> None:
    """A diversion is raised from inside the router's pipeline, at whatever depth
    the calling convention and the binding put it — a fixed constant would have to be pinned
    per combination, as `bind_tools` needed one and `with_structured_output` (the
    `StructuredRouter.stream`/`astream`, now delegating to the router's own rather than
    `Runnable`'s default) needed another; `_stacklevel` walks to the boundary instead,
    so every combination here names the same thing: the line that called the bound runnable."""
    router, _ = mixed()
    bound = bind(binder, router)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = await call_from_here(bound, convention, "cheap")

    (warning,) = routing_warnings(caught)
    assert (warning.filename, warning.lineno) == (__file__, line)


def test_the_walker_does_not_treat_a_module_that_merely_starts_with_langchain_as_library_code() -> (
    None
):
    """`_stacklevel` (`router.py`) matches `_LIBRARY_MODULES` on the dot: a module whose name
    starts with the same letters as `langchain` — someone's own `langchain_myapp` — is the
    application's, not the library's, and the walker must stop there rather than mistake it
    for library code and keep walking past it."""
    router, _ = router_of({"a": True}, default="a", strategy=Abstains())
    namespace: dict[str, Any] = {"__name__": "langchain_myapp", "router": router}
    # Building a frame whose module name is exactly "langchain_myapp" — a real one, since the
    # walker reads `frame.f_globals["__name__"]`, which only `exec` into a chosen namespace sets.
    exec(
        "def call_from_a_lookalike_module():\n    router.invoke('hello')\n",
        namespace,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        namespace["call_from_a_lookalike_module"]()

    (warning,) = routing_warnings(caught)
    assert warning.filename == "<string>"  # `exec`'s own compile unit, not this test file
