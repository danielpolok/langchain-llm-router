"""A route forced through runtime config is honoured.

`config_specs` declares the `"route"` configurable key, which `_plan` reads ahead of
the strategy: a forced route skips the strategy entirely, never calling its `decide` /
`adecide`. An unavailable forced route — unknown, or unable to use tools that
are bound — errors by default and falls back only under `on_unavailable_forced_route="fallback"`,
recording `forced=True` throughout and `fallback=True` with a reason naming the
forced route and why, when it gave way. Everything else in runtime config keeps
reaching the route exactly as before.

Neighbours, not repeated here: capability detection itself, and what `_divert` does with an
ordinary (unforced) diversion, are `test_tools.py`'s; the ordinary fallback — a strategy
that can't decide — is `test_fallback.py`'s.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import pytest
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.tools import tool
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from pydantic import BaseModel, Field

from langchain_llm_router import (
    ChatRouter,
    ForcedRouteError,
    ForcedRouteWarning,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    ToolSupportWarning,
    routing_decision,
)
from langchain_llm_router.decision import ROUTING_KEY
from tests.conventions import ALL_CONVENTIONS, AnyConvention, Convention, generated, respond
from tests.fakes import FakeChatModel, ToolCallingFakeChatModel, call_log


@tool
def get_weather(city: str) -> str:
    """Look up the weather in a city."""
    return f"sunny in {city}"


class Answer(BaseModel):
    """A schema simple enough for the fakes' structured output to satisfy trivially."""

    answer: str


class Counting(RoutingStrategy):
    """Would always choose `would_choose`; counts how often it is consulted.

    What proves "the strategy's `decide` is not called": a forced route that still
    left `calls == 0` was never given the chance to disagree.
    """

    def __init__(self, would_choose: str = "frontier") -> None:
        self.calls = 0
        self.would_choose = would_choose

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        self.calls += 1
        return RoutingChoice(route=self.would_choose, reason="the strategy's own pick")


class ConfigCapturingRoute(FakeChatModel):
    """A route that keeps a copy of every `RunnableConfig` it was invoked with."""

    configs_seen: list[dict[str, Any]] = Field(default_factory=list)

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.configs_seen.append(dict(ensure_config(config)))
        return super().invoke(input, config, stop=stop, **kwargs)


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


def router_of(**fields: Any) -> tuple[ChatRouter, dict[str, BaseChatModel]]:
    """`cheap` (the default), `frontier` (what `Counting` would pick) and `premium` (what a
    test forces) — three distinguishable routes, so an assertion can tell which one ran."""
    routes: dict[str, BaseChatModel] = {
        "cheap": FakeChatModel(model_name="model-cheap", reply="cheap answer"),
        "frontier": FakeChatModel(model_name="model-frontier", reply="frontier answer"),
        "premium": FakeChatModel(model_name="model-premium", reply="premium answer"),
    }
    router = ChatRouter(routes=routes, default_route="cheap", **fields)
    return router, routes


def tool_router(**fields: Any) -> tuple[ChatRouter, dict[str, BaseChatModel]]:
    """`cheap` (the default, tool-capable) and `local` (tool-incapable) — for the cases where a
    forced route's problem is what it can't do, not that it doesn't exist."""
    routes: dict[str, BaseChatModel] = {
        "cheap": ToolCallingFakeChatModel(model_name="model-cheap", reply="cheap answer"),
        "local": FakeChatModel(model_name="model-local", reply="local answer"),
    }
    router = ChatRouter(routes=routes, default_route="cheap", **fields)
    return router, routes


FORCED_CONVENTIONS = tuple(c for c in ALL_CONVENTIONS if c not in ("generate", "agenerate"))
"""Every calling convention that carries a `RunnableConfig` a forced route can ride in.

`generate` / `agenerate` cannot: they take `callbacks`, `tags`, `metadata`,
`run_name` and `run_id` as separate arguments and no config at all — see
`test_generate_and_agenerate_have_no_way_to_carry_a_forced_route`."""


# --- the key is declared, and both standard ways to set it force the route ---


def test_config_specs_declares_the_route_key() -> None:
    """Exactly one spec, `id="route"`, defaulting to unforced (`None`)."""
    router, routes = router_of()

    (spec,) = router.config_specs

    assert isinstance(spec, ConfigurableFieldSpec)
    assert spec.id == "route"
    assert spec.default is None
    assert "cheap" in (spec.description or "")  # names the routes it can force, for a caller
    assert list(routes) == ["cheap", "frontier", "premium"]


@pytest.mark.parametrize("convention", FORCED_CONVENTIONS)
async def test_config_configurable_route_forces_it(convention: AnyConvention) -> None:
    """`config={"configurable": {"route": ...}}` forces the named route on
    every convention that carries a config — including `batch`, `abatch` and `astream_events`,
    which reach it through `Runnable` rather than the router's own entry points."""
    strategy = Counting()
    router, routes = router_of(strategy=strategy)

    message = await respond(router, convention, "hello", {"configurable": {"route": "premium"}})

    assert message.content == "premium answer"
    assert strategy.calls == 0
    assert [len(call_log(route)) for route in routes.values()] == [0, 0, 1]
    assert routing_decision(message) == RoutingDecision(
        route="premium", reason="forced via runtime config", forced=True
    )


def test_with_config_configurable_route_forces_it_too() -> None:
    """`with_config(configurable={"route": ...})` — LangChain's other standard way to
    set a configurable field — forces it exactly the same way as passing `config=` directly."""
    strategy = Counting()
    router, _ = router_of(strategy=strategy)

    message = router.with_config(configurable={"route": "premium"}).invoke("hello")

    assert message.content == "premium answer"
    assert strategy.calls == 0
    assert routing_decision(message) == RoutingDecision(
        route="premium", reason="forced via runtime config", forced=True
    )


def test_a_route_named_none_is_not_forced() -> None:
    """`configurable["route"]` absent, or explicitly `None`, both mean "not forced" — the same
    as `config_specs`' own `default=None` promises: the strategy still decides."""
    strategy = Counting("premium")
    router, _ = router_of(strategy=strategy)

    absent = router.invoke("hello")
    explicit_none = router.invoke("hello", config={"configurable": {"route": None}})

    assert (absent.content, explicit_none.content) == ("premium answer", "premium answer")
    assert strategy.calls == 2
    assert (
        routing_decision(absent)
        == routing_decision(explicit_none)
        == RoutingDecision(route="premium", reason="the strategy's own pick", strategy="Counting")
    )


@pytest.mark.parametrize("convention", ["generate", "agenerate"])
async def test_generate_and_agenerate_have_no_way_to_carry_a_forced_route(
    convention: Literal["generate", "agenerate"],
) -> None:
    """`generate` / `agenerate` take no `RunnableConfig` at all, so there is nowhere
    for `configurable["route"]` to ride — expected, not a gap in the forced-route coverage."""
    router, _ = router_of()

    with pytest.raises(ValueError, match=r"generate\(\) has no argument for"):
        await generated(
            router,
            convention,
            [[HumanMessage("hello")]],
            cast("RunnableConfig", {"configurable": {"route": "premium"}}),
        )


# --- the strategy is skipped outright, and no strategy run opens for it ---


async def test_no_strategy_run_opens_for_a_forced_route() -> None:
    """Skipping the strategy means no strategy run — the route's call is the router
    run's only child — and the forced record is the one both places on the trace carry."""
    strategy = Counting()
    router, _ = router_of(strategy=strategy)
    collector = RunCollectorCallbackHandler()
    record = RoutingDecision(
        route="premium", reason="forced via runtime config", forced=True
    ).as_dict()

    message = router.invoke(
        "hello", config={"configurable": {"route": "premium"}, "callbacks": [collector]}
    )

    assert strategy.calls == 0
    (router_run,) = collector.traced_runs
    assert [run.run_type for run in router_run.child_runs] == ["llm"]
    (route_run,) = router_run.child_runs
    assert (route_run.extra or {})["metadata"][ROUTING_KEY] == record
    assert (router_run.outputs or {})[ROUTING_KEY] == record
    assert routing_decision(message) == RoutingDecision.from_dict(record)


# --- an unknown forced route ---


def test_an_unknown_forced_route_errors_by_default() -> None:
    """`on_unavailable_forced_route` defaults to `"error"` — an unknown name raises
    `ForcedRouteError` naming it, before the strategy or any route is ever touched."""
    strategy = Counting()
    router, routes = router_of(strategy=strategy)

    with pytest.raises(ForcedRouteError) as raised:
        router.invoke("hello", config={"configurable": {"route": "nope"}})

    assert str(raised.value) == (
        "forced route 'nope' is not one of the routes: 'cheap', 'frontier', 'premium'; set "
        "on_unavailable_forced_route='fallback' to fall back to the default route instead"
    )
    assert strategy.calls == 0
    assert [len(call_log(route)) for route in routes.values()] == [0, 0, 0]


def test_an_unknown_forced_route_falls_back_under_the_fallback_setting() -> None:
    """`on_unavailable_forced_route="fallback"` sends the request to the
    default route instead, with exactly one `ForcedRouteWarning` — not `FallbackWarning`, the
    strategy fallback's own — and both `forced=True` and `fallback=True` recorded, the reason naming
    the forced route and why it could not be used."""
    strategy = Counting()
    router, routes = router_of(strategy=strategy, on_unavailable_forced_route="fallback")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke("hello", config={"configurable": {"route": "nope"}})

    assert message.content == "cheap answer"
    assert strategy.calls == 0
    assert [w.category for w in routing_warnings(caught)] == [ForcedRouteWarning]
    assert str(routing_warnings(caught)[0].message) == (
        "forced route 'nope' is not one of the routes: 'cheap', 'frontier', 'premium'; "
        "falling back to the default route 'cheap'"
    )
    assert routing_decision(message) == RoutingDecision(
        route="cheap",
        reason=(
            "forced route 'nope' is not one of the routes: 'cheap', 'frontier', 'premium'; "
            "fell back to the default route"
        ),
        forced=True,
        fallback=True,
    )
    assert [len(call_log(route)) for route in routes.values()] == [1, 0, 0]


# --- a forced route that exists but can't use the bound tools ---


def test_a_tool_incapable_forced_route_errors_by_default() -> None:
    """When tools are bound, a forced route needs to be able to use them too —
    the same signals `_divert` reads for any other route — or it is refused just as an
    unknown name is."""
    router, routes = tool_router()
    with pytest.warns(ToolSupportWarning):
        bound = router.bind_tools([get_weather])

    with pytest.raises(ForcedRouteError) as raised:
        bound.invoke("hello", config={"configurable": {"route": "local"}})

    assert str(raised.value) == (
        "forced route 'local' can't use the bound tools; set on_unavailable_forced_route="
        "'fallback' to fall back to the default route instead"
    )
    assert [len(call_log(route)) for route in routes.values()] == [0, 0]


def test_a_tool_incapable_forced_route_falls_back_under_the_fallback_setting() -> None:
    """The same `"fallback"` path as the unknown-route case, for a route
    that exists but can't use the bound tools."""
    router, routes = tool_router(on_unavailable_forced_route="fallback")
    with pytest.warns(ToolSupportWarning):
        bound = router.bind_tools([get_weather])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = bound.invoke("hello", config={"configurable": {"route": "local"}})

    assert message.content == "cheap answer"
    assert [w.category for w in routing_warnings(caught)] == [ForcedRouteWarning]
    assert routing_decision(message) == RoutingDecision(
        route="cheap",
        reason="forced route 'local' can't use the bound tools; fell back to the default route",
        forced=True,
        fallback=True,
    )
    assert [len(call_log(route)) for route in routes.values()] == [1, 0]


def test_a_forced_route_that_cannot_use_structured_output_is_refused_too() -> None:
    """`with_structured_output` builds on tool binding, so the same rule
    applies through it — the parity tool-aware routing requires, exercised for a forced route."""
    router, _ = tool_router()
    with pytest.warns(ToolSupportWarning):
        bound = router.with_structured_output(Answer)

    with pytest.raises(ForcedRouteError, match="forced route 'local' can't use the bound tools"):
        bound.invoke("hello", config={"configurable": {"route": "local"}})


def test_a_forced_route_with_nothing_bound_needs_only_to_exist() -> None:
    """Design choice (documented in the final report): with nothing bound, tool capability is
    never in question for a forced route — only its existence is — so `local`, which cannot use
    tools, still runs when the call itself binds none."""
    router, routes = tool_router()  # "local" cannot use tools; nothing is bound in this call

    message = router.invoke("hello", config={"configurable": {"route": "local"}})

    assert message.content == "local answer"
    assert routing_decision(message) == RoutingDecision(
        route="local", reason="forced via runtime config", forced=True
    )
    assert [len(call_log(route)) for route in routes.values()] == [0, 1]


# --- Whatever a forced-route fallback resolves to is diverted like any other decision ---


async def test_a_forced_route_fallback_onto_an_incapable_default_is_diverted_too() -> None:
    """`_divert`'s own docstring promise: a forced-route fallback is not special-cased out of
    tool diversion. Here the default itself can't use the bound tools, so after the forced-route
    fallback lands on it, tool-aware routing diverts a second time — both facts land in the one
    record."""
    routes: dict[str, BaseChatModel] = {
        "cheap": FakeChatModel(model_name="model-cheap", reply="cheap answer"),
        "capable": ToolCallingFakeChatModel(model_name="model-capable", reply="capable answer"),
    }
    router = ChatRouter(
        routes=routes, default_route="cheap", on_unavailable_forced_route="fallback"
    )
    with pytest.warns(ToolSupportWarning):
        bound = router.bind_tools([get_weather])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = bound.invoke("hello", config={"configurable": {"route": "nope"}})

    assert message.content == "capable answer"
    assert [w.category for w in routing_warnings(caught)] == [
        ForcedRouteWarning,
        ToolSupportWarning,
    ]
    assert routing_decision(message) == RoutingDecision(
        route="capable",
        reason=(
            "forced route 'nope' is not one of the routes: 'cheap', 'capable'; fell back to "
            "the default route; 'cheap' can't use the bound tools, so it was diverted to "
            "'capable'"
        ),
        forced=True,
        fallback=True,
        diverted_from="cheap",
    )
    assert [len(call_log(route)) for route in routes.values()] == [0, 1]


# --- config_specs is visible through every wrapper ---


def test_config_specs_is_visible_bound_and_with_config() -> None:
    """`bind_tools`, `with_structured_output` and a plain `with_config` all still expose
    the `"route"` spec — `bind_tools`/`with_config` through `RunnableBinding.config_specs`
    forwarding to `self.bound.config_specs` (`runnables/base.py`), and
    `with_structured_output` through `StructuredRouter.config_specs` (`_tools.py`)."""
    router, _ = tool_router()
    expected = router.config_specs

    with pytest.warns(ToolSupportWarning):
        tools_bound = router.bind_tools([get_weather])
    with pytest.warns(ToolSupportWarning):
        structured = router.with_structured_output(Answer)
    plain_config = router.with_config(tags=["x"])

    assert tools_bound.config_specs == expected
    assert structured.config_specs == expected
    assert plain_config.config_specs == expected


def test_a_forced_route_works_end_to_end_through_bind_tools() -> None:
    """The spec being visible is only half of it — forcing has to actually work once a
    router is bound, not only on the bare router."""
    router, routes = tool_router()
    with pytest.warns(ToolSupportWarning):
        bound = router.bind_tools([get_weather])

    message = bound.invoke("hello", config={"configurable": {"route": "cheap"}})

    assert routing_decision(message) == RoutingDecision(
        route="cheap", reason="forced via runtime config", forced=True
    )
    assert len(call_log(routes["cheap"])) == 1


# --- everything else in runtime config keeps reaching the route unchanged ---


def test_other_runtime_configuration_still_reaches_the_route_unchanged() -> None:
    """`tags`, `metadata`, `callbacks` and `max_concurrency` set on the router's
    config keep reaching the route exactly as before forced routes existed — forcing a route changes
    nothing about the rest of the config. `run_name` names the *router's* own run, as it already did
    (`test_the_caller_s_configuration_reaches_the_route`); it is not repeated onto the
    route's own run, since `_route_call` replaces the route run's callbacks (which drops a
    `run_name` meant for the same run as the old callbacks) — unrelated to forcing, and
    unchanged by it."""
    route = ConfigCapturingRoute(model_name="model-cheap", reply="cheap answer")
    router = ChatRouter(routes={"cheap": route, "other": FakeChatModel()}, default_route="cheap")
    collector = RunCollectorCallbackHandler()

    router.invoke(
        "hello",
        config={
            "configurable": {"route": "cheap"},
            "tags": ["experiment"],
            "metadata": {"user": "ada"},
            "run_name": "routed call",
            "callbacks": [collector],
            "max_concurrency": 3,
        },
    )

    (router_run,) = collector.traced_runs
    assert router_run.name == "routed call"
    (route_run,) = router_run.child_runs
    assert "experiment" in (route_run.tags or [])
    assert (route_run.extra or {})["metadata"]["user"] == "ada"
    (seen,) = route.configs_seen
    assert seen["max_concurrency"] == 3
    assert seen["tags"] == ["experiment"]
    assert seen["metadata"]["user"] == "ada"


@pytest.mark.parametrize("convention", ["invoke", "ainvoke", "stream", "astream"])
async def test_forcing_a_route_does_not_disturb_config_on_any_convention(
    convention: Convention,
) -> None:
    """The regression check repeated across every convention the router overrides
    itself — a forced route is still just a request, and the rest of its config travels with
    it exactly as an unforced one's does."""
    routes: dict[str, BaseChatModel] = {
        "cheap": FakeChatModel(model_name="model-cheap", reply="cheap answer"),
        "other": FakeChatModel(model_name="model-other", reply="other answer"),
    }
    router = ChatRouter(routes=routes, default_route="cheap")
    collector = RunCollectorCallbackHandler()

    await respond(
        router,
        convention,
        "hello",
        {
            "configurable": {"route": "cheap"},
            "tags": ["experiment"],
            "metadata": {"user": "ada"},
            "callbacks": [collector],
        },
    )

    (router_run,) = collector.traced_runs
    (route_run,) = router_run.child_runs
    assert "experiment" in (route_run.tags or [])
    assert (route_run.extra or {})["metadata"]["user"] == "ada"
