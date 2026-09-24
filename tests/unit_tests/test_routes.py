"""Named routes, the default route, and the shape one routed call takes.

The fallback paths are in `test_fallback.py`; here are the routes themselves, what the
constructor accepts, and the runs one call opens.
"""

from __future__ import annotations

import copy
import sys
import warnings
from typing import Any, ClassVar, cast

import pytest
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from pydantic import ValidationError

from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    RoutingChoice,
    RoutingDecision,
    RoutingError,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    routing_decision,
)
from langchain_llm_router.decision import ROUTING_KEY
from tests.conventions import CONVENTIONS, Convention, respond
from tests.fakes import FakeChatModel, call_log
from tests.tracing import model_name_of, model_runs


class ByText(RoutingStrategy):
    """Routes to the route the request's text names — and keeps every request it saw."""

    def __init__(self) -> None:
        self.requests: list[RoutingRequest] = []

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        self.requests.append(request)
        return RoutingChoice(route=request.text, reason=f"the request named {request.text!r}")


class Consulting(RoutingStrategy):
    """Asks a model before deciding, as an opt-in classifier strategy does."""

    def __init__(self, model: BaseChatModel, *, pass_config: bool) -> None:
        self.model = model
        self.pass_config = pass_config

    def _choice(self) -> RoutingChoice:
        return RoutingChoice(route="frontier", reason="the classifier said so")

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        self.model.invoke(request.text, config=request.config if self.pass_config else None)
        return self._choice()

    async def adecide(self, request: RoutingRequest) -> RoutingChoice:
        await self.model.ainvoke(request.text, config=request.config if self.pass_config else None)
        return self._choice()


def fake_routes(*names: str) -> dict[str, BaseChatModel]:
    """One fake route per name, each answering with its own name."""
    return {
        name: FakeChatModel(model_name=f"model-{name}", reply=f"{name} answer") for name in names
    }


def configuration_of(route: BaseChatModel) -> dict[str, Any]:
    """A deep copy of everything the route is configured with, bar the fake's own call log."""
    return copy.deepcopy(
        {name: getattr(route, name) for name in type(route).model_fields if name != "calls"}
    )


def failures(**kwargs: Any) -> list[tuple[tuple[int | str, ...], Any]]:
    """Where constructing a router failed, and the error each validator raised."""
    with pytest.raises(ValidationError) as caught:
        ChatRouter(**kwargs)
    return [
        (error["loc"], error.get("ctx", {}).get("error", error["type"]))
        for error in caught.value.errors()
    ]


# --- The constructor ---


def test_a_router_needs_at_least_one_route() -> None:
    """Empty `routes` fails at construction, with a message naming the offender."""
    [(loc, error)] = failures(routes={}, default_route="cheap")

    assert loc == ("routes",)
    assert isinstance(error, RoutingError)
    assert str(error) == "routes is empty: a ChatRouter needs at least one route"


def test_a_default_route_is_mandatory() -> None:
    """A router without a default route cannot be built at all."""
    assert failures(routes=fake_routes("cheap")) == [(("default_route",), "missing")]


def test_the_default_route_must_name_a_route() -> None:
    """A default route that names none of the routes fails, naming it."""
    [(loc, error)] = failures(routes=fake_routes("cheap", "frontier"), default_route="nope")

    assert loc == ()
    assert isinstance(error, RoutingError)
    assert str(error) == "default_route 'nope' is not one of the routes: 'cheap', 'frontier'"


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
def test_a_route_name_may_not_be_blank(blank: str) -> None:
    """Names identify routes in records, configuration and warnings — so each route
    needs one."""
    [(loc, error)] = failures(
        routes={blank: FakeChatModel(), "cheap": FakeChatModel()}, default_route="cheap"
    )

    assert loc == ("routes",)
    assert str(error) == f"route name {blank!r} is blank: every route needs a name"


def test_tool_support_overrides_must_name_routes() -> None:
    """An override for a route that doesn't exist is a typo, caught at construction
    rather than silently ignored when tools are bound."""
    [(loc, error)] = failures(
        routes=fake_routes("cheap", "frontier"),
        default_route="cheap",
        tool_support_overrides={"frontier": True, "gpt-9": False},
    )

    assert loc == ()
    assert str(error) == (
        "tool_support_overrides names routes that don't exist: 'gpt-9'; "
        "the routes are 'cheap', 'frontier'"
    )


def test_the_pinned_fields_have_their_pinned_defaults() -> None:
    """Public API: a router is built from routes and a default route alone."""
    router = ChatRouter(routes=fake_routes("cheap"), default_route="cheap")

    assert router.strategy is None
    assert router.on_unavailable_forced_route == "error"
    assert router.tool_support_overrides == {}


def test_on_unavailable_forced_route_takes_only_the_two_settings() -> None:
    """The setting is `error` or `fallback`; anything else is a construction error."""
    [(loc, _)] = failures(
        routes=fake_routes("cheap"), default_route="cheap", on_unavailable_forced_route="maybe"
    )

    assert loc == ("on_unavailable_forced_route",)


def test_a_strategy_is_coerced_once_and_left_out_of_serialization() -> None:
    """The router holds one interface, whether it was given a callable or a strategy."""
    given = ByText()
    routes = fake_routes("cheap")

    assert ChatRouter(routes=routes, default_route="cheap", strategy=given).strategy is given

    router = ChatRouter(routes=routes, default_route="cheap", strategy=lambda request: request.text)
    assert isinstance(router.strategy, RoutingStrategy)
    assert "strategy" not in router.model_dump()


def test_something_that_is_not_a_strategy_is_refused_at_construction() -> None:
    """`as_strategy` is the one gate on what may be a strategy, so what it
    refuses is its own `TypeError` — pydantic wraps a validator's `ValueError`, not a
    `TypeError`, so this is not a `ValidationError` like the route-name failures above."""
    kwargs: dict[str, Any] = {
        "routes": fake_routes("cheap"),
        "default_route": "cheap",
        "strategy": 42,
    }

    with pytest.raises(TypeError, match="strategy"):
        ChatRouter(**kwargs)


# --- Routes are used as given ---


@pytest.mark.parametrize("count", [1, 2, 12], ids=["one", "two", "many"])
async def test_any_number_of_named_routes_answer_their_own_requests(count: int) -> None:
    """One, two or many ordinary chat models, each answering the requests the
    strategy sends it, through every calling convention."""
    names = [f"route-{index}" for index in range(count)]
    routes = fake_routes(*names)
    strategy = ByText()
    router = ChatRouter(routes=routes, default_route=names[0], strategy=strategy)

    for name in names:
        for convention in CONVENTIONS:
            message = await respond(router, convention, name)

            assert message.content == f"{name} answer"
            assert routing_decision(message) == RoutingDecision(
                route=name, reason=f"the request named {name!r}", strategy="ByText"
            )

    assert [len(call_log(route)) for route in routes.values()] == [len(CONVENTIONS)] * count
    assert strategy.requests[0].routes == tuple(names)


@pytest.mark.parametrize("count", [1, 2, 12], ids=["one", "two", "many"])
async def test_a_route_is_the_object_it_was_given_and_is_never_reconfigured(count: int) -> None:
    """The router holds each route object itself, in declaration order, and
    calling it changes nothing about it."""
    names = [f"route-{index}" for index in range(count)]
    routes = fake_routes(*names)
    before = {name: configuration_of(route) for name, route in routes.items()}
    router = ChatRouter(routes=routes, default_route=names[0], strategy=ByText())

    for convention in CONVENTIONS:
        await respond(router, convention, names[-1])

    assert list(router.routes) == names
    for name, route in routes.items():
        assert router.routes[name] is route
        assert configuration_of(route) == before[name]


def test_batch_routes_each_input_on_its_own() -> None:
    """`batch` goes through the routed `invoke`, so every input gets its own decision."""
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=ByText()
    )

    messages = router.batch(["frontier", "cheap"])

    assert [message.content for message in messages] == ["frontier answer", "cheap answer"]
    assert [routing_decision(message) for message in messages] == [
        RoutingDecision(route="frontier", reason="the request named 'frontier'", strategy="ByText"),
        RoutingDecision(route="cheap", reason="the request named 'cheap'", strategy="ByText"),
    ]


def test_the_record_is_added_to_a_copy_of_the_route_s_message() -> None:
    """The route may keep the object it returned — its response cache does — so the
    record goes on a copy, and nothing else about the message changes."""
    answered = AIMessage(content="kept", id="route-message")
    route = FakeChatModel(script=[answered])
    router = ChatRouter(routes={"only": route}, default_route="only")

    message = router.invoke("hello")

    assert ROUTING_KEY not in answered.response_metadata
    assert message.response_metadata == {
        **answered.response_metadata,
        ROUTING_KEY: RoutingDecision(route="only", reason="no strategy configured").as_dict(),
    }
    assert (message.content, message.id) == ("kept", "route-message")
    assert message.usage_metadata == answered.usage_metadata


# --- What one call puts on the trace ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_router_run_wraps_the_strategy_run_and_the_route_s_call(
    convention: Convention,
) -> None:
    """The router's own run is a chain run; the strategy decides in a child chain run of
    its own whose output is the decision; the selected route's call is the only model run,
    also a child of the router's run, and carries the record in its metadata."""
    routes = fake_routes("cheap", "frontier")
    strategy = ByText()
    router = ChatRouter(routes=routes, default_route="cheap", strategy=strategy)
    collector = RunCollectorCallbackHandler()
    record = RoutingDecision(
        route="frontier", reason="the request named 'frontier'", strategy="ByText"
    ).as_dict()

    message = await respond(router, convention, "frontier", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    assert (router_run.run_type, router_run.name) == ("chain", "ChatRouter")
    strategy_run, route_run = router_run.child_runs
    assert (strategy_run.run_type, strategy_run.name) == ("chain", "ByText")
    assert strategy_run.outputs == record
    assert strategy_run.inputs == {
        "text": "frontier",
        "modalities": ["text"],
        "routes": ["cheap", "frontier"],
        "tools_bound": False,
    }
    assert [run.id for run in model_runs(collector.traced_runs)] == [route_run.id]
    assert model_name_of(route_run) == "model-frontier"
    assert (route_run.extra or {})["metadata"][ROUTING_KEY] == record
    assert (router_run.outputs or {})[ROUTING_KEY] == record
    assert routing_decision(message) == RoutingDecision.from_dict(record)

    callbacks = strategy.requests[0].config.get("callbacks")
    assert isinstance(callbacks, BaseCallbackManager)
    assert callbacks.parent_run_id == strategy_run.id


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_without_a_strategy_the_default_route_answers_and_no_strategy_run_opens(
    convention: Convention,
) -> None:
    """With no strategy there is nothing to run and nothing to warn about — the
    default route answers, and the record says so."""
    routes = fake_routes("cheap", "frontier")
    router = ChatRouter(routes=routes, default_route="frontier")
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "hello", {"callbacks": [collector]})

    assert routing_decision(message) == RoutingDecision(
        route="frontier", reason="no strategy configured"
    )
    assert [warning for warning in caught if issubclass(warning.category, RoutingWarning)] == []
    (router_run,) = collector.traced_runs
    assert [run.run_type for run in router_run.child_runs] == ["llm"]
    assert (len(call_log(routes["cheap"])), len(call_log(routes["frontier"]))) == (0, 1)


@pytest.mark.parametrize("pass_config", [True, False], ids=["config passed", "config omitted"])
@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_a_model_the_strategy_calls_nests_under_the_strategy_s_run(
    convention: Convention, pass_config: bool
) -> None:
    """A strategy that calls a model has a parent run to nest under, so that call is
    traced and costed as the strategy's, not the router's or the route's. Passing
    `request.config` works everywhere; the context config covers a strategy that omits it."""
    if convention == "ainvoke" and not pass_config and sys.version_info < (3, 11):
        pytest.skip("below Python 3.11 an async call must be handed its config")
    classifier = FakeChatModel(model_name="classifier", reply="frontier")
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"),
        default_route="cheap",
        strategy=Consulting(classifier, pass_config=pass_config),
    )
    collector = RunCollectorCallbackHandler()

    await respond(router, convention, "hello", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    (classifier_run,) = strategy_run.child_runs
    assert model_name_of(classifier_run) == "classifier"
    assert model_name_of(route_run) == "model-frontier"
    assert [run.id for run in model_runs(collector.traced_runs)] == [
        classifier_run.id,
        route_run.id,
    ]


@pytest.mark.parametrize("convention", ["stream", "astream"])
async def test_exactly_one_streamed_chunk_carries_the_record(convention: Convention) -> None:
    """The route is chosen before the first chunk, and only that chunk carries the record
    — a record on every chunk would merge into `'frontierfrontier…'`."""
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=ByText()
    )
    record = RoutingDecision(
        route="frontier", reason="the request named 'frontier'", strategy="ByText"
    ).as_dict()

    if convention == "stream":
        chunks = list(router.stream("frontier"))
    else:
        chunks = [chunk async for chunk in router.astream("frontier")]

    assert [chunk.response_metadata.get(ROUTING_KEY) for chunk in chunks].count(record) == 1
    assert "".join(cast("str", chunk.content) for chunk in chunks) == "frontier answer"


def test_the_caller_s_configuration_reaches_the_route() -> None:
    """The router replaces the callbacks and adds the decision to the metadata; the rest
    of the caller's config — tags, metadata, run name — travels on."""
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=ByText()
    )
    collector = RunCollectorCallbackHandler()

    router.invoke(
        "frontier",
        config={
            "callbacks": [collector],
            "tags": ["experiment"],
            "metadata": {"user": "ada"},
            "run_name": "routed call",
        },
    )

    (router_run,) = collector.traced_runs
    assert router_run.name == "routed call"
    strategy_run, route_run = router_run.child_runs
    for run in (router_run, strategy_run, route_run):
        assert "experiment" in (run.tags or [])
        assert (run.extra or {})["metadata"]["user"] == "ada"
    assert (strategy_run.name, route_run.name) == ("ByText", "FakeChatModel")


class Recording(FakeChatModel):
    """A route that records how it was called — the object, the transcript and the arguments."""

    calls_seen: ClassVar[list[tuple[Any, list[Any], Any, dict[str, Any]]]] = []

    def _generate(
        self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        Recording.calls_seen.append((self, list(messages), stop, dict(kwargs)))
        return super()._generate(messages, stop, run_manager, **kwargs)

    def _stream(
        self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        Recording.calls_seen.append((self, list(messages), stop, dict(kwargs)))
        return super()._stream(messages, stop, run_manager, **kwargs)


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_route_is_called_as_given_with_the_whole_transcript(
    convention: Convention,
) -> None:
    """ "used as given" covers the call, not only the stored mapping — the route
    object itself is invoked, with the caller's whole transcript, through every convention."""
    route = Recording(model_name="model-only", reply="an answer")
    Recording.calls_seen.clear()
    router = ChatRouter(routes={"only": route}, default_route="only")
    transcript = [
        ("system", "be brief"),
        ("human", "first"),
        ("ai", "an answer"),
        ("human", "second"),
    ]

    await respond(router, convention, transcript, None)

    (called, messages, stop, kwargs) = Recording.calls_seen[0]
    assert called is route  # not a binding, a copy, or a reconfigured clone
    assert [message.text for message in messages] == ["be brief", "first", "an answer", "second"]
    assert (stop, kwargs) == (None, {})


def test_the_route_is_given_the_call_arguments_and_nothing_else() -> None:
    """`stop` and the caller's kwargs reach the route, and the router adds none."""
    route = Recording(model_name="model-only")
    Recording.calls_seen.clear()
    router = ChatRouter(routes={"only": route}, default_route="only")

    router.invoke("hi", stop=["END"], temperature=0.1)

    (_, _, stop, kwargs) = Recording.calls_seen[0]
    assert (stop, kwargs) == (["END"], {"temperature": 0.1})


def test_the_routers_own_callbacks_tags_and_metadata_stay_on_its_own_run() -> None:
    """A chat model's constructor-level callbacks, tags and metadata are local to its own
    run — the router's behave the same way, so the route's run doesn't inherit them."""
    local = RunCollectorCallbackHandler()
    inherited = RunCollectorCallbackHandler()
    router = ChatRouter(
        routes=fake_routes("only"),
        default_route="only",
        callbacks=[local],
        tags=["router-tag"],
        metadata={"owner": "router"},
    )

    router.invoke("hi", config={"callbacks": [inherited]})

    (own_run,) = local.traced_runs
    assert own_run.name == "ChatRouter"
    assert own_run.child_runs == []  # local callbacks don't reach the route's run
    assert "router-tag" in (own_run.tags or [])
    assert (own_run.extra or {})["metadata"]["owner"] == "router"
    (router_run,) = inherited.traced_runs
    (route_run,) = router_run.child_runs
    assert "router-tag" not in (route_run.tags or [])


def test_a_fallback_warning_escalated_to_an_error_leaves_no_run_open() -> None:
    """An application may turn warnings into errors; the strategy's run closes anyway."""
    collector = RunCollectorCallbackHandler()
    router = ChatRouter(
        routes=fake_routes("cheap"), default_route="cheap", strategy=lambda request: None
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        with pytest.raises(FallbackWarning):
            router.invoke("hi", config={"callbacks": [collector]})

    runs = [run for run in collector.traced_runs for run in (run, *run.child_runs)]
    assert [run.name for run in runs] == ["ChatRouter", "<lambda>"]
    assert all(run.end_time is not None for run in runs)
