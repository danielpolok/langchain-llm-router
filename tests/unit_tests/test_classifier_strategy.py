"""T-134: the opt-in small-LLM classifier strategy — REQ-R7-2 and REQ-R3-2.

Three threads, the same shape T-133's `test_embedding_strategy.py` uses for the sibling opt-in
strategy:

- **REQ-R7-2** — construction takes the classifier model and the route descriptions as required
  arguments, with no default for either: `test_construction_requires_*`.
- **Tracing (D9, REQ-R3-2)** — every per-request classifier call is a real `BaseChatModel` call
  made with `request.config`, so it is traced and costed *for free* by LangChain's own callback
  machinery, unlike `EmbeddingStrategy`'s manual run-opening for `Embeddings`. The classifier
  strategy module docstring explains why passing `request.config` (which `ClassifierStrategy`
  always does, sync and async) makes nesting version-independent; the async-context-propagation
  boundary that matters when a strategy *omits* the config is already proven, as a mechanism
  probe rather than a hard Python-version check, by
  `tests/unit_tests/test_tracing.py::test_a_model_the_strategy_calls_nests_under_the_strategy_s_run`
  — this module does not repeat that proof, only the classifier-specific shape and cost.
- **R9** — an empty request, a classifier answer that doesn't parse (an unknown route, or no
  tool call at all), and a genuine call failure all end the same way a built-in strategy's
  always have: the default route, one `FallbackWarning`, the cause recorded — but by two
  different code paths (`decide`/`adecide` turning a bad-but-successful answer into `None`,
  versus a raised exception left to propagate), tested separately.

`NativeStructuredFakeChatModel` (`tests/fakes.py`) answers `with_structured_output` the way a
real tool-calling provider does, so it stands in for the classifier model without needing a real
one; `FailingChatModel` (same module) is the same fake with its call replaced by a raise, for the
call-failure path.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    RoutingChoice,
    RoutingDecision,
    RoutingError,
    RoutingRequest,
    RoutingWarning,
    routing_decision,
)
from langchain_llm_router._extraction import build_request
from langchain_llm_router.strategies.classifier import ClassifierStrategy
from tests.conventions import Convention, respond
from tests.fakes import FailingChatModel, FakeChatModel, NativeStructuredFakeChatModel
from tests.tracing import StartLog, model_name_of, model_runs, priced_calls

ROUTE_DESCRIPTIONS = {
    "coder": "programming help: code, debugging, stack traces, refactors",
    "support": "account issues: password resets, billing, cancellations",
}
ROUTES = ("coder", "support")


def make_request(
    text: str, *, routes: tuple[str, ...] = ROUTES, config: RunnableConfig | None = None
) -> RoutingRequest:
    """The request the router would build from a one-turn conversation (R4)."""
    request = build_request(
        [HumanMessage(text)],
        routes=routes,
        tools_bound=False,
        wants_full_context=False,
        config=config if config is not None else RunnableConfig(),
    )
    assert request is not None
    return request


def fake_routes() -> dict[str, BaseChatModel]:
    return {
        "coder": FakeChatModel(model_name="model-coder", reply="coder answer"),
        "support": FakeChatModel(model_name="model-support", reply="support answer"),
    }


def classifier_answering(route: str) -> NativeStructuredFakeChatModel:
    """A classifier that always answers with `route`, the way a real tool-calling model would
    once it has decided — the tool name matches `classifier.py`'s `_SCHEMA_NAME` exactly, as a
    real provider's structured-output call would for whatever schema it was given."""
    return NativeStructuredFakeChatModel(
        model_name="model-classifier",
        tool_calls=[ToolCall(id="call_1", name="RouteClassification", args={"route": route})],
    )


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


# --- REQ-R7-2: construction requires the model and route_descriptions, no defaults ---


def test_construction_requires_the_model() -> None:
    """REQ-R7-2: no default -- this strategy can never be enabled by accident."""
    with pytest.raises(TypeError):
        ClassifierStrategy()  # type: ignore[call-arg]


def test_construction_requires_route_descriptions_too() -> None:
    with pytest.raises(TypeError):
        ClassifierStrategy(classifier_answering("coder"))  # type: ignore[call-arg]


# --- Configuration errors, at construction (never once per request) ---

BAD_ROUTE_DESCRIPTIONS = [
    pytest.param(
        {},
        "route_descriptions is empty: a ClassifierStrategy needs at least one route",
        id="empty",
    ),
    pytest.param(
        [],
        "route_descriptions is list: a ClassifierStrategy takes a mapping of route name to a "
        "human-readable description, such as {'coder': 'programming help: code, debugging, "
        "stack traces'}",
        id="not-a-mapping",
    ),
    pytest.param({" ": "ok"}, "the route ' ' is blank: every route needs a name", id="blank-route"),
    pytest.param(
        {"coder": ""},
        "route 'coder' has a blank or non-string description: ''",
        id="blank-description",
    ),
    pytest.param(
        {"coder": None},
        "route 'coder' has a blank or non-string description: None",
        id="non-string-description",
    ),
]


@pytest.mark.parametrize(("route_descriptions", "message"), BAD_ROUTE_DESCRIPTIONS)
def test_route_descriptions_that_could_never_decide_fail_at_construction(
    route_descriptions: Any, message: str
) -> None:
    """Everything judged without a request is judged here, once, not per request."""
    with pytest.raises(RoutingError, match=rf"^{re.escape(message)}$"):
        ClassifierStrategy(classifier_answering("coder"), route_descriptions)


# --- Choosing a route: valid, invalid and failing classifier answers ---


def test_a_valid_classification_decides() -> None:
    """The classifier's answer, constrained to the closed route set, decides directly."""
    strategy = ClassifierStrategy(classifier_answering("coder"), ROUTE_DESCRIPTIONS)

    choice = strategy.decide(make_request("fix this stack trace"))

    assert choice == RoutingChoice(
        route="coder",
        reason="the classifier chose 'coder': programming help: code, debugging, stack "
        "traces, refactors",
    )


async def test_a_valid_classification_decides_async_too() -> None:
    strategy = ClassifierStrategy(classifier_answering("support"), ROUTE_DESCRIPTIONS)

    choice = await strategy.adecide(make_request("cancel my subscription"))

    assert choice is not None
    assert choice.route == "support"


def test_an_unknown_route_in_the_answer_abstains_not_raises() -> None:
    """R9: the model's tool call names something outside the `Literal` -- a successful call, a
    bad answer -- so `with_structured_output(..., include_raw=True)` reports `parsed=None`
    rather than raising, and `decide` reads that as an abstain."""
    model = NativeStructuredFakeChatModel(
        model_name="model-classifier",
        tool_calls=[ToolCall(id="call_1", name="RouteClassification", args={"route": "billing"})],
    )
    strategy = ClassifierStrategy(model, ROUTE_DESCRIPTIONS)

    assert strategy.decide(make_request("fix this stack trace")) is None


def test_no_tool_call_at_all_abstains_not_raises() -> None:
    """The other "successful call, bad answer" shape: the model answers without using the tool."""
    model = NativeStructuredFakeChatModel(model_name="model-classifier", tool_calls=[])
    strategy = ClassifierStrategy(model, ROUTE_DESCRIPTIONS)

    assert strategy.decide(make_request("fix this stack trace")) is None


def test_an_empty_request_abstains_without_calling_the_classifier() -> None:
    """R9: nothing to classify -- checked before any call is made, sync or async."""
    model = classifier_answering("coder")
    strategy = ClassifierStrategy(model, ROUTE_DESCRIPTIONS)

    assert strategy.decide(make_request("   ")) is None

    assert model.calls == []


async def test_an_empty_request_abstains_async_too() -> None:
    model = classifier_answering("coder")
    strategy = ClassifierStrategy(model, ROUTE_DESCRIPTIONS)

    assert await strategy.adecide(make_request("   ")) is None

    assert model.calls == []


# --- A stray route_descriptions entry is never offered to the model (unlike Keyword/Embedding) ---


def test_a_route_the_router_doesn_t_have_is_never_in_the_schema() -> None:
    """Unlike `KeywordStrategy`/`EmbeddingStrategy`, `ClassifierStrategy` closes the schema to
    `request.routes`, so a stray `route_descriptions` entry never reaches the model at all."""
    descriptions = {**ROUTE_DESCRIPTIONS, "billing": "billing questions"}
    model = classifier_answering("coder")
    strategy = ClassifierStrategy(model, descriptions)

    strategy.decide(make_request("fix this stack trace", routes=ROUTES))

    (bind_call,) = model.bind_calls
    (schema,) = bind_call["tools"]
    assert schema.model_json_schema()["properties"]["route"]["enum"] == list(ROUTES)


def test_no_overlap_between_route_descriptions_and_the_router_s_routes_raises() -> None:
    """Same precedent as `KeywordStrategy`/`EmbeddingStrategy`: a mapping naming none of the
    router's routes can never decide anything -- raised up front, not abstained silently."""
    strategy = ClassifierStrategy(classifier_answering("coder"), ROUTE_DESCRIPTIONS)

    with pytest.raises(RoutingError, match="no route in route_descriptions is one this router has"):
        strategy.decide(make_request("fix this stack trace", routes=("billing", "sales")))


async def test_through_the_router_that_unroutable_mapping_becomes_a_fallback_too() -> None:
    strategy = ClassifierStrategy(classifier_answering("coder"), ROUTE_DESCRIPTIONS)
    router = ChatRouter(
        routes={"billing": FakeChatModel(model_name="model-billing", reply="billing answer")},
        default_route="billing",
        strategy=strategy,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke("fix this stack trace")

    assert message.content == "billing answer"
    (warning,) = routing_warnings(caught)
    assert warning.category is FallbackWarning
    assert (
        "ClassifierStrategy raised RoutingError: no route in route_descriptions is one this "
        "router has" in str(warning.message)
    )


# --- A genuine call failure propagates and falls back through the router (R9) ---


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_a_classifier_call_failure_falls_back_to_the_default_route(
    convention: Convention,
) -> None:
    """R9: the strategy does not catch the classifier call's exception (module doc) -- the
    router's own machinery turns it into the default route and one `FallbackWarning`."""
    model = FailingChatModel(
        model_name="model-classifier", error_type=ConnectionError, error_message="down"
    )
    strategy = ClassifierStrategy(model, ROUTE_DESCRIPTIONS)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "fix this stack trace")

    assert message.content == "support answer"
    (warning,) = routing_warnings(caught)
    assert warning.category is FallbackWarning
    assert str(warning.message) == (
        "ClassifierStrategy raised ConnectionError: down; falling back to the default route "
        "'support'"
    )
    record = routing_decision(message)
    assert record == RoutingDecision(
        route="support",
        reason="ClassifierStrategy raised ConnectionError: down; fell back to the default route",
        strategy="ClassifierStrategy",
        fallback=True,
    )


def test_a_classifier_model_with_no_structured_output_support_falls_back_too() -> None:
    """The documented limitation (module doc): a classifier model that never implements
    `bind_tools` raises `NotImplementedError` from `with_structured_output`, which is not
    special-cased -- it propagates like any other call failure and falls back (R9)."""
    model = FakeChatModel(model_name="model-classifier")  # no bind_tools override
    strategy = ClassifierStrategy(model, ROUTE_DESCRIPTIONS)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke("fix this stack trace")

    assert message.content == "support answer"
    (warning,) = routing_warnings(caught)
    assert warning.category is FallbackWarning
    assert "ClassifierStrategy raised NotImplementedError" in str(warning.message)


def test_the_failed_call_carries_the_error_and_the_route_still_answers() -> None:
    """D9: the classifier's own run closes as an error too, not only the strategy run around it,
    while the router's run and the route's call complete normally (C6)."""
    model = FailingChatModel(model_name="model-classifier")
    strategy = ClassifierStrategy(model, ROUTE_DESCRIPTIONS)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        router.invoke("fix this stack trace", config={"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    (classifier_run,) = model_runs(strategy_run.child_runs)
    assert "the model is unreachable" in (classifier_run.error or "")
    assert "the model is unreachable" in (strategy_run.error or "")
    assert (router_run.error, route_run.error) == (None, None)


# --- The classifier call is its own child run of the strategy run, costed separately (D9, R3) ---


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_the_classifier_call_nests_under_the_strategy_run_and_is_costed_separately(
    convention: Convention,
) -> None:
    """The acceptance criterion: router run -> strategy run -> the classifier's own chat-model
    run (a descendant, however many `Runnable` steps `with_structured_output` composes) -> and
    the route's call stays where it always is, directly under the router run, untouched. Both
    model runs carry their own `usage_metadata`, so nothing is billed twice (R3).

    `ClassifierStrategy` always passes `request.config` (module doc), so this nests identically
    whether or not the running Python's `asyncio` hands a coroutine its caller's context --
    `test_tracing.py`'s mechanism probe is what proves that boundary in general; this test only
    proves the classifier-specific shape holds, on whatever Python runs it.
    """
    text = "fix this stack trace"
    classifier = classifier_answering("coder")
    strategy = ClassifierStrategy(classifier, ROUTE_DESCRIPTIONS)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)
    starts, collector = StartLog(), RunCollectorCallbackHandler()

    message = await respond(router, convention, text, {"callbacks": [starts, collector]})

    assert message.content == "coder answer"
    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    assert (router_run.run_type, strategy_run.run_type) == ("chain", "chain")
    assert strategy_run.name == "ClassifierStrategy"

    (classifier_run,) = model_runs(strategy_run.child_runs)
    assert route_run.run_type in {"llm", "chat_model"}
    assert model_runs([route_run]) == [route_run]  # the route's own call, not touched
    assert [run.id for run in model_runs(collector.traced_runs)] == [
        classifier_run.id,
        route_run.id,
    ]
    assert (model_name_of(classifier_run), model_name_of(route_run)) == (
        "model-classifier",
        "model-coder",
    )

    calls = {call.model_name: call.usage for call in priced_calls(collector.traced_runs)}
    assert calls.keys() == {"model-classifier", "model-coder"}
    assert calls["model-classifier"]["total_tokens"] == 8  # the fake's default usage
    assert calls["model-coder"]["total_tokens"] == 8

    assert {start.run_id for start in starts.chat_models} == {classifier_run.id, route_run.id}
    assert starts.llms == []


# --- REQ-R6-1: the built-in is exercised only through the public interface ---


async def test_through_the_router_a_classifier_strategy_is_an_ordinary_strategy() -> None:
    """REQ-R6-1: nothing about running it through `ChatRouter` is special-cased."""
    strategy = ClassifierStrategy(classifier_answering("coder"), ROUTE_DESCRIPTIONS)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    message = router.invoke("fix this stack trace")

    assert message.content == "coder answer"
    record = routing_decision(message)
    assert record is not None
    assert record.route == "coder"
    assert record.fallback is False
