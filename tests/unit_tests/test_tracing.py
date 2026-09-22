"""T-117: what one routed call puts on the trace, for every way of calling the router (C5, D9).

The shape D9 settled: the router's run is a *chain* run whose output carries the decision; a
strategy, when there is one, decides in a child chain run of its own; and the selected route's
call is the only chat-model run outside the strategy's. That is what makes the trace show the
decision *and* the real call (C5) while a model is billed once (R3, `test_cost.py`).

The shape used to be asserted call by call, in the tests of whichever task first needed it —
`test_routes.py` for the four entry points the router overrides, `test_conventions.py` for the
ones `Runnable` builds on them. This module is where it is one guarantee, over every convention
a chat model has (`ALL_CONVENTIONS`): REQ-C5-1 and REQ-C5-2 for the shape itself, REQ-C5-5 for
what a strategy's own model call does to it.

Nesting is measured two ways, because they fail differently: a `StartLog` says which callback
announced each run and under which parent (`on_chat_model_start` is what REQ-C5-1 counts), and
the offline tracer's tree says where each run ended up and what it carried.
"""

from __future__ import annotations

from contextvars import ContextVar, copy_context

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.utils import coro_with_context
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from langchain_llm_router import (
    ChatRouter,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
    routing_decision,
)
from langchain_llm_router.decision import ROUTING_KEY
from tests.conventions import ALL_CONVENTIONS, AnyConvention, respond
from tests.fakes import FakeChatModel
from tests.tracing import ConsultingStrategy, StartLog, model_name_of, model_runs, walk

ASYNC: frozenset[AnyConvention] = frozenset({"ainvoke", "astream", "abatch", "events", "agenerate"})
"""The conventions that await the strategy — the ones whose context has to reach a coroutine."""

RECORD = RoutingDecision(
    route="frontier", reason="the request named 'frontier'", strategy="ByText"
).as_dict()
"""What `ByText` decides for the request `"frontier"`."""


class ByText(RoutingStrategy):
    """Routes to the route the request's text names, and makes no call of its own."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route=request.text, reason=f"the request named {request.text!r}")


def fake_routes(*names: str) -> dict[str, BaseChatModel]:
    """One fake route per name, each answering with its own name."""
    return {
        name: FakeChatModel(model_name=f"model-{name}", reply=f"{name} answer here")
        for name in names
    }


async def async_calls_inherit_the_context() -> bool:
    """Whether a coroutine awaited through `coro_with_context` runs in the context it was given.

    This is what D9 leans on for a strategy that omits `request.config`, and it is not the same
    everywhere. `asyncio.create_task` takes a `context` only from Python 3.11; below that,
    `langchain-core` 1.6 creates the task *inside* the context (`runnables/utils.py:157`), which
    has the same effect, while 1.1 hands back the bare coroutine (`create_task=False`) and the
    call runs in the caller's context instead. LangChain documents the consequence: there, an
    async call has to be handed its config. Asking the mechanism, rather than the version
    numbers, keeps the skip below true.
    """
    probe: ContextVar[bool] = ContextVar("probe", default=False)

    async def read() -> bool:
        return probe.get()

    context = copy_context()
    context.run(probe.set, True)
    return await coro_with_context(read(), context)


# --- The shape, for every convention (REQ-C5-1, REQ-C5-2) ---


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_every_convention_puts_one_chat_model_run_under_the_router_s_chain_run(
    convention: AnyConvention,
) -> None:
    """REQ-C5-1, REQ-C5-2: with a strategy that makes no call, a call through any convention
    announces exactly one `on_chat_model_start` — the route's — and its parent is the router's
    chain run, which is the root. The router's run carries the decision in its outputs, and the
    route's run in its metadata."""
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=ByText()
    )
    starts, collector = StartLog(), RunCollectorCallbackHandler()

    message = await respond(router, convention, "frontier", {"callbacks": [starts, collector]})

    router_start, strategy_start = starts.chains
    assert (router_start.name, router_start.parent_run_id) == ("ChatRouter", None)
    assert (strategy_start.name, strategy_start.parent_run_id) == ("ByText", router_start.run_id)
    (route_start,) = starts.chat_models
    assert route_start.parent_run_id == router_start.run_id  # not the strategy's run
    assert starts.llms == []

    (router_run,) = collector.traced_runs
    assert (router_run.id, router_run.run_type) == (router_start.run_id, "chain")
    strategy_run, route_run = router_run.child_runs
    assert (strategy_run.id, route_run.id) == (strategy_start.run_id, route_start.run_id)
    assert strategy_run.child_runs == []  # a strategy that calls nothing has nothing under it
    assert route_run.child_runs == []
    assert [run.id for run in model_runs(collector.traced_runs)] == [route_run.id]

    assert (router_run.outputs or {})[ROUTING_KEY] == RECORD
    assert strategy_run.outputs == RECORD
    assert (route_run.extra or {})["metadata"][ROUTING_KEY] == RECORD
    assert routing_decision(message) == RoutingDecision.from_dict(RECORD)
    assert all(run.end_time is not None and run.error is None for run in walk([router_run]))


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_without_a_strategy_the_router_run_holds_the_route_s_call_alone(
    convention: AnyConvention,
) -> None:
    """REQ-C5-1, REQ-C5-2, D9: nothing decides, so there is no strategy run — the router's chain
    run wraps the route's chat-model run and nothing else, and both carry the record."""
    router = ChatRouter(routes=fake_routes("cheap", "frontier"), default_route="frontier")
    starts, collector = StartLog(), RunCollectorCallbackHandler()
    record = RoutingDecision(route="frontier", reason="no strategy configured").as_dict()

    await respond(router, convention, "hello", {"callbacks": [starts, collector]})

    (router_start,) = starts.chains
    (route_start,) = starts.chat_models
    assert (router_start.name, route_start.parent_run_id) == ("ChatRouter", router_start.run_id)
    assert starts.llms == []
    (router_run,) = collector.traced_runs
    (route_run,) = router_run.child_runs
    assert (router_run.run_type, route_run.id) == ("chain", route_start.run_id)
    assert (router_run.outputs or {})[ROUTING_KEY] == record
    assert (route_run.extra or {})["metadata"][ROUTING_KEY] == record


# --- A strategy's own model call nests under the strategy's run (REQ-C5-5) ---


@pytest.mark.parametrize("pass_config", [True, False], ids=["config passed", "config omitted"])
@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_a_model_the_strategy_calls_nests_under_the_strategy_s_run(
    convention: AnyConvention, pass_config: bool
) -> None:
    """REQ-C5-5, D9: router run → strategy run → the strategy's model run, and router run → the
    route's model run — through every convention, sync and async.

    A strategy that passes `request.config` nests everywhere, Python 3.10 included, where an
    async call gets its callbacks only from the config it is handed. One that omits it nests
    wherever context propagates: every sync path, and every async path where a coroutine runs
    in the context it is given — from Python 3.11, and on 3.10 for the `langchain-core`
    versions that arrange it — which is the documented limit the skip below names.
    """
    if convention in ASYNC and not pass_config and not await async_calls_inherit_the_context():
        pytest.skip("here an async call must be handed its config to nest (D9)")
    classifier = FakeChatModel(model_name="model-classifier", reply="frontier")
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"),
        default_route="cheap",
        strategy=ConsultingStrategy(classifier, "frontier", pass_config=pass_config),
    )
    starts, collector = StartLog(), RunCollectorCallbackHandler()

    await respond(router, convention, "hello", {"callbacks": [starts, collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    (classifier_run,) = strategy_run.child_runs
    assert (router_run.run_type, strategy_run.run_type) == ("chain", "chain")
    assert (strategy_run.name, classifier_run.run_type, route_run.run_type) == (
        "ConsultingStrategy",
        "llm",
        "llm",
    )
    assert classifier_run.child_runs == route_run.child_runs == []
    assert (model_name_of(classifier_run), model_name_of(route_run)) == (
        "model-classifier",
        "model-frontier",
    )
    assert [run.id for run in model_runs(collector.traced_runs)] == [
        classifier_run.id,
        route_run.id,
    ]

    classifier_start, route_start = starts.chat_models
    assert (classifier_start.run_id, classifier_start.parent_run_id) == (
        classifier_run.id,
        strategy_run.id,
    )
    assert (route_start.run_id, route_start.parent_run_id) == (route_run.id, router_run.id)
    assert starts.llms == []
