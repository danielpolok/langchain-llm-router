"""T-117: the run shape is confirmed on a live LangSmith trace, not only offline (REQ-C5-4).

`test_tracing.py` and `test_cost.py` check the shape and the bill against the offline tracer,
and `tests.tracing.priced_calls` stands in for what LangSmith would charge. This is the same
claim made of the real service: one routed call per convention is traced to LangSmith, and the
trace is read back — what LangSmith *stored*, and what it *computed* from it.

The route is a fake that claims a real, priced model (`gpt-4o`) in its trace metadata, because
LangSmith derives cost from `usage_metadata` and `ls_provider` / `ls_model_name` and prices
nothing it doesn't recognise. No provider is called, so this needs LangSmith credentials and no
model key. It is skipped, not failed, when LangSmith does not accept the ones configured
(`requires_langsmith`, in the root `conftest.py`), and it needs `LANGSMITH_TRACING` neither on
nor off: the tracer is handed to each call explicitly.

The read-back uses `Client.list_runs`, which the LangSmith SDK deprecates in favour of the async
`client.traces.list_runs`, with removal after 2027-01-31 (migration guide:
`docs.langchain.com/langsmith/smithdb-sdk-migration-traces`). It is used because the run shape
it returns is the one the offline assertions are written against, and because this test has no
credentials to try the new call with; move it when the SDK pin moves past that date.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langsmith import Client
from langsmith.schemas import Run
from langsmith.utils import LangSmithNotFoundError

from langchain_llm_router import (
    ChatRouter,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
)
from langchain_llm_router.decision import ROUTING_KEY
from tests.conventions import ALL_CONVENTIONS, AnyConvention, respond
from tests.fakes import FakeChatModel

pytestmark = [
    pytest.mark.requires_langsmith,
    pytest.mark.filterwarnings("ignore:.*removed after Jan 31, 2027"),
]

PROJECT = "llm-router-tests"
"""Where the traces go, so they can be looked at afterwards. LangSmith creates it on first use."""

RECORD = RoutingDecision(
    route="frontier", reason="the request named 'frontier'", strategy="ByText"
).as_dict()

_LANDING = 120.0
"""Seconds to wait for a trace to be stored and priced. LangSmith ingests and prices runs after
the call returns, so the trace is read back until it is complete — this is only the patience
before a run that never arrives is reported as one, and never a delay on one that does."""


class ByText(RoutingStrategy):
    """Routes to the route the request's text names, and makes no call of its own."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route=request.text, reason=f"the request named {request.text!r}")


class PricedFake(FakeChatModel):
    """A fake route that tells LangSmith it is an OpenAI model, so its usage has a price."""

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return LangSmithParams(
            ls_provider="openai", ls_model_name=self.model_name, ls_model_type="chat"
        )


@pytest.fixture(scope="module")
def client() -> Client:
    return Client()


def own_usage(run: Run) -> Any:
    """The usage a run carries itself: `LangChainTracer` puts it on an LLM run's metadata."""
    return ((run.extra or {}).get("metadata") or {}).get("usage_metadata")


def stored_trace(client: Client, trace_id: Any) -> list[Run]:
    """The runs LangSmith holds for `trace_id`, once all three are finished and the model call is
    priced — read repeatedly, because ingestion and pricing both happen after the call."""
    deadline = time.monotonic() + _LANDING
    runs: list[Run] = []
    while time.monotonic() < deadline:
        try:
            runs = list(client.list_runs(project_name=PROJECT, trace_id=trace_id))
        except LangSmithNotFoundError:  # the project is created by its first run
            runs = []
        model_runs = [run for run in runs if run.run_type == "llm"]
        if (
            len(runs) >= 3
            and all(run.status == "success" for run in runs)
            and all(run.total_cost is not None for run in model_runs)
        ):
            return runs
        time.sleep(2.0)
    found = [(run.name, run.run_type, run.status, run.total_cost) for run in runs]
    pytest.fail(f"LangSmith did not finish storing and pricing the trace in time: {found}")


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_a_live_trace_shows_the_decision_and_the_real_call_priced_once(
    convention: AnyConvention, client: Client
) -> None:
    """REQ-C5-4: LangSmith stores the router's chain run with the decision in its outputs, the
    strategy's run with the decision as its output, and one LLM run — the route's — carrying the
    decision in its metadata; and it prices that one call once. The router's run and the
    strategy's carry no usage of their own, and the trace's cost is the model run's."""
    routes: dict[str, BaseChatModel] = {
        "cheap": PricedFake(model_name="gpt-4o-mini", reply="cheap answer"),
        "frontier": PricedFake(
            model_name="gpt-4o", reply="frontier answer", input_tokens=11, output_tokens=13
        ),
    }
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())
    tracer = LangChainTracer(client=client, project_name=PROJECT)
    collector = RunCollectorCallbackHandler()

    await respond(router, convention, "frontier", {"callbacks": [tracer, collector]})

    (sent,) = collector.traced_runs
    trace = await asyncio.to_thread(stored_trace, client, sent.id)

    by_id = {run.id: run for run in trace}
    router_run = by_id[sent.id]
    assert (router_run.run_type, router_run.name, router_run.parent_run_id) == (
        "chain",
        "ChatRouter",
        None,
    )
    assert sorted(run.run_type for run in trace) == ["chain", "chain", "llm"]
    (strategy_run,) = [run for run in trace if run.name == "ByText"]
    (model_run,) = [run for run in trace if run.run_type == "llm"]
    assert strategy_run.parent_run_id == router_run.id
    assert model_run.parent_run_id == router_run.id  # the route's call, not the strategy's

    assert (router_run.outputs or {})[ROUTING_KEY] == RECORD
    assert strategy_run.outputs == RECORD
    assert ((model_run.extra or {}).get("metadata") or {})[ROUTING_KEY] == RECORD

    assert [run.id for run in trace if own_usage(run)] == [model_run.id]
    assert model_run.total_tokens == 24
    assert model_run.total_cost is not None
    assert model_run.total_cost > 0
    assert router_run.total_tokens == model_run.total_tokens  # rolled up once, not twice
    assert router_run.total_cost == model_run.total_cost
