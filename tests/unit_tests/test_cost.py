"""T-117: cost is counted once, and to the model that ran (R3).

Two things read a trace as a bill. `UsageMetadataCallbackHandler` (through
`get_usage_metadata_callback`) adds up the `usage_metadata` of every chat-model run that ends,
keyed by the model name on its message; LangSmith prices each LLM run from its own usage and
`ls_model_name`. So a router is costed correctly when the trace holds exactly one model run per
model call, and the router's own run — a chain — is never one of them (spike surprise 3: a second
run bills the same tokens again, and nothing errors).

`tests.tracing.priced_calls` mirrors what LangSmith would charge from the offline trace, and the
handler is the other reader. Each test here has both look at the same call, over every convention
(`ALL_CONVENTIONS`), because a double count that shows on one path and not another is the one
that ships.
"""

from __future__ import annotations

import pytest
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from langchain_llm_router import ChatRouter, RoutingChoice, RoutingRequest, RoutingStrategy
from tests.conventions import ALL_CONVENTIONS, AnyConvention, generated, prompts, respond
from tests.fakes import FakeChatModel, call_log
from tests.tracing import (
    ConsultingStrategy,
    PricedCall,
    ls_metadata_of,
    model_name_of,
    priced_calls,
    usage_of,
    walk,
)

CHEAP = UsageMetadata(input_tokens=3, output_tokens=5, total_tokens=8)
FRONTIER = UsageMetadata(input_tokens=11, output_tokens=13, total_tokens=24)
CLASSIFIER = UsageMetadata(input_tokens=7, output_tokens=2, total_tokens=9)
"""What each model reports for one call. Distinct on purpose: a total that is a sum of the wrong
two, or counted twice, has to be a number nobody else could have produced."""


class ByText(RoutingStrategy):
    """Routes to the route the request's text names, and makes no call of its own."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route=request.text, reason=f"the request named {request.text!r}")


def priced_routes() -> dict[str, BaseChatModel]:
    """Two routes that each report their own usage under their own model name."""
    return {
        "cheap": FakeChatModel(
            model_name="model-cheap", reply="cheap answer", input_tokens=3, output_tokens=5
        ),
        "frontier": FakeChatModel(
            model_name="model-frontier", reply="frontier answer", input_tokens=11, output_tokens=13
        ),
    }


def times(usage: UsageMetadata, count: int) -> UsageMetadata:
    """`usage` for `count` calls of the same model."""
    return UsageMetadata(
        input_tokens=usage["input_tokens"] * count,
        output_tokens=usage["output_tokens"] * count,
        total_tokens=usage["total_tokens"] * count,
    )


# --- The handler's totals are the routes' own (REQ-R3-1) ---


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_the_usage_totals_equal_what_the_route_reports_when_called_directly(
    convention: AnyConvention,
) -> None:
    """REQ-R3-1: `UsageMetadataCallbackHandler` reports, for a call through the router, exactly
    what it reports for the same route called on its own — one model, counted once, under the
    name of the model that ran and not the router's."""
    router = ChatRouter(routes=priced_routes(), default_route="cheap", strategy=ByText())

    with get_usage_metadata_callback() as routed:
        message = await respond(router, convention, "frontier")
    with get_usage_metadata_callback() as direct:
        await respond(priced_routes()["frontier"], convention, "frontier")

    assert direct.usage_metadata == {"model-frontier": FRONTIER}
    assert routed.usage_metadata == direct.usage_metadata
    assert message.usage_metadata == FRONTIER  # the answer's own usage is the route's, unchanged


@pytest.mark.parametrize("convention", ["invoke", "ainvoke", "stream", "batch", "generate"])
async def test_the_usage_totals_are_kept_per_model_across_calls(convention: AnyConvention) -> None:
    """REQ-R3-1: calls that go to different routes add up under each route's own model name —
    two to the cheap model and one to the frontier model are 2x and 1x, not 3x of either."""
    routes = priced_routes()
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())

    with get_usage_metadata_callback() as usage:
        for text in ("cheap", "frontier", "cheap"):
            await respond(router, convention, text)

    assert usage.usage_metadata == {"model-cheap": times(CHEAP, 2), "model-frontier": FRONTIER}
    assert [len(call_log(route)) for route in routes.values()] == [2, 1]


async def test_one_generate_over_several_prompts_is_counted_per_prompt() -> None:
    """REQ-R3-1: `generate` over three prompts — the one call that is many routed requests — is
    counted three times, each under the model its own prompt was routed to."""
    router = ChatRouter(routes=priced_routes(), default_route="cheap", strategy=ByText())

    with get_usage_metadata_callback() as usage:
        result = await generated(router, "generate", prompts("cheap", "frontier", "cheap"))

    assert usage.usage_metadata == {"model-cheap": times(CHEAP, 2), "model-frontier": FRONTIER}
    assert len(result.generations) == 3


# --- The router's own run is never a charge (REQ-R3-1, C5) ---


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_the_trace_holds_exactly_one_charge_and_it_is_the_route_s(
    convention: AnyConvention,
) -> None:
    """REQ-R3-1, C5: what LangSmith would price is one line, for the model that ran. The router's
    own run and the strategy's are chain runs that carry no usage and no model identity — the
    two things that would turn either into a second charge for the same tokens."""
    router = ChatRouter(routes=priced_routes(), default_route="cheap", strategy=ByText())
    collector = RunCollectorCallbackHandler()

    await respond(router, convention, "frontier", {"callbacks": [collector]})

    assert priced_calls(collector.traced_runs) == [PricedCall("model-frontier", FRONTIER)]
    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    for run in (router_run, strategy_run):
        assert run.run_type == "chain"
        assert usage_of(run) is None
        assert model_name_of(run) is None
        assert ls_metadata_of(run) == []
        assert "usage_metadata" not in (run.extra or {}).get("metadata", {})
    assert "ls_model_name" in ls_metadata_of(route_run)  # the model's identity is on its run
    assert usage_of(route_run) == FRONTIER


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_the_router_run_s_outputs_are_the_response_and_the_record_never_generations(
    convention: AnyConvention,
) -> None:
    """R3: the router run's `output` is the answer, usage and all — the response, not a charge.
    What makes a run chargeable is the shape the tracer reads usage from (`generations` on an LLM
    run's outputs, `tracers/langchain.py:395`), and a chain run's outputs never have it."""
    router = ChatRouter(routes=priced_routes(), default_route="cheap", strategy=ByText())
    collector = RunCollectorCallbackHandler()

    await respond(router, convention, "frontier", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    assert sorted(router_run.outputs or {}) == ["output", "routing"]
    assert "generations" not in (router_run.outputs or {})


# --- A strategy's own call is costed to the strategy's model (REQ-R3-2) ---


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_a_strategy_s_model_is_costed_on_its_own_run_inside_the_strategy_run(
    convention: AnyConvention,
) -> None:
    """REQ-R3-2: a classifier's run sits inside the strategy's run with its own usage; the
    answering route's usage is what it always was; the handler reports the two models
    separately, each counted once. Neither is folded into the router or into the other."""
    classifier = FakeChatModel(
        model_name="model-classifier", reply="frontier", input_tokens=7, output_tokens=2
    )
    routes = priced_routes()
    router = ChatRouter(
        routes=routes,
        default_route="cheap",
        strategy=ConsultingStrategy(classifier, "frontier"),
    )
    collector = RunCollectorCallbackHandler()

    with get_usage_metadata_callback() as usage:
        message = await respond(router, convention, "hello", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    (classifier_run,) = strategy_run.child_runs
    assert usage_of(classifier_run) == CLASSIFIER
    assert usage_of(route_run) == FRONTIER
    assert usage_of(strategy_run) is None
    assert usage_of(router_run) is None
    assert priced_calls(collector.traced_runs) == [
        PricedCall("model-classifier", CLASSIFIER),
        PricedCall("model-frontier", FRONTIER),
    ]
    assert usage.usage_metadata == {"model-classifier": CLASSIFIER, "model-frontier": FRONTIER}
    assert message.usage_metadata == FRONTIER
    assert (len(call_log(classifier)), len(call_log(routes["frontier"]))) == (1, 1)
    assert call_log(routes["cheap"]) == []


@pytest.mark.parametrize("pass_config", [True, False], ids=["config passed", "config omitted"])
@pytest.mark.parametrize("convention", ["invoke", "ainvoke", "stream", "astream", "generate"])
async def test_a_strategy_s_call_is_counted_once_whether_or_not_it_passes_its_config(
    convention: AnyConvention, pass_config: bool
) -> None:
    """REQ-R3-2: where the strategy's call lands in the tree is REQ-C5-5's and depends on how it
    was made; that it is *counted*, once, does not — the handler is registered on the context,
    and every model run that ends reaches it."""
    classifier = FakeChatModel(
        model_name="model-classifier", reply="frontier", input_tokens=7, output_tokens=2
    )
    router = ChatRouter(
        routes=priced_routes(),
        default_route="cheap",
        strategy=ConsultingStrategy(classifier, "frontier", pass_config=pass_config),
    )

    with get_usage_metadata_callback() as usage:
        await respond(router, convention, "hello")

    assert usage.usage_metadata == {"model-classifier": CLASSIFIER, "model-frontier": FRONTIER}


async def test_the_usage_in_the_whole_tree_is_what_the_two_calls_reported() -> None:
    """R3: summing the usage of every run in the tree, chains included, gives what was spent —
    the classifier's call and the route's, and nothing that was not a call."""
    classifier = FakeChatModel(
        model_name="model-classifier", reply="frontier", input_tokens=7, output_tokens=2
    )
    router = ChatRouter(
        routes=priced_routes(),
        default_route="cheap",
        strategy=ConsultingStrategy(classifier, "frontier"),
    )
    collector = RunCollectorCallbackHandler()

    router.invoke("hello", {"callbacks": [collector]})

    reported = [usage for run in walk(collector.traced_runs) if (usage := usage_of(run))]
    assert sum(usage["total_tokens"] for usage in reported) == (
        CLASSIFIER["total_tokens"] + FRONTIER["total_tokens"]
    )
