"""T-004: is cost counted once while the real call stays traced? (R3, C5, R1)

This file first pins down what the naive design of T-002 does, then scores the candidates in
`spike/designs.py` against the same measurements.
"""

from __future__ import annotations

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.utils._gateway import GATEWAY_METADATA_RESPONSE_KEY

from spike.designs import DelegatingRouterChatModel, RelabellingRouterChatModel
from spike.fakes import FakeChatModel, ToolCallingFakeChatModel
from spike.router import ROUTING_KEY, SpikeRouterChatModel, routing_decision
from spike.tracing import model_runs, priced_calls

ONE_CALL = {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
TWO_CALLS = {"input_tokens": 6, "output_tokens": 10, "total_tokens": 16}
DEFAULTED = {"route": "only", "reason": "no strategy configured"}

GATEWAY_RUN_METADATA_KEY = "ls_gateway_info"
"""What the tracer calls the promoted gateway metadata (`tracers/core.py:41`)."""


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def router() -> SpikeRouterChatModel:
    return SpikeRouterChatModel(
        routes={"only": FakeChatModel(model_name="fake-1")}, default_route="only"
    )


def delegating(route: FakeChatModel | None = None) -> DelegatingRouterChatModel:
    return DelegatingRouterChatModel(
        routes={"only": route or FakeChatModel(model_name="fake-1")},
        default_route="only",
    )


def relabelling() -> RelabellingRouterChatModel:
    return RelabellingRouterChatModel(
        routes={"only": FakeChatModel(model_name="fake-1")}, default_route="only"
    )


# The naive design of T-002.


def test_invoke_counts_the_same_tokens_twice() -> None:
    """R3 broken: the router's run and the route's run report the same usage."""
    with get_usage_metadata_callback() as usage:
        router().invoke("hi")

    assert usage.usage_metadata == {"fake-1": TWO_CALLS}


def test_invoke_charges_the_trace_twice() -> None:
    """Both runs carry usage, so LangSmith sees two model calls for one request."""
    collector = RunCollectorCallbackHandler()

    router().invoke("hi", config={"callbacks": [collector]})

    assert len(model_runs(collector.traced_runs)) == 2
    charges = priced_calls(collector.traced_runs)
    assert [charge.usage for charge in charges] == [ONE_CALL, ONE_CALL]
    # The router's run has no model name of its own; the route's has.
    assert [charge.model_name for charge in charges] == [None, "fake-1"]


def test_streaming_counts_twice_and_loses_the_real_call() -> None:
    """The worse case: the tokens are still counted twice, but the call that produced
    them is missing from the trace.

    `BaseChatModel.stream` gives `_stream` no run manager, so the router can pass the route
    no callbacks. The caller's handlers never see the route's run (C5), while the usage
    callback — which installs itself into every callback manager through a context var —
    still sees both (R3).
    """
    collector = RunCollectorCallbackHandler()

    with get_usage_metadata_callback() as usage:
        list(router().stream("hi", config={"callbacks": [collector]}))

    assert usage.usage_metadata == {"fake-1": TWO_CALLS}
    assert len(model_runs(collector.traced_runs)) == 1


def test_the_caller_still_gets_the_routes_own_usage() -> None:
    """R1 holds in the naive design: nothing is stripped from the response."""
    answer = router().invoke("hi")

    assert answer.usage_metadata == ONE_CALL


# Candidate A — the router as a chain run that delegates.


def test_delegating_counts_usage_once() -> None:
    with get_usage_metadata_callback() as usage:
        delegating().invoke("hi")

    assert usage.usage_metadata == {"fake-1": ONE_CALL}


async def test_delegating_counts_usage_once_asynchronously() -> None:
    with get_usage_metadata_callback() as usage:
        await delegating().ainvoke("hi")

    assert usage.usage_metadata == {"fake-1": ONE_CALL}


def test_delegating_counts_usage_once_when_streaming() -> None:
    with get_usage_metadata_callback() as usage:
        list(delegating().stream("hi"))

    assert usage.usage_metadata == {"fake-1": ONE_CALL}


def test_delegating_charges_the_trace_once_and_keeps_the_real_call() -> None:
    """R3 and C5 together: one charge, and the model call is still a run of its own."""
    collector = RunCollectorCallbackHandler()

    delegating().invoke("hi", config={"callbacks": [collector]})

    assert [charge.model_name for charge in priced_calls(collector.traced_runs)] == ["fake-1"]
    (root,) = collector.traced_runs
    (model_run,) = model_runs(collector.traced_runs)
    assert root.run_type == "chain"
    assert model_run.parent_run_id == root.id


def test_delegating_puts_the_decision_in_the_trace() -> None:
    """C5's other half: the trace says which route was taken, and why."""
    collector = RunCollectorCallbackHandler()

    delegating().invoke("hi", config={"callbacks": [collector]})

    (root,) = collector.traced_runs
    assert root.extra["metadata"][ROUTING_KEY] == DEFAULTED


def test_delegating_keeps_the_route_run_nested_when_streaming() -> None:
    """Closes the gap T-002 found: streaming no longer hides the real call."""
    collector = RunCollectorCallbackHandler()

    list(delegating().stream("hi", config={"callbacks": [collector]}))

    (root,) = collector.traced_runs
    (model_run,) = model_runs(collector.traced_runs)
    assert model_run.parent_run_id == root.id


def test_delegating_returns_the_routes_response_unchanged() -> None:
    """R1: the usage the caller sees is the route's own, and the decision is readable."""
    answer = delegating().invoke("hi")

    assert answer.usage_metadata == ONE_CALL
    assert routing_decision(answer) == DEFAULTED


def test_delegating_still_binds_tools() -> None:
    """C3 keeps working: delegation happens after the route converts the tools."""
    route = ToolCallingFakeChatModel(model_name="fake-1")

    delegating(route).bind_tools([add]).invoke("add 2 and 3")

    assert [spec["function"]["name"] for spec in route.calls[0]["tools"]] == ["add"]


# Candidate B — one model run, wearing the selected model's identity.


def test_relabelling_counts_usage_once() -> None:
    with get_usage_metadata_callback() as usage:
        relabelling().invoke("hi")

    assert usage.usage_metadata == {"fake-1": ONE_CALL}


def test_relabelling_leaves_one_run_wearing_the_models_identity() -> None:
    """The tracer promotes the gateway identity over the request-time one."""
    collector = RunCollectorCallbackHandler()

    relabelling().invoke("hi", config={"callbacks": [collector]})

    (model_run,) = model_runs(collector.traced_runs)
    assert model_run.extra["metadata"]["ls_model_name"] == "fake-1"
    assert [charge.model_name for charge in priced_calls(collector.traced_runs)] == ["fake-1"]


def test_relabelling_puts_the_decision_in_the_trace() -> None:
    collector = RunCollectorCallbackHandler()

    relabelling().invoke("hi", config={"callbacks": [collector]})

    (model_run,) = model_runs(collector.traced_runs)
    assert model_run.extra["metadata"][GATEWAY_RUN_METADATA_KEY][ROUTING_KEY] == DEFAULTED


def test_relabelling_keeps_the_gateway_key_out_of_the_response() -> None:
    """R1: `_gen_info_and_msg_metadata` strips the key before it reaches the message."""
    answer = relabelling().invoke("hi")

    assert GATEWAY_METADATA_RESPONSE_KEY not in answer.response_metadata
    assert answer.usage_metadata == ONE_CALL
    assert routing_decision(answer) == DEFAULTED


def test_relabelling_does_not_fix_streaming() -> None:
    """A limitation of the candidate as built, not of the idea.

    Only `_generate` / `_agenerate` are overridden, so streaming falls back to the naive
    path and doubles again. Fixing it means the same treatment in `_stream`, with the
    gateway key on exactly one chunk.
    """
    with get_usage_metadata_callback() as usage:
        list(relabelling().stream("hi"))

    assert usage.usage_metadata == {"fake-1": TWO_CALLS}
