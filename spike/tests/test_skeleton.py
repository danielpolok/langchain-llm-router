"""T-002: the router skeleton delegates faithfully (C1, C2, R1, R2, R5, R9)."""

from __future__ import annotations

import operator
from functools import reduce
from typing import Any, NamedTuple
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from spike.fakes import FakeChatModel
from spike.router import (
    RoutingWarning,
    SpikeRouterChatModel,
    Strategy,
    routing_decision,
)


def cheap() -> FakeChatModel:
    return FakeChatModel(model_name="cheap-1", reply="cheap answer")


def frontier() -> FakeChatModel:
    return FakeChatModel(
        model_name="frontier-1",
        reply="frontier answer",
        input_tokens=11,
        output_tokens=13,
    )


def _router(strategy: Strategy | None = None) -> SpikeRouterChatModel:
    return SpikeRouterChatModel(
        routes={"cheap": cheap(), "frontier": frontier()},
        default_route="cheap",
        strategy=strategy,
    )


def pick_frontier(messages: list[BaseMessage]) -> str | None:
    return "frontier"


def by_content(messages: list[BaseMessage]) -> str | None:
    return "frontier" if "hard" in messages[-1].text else "cheap"


class StartedRun(NamedTuple):
    run_id: UUID
    parent_run_id: UUID | None


class RunTree(BaseCallbackHandler):
    """Records every chat model run the caller's callbacks can see, and its parent."""

    def __init__(self) -> None:
        self.runs: list[StartedRun] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.runs.append(StartedRun(run_id, parent_run_id))


def test_default_route_must_be_one_of_the_routes() -> None:
    """R9: the default route is mandatory, so it has to exist."""
    with pytest.raises(ValidationError, match="default_route"):
        SpikeRouterChatModel(routes={"cheap": FakeChatModel()}, default_route="missing")


def test_invoke_returns_the_route_output_unchanged() -> None:
    """R1: content, tool calls and usage are the selected route's, with the decision added."""
    reference = frontier().invoke("hi")

    answer = _router(pick_frontier).invoke("hi")

    assert answer.content == reference.content
    assert answer.tool_calls == reference.tool_calls
    assert answer.usage_metadata == reference.usage_metadata
    assert answer.response_metadata["model_name"] == "frontier-1"
    assert routing_decision(answer) == {"route": "frontier", "reason": "strategy"}


async def test_ainvoke_returns_the_route_output_unchanged() -> None:
    answer = await _router(pick_frontier).ainvoke("hi")

    assert answer.content == "frontier answer"
    assert answer.usage_metadata == {
        "input_tokens": 11,
        "output_tokens": 13,
        "total_tokens": 24,
    }
    assert routing_decision(answer) == {"route": "frontier", "reason": "strategy"}


def test_stream_yields_the_route_chunks_unchanged() -> None:
    expected = list(frontier().stream("hi"))

    chunks = list(_router(pick_frontier).stream("hi"))

    assert [chunk.content for chunk in chunks] == [chunk.content for chunk in expected]
    merged = reduce(operator.add, chunks)
    assert merged.usage_metadata == {
        "input_tokens": 11,
        "output_tokens": 13,
        "total_tokens": 24,
    }
    assert merged.response_metadata["model_name"] == "frontier-1"


async def test_astream_yields_the_route_chunks_unchanged() -> None:
    chunks = [chunk async for chunk in _router(pick_frontier).astream("hi")]

    assert "".join(str(chunk.content) for chunk in chunks) == "frontier answer"
    assert routing_decision(reduce(operator.add, chunks)) == {
        "route": "frontier",
        "reason": "strategy",
    }


def test_batch_routes_each_input_on_its_own() -> None:
    """C2: batch comes from the Runnable defaults and routes per request (R4)."""
    answers = _router(by_content).batch(["an easy one", "a hard one"])

    assert [answer.content for answer in answers] == ["cheap answer", "frontier answer"]
    assert [routing_decision(answer) for answer in answers] == [
        {"route": "cheap", "reason": "strategy"},
        {"route": "frontier", "reason": "strategy"},
    ]


def test_the_decision_record_rides_on_exactly_one_chunk() -> None:
    """merge_dicts concatenates strings repeated across chunks (R2)."""
    chunks = list(_router(pick_frontier).stream("hi"))

    records = [routing_decision(chunk) for chunk in chunks]
    assert [record for record in records if record is not None] == [
        {"route": "frontier", "reason": "strategy"}
    ]


def test_a_strategy_that_raises_falls_back_to_the_default_route() -> None:
    """R9: strategy failure degrades to the default route, with a warning and a reason."""

    def explode(messages: list[BaseMessage]) -> str | None:
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.warns(RoutingWarning, match="strategy raised"):
        answer = _router(explode).invoke("hi")

    assert answer.content == "cheap answer"
    assert routing_decision(answer) == {
        "route": "cheap",
        "reason": "strategy raised RuntimeError",
    }


def test_an_unknown_route_falls_back_to_the_default_route() -> None:
    with pytest.warns(RoutingWarning, match="unknown route"):
        answer = _router(lambda messages: "nope").invoke("hi")

    assert routing_decision(answer) == {
        "route": "cheap",
        "reason": "unknown route 'nope'",
    }


def test_an_undecided_strategy_falls_back_to_the_default_route() -> None:
    with pytest.warns(RoutingWarning, match="did not decide"):
        answer = _router(lambda messages: None).invoke("hi")

    assert routing_decision(answer) == {
        "route": "cheap",
        "reason": "strategy did not decide",
    }


def test_no_strategy_always_uses_the_default_route() -> None:
    answer = _router().invoke("hi")

    assert routing_decision(answer) == {
        "route": "cheap",
        "reason": "no strategy configured",
    }


def test_invoke_nests_the_route_run_under_the_router_run() -> None:
    """C5, and T-004 depends on it: the route's call is a child run of the router's."""
    tree = RunTree()

    _router(pick_frontier).invoke("hi", config={"callbacks": [tree]})

    router_run, route_run = tree.runs
    assert router_run.parent_run_id is None
    assert route_run.parent_run_id == router_run.run_id


async def test_ainvoke_nests_the_route_run_under_the_router_run() -> None:
    tree = RunTree()

    await _router(pick_frontier).ainvoke("hi", config={"callbacks": [tree]})

    router_run, route_run = tree.runs
    assert route_run.parent_run_id == router_run.run_id


def test_stream_cannot_nest_the_route_run() -> None:
    """A gap in the naive design, not a decision.

    `BaseChatModel.stream` calls `_stream` without a run manager (`chat_models.py:794`,
    and again at `:1981` for the streaming branch of invoke), so the router has no parent
    to hand the route. The route runs with no callbacks at all: its run is invisible to
    the caller's handlers, and to LangSmith. T-004 weighs designs that close this.
    """
    tree = RunTree()

    list(_router(pick_frontier).stream("hi", config={"callbacks": [tree]}))

    assert [run.parent_run_id for run in tree.runs] == [None]


async def test_astream_cannot_nest_the_route_run() -> None:
    tree = RunTree()

    async for _ in _router(pick_frontier).astream("hi", config={"callbacks": [tree]}):
        pass

    assert [run.parent_run_id for run in tree.runs] == [None]


def test_routes_are_called_with_the_bound_kwargs() -> None:
    """Call kwargs reach the selected route untouched (a preview of C3's binding)."""
    router = _router(pick_frontier)
    router.invoke("hi", stop=["STOP"], temperature=0.5)

    frontier = router.routes["frontier"]
    assert isinstance(frontier, FakeChatModel)
    assert frontier.calls == [{"temperature": 0.5}]


def test_an_ordinary_message_history_works_through_the_router() -> None:
    """C1: the router is a chat model, so ordinary message lists go in."""
    answer = _router(by_content).invoke(
        [HumanMessage("an easy one"), AIMessage("earlier answer"), HumanMessage("a hard one")]
    )

    assert routing_decision(answer) == {"route": "frontier", "reason": "strategy"}
