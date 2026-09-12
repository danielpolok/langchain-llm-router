"""T-003: tools and structured output through the router (C3, C1, and a preview of R10).

Each test answers one of the questions in the task, offline, with fake routes.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from spike.fakes import (
    FakeChatModel,
    NativeStructuredFakeChatModel,
    ToolCallingFakeChatModel,
)
from spike.router import (
    TOOL_BINDING_KEY,
    SpikeRouterChatModel,
    ToolBinding,
    routing_decision,
)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class Answer(BaseModel):
    """An answer, with how sure the model is of it."""

    summary: str = Field(description="the answer itself")
    confidence: float = Field(description="between 0 and 1")


def call_of(name: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[ToolCall(name=name, args=args, id="call_1", type="tool_call")],
    )


def router_over(**routes: BaseChatModel) -> SpikeRouterChatModel:
    """A router whose strategy picks the route whose name appears in the request.

    A request that names no route goes to the first one, so these tests warn (R9) only
    where that is the thing under test.
    """
    default = next(iter(routes))

    def by_name(messages: list[Any]) -> str | None:
        text = messages[-1].text
        return next((name for name in routes if name in text), default)

    return SpikeRouterChatModel(routes=dict(routes), default_route=default, strategy=by_name)


# Question 1 — binding.


def test_bind_tools_keeps_the_tools_in_their_original_form() -> None:
    """Nothing is converted at bind time: the route has not been chosen yet (C3)."""
    router = router_over(a=ToolCallingFakeChatModel())

    bound = router.bind_tools([add], tool_choice="any")

    assert isinstance(bound, RunnableBinding)  # bind_tools is declared as -> Runnable
    binding = bound.kwargs[TOOL_BINDING_KEY]
    assert isinstance(binding, ToolBinding)
    assert binding.tools == (add,)
    assert binding.tool_choice == "any"


def test_the_selected_route_converts_the_tools_at_call_time() -> None:
    route = ToolCallingFakeChatModel()
    router = router_over(a=route)

    router.bind_tools([add], tool_choice="any").invoke("add 2 and 3")

    (call,) = route.calls
    assert [spec["function"]["name"] for spec in call["tools"]] == ["add"]
    assert call["tool_choice"] == "any"


def test_binding_kwargs_are_replayed_on_the_routes_own_binder() -> None:
    """`strict=` and friends change how a provider builds the tool schema, so they
    have to reach `bind_tools`, not the call."""
    route = ToolCallingFakeChatModel()
    router = router_over(a=route)

    router.bind_tools([add], strict=True).invoke("hi")

    assert route.calls[0]["strict"] is True


def test_each_request_is_bound_by_its_own_route() -> None:
    cheap, frontier = ToolCallingFakeChatModel(), ToolCallingFakeChatModel()
    router = router_over(cheap=cheap, frontier=frontier)

    router.bind_tools([add]).batch(["ask cheap", "ask frontier"])

    assert len(cheap.calls) == 1
    assert len(frontier.calls) == 1


def test_tool_calls_survive_streaming_through_the_router() -> None:
    route = ToolCallingFakeChatModel(
        tool_calls=[ToolCall(name="add", args={"a": 2, "b": 3}, id="call_1", type="tool_call")]
    )
    router = router_over(a=route)

    chunks = list(router.bind_tools([add]).stream("add 2 and 3"))

    merged = reduce(operator.add, chunks)
    assert merged.tool_calls == [
        {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1", "type": "tool_call"}
    ]


# Question 2 — structured output.


def test_structured_output_returns_an_instance_of_the_schema() -> None:
    route = NativeStructuredFakeChatModel(script=[call_of("Answer", summary="42", confidence=0.9)])
    router = router_over(a=route)

    result = router.with_structured_output(Answer).invoke("the question")

    assert result == Answer(summary="42", confidence=0.9)


def test_structured_output_with_include_raw_keeps_the_raw_message() -> None:
    route = NativeStructuredFakeChatModel(script=[call_of("Answer", summary="42", confidence=0.9)])
    router = router_over(a=route)

    result = router.with_structured_output(Answer, include_raw=True).invoke("q")

    assert isinstance(result, dict)
    assert result["parsed"] == Answer(summary="42", confidence=0.9)
    assert result["parsing_error"] is None
    assert isinstance(result["raw"], AIMessage)


def test_structured_output_forwards_provider_arguments_to_the_route() -> None:
    """The point of overriding it: `method=` reaches the route's native implementation."""
    route = NativeStructuredFakeChatModel(script=[call_of("Answer", summary="42", confidence=0.9)])
    router = router_over(a=route)

    router.with_structured_output(Answer, method="json_schema").invoke("q")

    assert route.structured_output_calls == [{"method": "json_schema", "include_raw": False}]


def test_the_inherited_default_would_drop_provider_arguments() -> None:
    """What `BaseChatModel.with_structured_output` does instead: `method=` is discarded
    and every route is forced through function calling on the *router's* `bind_tools`."""
    route = NativeStructuredFakeChatModel(script=[call_of("Answer", summary="42", confidence=0.9)])
    router = router_over(a=route)

    inherited = BaseChatModel.with_structured_output(router, Answer, method="json_schema")
    result = inherited.invoke("q")

    assert result == Answer(summary="42", confidence=0.9)
    assert route.structured_output_calls == []


def test_structured_output_does_not_carry_the_decision_record() -> None:
    """Forwarding bypasses the router's own `_generate`, so nothing adds the record.

    The parsed value has nowhere to carry it either. R2 needs another answer for
    structured output — a v1 question (T-114).
    """
    route = NativeStructuredFakeChatModel(script=[call_of("Answer", summary="42", confidence=0.9)])
    router = router_over(a=route)

    result = router.with_structured_output(Answer, include_raw=True).invoke("q")

    assert isinstance(result, dict)
    assert routing_decision(result["raw"]) is None


# Question 3 — agents.


def test_create_agent_runs_a_tool_loop_through_the_router() -> None:
    """C1: the router is the model behind an agent, tool loop and all."""
    route = ToolCallingFakeChatModel(script=[call_of("add", a=2, b=3), AIMessage("2 + 3 = 5")])
    agent = create_agent(model=router_over(a=route), tools=[add])

    result = agent.invoke({"messages": [{"role": "user", "content": "add 2 and 3"}]})

    assert [
        message.content for message in result["messages"] if isinstance(message, ToolMessage)
    ] == ["5"]
    assert result["messages"][-1].content == "2 + 3 = 5"


def test_the_profile_reports_only_what_every_route_can_do() -> None:
    """Q3: `create_agent` reads `profile` to choose a structured-output strategy."""
    router = router_over(
        native=ToolCallingFakeChatModel(
            profile={
                "structured_output": True,
                "tool_calling": True,
                "max_input_tokens": 200_000,
            }
        ),
        small=ToolCallingFakeChatModel(
            profile={
                "structured_output": False,
                "tool_calling": True,
                "max_input_tokens": 8_000,
            }
        ),
    )

    assert router.profile == {
        "structured_output": False,
        "tool_calling": True,
        "max_input_tokens": 8_000,
    }


def test_create_agent_response_format_picks_a_strategy_every_route_can_serve() -> None:
    route = ToolCallingFakeChatModel(
        script=[call_of("Answer", summary="42", confidence=0.9)],
        profile={"structured_output": False, "tool_calling": True},
    )
    agent = create_agent(model=router_over(a=route), tools=[], response_format=Answer)

    result = agent.invoke({"messages": [{"role": "user", "content": "the question"}]})

    assert result["structured_response"] == Answer(summary="42", confidence=0.9)


# Question 4 — routes that cannot use tools (observation only; R10 is built in T-115).


def test_binding_tools_is_accepted_even_when_a_route_cannot_use_them() -> None:
    """No check happens at bind time: `bind_tools` cannot know which route will run."""
    router = router_over(capable=ToolCallingFakeChatModel(), plain=FakeChatModel())

    assert router.bind_tools([add]) is not None


def test_a_route_that_cannot_use_tools_fails_at_call_time() -> None:
    """The failure is `BaseChatModel.bind_tools`' `NotImplementedError`, raised only when
    that route is actually selected. R10 turns this into an up-front warning and a
    diversion to a route that can (T-115)."""
    router = router_over(capable=ToolCallingFakeChatModel(), plain=FakeChatModel())
    bound = router.bind_tools([add])

    bound.invoke("ask capable")

    with pytest.raises(NotImplementedError):
        bound.invoke("ask plain")
