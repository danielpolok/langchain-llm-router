"""T-003: the same tests against two real providers. Skipped unless both keys are set.

The routes are deliberately from different providers: tool schemas and structured-output
modes differ between them, which is what C3 has to survive.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from spike.router import SpikeRouterChatModel, routing_decision

OPENAI_MODEL = os.environ.get("LLM_ROUTER_OPENAI_MODEL", "openai:gpt-5.4-mini")
ANTHROPIC_MODEL = os.environ.get(
    "LLM_ROUTER_ANTHROPIC_MODEL", "anthropic:claude-haiku-4-5-20251001"
)

pytestmark = pytest.mark.requires_env("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@tool
def get_weather(city: str) -> str:
    """Return the current weather in a city."""
    return f"It is sunny in {city}."


class Answer(BaseModel):
    """An answer, with how sure the model is of it."""

    summary: str = Field(description="the answer itself")
    confidence: float = Field(description="between 0 and 1")


def router() -> SpikeRouterChatModel:
    """Routes on a marker in the request, so a test can choose the route it exercises."""

    def by_marker(messages: list[Any]) -> str | None:
        return "anthropic" if "[anthropic]" in messages[-1].text else "openai"

    return SpikeRouterChatModel(
        routes={
            "openai": init_chat_model(OPENAI_MODEL),
            "anthropic": init_chat_model(ANTHROPIC_MODEL),
        },
        default_route="openai",
        strategy=by_marker,
    )


@pytest.mark.parametrize("route", ["openai", "anthropic"])
def test_a_tool_call_works_through_the_router(route: str) -> None:
    answer = (
        router()
        .bind_tools([get_weather])
        .invoke(f"[{route}] What is the weather in Paris? Use the tool.")
    )

    assert [call["name"] for call in answer.tool_calls] == ["get_weather"]
    assert routing_decision(answer) == {"route": route, "reason": "strategy"}


@pytest.mark.parametrize("route", ["openai", "anthropic"])
def test_structured_output_works_through_the_router(route: str) -> None:
    result = (
        router()
        .with_structured_output(Answer)
        .invoke(f"[{route}] What is 2 + 3? Answer with the number only.")
    )

    assert isinstance(result, Answer)
    assert "5" in result.summary
