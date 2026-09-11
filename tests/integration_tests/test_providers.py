"""Real-provider smoke tests. Each skips unless its provider's API key is set."""

import os

import pytest
from langchain.chat_models import init_chat_model

OPENAI_MODEL = os.environ.get("LLM_ROUTER_OPENAI_MODEL", "openai:gpt-5.4-mini")
ANTHROPIC_MODEL = os.environ.get(
    "LLM_ROUTER_ANTHROPIC_MODEL", "anthropic:claude-haiku-4-5-20251001"
)


@pytest.mark.requires_env("OPENAI_API_KEY")
def test_openai_responds() -> None:
    assert init_chat_model(OPENAI_MODEL).invoke("Reply with one word: pong").text


@pytest.mark.requires_env("ANTHROPIC_API_KEY")
def test_anthropic_responds() -> None:
    assert init_chat_model(ANTHROPIC_MODEL).invoke("Reply with one word: pong").text
