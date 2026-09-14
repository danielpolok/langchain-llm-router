"""Real-provider smoke tests: one cloud model, one local.

Gemini skips unless `GEMINI_API_KEY` is set; Ollama skips unless a local server answers.
"""

import os

import pytest
from langchain.chat_models import init_chat_model

GEMINI_MODEL = os.environ.get("LLM_ROUTER_GEMINI_MODEL", "google_genai:gemini-3-flash-preview")
OLLAMA_MODEL = os.environ.get("LLM_ROUTER_OLLAMA_MODEL", "ollama:qwen3:8b")


@pytest.mark.requires_env("GEMINI_API_KEY")
def test_gemini_responds() -> None:
    assert init_chat_model(GEMINI_MODEL).invoke("Reply with one word: pong").text


@pytest.mark.requires_ollama
def test_ollama_responds() -> None:
    assert init_chat_model(OLLAMA_MODEL).invoke("Reply with one word: pong").text
