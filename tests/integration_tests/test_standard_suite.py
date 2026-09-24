"""`ChatRouter` against `langchain_tests`' standard real-provider suite.

Gated exactly as the repo's other real-provider tests are (`conftest.py`,
`tests/integration_tests/test_providers.py`): `requires_ollama` skips without a reachable local
server, `requires_env("GEMINI_API_KEY")` skips without the key. Neither subclass is required to
pass in every environment — Gemini's free tier is known to run out of quota —
only to be correctly gated and, given credentials, to exercise the router with a real chat model
behind it exactly as the standard suite would a bare one.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_llm_router import ChatRouter

GEMINI_MODEL = os.environ.get("LLM_ROUTER_GEMINI_MODEL", "google_genai:gemini-3-flash-preview")
OLLAMA_MODEL = os.environ.get("LLM_ROUTER_OLLAMA_MODEL", "ollama:qwen3:8b")


class _RouterIntegrationTests(ChatModelIntegrationTests):
    """Shared shape: one real chat model behind a single route, always the default —
    these tests exercise the standard chat-model contract *through* the router, not routing
    policy, so which route runs is never in question.

    Feature flags below are set conservatively rather than to whatever each provider might
    actually support: the point is that the router delegates unchanged, not a full
    capability audit of Gemini or Ollama, and an inaccurate `True` would fail for a provider
    reason that has nothing to do with the router.
    """

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatRouter

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        return False

    @property
    def supports_video_inputs(self) -> bool:
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def supports_json_mode(self) -> bool:
        return False

    @property
    def supports_model_override(self) -> bool:
        """`ChatRouter` has no `model` field of its own to override: its unit of
        selection is the *route name*, not a per-call model string, and a bare `model=` kwarg
        would reach whichever route already answered rather than choosing one — a different
        thing than what this flag is about, so it is left untested here rather than claimed.
        """
        return False

    # --- v3 streaming is refused by design ---

    @pytest.mark.xfail(
        reason="ChatRouter refuses stream_events(version='v3') by design; see "
        "router._V3_UNSUPPORTED — it drives the model through _stream/_generate directly, "
        "which would bypass routing, the decision record and the run shape entirely."
    )
    def test_stream_events_v3(self, model: BaseChatModel) -> None:
        pytest.skip("v3 streaming is refused, not supported: router._V3_UNSUPPORTED")

    @pytest.mark.xfail(
        reason="ChatRouter refuses astream_events(version='v3') by design; see "
        "router._V3_UNSUPPORTED — same reason as test_stream_events_v3."
    )
    async def test_astream_events_v3(self, model: BaseChatModel) -> None:
        pytest.skip("v3 streaming is refused, not supported: router._V3_UNSUPPORTED")


@pytest.mark.requires_ollama
class TestChatRouterOllamaIntegration(_RouterIntegrationTests):
    """A real Ollama route (`qwen3:8b` by default) behind the router."""

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"routes": {"ollama": init_chat_model(OLLAMA_MODEL)}, "default_route": "ollama"}

    @property
    def has_tool_choice(self) -> bool:
        """`qwen3:8b` via `ChatOllama` does not reliably honor a forced `tool_choice` — verified
        directly against `ChatOllama` (no router involved): `bind_tools([...],
        tool_choice="any").invoke(...)` answers with prose and no tool call. Not a routing gap:
        the router replays the binding on the route unchanged, so whatever the route
        does with it is what a bare `ChatOllama` does too. `test_tool_choice` reads this flag on
        its own (`langchain_tests`' sanctioned opt-out); `test_unicode_tool_call_integration`
        does not, so it is overridden below instead."""
        return False

    @pytest.mark.xfail(
        reason="qwen3:8b via ChatOllama does not reliably honor a forced "
        "tool_choice (see has_tool_choice above) — the same failure reproduces directly "
        "against ChatOllama, with no router involved."
    )
    def test_unicode_tool_call_integration(
        self,
        model: BaseChatModel,
        *,
        tool_choice: str | None = None,  # noqa: PT028 — matches the overridden signature
        force_tool_call: bool = True,  # noqa: PT028 — matches the overridden signature
    ) -> None:
        pytest.skip(
            "qwen3:8b via ChatOllama does not reliably honor a forced tool_choice; not a "
            "routing gap — see has_tool_choice's docstring on this class."
        )


@pytest.mark.requires_env("GEMINI_API_KEY")
class TestChatRouterGeminiIntegration(_RouterIntegrationTests):
    """A real Gemini route behind the router."""

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"routes": {"gemini": init_chat_model(GEMINI_MODEL)}, "default_route": "gemini"}
