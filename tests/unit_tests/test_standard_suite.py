"""`ChatRouter` passes `langchain_tests`' standard offline compliance suite.

`ChatModelUnitTests` is what every LangChain partner package runs against its own chat model;
subclassing it here and pointing it at `ChatRouter` is the acceptance criterion itself, not a
proxy for it. Nothing is overridden — `test_no_overrides_DO_NOT_OVERRIDE` (`langchain_tests.
base.BaseStandardTests`) asserts as much on its own — so there is nothing to document a skip
reason for: every one of the base suite's 8 tests runs against `ChatRouter` unchanged and passes.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_llm_router import ChatRouter
from tests.fakes import NativeStructuredFakeChatModel


class TestChatRouterUnit(ChatModelUnitTests):
    """The standard suite, run against `ChatRouter` itself.

    Routes are `NativeStructuredFakeChatModel`s — tool-capable and with a structured output of
    their own — so `has_tool_calling` and `has_structured_output` (both auto-detected `True`,
    since `ChatRouter` overrides `bind_tools` / `with_structured_output`) have something to bind
    to: with an incapable route instead, `test_bind_tool_pydantic`'s first `bind_tools` call
    would raise `NoToolCapableRouteError`, which is correct behaviour but not what this
    suite is testing. `strategy` is left `None` (always the default route): every standard test
    exercises construction, binding or `_get_ls_params`, never an actual generation, so which
    route would answer is never in question.
    """

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatRouter

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {
            "routes": {
                "primary": NativeStructuredFakeChatModel(model_name="primary"),
                "secondary": NativeStructuredFakeChatModel(model_name="secondary"),
            },
            "default_route": "primary",
        }
