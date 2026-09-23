"""Fake routes shared by the agent examples — no support code beyond this.

Every other example needs only ``GenericFakeChatModel`` (public API, `langchain_core`). The
two agent examples also bind a tool, and no `langchain-core` fake implements `bind_tools` — it
raises `NotImplementedError` at call time, same as a real provider without tool support. This
adds just enough to run `create_agent` offline, so the examples need no API key and no network
call; swap in any real chat model (`init_chat_model(...)`) and the rest of each example is
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class ScriptedToolChatModel(GenericFakeChatModel):
    """`GenericFakeChatModel` with a working `bind_tools` (D5): it accepts the tools and
    answers with whatever `messages` was given next, exactly as the base class already does.
    """

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)
