"""Fake routes for the spike's offline tests.

``GenericFakeChatModel`` drops ``usage_metadata`` and ``response_metadata`` when it streams,
and no core fake implements ``bind_tools`` — both of which the spike needs (T-003, T-004).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCall
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field


class FakeChatModel(BaseChatModel):
    """A route that answers with fixed content and reports usage like a real provider.

    Cannot use tools: `bind_tools` raises, as `BaseChatModel`'s default does (R10 preview).
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "fake-1"
    reply: str = "an answer"
    input_tokens: int = 3
    output_tokens: int = 5
    tool_calls: list[ToolCall] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)
    """Every kwargs dict this route was called with, for assertions."""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _usage(self) -> UsageMetadata:
        return UsageMetadata(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )

    def _message(self) -> AIMessage:
        return AIMessage(
            content=self.reply,
            tool_calls=list(self.tool_calls),
            usage_metadata=self._usage(),
            response_metadata={"model_name": self.model_name},
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(dict(kwargs))
        return ChatResult(generations=[ChatGeneration(message=self._message())])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.calls.append(dict(kwargs))
        for token in re.split(r"(\s)", self.reply):
            if token:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        # Usage and model name arrive on the final chunk, as providers send them.
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                chunk_position="last",
                usage_metadata=self._usage(),
                response_metadata={"model_name": self.model_name},
                tool_call_chunks=[
                    tool_call_chunk(
                        name=call["name"],
                        args=json.dumps(call["args"]),
                        id=call["id"],
                        index=index,
                    )
                    for index, call in enumerate(self.tool_calls)
                ],
            )
        )


class ToolCallingFakeChatModel(FakeChatModel):
    """A route that can use tools: `bind_tools` converts them and binds them as kwargs."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        converted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=converted, **kwargs)
