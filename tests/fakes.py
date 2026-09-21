"""Fake routes for offline tests — ported from `spike/fakes.py` (tests never import `spike/`).

``GenericFakeChatModel`` drops ``usage_metadata`` and ``response_metadata`` when it streams,
and no core fake implements ``bind_tools`` — both of which the router's tests need.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCall
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field


class GenerateOnlyFakeChatModel(BaseChatModel):
    """A route with no streaming API of its own: it implements `_generate` and nothing else.

    Providers without a streaming endpoint look like this, and LangChain covers for them —
    `BaseChatModel.stream` falls back to `invoke` and yields the whole message as the single
    chunk. The router has to preserve that (REQ-C2-4), so tests need a route that has it.

    Cannot use tools: it keeps `BaseChatModel`'s `bind_tools`, which raises at call time (R10).
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "fake-1"
    reply: str = "an answer"
    input_tokens: int = 3
    output_tokens: int = 5
    tool_calls: list[ToolCall] = Field(default_factory=list)
    script: list[AIMessage] = Field(default_factory=list)
    """Replies to give in order, for multi-turn runs such as an agent's tool loop."""
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
        if self.script:
            message = self.script.pop(0)
            message.response_metadata.setdefault("model_name", self.model_name)
            if message.usage_metadata is None:
                message.usage_metadata = self._usage()
            return message
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


class FakeChatModel(GenerateOnlyFakeChatModel):
    """A route that answers with fixed content and reports usage like a real provider.

    Streams its reply token by token, as a provider with a streaming API does.
    """

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.calls.append(dict(kwargs))
        message = self._message()
        for token in re.split(r"(\s)", str(message.content)):
            if token:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        # Usage and model name arrive on the final chunk, as providers send them.
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                chunk_position="last",
                usage_metadata=message.usage_metadata,
                response_metadata=dict(message.response_metadata),
                tool_call_chunks=[
                    tool_call_chunk(
                        name=call["name"],
                        args=json.dumps(call["args"]),
                        id=call["id"],
                        index=index,
                    )
                    for index, call in enumerate(message.tool_calls)
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
    ) -> Runnable[LanguageModelInput, AIMessage]:
        converted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=converted, **kwargs)


class NativeStructuredFakeChatModel(ToolCallingFakeChatModel):
    """A route with its own `with_structured_output`, as the real providers have.

    Records the provider arguments it was given, so a test can see whether the router
    forwarded them or dropped them.
    """

    structured_output_calls: list[dict[str, Any]] = Field(default_factory=list)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        method: str = "function_calling",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        self.structured_output_calls.append(
            {"method": method, "include_raw": include_raw, **kwargs}
        )
        return super().with_structured_output(schema, include_raw=include_raw, **kwargs)


def call_log(route: BaseChatModel) -> list[dict[str, Any]]:
    """Every call a fake route took, reached through `BaseChatModel`.

    `ChatRouter.routes` is typed to LangChain's base class, and `dict` is invariant, so tests
    hold their routes as `dict[str, BaseChatModel]` and come back here for the fake's own record.
    """
    return cast("GenerateOnlyFakeChatModel", route).calls
