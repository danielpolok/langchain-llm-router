"""Fake routes for offline tests — ported from `spike/fakes.py` (tests never import `spike/`).

``GenericFakeChatModel`` drops ``usage_metadata`` and ``response_metadata`` when it streams,
and no core fake implements ``bind_tools`` — both of which the router's tests need.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, cast

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

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """What a cache lookup's `_get_llm_string` folds in (T-119, REQ-C10-1).

        `BaseChatModel._identifying_params` defaults to `{}` (`language_models/base.py:430`),
        so two fakes with different names would otherwise be indistinguishable to a cache that
        isn't `is_lc_serializable` (none of these are, matching the base fake's own default) —
        the fakes would collide where two real routes never would, since a real provider's
        `model` (or similar) field rides along either through its own `_identifying_params` or
        through `is_lc_serializable`'s full `dumpd(self)`. Naming the model here is what keeps
        a test's routes as distinguishable to the cache as two real ones would be.
        """
        return {"model_name": self.model_name}

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
    """A route that can use tools: `bind_tools` converts them and binds them as kwargs.

    Behaves as the providers do where it matters to the router: `strict=` is consumed at bind
    time, changing how the schema is built (`convert_to_openai_tool(..., strict=...)`), and so
    never becomes a call kwarg; `tool_choice` and any other binding kwarg do.
    """

    tool_format: Literal["openai", "anthropic"] = "openai"
    """The shape this provider wants a converted tool in: OpenAI's function spec, or Anthropic's
    `name` / `description` / `input_schema` — so two routes can convert the same tool apart."""

    bind_calls: list[dict[str, Any]] = Field(default_factory=list)
    """Every `bind_tools` call this route took, as given — before any conversion."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bind_calls.append({"tools": list(tools), "tool_choice": tool_choice, **kwargs})
        strict = kwargs.pop("strict", None)
        converted = [self._convert(tool, strict) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=converted, **kwargs)

    def _convert(self, tool: Any, strict: bool | None) -> dict[str, Any]:
        spec = convert_to_openai_tool(tool, strict=strict)
        if self.tool_format == "openai":
            return spec
        function = spec["function"]
        return {
            "name": function["name"],
            "description": function.get("description", ""),
            "input_schema": function["parameters"],
        }


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


class StreamingStructuredFakeChatModel(NativeStructuredFakeChatModel):
    """A route whose structured output streams progressively, as a real provider's does.

    `NativeStructuredFakeChatModel._stream` (inherited from `FakeChatModel`) hands over a tool
    call's arguments in one final chunk — enough for most tests, but not for T-115's structured
    streaming (C1, C2), which needs a route that answers the way `ChatOllama` or `ChatOpenAI`
    do: a growing JSON string, several chunks wide, that the route's own parser turns into
    progressively more complete partials. `piece_size` controls how many chunks that takes.

    `method="json_mode"` streams the same JSON as plain text content instead of a tool call —
    the other native structured mode a provider offers, parsed by `JsonOutputParser` rather
    than a tool-call parser — because that path takes a different shape through
    `with_structured_output` (`llm | JsonOutputParser()` instead of `bind_tools(...) | parser`)
    and T-115's tests check both.
    """

    piece_size: int = 6

    def _pieces(self, message: AIMessage) -> list[str]:
        (call,) = message.tool_calls
        args = json.dumps(call["args"])
        return [args[i : i + self.piece_size] for i in range(0, len(args), self.piece_size)]

    def _for_json_mode(self, message: AIMessage) -> AIMessage:
        """`message`, with its tool call's arguments as plain JSON content instead — what
        `method="json_mode"` answers with, on this fake as on a real provider."""
        (call,) = message.tool_calls
        return message.model_copy(update={"content": json.dumps(call["args"]), "tool_calls": []})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(dict(kwargs))
        message = self._message()
        if kwargs.get("json_mode", False):
            message = self._for_json_mode(message)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.calls.append(dict(kwargs))
        message = self._message()
        json_mode = kwargs.get("json_mode", False)
        for index, piece in enumerate(self._pieces(message)):
            if json_mode:
                yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
            else:
                (call,) = message.tool_calls
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            tool_call_chunk(
                                name=call["name"] if index == 0 else None,
                                args=piece,
                                id=call["id"] if index == 0 else None,
                                index=0,
                            )
                        ],
                    )
                )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                chunk_position="last",
                usage_metadata=message.usage_metadata,
                response_metadata=dict(message.response_metadata),
            )
        )

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
        if method != "json_mode":
            return BaseChatModel.with_structured_output(
                self, schema, include_raw=include_raw, **kwargs
            )
        # The other native mode: the model answers with JSON text, not a tool call, so it is
        # bound with `json_mode=True` rather than through `bind_tools` — `_generate` and
        # `_stream` read it back off the call kwargs, exactly where a bound kwarg lands (C3).
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import RunnableMap, RunnablePassthrough

        llm = self.bind(json_mode=True)
        parser = JsonOutputParser()
        if include_raw:
            return RunnableMap(raw=llm) | RunnablePassthrough.assign(
                parsed=lambda output: parser.invoke(output["raw"]),
                parsing_error=lambda _: None,
            )
        return llm | parser


def call_log(route: BaseChatModel) -> list[dict[str, Any]]:
    """Every call a fake route took, reached through `BaseChatModel`.

    `ChatRouter.routes` is typed to LangChain's base class, and `dict` is invariant, so tests
    hold their routes as `dict[str, BaseChatModel]` and come back here for the fake's own record.
    """
    return cast("GenerateOnlyFakeChatModel", route).calls
