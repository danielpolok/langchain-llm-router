"""T-002 and T-003: the smallest router that T-004 can test against.

Spike quality: it answers questions, it is not v1. Principles: C1, C2, C3, R1, R2, R5, R9.

The naive delegation design on purpose — T-004 starts by confirming the double count this
produces, then scores alternatives against R1, R3 and C5.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.model_profile import ModelProfile
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from typing_extensions import Self

ROUTING_KEY = "routing"
"""Key the decision record rides under in ``response_metadata`` (R2). Shape is not final."""

TOOL_BINDING_KEY = "__router_tool_binding"
"""Call kwarg carrying the tools the caller bound, unconverted, until a route is chosen."""

Strategy = Callable[[list[BaseMessage]], str | None]
"""A policy applied to one request: pick a route by name, or ``None`` to abstain."""

MessageT = TypeVar("MessageT", bound=AIMessage)

ToolLike = dict[str, Any] | type | Callable[..., Any] | BaseTool


class RoutingWarning(UserWarning):
    """The strategy could not decide, or chose a route that does not exist (R9)."""


@dataclass(frozen=True)
class RoutingDecision:
    """Which route was taken, and why (R2)."""

    route: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"route": self.route, "reason": self.reason}


@dataclass(frozen=True)
class ToolBinding:
    """What `bind_tools` was called with, kept until a route can convert it (C3)."""

    tools: tuple[ToolLike, ...]
    tool_choice: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def apply(self, route: BaseChatModel) -> Runnable[LanguageModelInput, AIMessage]:
        """Let the *selected* route do the provider-specific conversion, at call time."""
        if self.tool_choice is None:
            return route.bind_tools(list(self.tools), **self.kwargs)
        return route.bind_tools(list(self.tools), tool_choice=self.tool_choice, **self.kwargs)


def _child_callbacks(
    run_manager: CallbackManagerForLLMRun | None,
) -> CallbackManager | None:
    """Build the callback manager for the route's run, nested under the router's run (C5).

    ``CallbackManagerForLLMRun`` is not a ``ParentRunManager`` and so has no ``get_child()``:
    LangChain does not expect an LLM run to have children. This reproduces what
    ``ParentRunManager.get_child`` does.
    """
    if run_manager is None:
        return None
    manager = CallbackManager(handlers=[], parent_run_id=run_manager.run_id)
    manager.set_handlers(run_manager.inheritable_handlers)
    manager.add_tags(run_manager.inheritable_tags)
    manager.add_metadata(run_manager.inheritable_metadata)
    return manager


def _achild_callbacks(
    run_manager: AsyncCallbackManagerForLLMRun | None,
) -> AsyncCallbackManager | None:
    """Async counterpart of `_child_callbacks`."""
    if run_manager is None:
        return None
    manager = AsyncCallbackManager(handlers=[], parent_run_id=run_manager.run_id)
    manager.set_handlers(run_manager.inheritable_handlers)
    manager.add_tags(run_manager.inheritable_tags)
    manager.add_metadata(run_manager.inheritable_metadata)
    return manager


class SpikeRouterChatModel(BaseChatModel):
    """A chat model that hands each request to one of several chat models (C1, R5)."""

    routes: dict[str, BaseChatModel]
    """Named routes; any number of them (R5). Each is an ordinary chat model."""

    default_route: str
    """Mandatory (R9): where requests go when the strategy fails or cannot decide."""

    strategy: Strategy | None = Field(default=None, exclude=True)
    """Applies the application's policy to one request. ``None`` always uses the default."""

    @model_validator(mode="after")
    def _default_route_exists(self) -> Self:
        if self.default_route not in self.routes:
            msg = (
                f"default_route {self.default_route!r} is not one of the routes: "
                f"{sorted(self.routes)}"
            )
            raise ValueError(msg)
        return self

    @property
    def _llm_type(self) -> str:
        return "spike_router"

    def _resolve_model_profile(self) -> ModelProfile | None:
        """Report only what *every* route can do.

        `create_agent` reads `profile` to choose a structured-output strategy. A router that
        claimed a capability only some of its routes have would have a strategy picked that a
        route cannot serve, so the shared capabilities are the honest answer (C3, R10).
        """
        profiles = [route.profile for route in self.routes.values()]
        if not profiles or any(profile is None for profile in profiles):
            return None

        known = [cast("dict[str, Any]", profile) for profile in profiles]
        shared: dict[str, Any] = {}
        for key in set.intersection(*(set(profile) for profile in known)):
            values = [profile[key] for profile in known]
            if all(isinstance(value, bool) for value in values):
                shared[key] = all(values)
            elif all(isinstance(value, int) for value in values):
                shared[key] = min(values)
            elif all(value == values[0] for value in values):
                shared[key] = values[0]
        return cast("ModelProfile", shared)

    def decide(self, messages: list[BaseMessage]) -> RoutingDecision:
        """Apply the strategy, falling back to the default route on failure (R9)."""
        if self.strategy is None:
            return RoutingDecision(self.default_route, "no strategy configured")

        try:
            chosen = self.strategy(messages)
        except Exception as exc:  # R9: a failing strategy must still answer the request
            self._warn(f"strategy raised {exc!r}")
            return RoutingDecision(self.default_route, f"strategy raised {type(exc).__name__}")

        if chosen is None:
            self._warn("strategy did not decide")
            return RoutingDecision(self.default_route, "strategy did not decide")
        if chosen not in self.routes:
            self._warn(f"strategy chose unknown route {chosen!r}")
            return RoutingDecision(self.default_route, f"unknown route {chosen!r}")
        return RoutingDecision(chosen, "strategy")

    def _warn(self, problem: str) -> None:
        warnings.warn(
            f"{problem}; falling back to the default route {self.default_route!r}",
            RoutingWarning,
            stacklevel=3,
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Keep the tools as they are; the chosen route converts them at call time (C3).

        `tools` is bound as well, in its usual place, so LangChain's own checks that look for
        it — `disable_streaming="tool_calling"`, for one — behave as on any chat model.
        """
        binding = ToolBinding(tools=tuple(tools), tool_choice=tool_choice, kwargs=dict(kwargs))
        return self.bind(**{TOOL_BINDING_KEY: binding, "tools": list(tools)})

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        """Forward to the selected route's own implementation, per request (C3).

        `BaseChatModel`'s default would build on *this* model's `bind_tools` and silently drop
        provider arguments such as `method=`, so every route would be forced through function
        calling. Forwarding keeps each route's native structured-output modes.
        """

        def pick(model_input: LanguageModelInput) -> Runnable[LanguageModelInput, Any]:
            decision = self.decide(self._convert_input(model_input).to_messages())
            return self.routes[decision.route].with_structured_output(
                schema, include_raw=include_raw, **kwargs
            )

        return cast(
            "Runnable[LanguageModelInput, dict[str, Any] | BaseModel]",
            RunnableLambda(pick),
        )

    def _target(
        self, decision: RoutingDecision, kwargs: dict[str, Any]
    ) -> tuple[Runnable[LanguageModelInput, AIMessage], dict[str, Any]]:
        """The runnable to call, and the call kwargs left once tool binding is replayed."""
        route = self.routes[decision.route]
        call_kwargs = dict(kwargs)
        binding = call_kwargs.pop(TOOL_BINDING_KEY, None)
        if binding is None:
            return route, call_kwargs
        call_kwargs.pop("tools", None)  # replayed through the route's own bind_tools
        return cast("ToolBinding", binding).apply(route), call_kwargs

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        decision = self.decide(messages)
        target, call_kwargs = self._target(decision, kwargs)
        message = target.invoke(
            messages,
            config={"callbacks": _child_callbacks(run_manager)},
            stop=stop,
            **call_kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=_record(message, decision))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        decision = self.decide(messages)
        target, call_kwargs = self._target(decision, kwargs)
        message = await target.ainvoke(
            messages,
            config={"callbacks": _achild_callbacks(run_manager)},
            stop=stop,
            **call_kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=_record(message, decision))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        decision = self.decide(messages)
        target, call_kwargs = self._target(decision, kwargs)
        first = True
        for message in target.stream(
            messages,
            config={"callbacks": _child_callbacks(run_manager)},
            stop=stop,
            **call_kwargs,
        ):
            chunk = cast("AIMessageChunk", message)
            if first:
                # Only one chunk may carry the record: merge_dicts concatenates strings
                # that repeat across chunks, so a record on every chunk would come out
                # as "cheapcheapcheap".
                _record(chunk, decision)
                first = False
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        decision = self.decide(messages)
        target, call_kwargs = self._target(decision, kwargs)
        first = True
        async for message in target.astream(
            messages,
            config={"callbacks": _achild_callbacks(run_manager)},
            stop=stop,
            **call_kwargs,
        ):
            chunk = cast("AIMessageChunk", message)
            if first:
                _record(chunk, decision)
                first = False
            yield ChatGenerationChunk(message=chunk)


def _record(message: MessageT, decision: RoutingDecision) -> MessageT:
    """Add the decision to the route's own message; nothing is dropped (R1, R2)."""
    message.response_metadata[ROUTING_KEY] = decision.as_dict()
    return message


def routing_decision(message: AIMessage) -> dict[str, str] | None:
    """Read the decision back off a response (R2)."""
    record = message.response_metadata.get(ROUTING_KEY)
    return record if isinstance(record, dict) else None
