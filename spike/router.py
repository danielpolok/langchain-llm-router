"""T-002: the smallest router that T-003 and T-004 can test against.

Spike quality: it answers questions, it is not v1. Principles: C1, C2, R1, R2, R5, R9.

The naive delegation design on purpose — T-004 starts by confirming the double count this
produces, then scores alternatives against R1, R3 and C5.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, model_validator

if TYPE_CHECKING:
    from typing_extensions import Self

ROUTING_KEY = "routing"
"""Key the decision record rides under in ``response_metadata`` (R2). Shape is not final."""

Strategy = Callable[[list[BaseMessage]], str | None]
"""A policy applied to one request: pick a route by name, or ``None`` to abstain."""

MessageT = TypeVar("MessageT", bound=AIMessage)


class RoutingWarning(UserWarning):
    """The strategy could not decide, or chose a route that does not exist (R9)."""


@dataclass(frozen=True)
class RoutingDecision:
    """Which route was taken, and why (R2)."""

    route: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"route": self.route, "reason": self.reason}


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

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        decision = self.decide(messages)
        message = self.routes[decision.route].invoke(
            messages,
            config={"callbacks": _child_callbacks(run_manager)},
            stop=stop,
            **kwargs,
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
        message = await self.routes[decision.route].ainvoke(
            messages,
            config={"callbacks": _achild_callbacks(run_manager)},
            stop=stop,
            **kwargs,
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
        first = True
        for chunk in self.routes[decision.route].stream(
            messages,
            config={"callbacks": _child_callbacks(run_manager)},
            stop=stop,
            **kwargs,
        ):
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
        first = True
        async for chunk in self.routes[decision.route].astream(
            messages,
            config={"callbacks": _achild_callbacks(run_manager)},
            stop=stop,
            **kwargs,
        ):
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
