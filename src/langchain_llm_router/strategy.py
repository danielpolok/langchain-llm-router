"""The strategy interface (R6, D6): one small, stable surface every strategy implements.

Ready-made, configured and custom strategies all implement `RoutingStrategy`. A strategy sees
the user's current request by default (R4) and returns a `RoutingChoice`, or `None` when it
can't decide — the router then uses the default route (R9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from langchain_core.messages import BaseMessage, ContentBlock
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import run_in_executor

__all__ = [
    "RoutingCallable",
    "RoutingChoice",
    "RoutingRequest",
    "RoutingStrategy",
]


@dataclass(frozen=True)
class RoutingRequest:
    """What a strategy decides on: the current request, and the context it may use."""

    text: str
    """The current request's text (R4, D6)."""

    content_blocks: list[ContentBlock]
    """The current request's content, as LangChain defines content blocks (C7)."""

    modalities: frozenset[str]
    """The modalities present in the request — `{"text", "image", ...}`."""

    routes: tuple[str, ...]
    """The available route names, in declaration order."""

    tools_bound: bool
    """Tools or structured output are bound to this call."""

    messages: list[BaseMessage] | None = None
    """The whole transcript — only when the strategy sets `wants_full_context` (R4)."""

    config: RunnableConfig = field(
        default_factory=lambda: RunnableConfig(), compare=False, repr=False
    )
    """The strategy run's child config (D9). Pass it on to any model or embeddings call, so
    that call is traced and costed under the strategy's run. It identifies a run, not a
    request, so it takes no part in equality or repr."""


@dataclass(frozen=True)
class RoutingChoice:
    """A strategy's answer: the route to take, and why (R2)."""

    route: str
    reason: str


class RoutingStrategy(ABC):
    """Applies the application's routing policy to one request (R6)."""

    wants_full_context: ClassVar[bool] = False
    """Set to `True` to receive the whole transcript in `RoutingRequest.messages` (R4)."""

    @abstractmethod
    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Choose a route for `request`, or return `None` if this strategy can't decide."""

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Async `decide`. Defaults to `decide` in an executor, so async callers never block (C2).

        Strategies that call models override this with a native async implementation.
        """
        return await run_in_executor(request.config, self.decide, request)


RoutingCallable = Callable[[RoutingRequest], "RoutingChoice | str | None"]
"""A plain function accepted as `strategy=` and coerced to a `RoutingStrategy` (REQ-R6-2).

A bare string return names the route."""


class _CallableStrategy(RoutingStrategy):
    """A `RoutingCallable` wearing the `RoutingStrategy` interface (REQ-R6-2)."""

    def __init__(self, func: RoutingCallable) -> None:
        self.func = func
        self.name: str = getattr(func, "__name__", type(func).__name__)

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        choice = self.func(request)
        if isinstance(choice, str):
            return RoutingChoice(route=choice, reason=f"{self.name} chose {choice!r}")
        return choice


def as_strategy(strategy: RoutingStrategy | RoutingCallable) -> RoutingStrategy:
    """Coerce what `strategy=` accepts into a `RoutingStrategy`."""
    if isinstance(strategy, RoutingStrategy):
        return strategy
    if callable(strategy):
        return _CallableStrategy(strategy)
    msg = f"strategy must be a RoutingStrategy or a callable, not {type(strategy).__name__}"
    raise TypeError(msg)


def strategy_name(strategy: RoutingStrategy) -> str:
    """The name a decision record and the strategy's trace run carry (D8, D9)."""
    if isinstance(strategy, _CallableStrategy):
        return strategy.name
    return type(strategy).__name__
