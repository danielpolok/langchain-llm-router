"""The strategy interface (R6, D6): one small, stable surface every strategy implements.

Ready-made, configured and custom strategies are all a `RoutingStrategy`: one method, `decide`,
over a `RoutingRequest`, returning a `RoutingChoice` — or `None` when it can't decide, and the
router uses the default route (R9). A strategy that raises, or names a route that doesn't
exist, is treated the same way, with a `FallbackWarning` and the cause recorded.

- **Current request by default (R4).** A strategy sees the user's current request. It gets the
  whole transcript in `RoutingRequest.messages` only if it sets `wants_full_context = True`.
- **Sync and async (C2).** The router calls `decide` on sync paths and `adecide` on async ones.
  `adecide` defaults to `decide` in a worker thread, so `decide` must be thread-safe. A strategy
  that calls a model overrides `adecide` with a native async implementation.
- **Its own calls are traced (D9).** A strategy that calls a model or embeddings passes
  `request.config` to the call, so it is traced and costed under the strategy's run. Below
  Python 3.11 that is the only way an async call nests there.
- **A plain function is a strategy too (REQ-R6-2).** `strategy=pick`, where `pick(request)`
  returns a `RoutingChoice`, a bare route name, or `None`. It must be synchronous and sees only
  the current request; for an async strategy or the whole transcript, subclass
  `RoutingStrategy`.

Stability promise
-----------------
The interface is the four names `langchain_llm_router` exports from here: `RoutingStrategy`,
`RoutingRequest`, `RoutingChoice` and `RoutingCallable` — their members and what each means.
Versions follow semantic versioning; until 1.0, the minor version stands in for the major one.

- **Minor release (compatible):** a new `RoutingRequest` field, added last with a default; a new
  optional `RoutingStrategy` member whose default keeps today's behaviour, as
  `wants_full_context` does; new values in `modalities` as LangChain adds content-block types;
  new wording in the reasons the package writes; a widened type that keeps accepting everything
  it accepts today, such as `ClassVar[bool]` becoming `bool`.
- **Major release (breaking):** removing or renaming anything above, or retyping it so that code
  which type-checks today no longer does; a new abstract method; changing what `None` or a bare
  route name means; changing the signature of `decide` or `adecide`; changing
  `wants_full_context`'s default (R4).

Everything else in this module — `as_strategy`, `strategy_name` and the underscore names — is
the router's own and may change in any release.
"""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

from langchain_core.messages import BaseMessage, ContentBlock
from langchain_core.runnables import RunnableConfig, run_in_executor

__all__ = [
    "RoutingCallable",
    "RoutingChoice",
    "RoutingRequest",
    "RoutingStrategy",
]


@dataclass(frozen=True)
class RoutingRequest:
    """What a strategy decides on: the current request, and the context it may use.

    Frozen but not hashable: `content_blocks` and `messages` are lists, so a generated hash
    would fail on them anyway. To cache decisions, key on a hashable part such as `text`.
    """

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
    request, so it takes no part in equality or repr (REQ-R6-5)."""

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RoutingChoice:
    """A strategy's answer: the route to take, and why (R2)."""

    route: str
    reason: str


class RoutingStrategy(ABC):
    """Applies the application's routing policy to one request (R6).

    Subclass it and implement `decide`. The router calls it once per request, and not at all
    when the route is forced (D2).
    """

    wants_full_context: ClassVar[bool] = False
    """Set to `True` to receive the whole transcript in `RoutingRequest.messages` (R4)."""

    @abstractmethod
    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Choose a route for `request`, or return `None` if this strategy can't decide.

        `None` sends the request to the default route, with a warning and the reason
        recorded (R9) — better than a guess.
        """

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Async `decide`. Defaults to `decide` in an executor, so async callers never block (C2).

        `decide` therefore has to be thread-safe. Strategies that call models override this
        with a native async implementation, passing `request.config` to their calls (D9).
        """
        # A config sends `run_in_executor` down the same path as no executor at all: the loop's
        # default one, running `decide` in a copy of the *current* context
        # (`runnables/config.py:705`). That copy — not the argument — is what carries the config
        # and tracing context the router sets with `set_config_context` into `decide`'s thread
        # (D9). The config is passed because it is the one a strategy's calls take, and because
        # `Runnable.ainvoke` passes its own here too (`runnables/base.py:929`).
        return await run_in_executor(request.config, self.decide, request)


RoutingCallable: TypeAlias = Callable[[RoutingRequest], RoutingChoice | str | None]
"""A plain function accepted as `strategy=` and coerced to a `RoutingStrategy` (REQ-R6-2).

It returns a `RoutingChoice`, a bare route name (the reason is then written for it), or `None`
to abstain. It must be synchronous, and it sees only the current request."""


class _CallableStrategy(RoutingStrategy):
    """A `RoutingCallable` wearing the `RoutingStrategy` interface (REQ-R6-2).

    It keeps the defaults: the current request only (R4), and `decide` in an executor for
    async callers (C2).
    """

    def __init__(self, func: RoutingCallable) -> None:
        self.func = func
        self.name = _callable_name(func)

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        choice = self.func(request)
        if isinstance(choice, str):
            return RoutingChoice(route=choice, reason=f"{self.name} chose {choice!r}")
        if choice is None or isinstance(choice, RoutingChoice):
            return choice
        # Reached by untyped code, e.g. a lambda that returns an un-awaited coroutine; the
        # router reports it like any other strategy failure (R9).
        msg = (
            f"strategy {self.name!r} returned {type(choice).__name__}; "
            "expected a RoutingChoice, a route name or None"
        )
        raise TypeError(msg)


def _unwrap(func: object) -> object:
    """What a `functools.partial` chain ends in — what a call actually reaches."""
    while isinstance(func, functools.partial):
        func = func.func
    return func


def _callable_name(func: Callable[..., object]) -> str:
    """The function's own name, seen through `functools.partial`.

    A lambda is `<lambda>`, as Python names it; a callable object goes by its class name.
    """
    unwrapped = _unwrap(func)
    name = getattr(unwrapped, "__name__", None)
    return name if isinstance(name, str) else type(unwrapped).__name__


def _is_async(func: object) -> bool:
    """Whether calling `func` hands back something to await rather than a choice.

    Checked on the type's `__call__` as well, as a call looks it up, so a callable object with
    an `async def __call__` is caught too. A sync function that hides an async one behind
    `functools.wraps` can't be told apart — that one surfaces on the first request (R9).
    """
    return inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)


def as_strategy(strategy: RoutingStrategy | RoutingCallable) -> RoutingStrategy:
    """Coerce what `strategy=` accepts into a `RoutingStrategy` (REQ-R6-2).

    Raises `TypeError` for anything else — including two near misses that would otherwise fail
    on every request rather than once, here: a strategy class passed without instantiating it,
    and an `async def` function, which `RoutingCallable` (synchronous) doesn't cover.
    """
    if isinstance(strategy, RoutingStrategy):
        return strategy
    # `__mro__` rather than `issubclass`, which would leave mypy seeing `type[object]` below.
    if inspect.isclass(strategy) and RoutingStrategy in strategy.__mro__:
        msg = f"strategy= takes an instance: pass {strategy.__name__}(), not the class"
        raise TypeError(msg)
    target = _unwrap(strategy)
    if _is_async(target) or _is_async(type(target).__call__):
        msg = (
            f"strategy {_callable_name(strategy)!r} is async, and a plain function must be "
            "synchronous: subclass RoutingStrategy and override adecide instead"
        )
        raise TypeError(msg)
    if callable(strategy):
        return _CallableStrategy(strategy)
    msg = f"strategy must be a RoutingStrategy or a callable, not {type(strategy).__name__}"
    raise TypeError(msg)


def strategy_name(strategy: RoutingStrategy) -> str:
    """The name a decision record and the strategy's trace run carry (D8, D9).

    A strategy's class name, or a coerced function's own name.
    """
    if isinstance(strategy, _CallableStrategy):
        return strategy.name
    return type(strategy).__name__
