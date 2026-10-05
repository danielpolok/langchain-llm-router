"""The strategy interface: one small, stable surface every strategy implements.

Ready-made, configured and custom strategies are all a `RoutingStrategy`: one method, `decide`,
over a `RoutingRequest`, returning a `RoutingChoice` — or `None` when it can't decide, and the
router uses the default route. A strategy that raises, or names a route that doesn't
exist, is treated the same way, with a `FallbackWarning` and the cause recorded.

- **Current request by default.** A strategy sees the user's current request. It gets the
  user's previous messages in `RoutingRequest.previous_requests` only if it sets `lookback`
  to how many it wants, and the whole transcript in `RoutingRequest.messages` only if it sets
  `wants_full_context = True`. The two are independent.
- **Wrappers pass `lookback` on.** The router hands over as many previous messages as the
  strategy it was given asks for, and a built-in reads at most its own `lookback`. A strategy
  that hands the request to another one therefore declares that one's `lookback`, and
  `with_lookback` gives it a copy that reads a different number.
- **A choice can say which message decided.** `RoutingChoice.messages_back` is how far back the
  user message that decided was, as `lookback` counts; the router records it on the decision.
  A wrapper keeps it by returning the inner strategy's choice, or a `dataclasses.replace` of it.
- **Sync and async.** The router calls `decide` on sync paths and `adecide` on async ones.
  `adecide` defaults to `decide` in a worker thread, so `decide` must be thread-safe. A strategy
  that calls a model overrides `adecide` with a native async implementation.
- **Its own calls are traced.** A strategy that calls a model or embeddings passes
  `request.config` to the call, so it is traced and costed under the strategy's run. Below
  Python 3.11 that is the only way an async call nests there.
- **A plain function is a strategy too.** `strategy=pick`, where `pick(request)`
  returns a `RoutingChoice`, a bare route name, or `None`. It must be synchronous and sees only
  the current request; for an async strategy, earlier messages or the whole transcript,
  subclass `RoutingStrategy`.

Stability promise
-----------------
The interface is the four names `langchain_model_router` exports from here: `RoutingStrategy`,
`RoutingRequest`, `RoutingChoice` and `RoutingCallable` — their members and what each means.
Versions follow semantic versioning; until 1.0, the minor version stands in for the major one.

- **Minor release (compatible):** a new `RoutingRequest` or `RoutingChoice` field, added last with a
  default, as `previous_requests` and `messages_back` were; a new optional `RoutingStrategy` member
  whose default keeps today's behaviour, as `wants_full_context`, `lookback` and `with_lookback`
  are; new values in `modalities` as LangChain adds content-block types; new wording in the reasons
  the package writes; a widened type that keeps accepting everything it accepts today, such as
  `ClassVar[bool]` becoming `bool`.
- **Major release (breaking):** removing or renaming anything above, or retyping it so that code
  which type-checks today no longer does; a new abstract method; changing what `None` or a bare
  route name means; changing the signature of `decide` or `adecide`; changing the default of
  `wants_full_context` or `lookback`, what `lookback` or `messages_back` counts, or the order
  of `previous_requests`.

Everything else in this module — `as_strategy`, `strategy_name` and the underscore names — is
the router's own and may change in any release.
"""

from __future__ import annotations

import copy
import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias, TypeVar

from langchain_core.messages import BaseMessage, ContentBlock
from langchain_core.runnables import RunnableConfig, run_in_executor

from langchain_model_router.errors import RoutingError

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
    would fail on them anyway. To cache decisions, key on a hashable part such as `text` — and,
    for a strategy with a `lookback`, on the previous requests' `text` as well, since the same
    follow-up can decide differently after different messages.

    Attributes:
        text: The current request's text.
        content_blocks: The current request's content, as LangChain defines content blocks.
        modalities: The modalities present, from a closed vocabulary.
        routes: The available route names, in declaration order.
        tools_bound: Whether tools or structured output are bound to this call.
        messages: The transcript up to now, or `None` unless the strategy opts in.
        config: The strategy run's child config, to pass on to any model call.
        previous_requests: The user's previous messages, newest first, up to the strategy's
            `lookback`.

    Example:
        ```python
        def pick(request: RoutingRequest) -> str | None:
            if "image" in request.modalities:
                return "vision"
            return "small" if len(request.text) < 200 else None
        ```
    """

    text: str
    """The current request's text."""

    content_blocks: list[ContentBlock]
    """The current request's content, as LangChain defines content blocks."""

    modalities: frozenset[str]
    """The modalities present in the request, from a closed vocabulary.

    `"text"` (whenever `text` is non-empty), `"image"`, `"audio"`, `"video"`, `"file"` — a
    document, including a plain-text one, whose text is not part of `text` — and `"other"` for
    content LangChain does not translate into a standard block. New values arrive only as
    LangChain adds content-block types."""

    routes: tuple[str, ...]
    """The available route names, in declaration order."""

    tools_bound: bool
    """Tools or structured output are bound to this call."""

    messages: list[BaseMessage] | None = None
    """The transcript — only when the strategy sets `wants_full_context`.

    The whole of it on the current request. On a previous request, the conversation as it stood
    when that message was sent: everything up to and including it."""

    config: RunnableConfig = field(
        default_factory=lambda: RunnableConfig(), compare=False, repr=False
    )
    """The strategy run's child config. Pass it on to any model or embeddings call, so
    that call is traced and costed under the strategy's run. It identifies a run, not a
    request, so it takes no part in equality or repr."""

    previous_requests: tuple[RoutingRequest, ...] = ()
    """The user's previous messages, newest first — up to the strategy's `lookback` of them.

    Empty unless the strategy sets `lookback`, and on a conversation's first turn. Each is read
    by the rules the current request is read by, so system prompts, the model's answers and tool
    results are never among them, and inside an agent's tool loop they are the messages before
    the one that started it. Each is itself a `RoutingRequest`, so it can be handed to any
    strategy's `decide`: it is the request the router read when that message was the current
    one, as far as the transcript records it. Its `text`, `content_blocks` and `modalities` are
    that message's, and its `messages` end at that message. What the transcript doesn't record
    is this call's: `routes`, `tools_bound` and `config`. Its own `previous_requests` is
    empty."""

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RoutingChoice:
    """A strategy's answer: the route to take, and why.

    Args:
        route: The name of the route to take. It should be one of `RoutingRequest.routes`; a
            name the router doesn't have is a fallback to the default route.
        reason: Why, in words, for a human reading a trace. It is recorded on the decision.
        messages_back: How many of the user's messages back the message that decided was: `0`
            for the current request, `1` for the one before it, and so on. It is recorded on
            the decision, so a dashboard can count follow-ups decided by an earlier message
            without reading the reason. `None`, the default, says nothing.

    Raises:
        RoutingError: if `messages_back` is neither `None` nor a non-negative integer.

    Example:
        ```python
        RoutingChoice(route="coder", reason="mentions a stack trace")
        RoutingChoice("legal", "matched 'contract' (1 message back)", messages_back=1)
        ```
    """

    route: str
    reason: str
    messages_back: int | None = None

    def __post_init__(self) -> None:
        """Refuse a `messages_back` that isn't a count.

        Checked here rather than by the router: a strategy that builds a bad choice raises
        inside `decide`, and the router falls back with the cause, as for any strategy error.
        """
        if self.messages_back is not None and not _is_count(self.messages_back):
            msg = (
                f"messages_back must be None or a non-negative integer, the number of the "
                f"user's messages back that decided, got {self.messages_back!r}"
            )
            raise RoutingError(msg)


class RoutingStrategy(ABC):
    """Applies the application's routing policy to one request.

    Subclass it and implement `decide`. The router calls it once per request, and not at all
    when the route is forced.

    Example:
        ```python
        class ByLength(RoutingStrategy):
            def decide(self, request: RoutingRequest) -> RoutingChoice | None:
                if len(request.text) > 2000:
                    return RoutingChoice("large", "long request")
                return None  # abstain: the default route answers
        ```
    """

    wants_full_context: ClassVar[bool] = False
    """Set to `True` to receive the whole transcript in `RoutingRequest.messages`."""

    lookback: int = 0
    """How many of the user's previous messages to read as well as the current one.

    The router hands up to that many to `decide` in `RoutingRequest.previous_requests`, newest
    first. `0`, the default, is the current request alone. An ordinary attribute rather than a
    class constant, so a strategy can take it from its constructor, as the built-in ones do; the
    router reads it on every request, before it builds the request. It must be a non-negative
    integer: the router refuses a strategy whose `lookback` isn't one when it is built, and
    answers from the default route, with a `FallbackWarning`, if it stops being one later."""

    def with_lookback(self: _Strategy, lookback: int) -> _Strategy:
        """A copy of this strategy that reads `lookback` of the user's previous messages.

        The strategy itself is left as it was. This is for a strategy that hands the request to
        another one. The router hands over as many previous messages as the strategy it was given
        asks for, and a built-in reads at most its own `lookback`, so a wrapper declares its
        inner strategy's `lookback`. To make the inner strategy read a different number, the
        wrapper gives it this copy.

        The copy is shallow: it shares everything else with the original. That suits a strategy
        that reads `self.lookback` when it decides, as the built-in ones do. A strategy that works
        something out from its `lookback` when it is built, or that wraps another strategy which
        should read the new number too, overrides this.

        Args:
            lookback: How many of the user's previous messages the copy reads.

        Returns:
            A copy of this strategy, with that `lookback`.

        Raises:
            RoutingError: if `lookback` isn't a non-negative integer.

        Example:
            ```python
            class Logged(RoutingStrategy):
                def __init__(self, inner, *, lookback=None):
                    # None keeps the inner strategy's lookback; a number overrides it.
                    self.inner = inner if lookback is None else inner.with_lookback(lookback)
                    self.lookback = self.inner.lookback

                def decide(self, request):
                    choice = self.inner.decide(request)
                    print(choice)
                    return choice
            ```
        """
        if not _is_count(lookback):
            msg = (
                f"lookback must be a non-negative integer, the number of the user's previous "
                f"messages to read, got {lookback!r}"
            )
            raise RoutingError(msg)
        clone = copy.copy(self)
        clone.lookback = lookback
        return clone

    @abstractmethod
    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Choose a route for `request`, or return `None` if this strategy can't decide.

        `None` sends the request to the default route, with a warning and the reason
        recorded — better than a guess.

        Args:
            request: The current request and the context the strategy may use.

        Returns:
            The route to take and why, or `None` to abstain.
        """

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Async `decide`. Defaults to `decide` in an executor, so async callers never block.

        `decide` therefore has to be thread-safe. Strategies that call models override this
        with a native async implementation, passing `request.config` to their calls.

        Args:
            request: The current request and the context the strategy may use.

        Returns:
            The route to take and why, or `None` to abstain.
        """
        # A config sends `run_in_executor` down the same path as no executor at all: the loop's
        # default one, running `decide` in a copy of the *current* context
        # (`runnables/config.py:705`). That copy — not the argument — is what carries the config
        # and tracing context the router sets with `set_config_context` into `decide`'s thread.
        # The config is passed because it is the one a strategy's calls take, and because
        # `Runnable.ainvoke` passes its own here too (`runnables/base.py:929`).
        return await run_in_executor(request.config, self.decide, request)


_Strategy = TypeVar("_Strategy", bound=RoutingStrategy)


def _is_count(value: object) -> bool:
    """Whether `value` can be a `lookback`: a non-negative integer.

    A `bool` is refused although Python counts it as an `int`: `True` is a flag where a count
    was meant.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


RoutingCallable: TypeAlias = Callable[[RoutingRequest], RoutingChoice | str | None]
"""A plain function accepted as `strategy=` and coerced to a `RoutingStrategy`.

It returns a `RoutingChoice`, a bare route name (the reason is then written for it), or `None`
to abstain. It must be synchronous, and it sees only the current request.

Example:
    ```python
    def pick(request: RoutingRequest) -> str | None:
        return "coder" if "def " in request.text else None

    router = ChatRouter(routes=routes, default_route="small", strategy=pick)
    ```
"""


class _CallableStrategy(RoutingStrategy):
    """A `RoutingCallable` wearing the `RoutingStrategy` interface.

    It keeps the defaults: the current request only, and `decide` in an executor for
    async callers.
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
        # router reports it like any other strategy failure.
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
    `functools.wraps` can't be told apart — that one surfaces on the first request.
    """
    return inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)


def as_strategy(strategy: RoutingStrategy | RoutingCallable) -> RoutingStrategy:
    """Coerce what `strategy=` accepts into a `RoutingStrategy`.

    Args:
        strategy: A strategy instance, or a synchronous function of a `RoutingRequest`.

    Returns:
        The strategy itself, or the function wrapped in the `RoutingStrategy` interface.

    Raises:
        TypeError: for anything else — including two near misses that would otherwise fail
            on every request rather than once, here: a strategy class passed without
            instantiating it, and an `async def` function, which `RoutingCallable`
            (synchronous) doesn't cover.
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
    """The name a decision record and the strategy's trace run carry.

    A strategy's class name, or a coerced function's own name.
    """
    if isinstance(strategy, _CallableStrategy):
        return strategy.name
    return type(strategy).__name__
