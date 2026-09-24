"""The routing decision record and the ways to read it back.

A caller has three ways to the record, in the order they are worth reaching for:
`routing_decision(response)` off the message, the trace (where it is placed three times), and
`last_routing_decision()` for the one path where nothing comes back that could carry it —
structured output without `include_raw=True`, which returns a parsed object.

`record_decision` and `discard_decision` are the router's own, not public API; everything
`langchain_llm_router` exports from here is.

A route may itself be a `ChatRouter`. The inner router records its decision on its answer, and
the outer router then replaces it with its own, so the message and `last_routing_decision()`
both report the *outermost* decision — the one the caller made. Nothing is lost: the inner
router's chain run carries its own record on the trace, as every router's run does.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, NamedTuple

from langchain_core.messages import BaseMessage

__all__ = ["ROUTING_KEY", "RoutingDecision", "last_routing_decision", "routing_decision"]

ROUTING_KEY = "routing"
"""Where the record rides: `response_metadata["routing"]`, and the same key on the trace."""


@dataclass(frozen=True)
class RoutingDecision:
    """Which route ran, and why. These six fields are the whole schema.

    Attributes:
        route: The route that ran.
        reason: Why, in words, for a human reading a trace.
        strategy: The strategy's name, or `None` when no strategy ran.
        fallback: Whether the default route ran because the strategy could not decide.
        forced: Whether the route came from runtime config.
        diverted_from: The tool-incapable route the request was diverted from, if any.

    Example:
        ```python
        response = router.invoke("Write a regex for ISO dates")
        decision = routing_decision(response)
        print(decision.route, decision.reason)
        ```
    """

    route: str
    """The route that ran."""

    reason: str
    """Why, in words, for a human reading a trace."""

    strategy: str | None = None
    """The strategy's name, or `None` when no strategy ran (no strategy, or a forced route)."""

    fallback: bool = False
    """The default-route fallback was taken: the strategy could not decide, so the default route
    ran."""

    forced: bool = False
    """The route came from runtime config."""

    diverted_from: str | None = None
    """The tool-incapable route the request was diverted from."""

    def as_dict(self) -> dict[str, Any]:
        """The record as it rides under `response_metadata["routing"]` and on the trace.

        Returns:
            A plain dict of the six fields, safe to serialize.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> RoutingDecision:
        """Rebuild a record from `as_dict()` output. Unknown keys are ignored.

        Args:
            record: A mapping holding at least `route` and `reason`.

        Returns:
            The rebuilt record.
        """
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in record.items() if key in names})


_REQUIRED_TEXT = tuple(field.name for field in fields(RoutingDecision) if field.default is MISSING)
"""What a mapping must hold to be one of ours, and hold as text: `route` and `reason`, the two
fields a `RoutingDecision` has no default for.

`"routing"` is a plain key on `response_metadata`, so another library is free to give it its own
meaning. Reading one of those gives the caller `None`, rather than a `TypeError` from inside
this package or a record whose `route` is someone else's integer. The fields with defaults are
taken as they come: past those two, what is under this key is a record we wrote."""


def routing_decision(message: BaseMessage) -> RoutingDecision | None:
    """Read the decision record back off a response, or `None` if it carries none.

    Args:
        message: A response from a `ChatRouter`.

    Returns:
        The decision it carries, or `None` if the message has no routing record.

    Example:
        ```python
        decision = routing_decision(router.invoke("Hello"))
        ```
    """
    record = message.response_metadata.get(ROUTING_KEY)
    if not isinstance(record, Mapping):
        return None
    if not all(isinstance(record.get(key), str) for key in _REQUIRED_TEXT):
        return None
    return RoutingDecision.from_dict(record)


class _Published(NamedTuple):
    """One routed call's record, and where it sits among the others."""

    order: int
    """When it was published. Higher is later; `_order` hands them out."""

    inherited: int
    """The `order` of the record this call found in its context when it published, or 0.

    What lets a call that ran *inside* a context supersede the record already there: a sequence
    step runs in a copy of the caller's context and inherits that record, while a call that
    started before it did inherits something older. It is not proof of nesting — a task started
    from this context after my call inherits exactly my record, and an async sequence step *is*
    such a task, so the two cannot be told apart. `last_routing_decision` says which way that
    goes."""

    decision: RoutingDecision | None
    """`None` withdraws the record: a routed call that failed has none, and nothing older
    stands in for it."""


_order = itertools.count(1)
"""Stamps each record. `next()` on an `itertools.count` is atomic, and two records sharing a
stamp would decide nothing this docstring does not already leave undefined."""

_in_context: ContextVar[_Published | None] = ContextVar(
    "langchain_llm_router_last_decision", default=None
)
"""The record for the context that routed — per call and per task, and discarded with a copy."""

_on_thread = threading.local()
"""The record for the thread that routed: what a copied context cannot discard."""


def record_decision(decision: RoutingDecision) -> None:
    """Publish `decision` for `last_routing_decision()`. The router's own, not public API.

    Published in both places because neither alone answers every caller:

    - the **context variable** is the exact one — per call, per task — and it is what a
      caller reads after `router.invoke(…)` or an awaited `ainvoke`, including two of those
      running concurrently, which each have a context of their own;
    - but LangChain runs a sequence's steps in a *copy* of the caller's context
      (`RunnableSequence.invoke` does `context.run(step.invoke, …)`, `runnables/base.py:3454`;
      `ainvoke` awaits a task carrying the copy, `:3496`), and a `ContextVar.set` inside a copy
      is discarded when the step returns. That copy is exactly the path this second placement exists
      for: `with_structured_output` builds `llm | output_parser` (`chat_models.py:2565`), so the
      router runs as a step while the caller reads from outside it. The **thread's** own
      storage is the finest-grained place a record can survive that, so it goes there too.
    """
    _publish(decision)


def discard_decision() -> None:
    """Withdraw this call's record: it failed, so it has no decision to report.

    Also the router's own. A failed call publishes nothing *and* supersedes what came before
    it, so an earlier call's record is never read back as the failed call's — which matters
    most under `with_fallbacks` and `with_retry`, where the answer the caller ends up holding
    did not come from the routed call at all.
    """
    _publish(None)


def _publish(decision: RoutingDecision | None) -> None:
    inherited = _in_context.get()
    published = _Published(next(_order), inherited.order if inherited else 0, decision)
    _in_context.set(published)
    _on_thread.published = published


def last_routing_decision() -> RoutingDecision | None:
    """The most recent routed call's decision as seen from here, or `None` if none was.

    The escape hatch for `with_structured_output` without `include_raw=True`, where the parsed
    object has nowhere to carry a record. It answers for:

    - a routed call made in this context — `router.invoke(…)`, `await router.ainvoke(…)`, or a
      stream whose first chunk has arrived (the route is chosen before it). Two calls
      started side by side, as `asyncio.gather` starts them, each have a context of their own
      and each read their own;
    - a routed call one or more LangChain steps deep, whose record reached only this thread:
      the parsed-only structured-output path, `router.with_structured_output(S).invoke(…)`. A
      call made *from* this context supersedes the record this context already had, which is
      what `_Published.inherited` is for.

    That last rule is what it cannot do precisely. A routed call started from this context
    while mine is in flight, or after it, inherits my record exactly as a nested step does, and
    will answer in its place — an `asyncio.gather` issued *after* my own call, a background
    task, a callback. There is no local way to tell those from the sequence step this exists
    for, and the sequence step has to win.

    Nor is it per call: the record outlives the call that made it, in the context *and* on the
    thread. So a task that runs later on a pooled thread, or the next request on a server's
    worker thread, reads back a neighbour's decision when it has made no routed call of its
    own. And when calls share one thread:

    - two parsed-only calls overlapping there overwrite each other, and the last to return
      answers for both;
    - `batch` over more than one input runs each input in a worker thread, as does a step
      LangChain evaluates in parallel — `RunnableMap(raw=llm)`, which is how `include_raw=True`
      calls the model (`chat_models.py:2564`). Those records stay on their own threads, so the
      caller reads back the last call it made itself, which may be an older one;
    - `abatch` instead runs its inputs as tasks on the caller's own thread, so the caller reads
      back one of the batch's records — whichever returned last — and not its own earlier call.

    Read the record off the response with `routing_decision`, or ask for `include_raw=True` and
    read it off the raw message, whenever calls can overlap like that. This is for one call at
    a time, and it is exact there.

    Returns:
        The decision, or `None` if no routed call is visible from here.

    Example:
        ```python
        parsed = router.with_structured_output(Answer).invoke("Q?")
        decision = last_routing_decision()
        ```
    """
    own = _in_context.get()
    on_thread: _Published | None = getattr(_on_thread, "published", None)
    if own is None:
        return on_thread.decision if on_thread is not None else None
    if on_thread is not None and on_thread.inherited >= own.order:
        return on_thread.decision  # published by a call that ran inside this context
    return own.decision
