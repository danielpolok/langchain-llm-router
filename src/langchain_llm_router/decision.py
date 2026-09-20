"""The routing decision record (R2, D8) and the ways to read it back (D3).

A caller has three ways to the record, in the order they are worth reaching for:
`routing_decision(response)` off the message, the trace (D9 places it three times), and
`last_routing_decision()` for the one path where nothing comes back that could carry it —
structured output without `include_raw=True`, which returns a parsed object.

`record_decision` is the router's own, not public API; everything `langchain_llm_router`
exports from here is.
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
"""Where the record rides: `response_metadata["routing"]`, and the same key on the trace (D9)."""


@dataclass(frozen=True)
class RoutingDecision:
    """Which route ran, and why (R2). The six fields are D8's schema."""

    route: str
    """The route that ran."""

    reason: str
    """Why, in words, for a human reading a trace."""

    strategy: str | None = None
    """The strategy's name, or `None` when no strategy ran (no strategy, or a forced route)."""

    fallback: bool = False
    """The R9 path was taken: the strategy could not decide, so the default route ran."""

    forced: bool = False
    """The route came from runtime config (R11)."""

    diverted_from: str | None = None
    """The tool-incapable route the request was diverted from (R10)."""

    def as_dict(self) -> dict[str, Any]:
        """The record as it rides under `response_metadata["routing"]` and on the trace."""
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> RoutingDecision:
        """Rebuild a record from `as_dict()` output. Unknown keys are ignored."""
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in record.items() if key in names})


_REQUIRED = frozenset(field.name for field in fields(RoutingDecision) if field.default is MISSING)
"""What a mapping must hold to be one of ours: the fields a `RoutingDecision` has no default
for. `"routing"` is a plain key on `response_metadata`, so a provider is free to put its own
meaning there, and reading one of those must give a caller `None` rather than a `TypeError`."""


def routing_decision(message: BaseMessage) -> RoutingDecision | None:
    """Read the decision record back off a response, or `None` if it carries none (R2)."""
    record = message.response_metadata.get(ROUTING_KEY)
    if not isinstance(record, Mapping) or not record.keys() >= _REQUIRED:
        return None
    return RoutingDecision.from_dict(record)


class _Published(NamedTuple):
    """One routed call's record, and where it sits among the others (D3)."""

    order: int
    """When it was published. Higher is later; `_order` hands them out."""

    inherited: int
    """The `order` of the record this call found in its context when it published, or 0.

    How a call that ran *inside* another's context is told from one that ran beside it: a
    sequence step runs in a copy of the caller's context and so inherits the caller's record,
    while a concurrent sibling inherits only what they both started from."""

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
    """Publish `decision` for `last_routing_decision()` (D3). The router's own, not public API.

    Published in both places because neither alone answers every caller:

    - the **context variable** D3 names is the exact one — per call, per task — and it is what a
      caller reads after `router.invoke(…)` or an awaited `ainvoke`, including two of those
      running concurrently, which each have a context of their own;
    - but LangChain runs a sequence's steps in a *copy* of the caller's context
      (`RunnableSequence.invoke` does `context.run(step.invoke, …)`, `runnables/base.py:3454`;
      `ainvoke` awaits a task carrying the copy, `:3496`), and a `ContextVar.set` inside a copy
      is discarded when the step returns. That copy is exactly the path D3 exists for:
      `with_structured_output` builds `llm | output_parser` (`chat_models.py:2565`), so the
      router runs as a step while the caller reads from outside it. The **thread's** own
      storage is the finest-grained place a record can survive that, so it goes there too.
    """
    _publish(decision)


def discard_decision() -> None:
    """Withdraw this call's record: it failed, so it has no decision to report (C6).

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
    """The most recent routed call's decision as seen from here (D3), or `None` if none was.

    The escape hatch for `with_structured_output` without `include_raw=True`, where the parsed
    object has nowhere to carry a record. It answers for:

    - a routed call made in this context — `router.invoke(…)`, `await router.ainvoke(…)`, or a
      stream whose first chunk has arrived (the route is chosen before it, D8). Concurrent
      calls each have their own context, so each reads its own;
    - a routed call one or more LangChain steps deep, whose record reached only this thread:
      the parsed-only structured-output path, `router.with_structured_output(S).invoke(…)`.
      A call made inside this context supersedes the record this context already had, which is
      what `_Published.inherited` is for.

    What it cannot do is tell *those* calls apart when they overlap, because a record that
    survives a copied context can be no finer-grained than the thread it ran on:

    - two parsed-only calls overlapping on one thread overwrite each other, and the last to
      return answers for both;
    - `batch` over more than one input runs each input in a worker thread, and so does a step
      LangChain evaluates in parallel — `RunnableMap(raw=llm)`, which is how `include_raw=True`
      calls the model (`chat_models.py:2564`). Those records stay on their own threads, and the
      caller reads back the last call it made itself, which may be an older one.

    Read the record off the response with `routing_decision`, or ask for `include_raw=True` and
    read it off the raw message, whenever calls like those can overlap.
    """
    own = _in_context.get()
    on_thread: _Published | None = getattr(_on_thread, "published", None)
    if own is None:
        return on_thread.decision if on_thread is not None else None
    if on_thread is not None and on_thread.inherited >= own.order:
        return on_thread.decision  # published by a call that ran inside this context
    return own.decision
