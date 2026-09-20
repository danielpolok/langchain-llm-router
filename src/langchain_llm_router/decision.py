"""The routing decision record (R2, D8) and the ways to read it back (D3).

A caller has three ways to the record, in the order they are worth reaching for:
`routing_decision(response)` off the message, the trace (D9 places it three times), and
`last_routing_decision()` for the one path where nothing comes back that could carry it —
structured output without `include_raw=True`, which returns a parsed object.

`record_decision` is the router's own, not public API; everything `langchain_llm_router`
exports from here is.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

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


def routing_decision(message: BaseMessage) -> RoutingDecision | None:
    """Read the decision record back off a response, or `None` if it carries none (R2)."""
    record = message.response_metadata.get(ROUTING_KEY)
    if not isinstance(record, Mapping):
        return None
    return RoutingDecision.from_dict(record)


_published = threading.local()
"""Where a routed call leaves its record for `last_routing_decision()` — see `record_decision`
for why this is a thread's own storage and not the context variable D3 first named."""


def record_decision(decision: RoutingDecision) -> None:
    """Publish `decision` for `last_routing_decision()` (D3). The router's own, not public API.

    D3 named a context variable, and a `ContextVar` would be the exact answer — per context, so
    per call and per task. It cannot be the answer here: LangChain runs a sequence's steps in a
    *copy* of the caller's context (`RunnableSequence.invoke` does
    `context.run(step.invoke, …)`, `runnables/base.py:3454`; `ainvoke` awaits a task carrying
    the copy, `:3496`), and a `ContextVar.set` inside a copy is discarded when the step
    returns. That copy is exactly the path D3 exists for: `with_structured_output` builds
    `llm | output_parser` (`chat_models.py:2565`), so the router runs as a step and the caller
    reads from outside it, where the record would never arrive.

    A thread's own storage is the finest-grained place that survives a copied context, and it
    is where the record goes. What that costs is in `last_routing_decision`'s own docstring.
    """
    _published.decision = decision


def last_routing_decision() -> RoutingDecision | None:
    """The most recent routed call on this thread (D3), or `None` if there has been none.

    The escape hatch for `with_structured_output` without `include_raw=True`, where the parsed
    object has nowhere to carry a record. It answers for:

    - a routed call this thread made itself — `router.invoke(…)`, `await router.ainvoke(…)`, or
      a stream whose first chunk has arrived (the route is chosen before it, D8);
    - a routed call one or more LangChain steps deep, which is the parsed-only structured-output
      path: `router.with_structured_output(Schema).invoke(…)`.

    What it cannot do is tell two routed calls on one thread apart, because a record that
    survives a copied context can be no finer-grained than the thread the call ran on:

    - calls that overlap on one thread — `abatch`, `asyncio.gather` over several routed calls —
      overwrite each other, and the last to return answers for all of them;
    - `batch` over more than one input runs each input in a worker thread, and so does a step
      LangChain evaluates in parallel — `RunnableMap(raw=llm)`, which is how `include_raw=True`
      calls the model (`chat_models.py:2564`). Those records stay on their own threads, and the
      caller reads back whatever it last routed itself, which may be an older call's.

    So this is for one call at a time. Read the record off the response with `routing_decision`,
    or ask for `include_raw=True` and read it off the raw message, whenever calls can overlap.
    """
    decision: RoutingDecision | None = getattr(_published, "decision", None)
    return decision
