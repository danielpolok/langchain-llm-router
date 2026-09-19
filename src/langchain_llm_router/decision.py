"""The routing decision record (R2, D8) and the ways to read it back (D3)."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
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


_last_decision: ContextVar[RoutingDecision | None] = ContextVar(
    "langchain_llm_router_last_decision", default=None
)


def last_routing_decision() -> RoutingDecision | None:
    """The decision of the most recent routed call in this context (D3).

    The escape hatch for `with_structured_output` without `include_raw=True`, where the parsed
    object has nowhere to carry a record.
    """
    return _last_decision.get()
