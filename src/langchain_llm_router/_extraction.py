"""Current-request extraction (R4, C7): the `RoutingRequest` a strategy sees.

Contract stub — T-112 owns the behaviour and its tests. The router calls `build_request` and
nothing else from here, so T-112 can change the internals freely.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langchain_llm_router.strategy import RoutingRequest


def build_request(
    messages: Sequence[BaseMessage],
    *,
    routes: tuple[str, ...],
    tools_bound: bool,
    wants_full_context: bool,
    config: RunnableConfig,
) -> RoutingRequest | None:
    """The request a strategy decides on, or `None` when there is no user message (REQ-R4-4).

    "Current request" is the most recent `HumanMessage`, ignoring trailing AI and tool
    messages (REQ-R4-1). `None` means the strategy can't decide, and the router falls back to
    the default route (R9).
    """
    current = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if current is None:
        return None
    blocks = current.content_blocks
    return RoutingRequest(
        text=current.text,
        content_blocks=list(blocks),
        modalities=frozenset(str(block.get("type")) for block in blocks),
        routes=routes,
        tools_bound=tools_bound,
        messages=list(messages) if wants_full_context else None,
        config=config,
    )
