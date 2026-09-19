"""Warnings and errors the router raises (R9, R10, R11).

Every routing warning subclasses `RoutingWarning`, so one filter sees them all. `RoutingError`
subclasses `ValueError`: a construction-time failure raised inside a pydantic validator surfaces
as `pydantic.ValidationError` (pydantic wraps `ValueError`), while a call-time failure raises the
`RoutingError` subclass itself.
"""

from __future__ import annotations

__all__ = [
    "FallbackWarning",
    "ForcedRouteError",
    "ForcedRouteWarning",
    "NoToolCapableRouteError",
    "RoutingError",
    "RoutingWarning",
    "ToolSupportWarning",
]


class RoutingWarning(UserWarning):
    """Base of every routing warning — filter on this to see them all."""


class FallbackWarning(RoutingWarning):
    """R9: the strategy failed, abstained or named an unknown route; the default route ran."""


class ToolSupportWarning(RoutingWarning):
    """R10: routes that can't use the bound tools, at bind time and on each diverted request."""


class ForcedRouteWarning(RoutingWarning):
    """R11: a forced route gave way — only under `on_unavailable_forced_route="fallback"`."""


class RoutingError(ValueError):
    """Base of configuration and forced-route failures."""


class NoToolCapableRouteError(RoutingError):
    """R10: tools or structured output were bound, and no route can use them."""


class ForcedRouteError(RoutingError):
    """R11: the forced route doesn't exist, or can't use the bound tools."""
