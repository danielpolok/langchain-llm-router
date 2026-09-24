"""Warnings and errors the router raises.

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
    """Base of every routing warning — filter on this to see them all.

    Example:
        ```python
        import warnings

        warnings.simplefilter("error", RoutingWarning)  # turn every routing warning into an error
        ```
    """


class FallbackWarning(RoutingWarning):
    """The strategy failed, abstained or named an unknown route; the default route ran.

    Example:
        ```python
        import warnings

        warnings.filterwarnings("ignore", category=FallbackWarning)
        ```
    """


class ToolSupportWarning(RoutingWarning):
    """Routes that can't use the bound tools, at bind time and on each diverted request.

    Example:
        ```python
        import warnings

        warnings.filterwarnings("ignore", category=ToolSupportWarning)
        ```
    """


class ForcedRouteWarning(RoutingWarning):
    """A forced route gave way — only under `on_unavailable_forced_route="fallback"`.

    Example:
        ```python
        import warnings

        warnings.filterwarnings("ignore", category=ForcedRouteWarning)
        ```
    """


class RoutingError(ValueError):
    """Base of configuration and forced-route failures.

    Example:
        ```python
        try:
            router = ChatRouter(routes={"small": small}, default_route="large")
        except ValueError as error:  # RoutingError is a ValueError
            print(error)
        ```
    """


class NoToolCapableRouteError(RoutingError):
    """Tools or structured output were bound, and no route can use them.

    Raised at bind time by `bind_tools` / `with_structured_output`, and on a request that
    reaches a nested `ChatRouter` none of whose routes can use them.

    Example:
        ```python
        try:
            router.bind_tools([my_tool])
        except NoToolCapableRouteError as error:
            print(error)
        ```
    """


class ForcedRouteError(RoutingError):
    """The forced route doesn't exist, or can't use the bound tools.

    Raised at call time, and only under the default `on_unavailable_forced_route="error"`.

    Example:
        ```python
        try:
            router.invoke("hi", config={"configurable": {"route": "nope"}})
        except ForcedRouteError as error:
            print(error)
        ```
    """
