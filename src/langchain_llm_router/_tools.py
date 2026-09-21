"""What the router keeps until a route is chosen, and how it tells which route can use it.

Two jobs, both forced by the same fact: tools are provider-specific, and which provider will
answer is not known at bind time (C3, R10).

**Binding is kept, not applied (REQ-C3-1).** `bind_tools` and `with_structured_output` store
what they were given in a `ToolBinding` / `StructuredOutputBinding`, which rides to the route's
call as a kwarg under the keys below and is replayed there by `bound_route` — on the *route's*
own binder, so the provider does its own conversion, and `tool_choice` and binding kwargs such
as `strict=` take effect where they belong: at bind time, changing how the schema is built,
rather than as call kwargs the provider never reads.

**Capability is detected, not discovered (D5, REQ-R10-1).** `BaseChatModel.bind_tools` raises
`NotImplementedError` at *call* time (`chat_models.py:2383`), so a route that can't use tools
fails only once it is selected — the late failure R10 exists to pre-empt. `supports_tools`
answers before that, from the route's `profile` when it reports one, otherwise from whether its
class overrides `bind_tools`, and from `tool_support_overrides` ahead of both.

Nothing here warns or raises: what the router does with an incapable route — the bind-time
notice (REQ-R10-2) and the per-request diversion (D1, REQ-R10-3) — is `router.py`'s, next to
the decision record it goes into.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_llm_router.router import ChatRouter

__all__ = [
    "STRUCTURED_OUTPUT_KEY",
    "TOOL_BINDING_KEY",
    "StructuredOutputBinding",
    "StructuredRouter",
    "ToolBinding",
    "bound_route",
    "supports_tools",
    "tools_bound",
]

TOOL_BINDING_KEY = "__llm_router_tool_binding"
"""Call kwarg carrying a `ToolBinding` from `bind_tools` to the route that will convert it."""

STRUCTURED_OUTPUT_KEY = "__llm_router_structured_output"
"""Call kwarg carrying a `StructuredOutputBinding` from `with_structured_output`."""

StructuredOutput: TypeAlias = dict[str, Any] | BaseModel
"""What a structured-output runnable answers with, as LangChain types it."""


@dataclass(frozen=True)
class ToolBinding:
    """What `bind_tools` was called with, kept as given until a route can convert it (C3)."""

    tools: tuple[Any, ...]
    """The tools exactly as the caller passed them — dicts, classes, functions or `BaseTool`s.

    Untouched: converting them here would mean picking one provider's form for all of them."""

    tool_choice: Any = None
    """Replayed on the route's binder, not passed as a call kwarg. `None` is not passed on at
    all, so a route whose `bind_tools` doesn't take it still works."""

    kwargs: Mapping[str, Any] = field(default_factory=dict)
    """The rest of `bind_tools`' keyword arguments, `strict=` among them — they change how the
    route builds the tool schema, which only its own `bind_tools` can do."""

    def apply(self, route: BaseChatModel) -> Runnable[LanguageModelInput, AIMessage]:
        """The route with these tools bound, converted by the route itself (REQ-C3-1)."""
        if self.tool_choice is None:
            return route.bind_tools(list(self.tools), **self.kwargs)
        return route.bind_tools(list(self.tools), tool_choice=self.tool_choice, **self.kwargs)


@dataclass(frozen=True)
class StructuredOutputBinding:
    """What `with_structured_output` was called with, forwarded per request (REQ-C3-3)."""

    schema: dict[str, Any] | type
    include_raw: bool = False
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    """`method=` and `strict=` among them: the base implementation drops both
    (`chat_models.py:2530`), and forwarding is what gets them to the route that reads them."""

    def apply(self, route: BaseChatModel) -> Runnable[LanguageModelInput, StructuredOutput]:
        """The route's own structured output for this schema, its own way (REQ-C3-3)."""
        return route.with_structured_output(
            self.schema, include_raw=self.include_raw, **self.kwargs
        )


def supports_tools(route: BaseChatModel, *, override: bool | None = None) -> bool:
    """Whether `route` can use tools, by D5's three signals in order (REQ-R10-1).

    `override` is the route's entry in `tool_support_overrides` and always wins — it is the
    only signal an application controls, and the escape hatch for a route the other two read
    wrongly. Then the route's `profile`, LangChain's own statement of what a model can do,
    when it reports one with a `tool_calling` key: `profile` is beta and may be absent or
    partial, so it can't be the only signal. Otherwise, whether the route's class overrides
    `bind_tools` at all — the base one exists only to raise (`chat_models.py:2383`).
    """
    if override is not None:
        return override
    profile = route.profile
    if profile is not None and "tool_calling" in profile:
        return bool(profile["tool_calling"])
    return type(route).bind_tools is not BaseChatModel.bind_tools


def tools_bound(kwargs: Mapping[str, Any]) -> bool:
    """Whether this call has tools or structured output bound (R10, `RoutingRequest`).

    What a strategy routes on, and what decides whether a tool-incapable route may answer. The
    two bindings above account for `bind_tools` and `with_structured_output`; a bare `tools`
    kwarg counts too, for a caller who reached for `bind(tools=...)` — or any other route into
    the place LangChain expects tools to be.
    """
    return (
        TOOL_BINDING_KEY in kwargs
        or STRUCTURED_OUTPUT_KEY in kwargs
        or bool(kwargs.get("tools"))
    )


def bound_route(
    route: BaseChatModel, kwargs: dict[str, Any]
) -> tuple[Runnable[LanguageModelInput, AIMessage], dict[str, Any]]:
    """The runnable the selected route's call goes to, and the kwargs left for that call.

    Replaying a binding here is what keeps the route used as given (REQ-R5-1): `bind_tools`
    returns a new runnable over the route, and the route itself is never touched.

    The raw `tools` list `bind_tools` also bound (REQ-C3-2) is dropped from the call kwargs
    when a binding is replayed: the route's own binder puts its converted form back in that
    same place, and the caller's unconverted list would otherwise reach the provider too.
    """
    call_kwargs = dict(kwargs)
    tool_binding = call_kwargs.pop(TOOL_BINDING_KEY, None)
    structured = call_kwargs.pop(STRUCTURED_OUTPUT_KEY, None)
    if structured is not None:
        call_kwargs.pop("tools", None)
        # A structured-output runnable answers with the parsed object rather than a message.
        # The router's entry points are typed for a chat model's answer; `_with_record` puts
        # the record wherever the answer can hold one, and `with_structured_output` casts
        # back to what it promised the caller (D3).
        target = cast(
            "Runnable[LanguageModelInput, AIMessage]",
            cast("StructuredOutputBinding", structured).apply(route),
        )
        return target, call_kwargs
    if tool_binding is not None:
        call_kwargs.pop("tools", None)
        return cast("ToolBinding", tool_binding).apply(route), call_kwargs
    return route, call_kwargs


class StructuredRouter(Runnable[LanguageModelInput, Any]):
    """What `ChatRouter.with_structured_output` hands back: route first, then parse (REQ-C3-3).

    A `Runnable` of its own rather than `llm | parser` — the shape LangChain builds
    (`chat_models.py:2565`) — because the parser here is the *route's*, and the router's own
    pipeline has to be the one that runs: one chain run, one strategy run, one decision, and a
    record published where `last_routing_decision()` reads it (D3, REQ-R2-4). It adds no run of
    its own to the trace; the router's chain run stays the root.

    Streaming it yields the finished object once, from `Runnable.stream`'s default. Partial
    parses belong to the route's parser, and reaching them would mean streaming through a
    sequence the router sits inside — the arrangement D3 exists to work around.
    """

    def __init__(self, router: ChatRouter, binding: StructuredOutputBinding) -> None:
        self.router = router
        self.binding = binding

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> StructuredOutput:
        """Route this request and answer with the selected route's parsed output."""
        return cast(
            "StructuredOutput", self.router.invoke(input, config, **self._call_kwargs(kwargs))
        )

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> StructuredOutput:
        """Async `invoke`, so an async caller never falls back to a worker thread (C2)."""
        return cast(
            "StructuredOutput",
            await self.router.ainvoke(input, config, **self._call_kwargs(kwargs)),
        )

    def _call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The caller's kwargs, with the binding riding along to `bound_route`."""
        return {STRUCTURED_OUTPUT_KEY: self.binding, **kwargs}
