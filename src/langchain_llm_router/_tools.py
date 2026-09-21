"""What the router keeps until a route is chosen, and how it tells which route can use it.

Two jobs, both forced by the same fact: tools are provider-specific, and which provider will
answer is not known at bind time (C3, R10).

**Binding is kept, not applied (REQ-C3-1, REQ-C3-3).** `bind_tools` and `with_structured_output`
store what they were given in a `ToolBinding` / `StructuredOutputBinding`. It rides to the
router's entry points as a call kwarg under `BINDING_KEY` and is replayed there, by
`bound_route`, on the *selected route's own* binder: the provider does its own conversion, and
`tool_choice` and binding kwargs such as `strict=` take effect where they belong — at bind time,
changing how the schema is built — rather than as call kwargs a provider never reads.

**Capability is detected, not discovered (D5, REQ-R10-1).** `BaseChatModel.bind_tools` raises
`NotImplementedError` at *call* time (`chat_models.py:2383`), so a route that can't use tools
fails only once it is selected — the late failure R10 exists to pre-empt. `supports_tools`
answers before that, from `tool_support_overrides` first, then the route's `profile`, then
whether its class overrides `bind_tools`.

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
    "BINDING_KEY",
    "StructuredOutput",
    "StructuredOutputBinding",
    "StructuredRouter",
    "ToolBinding",
    "bound_route",
    "supports_tools",
    "tools_are_bound",
]

BINDING_KEY = "__llm_router_binding"
"""The call kwarg that carries a `ToolBinding` or `StructuredOutputBinding` to the router.

An ordinary key in `RunnableBinding.kwargs`, so it survives `bind`, `with_config` and every
other wrapper LangChain puts around a runnable — and is spelled so that no provider argument
can collide with it."""

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

    def apply(self, route: BaseChatModel) -> Runnable[LanguageModelInput, Any]:
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

    def apply(self, route: BaseChatModel) -> Runnable[LanguageModelInput, Any]:
        """The route's own structured output for this schema, built its own way (REQ-C3-3)."""
        return route.with_structured_output(
            self.schema, include_raw=self.include_raw, **self.kwargs
        )


def supports_tools(route: BaseChatModel, *, override: bool | None = None) -> bool:
    """Whether `route` can use tools, by D5's signals in order of authority (REQ-R10-1).

    1. `override`, the route's entry in `tool_support_overrides`: always wins. It is the only
       signal the application controls, and the escape hatch for a route the others read wrong.
    2. The route's `profile`, LangChain's own statement of what a model can do, when it reports
       one with a `tool_calling` key. `profile` is beta and `total=False`, so a profile with no
       such key says nothing and the next signal decides, rather than reading as "no".
    3. Whether the route's class overrides `bind_tools` at all: the base one exists only to
       raise (`chat_models.py:2383`).
    """
    if override is not None:
        return override
    profile = route.profile
    if profile is not None and "tool_calling" in profile:
        return bool(profile["tool_calling"])
    return type(route).bind_tools is not BaseChatModel.bind_tools


def tools_are_bound(kwargs: Mapping[str, Any]) -> bool:
    """Whether this call has tools or structured output bound (R10, `RoutingRequest`).

    What a strategy routes on, and what decides whether a tool-incapable route may answer.
    `bind_tools` and `with_structured_output` leave a binding; a bare `tools` kwarg counts
    too, for a caller who reached for `bind(tools=...)` — the place LangChain itself looks.
    """
    return BINDING_KEY in kwargs or bool(kwargs.get("tools"))


def bound_route(
    route: BaseChatModel, kwargs: dict[str, Any]
) -> tuple[Runnable[LanguageModelInput, AIMessage], dict[str, Any]]:
    """The runnable the selected route's call goes to, and the kwargs left for that call.

    A binding is replayed on the route's own binder (REQ-C3-1, REQ-C3-3): the route is never
    touched (REQ-R5-1) — `bind_tools` returns a new runnable over it.

    The raw `tools` list the router also bound in its usual place (REQ-C3-2) is dropped from
    the call kwargs when a binding is replayed: the route's binder puts its own, converted
    form in that same place, and the caller's unconverted list would otherwise override it.

    Typed as the answer of a chat model, because the router's entry points are. A
    `StructuredOutputBinding` makes it answer with the parsed output instead — only
    `StructuredRouter`, which asked for that, reads it so — and `_with_record` in `router.py`
    tells the two apart.
    """
    call_kwargs = dict(kwargs)
    binding = cast(
        "ToolBinding | StructuredOutputBinding | None", call_kwargs.pop(BINDING_KEY, None)
    )
    if binding is None:
        return route, call_kwargs
    call_kwargs.pop("tools", None)
    return cast("Runnable[LanguageModelInput, AIMessage]", binding.apply(route)), call_kwargs


class StructuredRouter(Runnable[LanguageModelInput, Any]):
    """What `ChatRouter.with_structured_output` hands back: route first, then parse (REQ-C3-3).

    A `Runnable` of its own rather than `llm | parser` — the shape LangChain builds
    (`chat_models.py:2565`) — because the parser is the *route's*, and the router's own
    pipeline has to be the one that runs: one chain run, one strategy run, one decision, and a
    record published for `last_routing_decision()` (D3, REQ-R2-4). It opens no run of its own;
    the router's chain run stays the root of the trace.

    It answers once: `stream` and `astream` are `Runnable`'s defaults and yield the finished
    object as their single item. Progressive partial parses belong to the route's own parser,
    and reaching them would mean streaming *through* the router — whose chunks are merged
    with `+`, which a parsed object doesn't support.
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
            "StructuredOutput",
            self.router.invoke(input, config, **{BINDING_KEY: self.binding, **kwargs}),
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
            await self.router.ainvoke(input, config, **{BINDING_KEY: self.binding, **kwargs}),
        )
