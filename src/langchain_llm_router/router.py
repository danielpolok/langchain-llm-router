"""`ChatRouter`: a chat model that hands each request to one of its named routes.

The router's own run is a *chain* run that delegates: it overrides the public entry
points, not `_generate` / `_stream`, so the selected route's call is the only model run outside
the strategy run — cost is counted once and the real call stays in the trace.

Every entry point runs the same pipeline, in this order:

1. `_start_run` opens the router's chain run.
2. `_decide` / `_adecide` settle the route, diversion included. `_plan` says whether there is
   anything to run: a route forced via runtime config (`configurable["route"]`) or no
   strategy at all each settle the decision outright, with no strategy run — and
   otherwise the strategy runs in a child chain run of its own, with that run's child config as
   `RoutingRequest.config` and set as the context config. `_conclude` turns what the strategy
   did into a `RoutingDecision`, falling back to the default route when it can't;
   `_forced_decision` does the same for a forced route that can't be used, with
   `ForcedRouteError` or `ForcedRouteWarning` standing in for `FallbackWarning`. `_divert`
   then applies tool-aware routing to whatever settled the decision: a request whose route
   can't use the tools bound to it goes to the default route instead (or, if that can't use
   them either, the first route in declaration order that can), recorded as a diversion —
   before the strategy run closes, so its output is the same record the router run and the
   route run end up with.
3. `_route_call` prepares the selected route's call: nested under the router's run, with the
   decision in its metadata, and the caller's tool or structured-output binding replayed on
   the route itself.
4. `_with_record` adds the decision to the response (on exactly one chunk when streaming)
   and publishes it for `last_routing_decision()`, and the router's run closes with the
   record in its outputs. A route's error closes it as an error and propagates unchanged,
   and withdraws the record: a call that failed has no decision to report. A caller that
   abandons a stream is not a failure, and keeps its record.

`batch`, `abatch` and `astream_events` need nothing of their own: `Runnable` builds them on the
four entry points above. `generate` and `agenerate` — and `generate_prompt` and
`agenerate_prompt`, which call them — are built on `invoke` and `ainvoke`, one prompt at a time.
The one door that does not lead through them is the v3 streaming protocol, which is
refused — `_V3_UNSUPPORTED` says why.
"""

from __future__ import annotations

import asyncio
import sys
import types
import uuid
import warnings
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, TypeVar, cast

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput, ModelProfile
from langchain_core.language_models.base import LangSmithParams
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult, RunInfo
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config, patch_config
from langchain_core.runnables.config import set_config_context
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import ConfigurableFieldSpec, coro_with_context
from langchain_core.tools import BaseTool
from pydantic import Field, field_validator, model_validator

from langchain_llm_router._extraction import build_request
from langchain_llm_router._profile import resolve_profile
from langchain_llm_router._tools import (
    BINDING_KEY,
    StructuredOutput,
    StructuredOutputBinding,
    StructuredRouter,
    ToolBinding,
    bound_route,
    supports_tools,
    tools_are_bound,
)
from langchain_llm_router.decision import (
    ROUTING_KEY,
    RoutingDecision,
    discard_decision,
    record_decision,
)
from langchain_llm_router.errors import (
    FallbackWarning,
    ForcedRouteError,
    ForcedRouteWarning,
    NoToolCapableRouteError,
    RoutingError,
    ToolSupportWarning,
)
from langchain_llm_router.strategy import (
    RoutingCallable,
    RoutingChoice,
    RoutingRequest,
    RoutingStrategy,
    as_strategy,
    strategy_name,
)

__all__ = ["ChatRouter"]

AnswerT = TypeVar("AnswerT")
"""What a routed call answers with: a message, or a structured-output payload."""

_LIBRARY_MODULES = ("langchain_llm_router", "langchain_core", "langchain", "langgraph")
"""Module-name prefixes `_stacklevel` walks past to find the application's own frame.

Matched on the dot (`name == prefix or name.startswith(prefix + ".")`), so `langchain_myapp` —
someone's own package that merely starts with the same letters as `langchain` — is never
mistaken for the library's."""


def _is_library_frame(frame: types.FrameType) -> bool:
    """Whether `frame` belongs to this package or to LangChain (`_LIBRARY_MODULES`)."""
    name = frame.f_globals.get("__name__", "")
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in _LIBRARY_MODULES)


def _stacklevel() -> int:
    """The `stacklevel` that points a warning at the application's own code.

    That is the first frame outside the library, above the router's own caller's frame.

    A hand-counted constant breaks whenever a frame is added or removed between the router and
    its caller: bound through `bind_tools` versus called bare, `StructuredRouter.stream`
    delegating to `ChatRouter.stream`, `create_agent` wrapping the router in its own runnables,
    a later `langchain-core` release adding a frame of its own. Walking the stack instead finds
    the boundary itself, at the cost of a stdlib-private call: `sys._getframe` is what the
    stdlib `warnings` module uses internally (`warnings.py`, `_filters_mutated`'s callers) to
    find *its* caller, and is documented as CPython-implementation-specific rather than
    guaranteed portable — confirmed to behave the same, for this walk, on 3.10, 3.12 and 3.13.

    Frame 1 (`sys._getframe(1)`) is the function that calls `warnings.warn` — `stacklevel=1`'s
    frame, by `warnings`' own definition — which is always this package's own code and so
    always skipped; the walk starts there and moves outward until a frame's module is not one
    of `_LIBRARY_MODULES`, or there are no more frames. A call with genuinely no application
    frame above it — a `batch` worker thread, `agenerate`'s gathered tasks, an `astream_events`
    generator driven by `anyio` — runs out of stack first and lands on the outermost frame
    reached, which is documented, not a bug: there is no caller line to point at.
    """
    level = 1
    frame: types.FrameType | None = sys._getframe(1)  # private API: see the docstring above
    while frame is not None and _is_library_frame(frame):
        frame = frame.f_back
        level += 1
    return level if frame is not None else level - 1


_V3_UNSUPPORTED = (
    "ChatRouter does not support the v3 streaming protocol "
    "(stream_events / astream_events with version='v3'): it drives the model through "
    "_stream / _generate directly, which would bypass routing, the decision record and the "
    "run shape entirely. Use stream(), astream(), or version='v2' events."
)
"""Why v3 is refused rather than delegated.

`_chat_model_stream_v3` (`language_models/chat_models.py:995` in `langchain-core` 1.6.3; new in
1.4, and still beta) calls `self._stream` — the one hook a *delegating* router deliberately
does not implement, so none of the pipeline runs. Delegating to the selected route's own v3
stream would need the router to own the returned `ChatModelStream`: that type lives in a module
`langchain_core` doesn't export, it has no completion hook to close the router's run with, and
it assembles its `response_metadata` only from protocol events, so the record could only
be put there by forging one. None of it exists in `langchain-core` 1.1, the minimum supported.
Refusing at the two public doors keeps the failure honest and, unlike the base class's bare
`NotImplementedError`, stops a chat-model run being opened for the router itself."""


class ChatRouter(BaseChatModel):
    """A chat model that picks one of several named chat models per request.

    The response is the selected route's own, with the routing decision added under
    `response_metadata["routing"]`. The routing policy is the application's: a
    strategy applies it, and when it can't decide the default route answers.

    Because the router delegates to the route rather than generating itself, the settings that
    govern a call are the *route's*: its `cache`, `rate_limiter` and `disable_streaming` apply,
    and the router's own would never be consulted (the same reasoning that makes the router
    reject a `cache=` of its own outright rather than let it look effective).

    Args:
        routes: Named routes, any number of them; declaration order is significant. Each is an
            ordinary chat model, used as given. A runnable that wraps one, such as
            `model.with_retry()`, is not a route.
        default_route: The route that answers when no strategy is set or the strategy can't
            decide. Must name one of `routes`.
        strategy: A `RoutingStrategy`, or a plain function of a `RoutingRequest`. `None`
            always uses the default route.
        on_unavailable_forced_route: What happens when a route forced through runtime config
            is unknown or can't use bound tools: `"error"` (the default) raises, `"fallback"`
            warns and uses the default route.
        tool_support_overrides: Per-route override of tool-capability detection, by route
            name; it always wins.

    Returns:
        The router is itself a chat model: `invoke`, `stream`, `batch` and their async
        forms return the selected route's own answer, with the decision under
        `response_metadata["routing"]`.

    Raises:
        RoutingError: at construction, for no routes, a blank route name, a `default_route` or
            `tool_support_overrides` key that isn't a route, a route wrapped in another
            runnable, or a `cache=` on the router. Constructing inside a pydantic validator
            surfaces it as `pydantic.ValidationError`.
        TypeError: for a `strategy` that is neither a strategy instance nor a synchronous
            callable.
        ForcedRouteError: at call time, for a forced route that doesn't exist or can't use the
            bound tools.
        NoToolCapableRouteError: from `bind_tools` / `with_structured_output`, or at call
            time, when no route can use the bound tools.

    Example:
        ```python
        from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision

        router = ChatRouter(
            routes={"small": small_model, "coder": coder_model},
            default_route="small",
            strategy=KeywordStrategy({"coder": ["python", "stack trace"]}),
        )
        response = router.invoke("Why does this Python stack trace happen?")
        print(routing_decision(response).route)  # "coder"
        ```
    """

    routes: dict[str, BaseChatModel]
    """Named routes, any number of them; declaration order is significant.

    Each is an ordinary chat model, used as given — the router never reconfigures or mutates
    one. The names are the mapping's keys, so they are unique.

    A route is a chat model and not a runnable wrapped around one, because the router has to
    ask it what it can do — `bind_tools` and `profile` for tool capability, its `cache`
    for who owns response caching — and a wrapper answers none of those. Retries go on the router or
    in the provider's client; `_reject_wrapped_routes` says so."""

    default_route: str
    """Mandatory. Must name one of `routes`."""

    strategy: RoutingStrategy | RoutingCallable | None = Field(default=None, exclude=True)
    """None always uses the default route. A plain callable is coerced.

    Kept out of serialization: a strategy is code, not configuration."""

    on_unavailable_forced_route: Literal["error", "fallback"] = "error"
    """What happens when a forced route is unknown or can't use bound tools."""

    tool_support_overrides: dict[str, bool] = Field(default_factory=dict)
    """Per-route override of capability detection; always wins."""

    @field_validator("routes", mode="before")
    @classmethod
    def _reject_wrapped_routes(cls, routes: object) -> object:
        """A runnable that wraps a chat model is not a route, and says why.

        `model.with_retry()` and `model.bind(...)` return a `RunnableBinding`, and
        `init_chat_model(configurable_fields=...)` a `_ConfigurableModel` — none of them a
        `BaseChatModel`, so pydantic would reject them anyway, with a message that names the
        expected type and not the thing to do instead. Retrying a single route is the reason
        people reach for this, and it would not work: `RunnableBindingBase.stream` yields
        straight from `self.bound.stream(...)`, so a wrapped route retries on `invoke` and not
        when streamed.
        """
        if not isinstance(routes, Mapping):
            return routes
        for name, route in routes.items():
            if isinstance(route, Runnable) and not isinstance(route, BaseChatModel):
                msg = (
                    f"route {name!r} is a {type(route).__name__}, not a chat model: a route "
                    f"is used as given, and a wrapper hides what the router must ask it "
                    f"(bind_tools, profile, cache). To retry, wrap the router — "
                    f"router.with_retry(...) — or set the provider client's own max_retries."
                )
                raise RoutingError(msg)
        return routes

    @field_validator("routes")
    @classmethod
    def _check_routes(cls, routes: dict[str, BaseChatModel]) -> dict[str, BaseChatModel]:
        """At least one route, each with a name."""
        if not routes:
            msg = "routes is empty: a ChatRouter needs at least one route"
            raise RoutingError(msg)
        for name in routes:
            if not name.strip():
                msg = f"route name {name!r} is blank: every route needs a name"
                raise RoutingError(msg)
        return routes

    @field_validator("cache", mode="before")
    @classmethod
    def _reject_cache(cls, cache: object) -> object:
        """The router's own `cache=` would silently no-op otherwise.

        `BaseChatModel` declares `cache` (`language_models/base.py:190`) with no validator of
        its own, and `ChatRouter` inherits the field but never reads it: it delegates to each
        route's `invoke` / `stream` rather than running `_generate_with_cache` itself (the
        module docstring's pipeline), so nothing on the router would ever consult a cache set
        here. `None`, the field's own default, is the only value that passes — it means "not
        set", not "use the global cache", since the router never checks it either way; anything
        else (an explicit `BaseCache`, `True` or `False`) would look like it took effect and
        never would.

        `mode="before"`, like `_reject_wrapped_routes`: pydantic only runs a validator against
        an explicitly passed value by default (`validate_default` is off), so a `ChatRouter`
        built without `cache=` at all never reaches this at all.
        """
        if cache is not None:
            msg = (
                "cache belongs on the route, not the router: ChatRouter delegates to "
                "each route's own invoke/stream rather than running _generate_with_cache "
                "itself, so its own cache would never be consulted — set cache= on each "
                "route instead."
            )
            raise RoutingError(msg)
        return cache

    @field_validator("strategy", mode="before")
    @classmethod
    def _coerce_strategy(cls, strategy: object) -> RoutingStrategy | None:
        """One interface for every strategy: a plain callable is wrapped.

        Before the field's own type check, so `as_strategy` is the single gate on what may be
        a strategy and says so itself. What it rejects is a `TypeError`, which pydantic does
        not wrap — unlike the `RoutingError` a misnamed route raises.
        """
        return None if strategy is None else as_strategy(cast("RoutingCallable", strategy))

    @model_validator(mode="after")
    def _check_route_references(self) -> ChatRouter:
        """Every name that refers to a route names one."""
        if self.default_route not in self.routes:
            msg = (
                f"default_route {self.default_route!r} is not one of the routes: "
                f"{_names(self.routes)}"
            )
            raise RoutingError(msg)
        unknown = [name for name in self.tool_support_overrides if name not in self.routes]
        if unknown:
            msg = (
                f"tool_support_overrides names routes that don't exist: {_names(unknown)}; "
                f"the routes are {_names(self.routes)}"
            )
            raise RoutingError(msg)
        return self

    @property
    def _llm_type(self) -> str:
        return "chat_router"

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """The `"route"` key: what forces a route for one call.

        `Runnable`'s own default is `[]`, which would hide the key from `with_config`,
        `config={"configurable": {"route": ...}}` and config-schema introspection —
        `ConfigurableFieldSpec` (`runnables/utils.py:655`) is the mechanism LangChain gives
        every `Runnable` for exactly this. `default=None` matches what leaving the key unset
        means: no force, the strategy decides. `StructuredRouter.config_specs` forwards
        this (`_tools.py`), so a router bound through `with_structured_output` exposes it too,
        and a `RunnableBinding` — what `bind_tools` and `with_config` return — forwards a
        wrapped `Runnable`'s `config_specs` on its own (`RunnableBindingBase.config_specs`),
        so no extra forwarding is needed for those.
        """
        return [
            ConfigurableFieldSpec(
                id="route",
                annotation=str | None,
                name="Route",
                description=(
                    "Force one of the routes for this call, skipping the strategy entirely: "
                    f"{_names(self.routes)}."
                ),
                default=None,
            )
        ]

    def _resolve_model_profile(self) -> ModelProfile | None:
        """What the router can promise on every route's behalf.

        Called by the base class's `_set_model_profile` validator (`chat_models.py:417`), and
        only when `profile` was not supplied directly — an explicit `ChatRouter(..., profile=…)`
        wins over this without this method being consulted at all. By the time any
        `model_validator(mode="after")` runs — this one included — pydantic v2 has already run
        field validation, so `self.routes` is populated and `_check_routes` has already required
        at least one route (verified in `tests/unit_tests/test_profile.py`).

        `_profile.resolve_profile` does the reduction; this is the one line `create_agent` needs
        (`model.profile`, read by `_supports_provider_strategy`,
        `langchain/agents/factory.py:560`).
        """
        return resolve_profile(self.routes)

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        """Standard LangSmith params: `ls_model_name` names the default route.

        Dead code for the router's own pipeline — `invoke`, `stream` and the rest never call
        `_generate_with_cache` or any other base-class path that reads this (module docstring);
        it exists only for callers that read it directly, and for `ChatModelUnitTests`'
        `test_standard_params`, whose `Check` is part of this task's own. The base
        implementation (`chat_models.py:1502`) falls back to `self.model` / `self.model_name`,
        neither of which `ChatRouter` has, so `ls_model_name` would otherwise be missing rather
        than merely wrong — a hole the base class's own docstring invites subclasses to close
        ("Subclasses **should override** this method").

        The base implementation already reads `kwargs["model"]` first
        (`test_standard_params_model_override`'s per-call override, which needs no change here);
        this only adds a fallback for the no-override case, the same way `self.model` would for
        an ordinary chat model — the default route is the one route guaranteed to answer when
        nothing else does, so it stands in for a nominal "current model" here.
        """
        params = super()._get_ls_params(stop=stop, **kwargs)
        params.setdefault("ls_model_name", self.default_route)
        return params

    # --- Entry points: each runs the pipeline in the module docstring. ---

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """The selected route's own answer, with the decision record added.

        Args:
            input: A string, a list of messages or a prompt value, as any chat model takes.
            config: The run's config; `configurable["route"]` forces a route.
            stop: Stop sequences, passed to the route.
            **kwargs: Passed to the route's own `invoke`.

        Returns:
            The route's message, with the decision under `response_metadata["routing"]`.

        Raises:
            ForcedRouteError: if a forced route doesn't exist or can't use the bound tools.
        """
        config = ensure_config(config)
        # private API: `_convert_input` is how every chat model turns its input into messages.
        # Calling it keeps a string, a list of dicts, `BaseMessage`s and a `ChatPromptValue`
        # identical through the router. `set_config_context` and
        # `coro_with_context`, used below, are public names their modules leave out of
        # `__all__`; both are how the strategy's run is entered.
        messages = self._convert_input(input).to_messages()
        run_manager = self._start_run(config, messages)
        try:
            decision = self._decide(messages, config, run_manager, kwargs)
            call = self._route_call(decision, config, run_manager, kwargs)
            message = call.route.invoke(messages, call.config, stop=stop, **call.kwargs)
        except BaseException as error:
            discard_decision()  # this call has no decision to report
            run_manager.on_chain_error(error)
            raise
        message = _with_record(message, decision)
        run_manager.on_chain_end(_run_outputs(message, decision))
        return message

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Async `invoke`: the strategy is awaited too, so neither blocks the loop."""
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()  # private API: see `invoke`
        run_manager = await self._astart_run(config, messages)
        try:
            decision = await self._adecide(messages, config, run_manager, kwargs)
            call = self._route_call(decision, config, run_manager, kwargs)
            message = await call.route.ainvoke(messages, call.config, stop=stop, **call.kwargs)
        except BaseException as error:
            discard_decision()  # this call has no decision to report
            await run_manager.on_chain_error(error)
            raise
        message = _with_record(message, decision)
        await run_manager.on_chain_end(_run_outputs(message, decision))
        return message

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """The route's items, passed through unchanged bar the record on one of them.

        "Items" and not "chunks": what `call.route.stream` yields is `AIMessageChunk`s for a
        plain or tool-bound call, but the route's own progressive partials — Pydantic objects,
        dicts, or `include_raw=True`'s `{"raw", "parsed", "parsing_error"}` — when a
        `StructuredOutputBinding` was replayed onto it (only `StructuredRouter.stream`
        asks for that, and reads the result so). The annotation is the chat-model contract this
        method keeps for its ordinary callers.

        `stop` reaches the route only when the caller gave one: `include_raw=True` builds
        `RunnableMap(raw=llm) | ...` (`chat_models.py:2564`), whose `_transform` — the hook
        `.stream()` runs on — takes no `**kwargs` at all, so even `stop=None` explicitly passed
        raises `TypeError`. `.invoke()` tolerates it (`RunnableParallel.invoke` accepts and
        drops stray kwargs), which is why this only bites streaming, and only once a route
        answers with more than the one item `StructuredRouter.stream` used to stop at.
        """
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()  # private API: see `invoke`
        run_manager = self._start_run(config, messages)
        output: Any = None
        try:
            decision = self._decide(messages, config, run_manager, kwargs)
            call = self._route_call(decision, config, run_manager, kwargs)
            stream_kwargs = call.kwargs if stop is None else {**call.kwargs, "stop": stop}
            for item in call.route.stream(messages, call.config, **stream_kwargs):
                if output is None:
                    # Decided before the first item; only it carries the record —
                    # wherever `_recorded_on` finds somewhere to put one.
                    item = output = _with_record(item, decision)
                else:
                    output = _merge(output, item)
                yield cast("AIMessageChunk", item)
        except GeneratorExit as error:
            # A caller that stops consuming — a `break`, or the generator being finalized —
            # is not a failed call: the chunks it did take carry the record, and the record
            # stays readable. Finalization can also run on another thread or context,
            # where a tombstone would erase a record that is not this call's at all.
            run_manager.on_chain_error(error)
            raise
        except BaseException as error:
            discard_decision()  # this call has no decision to report
            run_manager.on_chain_error(error)
            raise
        run_manager.on_chain_end(_run_outputs(output, decision))

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Async `stream`. `stop` is omitted when unset, for the reason `stream` documents."""
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()  # private API: see `invoke`
        run_manager = await self._astart_run(config, messages)
        output: Any = None
        try:
            decision = await self._adecide(messages, config, run_manager, kwargs)
            call = self._route_call(decision, config, run_manager, kwargs)
            stream_kwargs = call.kwargs if stop is None else {**call.kwargs, "stop": stop}
            async for item in call.route.astream(messages, call.config, **stream_kwargs):
                if output is None:
                    # Decided before the first item; only it carries the record —
                    # wherever `_recorded_on` finds somewhere to put one.
                    item = output = _with_record(item, decision)
                else:
                    output = _merge(output, item)
                yield cast("AIMessageChunk", item)
        except GeneratorExit as error:
            # Not a failed call — see `stream`.
            await run_manager.on_chain_error(error)
            raise
        except BaseException as error:
            discard_decision()  # this call has no decision to report
            await run_manager.on_chain_error(error)
            raise
        await run_manager.on_chain_end(_run_outputs(output, decision))

    def stream_events(  # type: ignore[override]
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """v1 and v2 events, which are produced from `stream` and so are routed.

        The narrower return type than the base class's is the point: v3 is refused, so a
        router only ever hands back `StreamEvent`s.
        """
        if version == "v3":
            raise NotImplementedError(_V3_UNSUPPORTED)
        # `Runnable` grew a synchronous `stream_events` in `langchain-core` 1.4, alongside v3.
        # Below that, `super()` has none and this raises the `AttributeError` a bare chat model
        # raises there too — and `version="v3"` is unreachable, so the guard above never fires.
        return super().stream_events(input, config, version=version, **kwargs)

    def astream_events(  # type: ignore[override]
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Async `stream_events`: v1 and v2 events, produced from `astream`.

        Not an `async def`, as the base class's isn't: it hands back the iterator rather than
        being one, so a v3 caller — who would `await` this — is refused at the call itself.
        """
        if version == "v3":
            raise NotImplementedError(_V3_UNSUPPORTED)
        return super().astream_events(input, config, version=version, **kwargs)

    def generate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Route every prompt as `invoke` does, and answer in an `LLMResult`.

        The base `generate` opens a model run per prompt and calls `_generate` inside it
        (`chat_models.py:1668`), so left alone it would put a model run of the router's own
        beside the route's and bill the same tokens twice. This runs
        `invoke` once per prompt instead, so each prompt gets what an `invoke` gets: a router
        chain run of its own, its own decision — prompts are decided one by one, not
        once for the batch — and the route's call as the only model run.

        What carries over from the base `generate`:

        - `callbacks`, `tags`, `metadata` and `run_name` apply to every prompt's run, and
          `run_id` names the first prompt's (`manager.py:1452`); the others get one of their own.
          `BaseChatModel.invoke` takes a config apart into these arguments (`:488`), and
          `_generate_config` puts them back together.
        - `stop` and `kwargs` reach the route, as they reach it from `invoke`.
        - `LLMResult.run` lists the run each prompt got: the router's chain run, which is the
          top-level run the caller called.
        - Prompts run in order, and the first failure stops the call.

        What does not is the shape of the route's own `LLMResult`. A prompt is answered by the
        route's message, with its record — `usage_metadata`, `response_metadata`, `id` and all —
        so a route's extra candidates, `generation_info` and `llm_output` stay on the route's own
        run, where its callbacks see them: a cost callback reads `on_llm_end`, not this result.
        LangChain itself tells callers not to rely on `llm_output` and to read the message
        (`outputs/llm_result.py:40`), so it is `{}` here, as it is for any chat model that does
        not combine its outputs.

        A `FallbackWarning` from a prompt run here points at the caller exactly as it does
        under `invoke`: `_stacklevel` walks past every frame that is this package's or
        LangChain's own, not a fixed count, so the extra frame `generate` adds before reaching
        `invoke` costs it nothing. (`agenerate`, whose prompts run as tasks, has no caller
        frame above them either, and neither has `abatch`'s worker threads — both still land
        on the outermost frame the walk can reach.)
        """
        run_ids = _run_ids(len(messages), run_id)
        answers = [
            self.invoke(
                prompt,
                _generate_config(callbacks, tags, metadata, run_name, prompt_run_id),
                stop=stop,
                **kwargs,
            )
            for prompt, prompt_run_id in zip(messages, run_ids, strict=True)
        ]
        return _llm_result(answers, run_ids)

    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async `generate`: the prompts overlap, as the base `agenerate` overlaps them.

        Every prompt runs to the end before a failure is raised — the first, in prompt order —
        because the others' route calls are already in flight and have been paid for
        (`chat_models.py:1808`); each closes its own router run either way.
        """
        run_ids = _run_ids(len(messages), run_id)
        outcomes = await asyncio.gather(
            *(
                self.ainvoke(
                    prompt,
                    _generate_config(callbacks, tags, metadata, run_name, prompt_run_id),
                    stop=stop,
                    **kwargs,
                )
                for prompt, prompt_run_id in zip(messages, run_ids, strict=True)
            ),
            return_exceptions=True,
        )
        answers: list[AIMessage] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            answers.append(outcome)
        return _llm_result(answers, run_ids)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Refuse: the router has no model call of its own.

        A call that reaches this went around the routed entry points.

        `invoke`, `ainvoke`, `stream`, `astream`, `generate` and `agenerate` all hand the request
        to a route, and nothing reachable through them comes here. What could is a door left
        open — the v3 streaming protocol before it was refused, or a new base-class path in a
        later `langchain-core` — and the base class would run this inside a model run of the
        router's own, beside the route's, billing the same tokens twice without an error. Refusing
        turns that silent miscount into a failure someone sees. It is also what stops `_agenerate`,
        which the base class builds on this, from working.
        """
        msg = (
            "ChatRouter has no model call of its own: invoke, ainvoke, stream, astream, "
            "generate and agenerate hand each request to a route, and this call reached the "
            "router another way"
        )
        raise NotImplementedError(msg)

    # --- Binding: kept as given until a route is chosen ---

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools unconverted; the route that answers converts them.

        Converting a tool is provider-specific work, and which provider does it isn't known
        until a route is chosen — so `tools` is kept exactly as given and replayed on the
        route's own `bind_tools` at call time. `tool_choice` and the rest of the keyword
        arguments go with it rather than becoming call kwargs: `strict=` and its kind change
        how the route *builds* the schema, which only its binder can do.

        That private binding is all that is bound: the router sets no `tools` kwarg of its own.
        A plain chat model binds *converted* tool dicts there, and LangChain reads
        them that way — `create_react_agent` calls `.get("type")` on each entry, so a raw
        `StructuredTool` in the slot crashed it, and converting to fill it would reject a
        provider-native entry such as `{"google_search": {}}`, which `convert_to_openai_tool`
        can't read. Nothing needs the slot filled: the route's own binder sets its `tools` when
        the binding is replayed, which is where `disable_streaming="tool_calling"` looks, and
        a later bare `bind(tools=...)` reaches the route as a call kwarg, as it does on a
        plain chat model.

        Binding is also when the application learns which routes can't use tools
        (`_check_tool_support`), rather than on the first request that picks one. That is
        the whole of what is checked here, and two things it cannot see are left to the first
        request that reaches them:

        - An invalid binding kwarg. Conversion is deferred to the route, so a
          `strict=` its binder rejects fails on the first request routed to that route, as the
          route's own error, and not on this call.
        - A route that is itself a `ChatRouter`. It passes the check whatever *its* routes can
          do, because it overrides `bind_tools`; if none of them can use tools, the request
          fails with `NoToolCapableRouteError` once it gets there.

        Binding again replaces the binding, as on any chat model: a binding over a binding
        hands `bind_tools` back to the model (`RunnableBinding.__getattr__`), so what the
        first held is not part of the second.

        `tool_choice` is typed wider than `BaseChatModel.bind_tools` types it: providers take
        dicts and booleans too, and whatever a route accepts has to reach it unchanged.

        Args:
            tools: The tools to bind, in any form a route's own `bind_tools` accepts.
            tool_choice: Passed to the route's `bind_tools` unchanged.
            **kwargs: Also passed to the route's `bind_tools`.

        Returns:
            A runnable that replays the binding on whichever route answers.

        Raises:
            NoToolCapableRouteError: if no route can use tools.
        """
        self._check_tool_support()
        binding = ToolBinding(tools=tuple(tools), tool_choice=tool_choice, kwargs=dict(kwargs))
        return self.bind(**{BINDING_KEY: binding})

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, StructuredOutput]:
        """Forward to the selected route's own structured output, per request.

        `BaseChatModel`'s default would build this on the *router's* `bind_tools` and drop
        `method=` and `strict=` on the way (`chat_models.py:2530`), forcing every route
        through function calling and past whatever native JSON-schema mode it has. Forwarding
        keeps each route's own — and each route's own parser with it.

        LangChain builds structured output on tool binding, so tool-aware routing applies as it does
        to `bind_tools`: the same bind-time check here, and the same per-request diversion off a
        route that can't use tools. As there, an invalid kwarg — `method=` a route
        doesn't support, say — fails on the first request routed to that route rather than
        here (each route converts at call time), and a route that is itself a `ChatRouter`
        passes this check whatever its own routes can do, failing mid-request with
        `NoToolCapableRouteError` if none of them can use tools.

        What comes back routes through the router's pipeline, so the decision reaches the
        trace and `last_routing_decision()`; with `include_raw=True` the raw message carries
        it too. Streamed progressively — the route's own parser runs — as it is
        on a plain chat model: `stream` and `astream` delegate to the router's own.

        Args:
            schema: A Pydantic class, a `TypedDict` or a JSON schema dict.
            include_raw: Also return the raw message, which carries the decision record.
            **kwargs: Passed to the selected route's own `with_structured_output`.

        Returns:
            A runnable whose output is the route's parsed answer — or, with `include_raw=True`,
            a dict of `raw`, `parsed` and `parsing_error`.

        Raises:
            NoToolCapableRouteError: if no route can use tools.
        """
        self._check_tool_support()
        binding = StructuredOutputBinding(
            schema=schema, include_raw=include_raw, kwargs=dict(kwargs)
        )
        return StructuredRouter(self, binding)

    # --- 1. The router's run ---

    def _start_run(
        self, config: RunnableConfig, messages: list[BaseMessage]
    ) -> CallbackManagerForChainRun:
        """Open the router's run: a chain run, so the route's call is the only model run.

        The router's own `callbacks`, `tags` and `metadata` apply to this run alone, as they do
        to a chat model's run; the caller's config is inherited by everything under it.
        """
        manager = CallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            config.get("metadata"),
            self.metadata,
        )
        return manager.on_chain_start(
            None,
            {"messages": messages},
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

    async def _astart_run(
        self, config: RunnableConfig, messages: list[BaseMessage]
    ) -> AsyncCallbackManagerForChainRun:
        """Async `_start_run`."""
        manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            config.get("metadata"),
            self.metadata,
        )
        return await manager.on_chain_start(
            None,
            {"messages": messages},
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

    # --- 2. The decision ---

    def _plan(
        self, messages: list[BaseMessage], config: RunnableConfig, kwargs: dict[str, Any]
    ) -> RoutingDecision | _StrategyCall:
        """What deciding this request takes: a decision already settled, or a strategy to run.

        A route forced through runtime config (`configurable["route"]`) settles it outright
        via `_forced_decision` — and so does having no strategy at all — either way with no
        strategy run. A request with no user message gives a strategy nothing
        to decide on, so it is not consulted either, and the default route answers.
        """
        forced = config.get("configurable", {}).get("route")
        if forced is not None:
            return self._forced_decision(cast("str", forced), kwargs)
        # `_coerce_strategy` has wrapped any plain callable.
        strategy = cast("RoutingStrategy | None", self.strategy)
        if strategy is None:
            return RoutingDecision(route=self.default_route, reason="no strategy configured")
        request = build_request(
            messages,
            routes=tuple(self.routes),
            tools_bound=tools_are_bound(kwargs),
            wants_full_context=strategy.wants_full_context,
            # Replaced by the strategy run's child config once that run is open.
            config=config,
        )
        if request is None:
            return self._fallback("the request has no user message to route on", strategy=None)
        return _StrategyCall(strategy, strategy_name(strategy), request)

    def _forced_decision(self, route: str, kwargs: dict[str, Any]) -> RoutingDecision:
        """What a forced route resolves to.

        Itself, when it can serve this call, or whatever `on_unavailable_forced_route` says to
        do when it can't.

        Never runs the strategy — this is `_plan`'s other way of settling the
        decision outright, taken before a `_StrategyCall` would even be built. Whatever comes
        back still goes through `_divert` afterwards, exactly like every other decision (see
        `_divert`'s own docstring) — this only rules out the one case the router must refuse
        first: the forced route itself being unusable.
        """
        problem = self._forced_route_problem(route, kwargs)
        if problem is None:
            return RoutingDecision(route=route, reason="forced via runtime config", forced=True)
        if self.on_unavailable_forced_route == "error":
            raise ForcedRouteError(
                f"{problem}; set on_unavailable_forced_route='fallback' to fall back to the "
                "default route instead"
            )
        return self._forced_fallback(problem)

    def _forced_route_problem(self, route: str, kwargs: dict[str, Any]) -> str | None:
        """Why a forced route can't be used, or `None` when it can.

        Existence first. Tool capability only when something is bound — tools, or a schema,
        which LangChain builds on tool binding — using the same signal `_divert` uses for every
        other route: with nothing bound, a forced route is always available on that
        count, and only its existence is ever in question.
        """
        if route not in self.routes:
            return f"forced route {route!r} is not one of the routes: {_names(self.routes)}"
        if tools_are_bound(kwargs) and not self._supports_tools(route):
            return f"forced route {route!r} can't use the bound tools"
        return None

    def _decide(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
        run_manager: CallbackManagerForChainRun,
        kwargs: dict[str, Any],
    ) -> RoutingDecision:
        """Settle the route, diversion included; the strategy runs in a child run.

        Any call the strategy makes with `request.config` — or without it, through the context
        config — nests under the strategy's run, so it is traced and costed there.

        `_divert` is applied at every return path, before the strategy run closes: the
        decision is one record in three places — the strategy run's output, the
        router run's output, the route run's metadata — and a diversion is part of settling
        the decision, not something layered on afterwards. The strategy run still ends a
        success: diverting is not a failure of the strategy, which chose correctly given what
        it knew, so `on_chain_end` runs even though the outputs it gets are the diverted
        record, not the strategy's raw choice — the same as for a fallback, whose
        strategy run likewise holds the final record.
        """
        pending = self._plan(messages, config, kwargs)
        if isinstance(pending, RoutingDecision):
            return self._divert(pending, kwargs)
        strategy_run = run_manager.get_child().on_chain_start(
            None, pending.inputs, name=pending.name
        )
        request = replace(
            pending.request, config=patch_config(config, callbacks=strategy_run.get_child())
        )
        try:
            with set_config_context(request.config) as context:
                choice: object = context.run(pending.strategy.decide, request)
        except BaseException as error:
            strategy_run.on_chain_error(error)
            if not isinstance(error, Exception):
                raise
            return self._divert(self._conclude(pending, error), kwargs)
        try:
            decision = self._divert(self._conclude(pending, choice), kwargs)
        except BaseException as error:
            # `_conclude` or `_divert` warns, and an application may have escalated that
            # warning to an error; the strategy's run has to close either way.
            strategy_run.on_chain_error(error)
            raise
        strategy_run.on_chain_end(decision.as_dict())
        return decision

    async def _adecide(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForChainRun,
        kwargs: dict[str, Any],
    ) -> RoutingDecision:
        """Async `_decide`: awaits `adecide`, so the strategy never blocks the loop."""
        pending = self._plan(messages, config, kwargs)
        if isinstance(pending, RoutingDecision):
            return self._divert(pending, kwargs)
        strategy_run = await run_manager.get_child().on_chain_start(
            None, pending.inputs, name=pending.name
        )
        request = replace(
            pending.request, config=patch_config(config, callbacks=strategy_run.get_child())
        )
        try:
            with set_config_context(request.config) as context:
                choice: object = await coro_with_context(pending.strategy.adecide(request), context)
        except BaseException as error:
            await strategy_run.on_chain_error(error)
            if not isinstance(error, Exception):
                raise
            return self._divert(self._conclude(pending, error), kwargs)
        try:
            decision = self._divert(self._conclude(pending, choice), kwargs)
        except BaseException as error:
            # As in `_decide`: a warning escalated to an error must still close the run.
            await strategy_run.on_chain_error(error)
            raise
        await strategy_run.on_chain_end(decision.as_dict())
        return decision

    def _conclude(self, pending: _StrategyCall, outcome: object) -> RoutingDecision:
        """The decision a strategy's outcome leads to.

        Every way a strategy can fail to decide — abstaining, raising, naming a route that
        doesn't exist, or returning something that isn't a choice — ends on the default route.
        A raised error's traceback stays on the strategy's run in the trace.
        """
        if isinstance(outcome, RoutingChoice) and outcome.route in self.routes:
            return RoutingDecision(
                route=outcome.route, reason=outcome.reason, strategy=pending.name
            )
        if outcome is None:
            cause = "could not decide"
        elif isinstance(outcome, BaseException):
            cause = f"raised {_describe(outcome)}"
        elif isinstance(outcome, RoutingChoice):
            cause = f"chose {outcome.route!r}, which is not one of the routes"
        else:
            cause = f"returned {type(outcome).__name__}, not a RoutingChoice"
        return self._fallback(f"{pending.name} {cause}", strategy=pending.name)

    def _fallback(self, cause: str, *, strategy: str | None) -> RoutingDecision:
        """The default route, with one `FallbackWarning` and the cause recorded."""
        warnings.warn(
            f"{cause}; falling back to the default route {self.default_route!r}",
            FallbackWarning,
            stacklevel=_stacklevel(),
        )
        return RoutingDecision(
            route=self.default_route,
            reason=f"{cause}; fell back to the default route",
            strategy=strategy,
            fallback=True,
        )

    def _forced_fallback(self, cause: str) -> RoutingDecision:
        """The forced-route fallback.

        The default route stands in for a forced route that can't be
        used, with one `ForcedRouteWarning` — not `FallbackWarning`, which is for a
        strategy that failed rather than a forced route that was refused — and both
        `forced=True` and `fallback=True` recorded. `strategy` stays unset: no
        strategy ran, so there is none to name.

        A dedicated method rather than a parameter on `_fallback`: that method's own callers (a
        strategy that couldn't decide, and a request with no user message) are that path's cases,
        and `_fallback` always uses `FallbackWarning` and never sets `forced=True` — widening its
        signature to cover a different warning class would let this case leak into theirs
        through a missed keyword, for two call sites that are otherwise unrelated.
        """
        warnings.warn(
            f"{cause}; falling back to the default route {self.default_route!r}",
            ForcedRouteWarning,
            stacklevel=_stacklevel(),
        )
        return RoutingDecision(
            route=self.default_route,
            reason=f"{cause}; fell back to the default route",
            forced=True,
            fallback=True,
        )

    # --- 3. Tool capability ---

    def _supports_tools(self, route: str) -> bool:
        """Whether `route` can use tools, by the signals `supports_tools` documents."""
        return supports_tools(self.routes[route], override=self.tool_support_overrides.get(route))

    def _tool_capable_route(self) -> str | None:
        """Where a request diverted off a tool-incapable route goes, or `None` if nowhere.

        The default route when it can use tools — the route the application already nominated
        for everything it didn't have an opinion about — and otherwise the first route in
        declaration order that can. Deterministic, and explainable without re-running the
        strategy.
        """
        if self._supports_tools(self.default_route):
            return self.default_route
        return next((name for name in self.routes if self._supports_tools(name)), None)

    def _check_tool_support(self) -> None:
        """What binding tools to this router means for the routes that can't use them.

        Once, at bind time, where the application can still do something about
        it: one warning naming every route that will be skipped, and an error when there is
        no route left to skip to — binding tools no route can use has no outcome worth
        waiting for a request to discover.
        """
        incapable = [name for name in self.routes if not self._supports_tools(name)]
        if not incapable:
            return
        target = self._tool_capable_route()
        if target is None:
            raise NoToolCapableRouteError(_no_tool_capable_route(self.routes))
        warnings.warn(
            f"routes that can't use tools: {_names(incapable)}; a request routed to one of "
            f"them goes to {target!r} instead",
            ToolSupportWarning,
            stacklevel=_stacklevel(),
        )

    def _divert(self, decision: RoutingDecision, kwargs: dict[str, Any]) -> RoutingDecision:
        """What tool-aware routing allows: a tool-incapable route gives way to a capable one.

        Only when something is bound — tools, or a schema, which LangChain builds on tool
        binding. The strategy is not consulted again: it has decided, a second run would
        buy a second call under the strategies that make them, and re-deciding could
        divert in a loop.

        One warning per diverted request: a diversion is not the route the policy
        asked for, and it happens per request, so it is per request that the application
        hears about it. `diverted_from` keeps the route it came from.

        Called from `_decide` / `_adecide`, before the strategy's run closes: the decision
        is one record in three places, so the strategy run's output is the *diverted* record
        too, the same as the router run's outputs and the route run's metadata — not what the
        strategy chose before the diversion stepped in. The `forced` guard runs earlier, in `_plan`:
        a forced route that can't use the bound tools is refused —
        `ForcedRouteError`, or `ForcedRouteWarning` under `on_unavailable_forced_route`'s
        `"fallback"` setting — before a request ever gets here, so this never has to tell a
        forced route from any other. Whatever a forced route's refusal falls back to is
        diverted like any other decision.
        """
        if not tools_are_bound(kwargs) or self._supports_tools(decision.route):
            return decision
        target = self._tool_capable_route()
        if target is None:
            # Unreachable through `bind_tools`, which refuses at bind time; reachable by
            # putting tools straight into the call kwargs with `bind(tools=...)`.
            raise NoToolCapableRouteError(_no_tool_capable_route(self.routes))
        cause = f"{decision.route!r} can't use the bound tools"
        warnings.warn(
            f"{cause}; diverted to {target!r}", ToolSupportWarning, stacklevel=_stacklevel()
        )
        return replace(
            decision,
            route=target,
            reason=f"{decision.reason}; {cause}, so it was diverted to {target!r}",
            diverted_from=decision.route,
        )

    # --- 4. The route's call ---

    def _route_call(
        self,
        decision: RoutingDecision,
        config: RunnableConfig,
        run_manager: CallbackManagerForChainRun | AsyncCallbackManagerForChainRun,
        kwargs: dict[str, Any],
    ) -> _RouteCall:
        """The selected route's call, nested under the router's run with the decision.

        The route is called as given — with whatever the caller bound to the
        router replayed on it here, by the route's own binder. The caller's config
        passes on unchanged but for its callbacks, and the decision rides in the route run's
        metadata.
        """
        route_config = patch_config(config, callbacks=run_manager.get_child())
        route_config["metadata"] = {
            **route_config.get("metadata", {}),
            ROUTING_KEY: decision.as_dict(),
        }
        route, call_kwargs = bound_route(self.routes[decision.route], kwargs)
        return _RouteCall(route, route_config, call_kwargs)


@dataclass(frozen=True)
class _StrategyCall:
    """A strategy about to decide one request."""

    strategy: RoutingStrategy
    name: str
    """What the strategy's run and the decision record call it."""
    request: RoutingRequest

    @property
    def inputs(self) -> dict[str, Any]:
        """The strategy run's inputs: what it decides on.

        The content blocks and transcript are left out — they are already on the router's run
        and the route's, and may hold large images.
        """
        return {
            "text": self.request.text,
            "modalities": sorted(self.request.modalities),
            "routes": list(self.request.routes),
            "tools_bound": self.request.tools_bound,
        }


class _RouteCall(NamedTuple):
    """What the selected route is called with."""

    route: Runnable[LanguageModelInput, AIMessage]
    config: RunnableConfig
    kwargs: dict[str, Any]


def _with_record(answer: AnswerT, decision: RoutingDecision) -> AnswerT:
    """The route's answer with the decision record added; nothing else changes.

    A copy, not an edit in place: the route may keep the object it returned — its response
    cache does — and must not find one call's record on another call's answer.

    Also publishes the decision for `last_routing_decision()`. Every entry point passes
    through here, and so does anything built on them — including the structured-output path
    whose parsed object has nowhere to carry a record, which is what `last_routing_decision()`
    exists for.
    """
    record_decision(decision)
    return cast("AnswerT", _recorded_on(answer, decision))


def _merge(output: Any, item: Any) -> Any:
    """Fold `item` into the running `output` the router keeps for its own run's outputs.

    Not what the caller sees, which is `item` itself, unchanged.

    `AIMessageChunk`s merge with `+`, and so do `include_raw=True`'s `AddableDict` deltas. A
    structured parser's own partial — a Pydantic object, or a plain dict from a JSON-schema or
    JSON-mode parse — is cumulative already, not a delta: no route's parser gives it a `+`, so
    the attempt raises `TypeError`, and the newest item replaces the running total instead of
    adding to it, since it already *is* the full state so far.
    """
    try:
        return output + item
    except TypeError:
        return item


def _recorded_on(answer: object, decision: RoutingDecision) -> object:
    """The record put wherever this answer can hold one.

    The same rule serves a whole answer and the one streamed item that carries it.

    A message holds it in `response_metadata`. `with_structured_output(include_raw=True)`
    answers with `{"raw", "parsed", "parsing_error"}`, and the raw message holds it there
    — including the first streamed item, an `AddableDict` delta that reliably has a
    `"raw"` key before any `"parsed"` key can (the parser needs raw content to parse). A parsed
    object, or a bare dict from a JSON-schema or JSON-mode parse, holds nothing and is handed
    back untouched: reaching it means `last_routing_decision()`, which `_with_record` has just
    published to — true of the whole answer and of a parsed stream's first partial alike.
    """
    if isinstance(answer, BaseMessage):
        return answer.model_copy(
            update={
                "response_metadata": {**answer.response_metadata, ROUTING_KEY: decision.as_dict()}
            }
        )
    if isinstance(answer, Mapping) and isinstance(answer.get("raw"), BaseMessage):
        return {**answer, "raw": _recorded_on(answer["raw"], decision)}
    return answer


def _run_outputs(answer: object, decision: RoutingDecision) -> dict[str, Any]:
    """The router run's outputs: the response, and the record on its own."""
    return {"output": answer, ROUTING_KEY: decision.as_dict()}


def _no_tool_capable_route(routes: Iterable[str]) -> str:
    """The message for a router none of whose routes can use tools."""
    return (
        f"no route can use tools: {_names(routes)}; binding tools or structured output needs "
        "at least one tool-capable route — tool_support_overrides can name one"
    )


def _run_ids(count: int, first: uuid.UUID | None) -> list[uuid.UUID]:
    """One router run id per prompt of a `generate`: the caller's for the first, if it gave one.

    The base `generate` names only the first prompt's run after `run_id` (`manager.py:1452`),
    and reports every run's id in `LLMResult.run`. `invoke` opens its own run, so the ids are
    chosen here and handed to it, which is what lets the result report them.
    """
    ids = [uuid.uuid4() for _ in range(count)]
    if first is not None and ids:
        ids[0] = first
    return ids


def _generate_config(
    callbacks: Callbacks,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    run_name: str | None,
    run_id: uuid.UUID,
) -> RunnableConfig:
    """The config a `generate` call's arguments amount to, for one prompt's `invoke`.

    `generate` has no `configurable`, `max_concurrency` or `recursion_limit` to give: a chat
    model's `generate` never took them, and `invoke` finds any that are set on the calling
    context through `ensure_config`.
    """
    config = RunnableConfig(run_id=run_id)
    if callbacks is not None:
        config["callbacks"] = callbacks
    if tags is not None:
        config["tags"] = tags
    if metadata is not None:
        config["metadata"] = metadata
    if run_name is not None:
        config["run_name"] = run_name
    return config


def _llm_result(answers: list[AIMessage], run_ids: list[uuid.UUID]) -> LLMResult:
    """One generation per prompt, in order; `run` is left unset for no prompts, as the base's is."""
    return LLMResult(
        generations=[[ChatGeneration(message=answer)] for answer in answers],
        llm_output={},
        run=[RunInfo(run_id=run_id) for run_id in run_ids] or None,
    )


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def _names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)
