"""`ChatRouter`: a chat model that hands each request to one of its named routes (C1, R5).

The router's own run is a *chain* run that delegates (PRD §11): it overrides the public entry
points, not `_generate` / `_stream`, so the selected route's call is the only model run outside
the strategy run — cost is counted once (R3) and the real call stays in the trace (C5).

Every entry point runs the same pipeline, in D9's order:

1. `_start_run` opens the router's chain run.
2. `_decide` / `_adecide` settle the route. `_plan` says whether there is anything to run —
   no strategy means the default route, with no strategy run — and otherwise the strategy
   runs in a child chain run of its own, with that run's child config as
   `RoutingRequest.config` and set as the context config. `_conclude` turns what the strategy
   did into a `RoutingDecision`, falling back to the default route when it can't (R9).
3. `_route_call` prepares the selected route's call: nested under the router's run, with the
   decision in its metadata.
4. `_with_record` adds the decision to the response (on exactly one chunk when streaming, D8),
   and the router's run closes with the record in its outputs. A route's error closes it as an
   error and propagates unchanged (C6).
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, TypeVar, cast

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config, patch_config
from langchain_core.runnables.config import set_config_context
from langchain_core.runnables.utils import coro_with_context
from pydantic import Field, field_validator, model_validator

from langchain_llm_router._extraction import build_request
from langchain_llm_router.decision import ROUTING_KEY, RoutingDecision
from langchain_llm_router.errors import FallbackWarning, RoutingError
from langchain_llm_router.strategy import (
    RoutingCallable,
    RoutingChoice,
    RoutingRequest,
    RoutingStrategy,
    as_strategy,
    strategy_name,
)

__all__ = ["ChatRouter"]

MessageT = TypeVar("MessageT", bound=BaseMessage)

_CALLER = 5
"""`stacklevel` that points a `FallbackWarning` at the code that called the router: the frames
are `_fallback` ← `_conclude` or `_plan` ← `_decide` / `_adecide` ← the entry point ← caller.

It counts *this* path, not every warning the router raises: a warning raised at another depth
(T-115's `ToolSupportWarning`, T-116's `ForcedRouteWarning`) needs its own count, and a test
that asserts where the warning points."""


class ChatRouter(BaseChatModel):
    """A chat model that picks one of several named chat models per request (C1, R5).

    The response is the selected route's own, with the routing decision added under
    `response_metadata["routing"]` (R1, R2). The routing policy is the application's: a
    strategy applies it, and when it can't decide the default route answers (R9).

    Because the router delegates to the route rather than generating itself, the settings that
    govern a call are the *route's*: its `cache`, `rate_limiter` and `disable_streaming` apply,
    and the router's own would never be consulted (the same reasoning as D4 gives for the
    cache, which T-119 rejects outright rather than letting it look effective).
    """

    routes: dict[str, BaseChatModel]
    """Named routes, any number of them; declaration order is significant (D1).

    Each is an ordinary chat model, used as given — the router never reconfigures or mutates
    one (REQ-R5-1). The names are the mapping's keys, so they are unique."""

    default_route: str
    """Mandatory (R9). Must name one of `routes`."""

    strategy: RoutingStrategy | RoutingCallable | None = Field(default=None, exclude=True)
    """None always uses the default route. A plain callable is coerced (REQ-R6-2).

    Kept out of serialization: a strategy is code, not configuration."""

    on_unavailable_forced_route: Literal["error", "fallback"] = "error"
    """R11: what happens when a forced route is unknown or can't use bound tools."""

    tool_support_overrides: dict[str, bool] = Field(default_factory=dict)
    """Per-route override of capability detection (D5); always wins."""

    @field_validator("routes")
    @classmethod
    def _check_routes(cls, routes: dict[str, BaseChatModel]) -> dict[str, BaseChatModel]:
        """At least one route, each with a name (REQ-R5-2)."""
        if not routes:
            msg = "routes is empty: a ChatRouter needs at least one route"
            raise RoutingError(msg)
        for name in routes:
            if not name.strip():
                msg = f"route name {name!r} is blank: every route needs a name"
                raise RoutingError(msg)
        return routes

    @field_validator("strategy", mode="before")
    @classmethod
    def _coerce_strategy(cls, strategy: object) -> RoutingStrategy | None:
        """One interface for every strategy (D6): a plain callable is wrapped (REQ-R6-2).

        Before the field's own type check, so `as_strategy` is the single gate on what may be
        a strategy and says so itself. What it rejects is a `TypeError`, which pydantic does
        not wrap — unlike the `RoutingError` a misnamed route raises.
        """
        return None if strategy is None else as_strategy(cast("RoutingCallable", strategy))

    @model_validator(mode="after")
    def _check_route_references(self) -> ChatRouter:
        """Every name that refers to a route names one (REQ-R5-2, REQ-R9-1)."""
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

    # --- Entry points: each runs the pipeline in the module docstring (D9). ---

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """The selected route's own answer, with the decision record added (R1, R2)."""
        config = ensure_config(config)
        # private API: `_convert_input` is how every chat model turns its input into messages.
        # Calling it keeps a string, a list of dicts, `BaseMessage`s and a `ChatPromptValue`
        # identical through the router (C2, C7, REQ-C7-2). `set_config_context` and
        # `coro_with_context`, used below, are public names their modules leave out of
        # `__all__`; D9 names both as how the strategy's run is entered.
        messages = self._convert_input(input).to_messages()
        run_manager = self._start_run(config, messages)
        try:
            decision = self._decide(messages, config, run_manager, kwargs)
            call = self._route_call(decision, config, run_manager, kwargs)
            message = call.route.invoke(messages, call.config, stop=stop, **call.kwargs)
        except BaseException as error:
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
        """Async `invoke`: the strategy is awaited too, so neither blocks the loop (C2)."""
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()  # private API: see `invoke`
        run_manager = await self._astart_run(config, messages)
        try:
            decision = await self._adecide(messages, config, run_manager, kwargs)
            call = self._route_call(decision, config, run_manager, kwargs)
            message = await call.route.ainvoke(messages, call.config, stop=stop, **call.kwargs)
        except BaseException as error:
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
        """The route's chunks, passed through unchanged bar the record on one of them (D8)."""
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()  # private API: see `invoke`
        run_manager = self._start_run(config, messages)
        output: AIMessageChunk | None = None
        try:
            decision = self._decide(messages, config, run_manager, kwargs)
            call = self._route_call(decision, config, run_manager, kwargs)
            for message in call.route.stream(messages, call.config, stop=stop, **call.kwargs):
                chunk = cast("AIMessageChunk", message)
                if output is None:
                    # Decided before the first chunk; only that chunk carries the record (D8).
                    chunk = output = _with_record(chunk, decision)
                else:
                    output += chunk
                yield chunk
        except BaseException as error:
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
        """Async `stream`."""
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()  # private API: see `invoke`
        run_manager = await self._astart_run(config, messages)
        output: AIMessageChunk | None = None
        try:
            decision = await self._adecide(messages, config, run_manager, kwargs)
            call = self._route_call(decision, config, run_manager, kwargs)
            async for message in call.route.astream(
                messages, call.config, stop=stop, **call.kwargs
            ):
                chunk = cast("AIMessageChunk", message)
                if output is None:
                    # Decided before the first chunk; only that chunk carries the record (D8).
                    chunk = output = _with_record(chunk, decision)
                else:
                    output += chunk
                yield chunk
        except BaseException as error:
            await run_manager.on_chain_error(error)
            raise
        await run_manager.on_chain_end(_run_outputs(output, decision))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Not routed: the base `generate` path would open a model run of the router's own.

        That run and the route's would bill the same tokens twice, silently (R3, spike
        surprise 3), so this refuses rather than miscount. Routing `generate()` and
        `agenerate()` the way `invoke` does is REQ-C2-2.
        """
        msg = (
            "ChatRouter routes invoke, ainvoke, stream, astream and batch; "
            "generate() and agenerate() are not supported yet"
        )
        raise NotImplementedError(msg)

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

        No strategy means the default route and no strategy run (D9). A request with no user
        message gives a strategy nothing to decide on, so it is not consulted and the default
        route answers (R9, REQ-R4-4).
        """
        # `_coerce_strategy` has wrapped any plain callable (REQ-R6-2).
        strategy = cast("RoutingStrategy | None", self.strategy)
        if strategy is None:
            return RoutingDecision(route=self.default_route, reason="no strategy configured")
        request = build_request(
            messages,
            routes=tuple(self.routes),
            tools_bound=bool(kwargs.get("tools")),
            wants_full_context=strategy.wants_full_context,
            # Replaced by the strategy run's child config once that run is open (D9).
            config=config,
        )
        if request is None:
            return self._fallback("the request has no user message to route on", strategy=None)
        return _StrategyCall(strategy, strategy_name(strategy), request)

    def _decide(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
        run_manager: CallbackManagerForChainRun,
        kwargs: dict[str, Any],
    ) -> RoutingDecision:
        """Settle the route; the strategy runs in a child run of the router's run (D9).

        Any call the strategy makes with `request.config` — or without it, through the context
        config — nests under the strategy's run, so it is traced and costed there (R3).
        """
        pending = self._plan(messages, config, kwargs)
        if isinstance(pending, RoutingDecision):
            return pending
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
            return self._conclude(pending, error)
        try:
            decision = self._conclude(pending, choice)
        except BaseException as error:
            # `_conclude` warns, and an application may have escalated that warning to an
            # error; the strategy's run has to close either way.
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
        """Async `_decide`: awaits `adecide`, so the strategy never blocks the loop (C2)."""
        pending = self._plan(messages, config, kwargs)
        if isinstance(pending, RoutingDecision):
            return pending
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
            return self._conclude(pending, error)
        try:
            decision = self._conclude(pending, choice)
        except BaseException as error:
            # As in `_decide`: a warning escalated to an error must still close the run.
            await strategy_run.on_chain_error(error)
            raise
        await strategy_run.on_chain_end(decision.as_dict())
        return decision

    def _conclude(self, pending: _StrategyCall, outcome: object) -> RoutingDecision:
        """The decision a strategy's outcome leads to (R2).

        Every way a strategy can fail to decide — abstaining, raising, naming a route that
        doesn't exist, or returning something that isn't a choice — ends on the default route
        (R9). A raised error's traceback stays on the strategy's run in the trace.
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
        """The default route, with one `FallbackWarning` and the cause recorded (R9, R2)."""
        warnings.warn(
            f"{cause}; falling back to the default route {self.default_route!r}",
            FallbackWarning,
            stacklevel=_CALLER,
        )
        return RoutingDecision(
            route=self.default_route,
            reason=f"{cause}; fell back to the default route",
            strategy=strategy,
            fallback=True,
        )

    # --- 3. The route's call ---

    def _route_call(
        self,
        decision: RoutingDecision,
        config: RunnableConfig,
        run_manager: CallbackManagerForChainRun | AsyncCallbackManagerForChainRun,
        kwargs: dict[str, Any],
    ) -> _RouteCall:
        """The selected route's call, nested under the router's run with the decision (D9).

        The route is called as given (REQ-R5-1). The caller's config passes on unchanged but
        for its callbacks, and the decision rides in the route run's metadata.
        """
        route_config = patch_config(config, callbacks=run_manager.get_child())
        route_config["metadata"] = {
            **route_config.get("metadata", {}),
            ROUTING_KEY: decision.as_dict(),
        }
        return _RouteCall(self.routes[decision.route], route_config, dict(kwargs))


@dataclass(frozen=True)
class _StrategyCall:
    """A strategy about to decide one request."""

    strategy: RoutingStrategy
    name: str
    """What the strategy's run and the decision record call it (D8, D9)."""
    request: RoutingRequest

    @property
    def inputs(self) -> dict[str, Any]:
        """The strategy run's inputs: what it decides on (D9).

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


def _with_record(message: MessageT, decision: RoutingDecision) -> MessageT:
    """The route's message with the decision record added; nothing else changes (R1, R2).

    A copy, not an edit in place: the route may keep the object it returned — its response
    cache does — and must not find one call's record on another call's answer (C10).
    """
    return message.model_copy(
        update={"response_metadata": {**message.response_metadata, ROUTING_KEY: decision.as_dict()}}
    )


def _run_outputs(message: BaseMessage | None, decision: RoutingDecision) -> dict[str, Any]:
    """The router run's outputs: the response, and the record on its own (D9)."""
    return {"output": message, ROUTING_KEY: decision.as_dict()}


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def _names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)
