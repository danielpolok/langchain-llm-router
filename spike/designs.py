"""T-004 candidate designs, scored against R1, R3 and C5 in `tests/test_cost_and_tracing.py`.

The naive design in `router.py` has both the router and the route emit an LLM run, so one
request is billed twice. Each candidate here breaks that a different way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, cast

from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableBinding, RunnableConfig, ensure_config
from langchain_core.runnables.config import patch_config
from langchain_core.utils._gateway import GATEWAY_METADATA_RESPONSE_KEY

from spike.router import ROUTING_KEY, RoutingDecision, SpikeRouterChatModel, _record

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForChainRun,
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForChainRun,
        CallbackManagerForLLMRun,
    )


class DelegatingRouterChatModel(SpikeRouterChatModel):
    """Candidate A — the router is a *chain* run; the route is the only model run.

    It overrides the public entry points rather than `_generate` / `_stream`. That is how
    LangChain's own dynamic model works: `_ConfigurableModel`, behind
    `init_chat_model(configurable_fields=...)`, resolves a model per call and delegates to it.

    The router opens a chain run whose metadata carries the decision, then hands the request to
    the route with that run's child callbacks. One LLM run exists — the route's — so usage is
    counted once (R3) and the real call is still in the trace, nested under a run that says
    which route was taken and why (C5). The response is the route's own (R1).
    """

    def _start_run(
        self,
        config: RunnableConfig,
        decision: RoutingDecision,
        messages: list[BaseMessage],
    ) -> CallbackManagerForChainRun:
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            {**(config.get("metadata") or {}), ROUTING_KEY: decision.as_dict()},
            self.metadata,
        )
        return callback_manager.on_chain_start(
            {"name": type(self).__name__},
            {"messages": messages, ROUTING_KEY: decision.as_dict()},
            name=config.get("run_name") or type(self).__name__,
        )

    async def _astart_run(
        self,
        config: RunnableConfig,
        decision: RoutingDecision,
        messages: list[BaseMessage],
    ) -> AsyncCallbackManagerForChainRun:
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            {**(config.get("metadata") or {}), ROUTING_KEY: decision.as_dict()},
            self.metadata,
        )
        return await callback_manager.on_chain_start(
            {"name": type(self).__name__},
            {"messages": messages, ROUTING_KEY: decision.as_dict()},
            name=config.get("run_name") or type(self).__name__,
        )

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()
        decision = self.decide(messages)
        target, call_kwargs = self._target(decision, kwargs)
        run_manager = self._start_run(config, decision, messages)
        try:
            message = target.invoke(
                messages,
                config=patch_config(config, callbacks=run_manager.get_child()),
                stop=stop,
                **call_kwargs,
            )
        except BaseException as error:
            run_manager.on_chain_error(error)
            raise
        _record(message, decision)
        run_manager.on_chain_end({"output": message})
        return message

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()
        decision = self.decide(messages)
        target, call_kwargs = self._target(decision, kwargs)
        run_manager = await self._astart_run(config, decision, messages)
        try:
            message = await target.ainvoke(
                messages,
                config=patch_config(config, callbacks=run_manager.get_child()),
                stop=stop,
                **call_kwargs,
            )
        except BaseException as error:
            await run_manager.on_chain_error(error)
            raise
        _record(message, decision)
        await run_manager.on_chain_end({"output": message})
        return message

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()
        decision = self.decide(messages)
        target, call_kwargs = self._target(decision, kwargs)
        run_manager = self._start_run(config, decision, messages)
        merged: AIMessageChunk | None = None
        first = True
        try:
            for message in target.stream(
                messages,
                config=patch_config(config, callbacks=run_manager.get_child()),
                stop=stop,
                **call_kwargs,
            ):
                chunk = cast("AIMessageChunk", message)
                if first:
                    _record(chunk, decision)
                    first = False
                merged = chunk if merged is None else merged + chunk
                yield chunk
        except BaseException as error:
            run_manager.on_chain_error(error)
            raise
        run_manager.on_chain_end({"output": merged})

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()
        decision = self.decide(messages)
        target, call_kwargs = self._target(decision, kwargs)
        run_manager = await self._astart_run(config, decision, messages)
        merged: AIMessageChunk | None = None
        first = True
        try:
            async for message in target.astream(
                messages,
                config=patch_config(config, callbacks=run_manager.get_child()),
                stop=stop,
                **call_kwargs,
            ):
                chunk = cast("AIMessageChunk", message)
                if first:
                    _record(chunk, decision)
                    first = False
                merged = chunk if merged is None else merged + chunk
                yield chunk
        except BaseException as error:
            await run_manager.on_chain_error(error)
            raise
        await run_manager.on_chain_end({"output": merged})


class RelabellingRouterChatModel(SpikeRouterChatModel):
    """Candidate B — one model run, the router's, wearing the selected model's identity.

    The route is called through its own `_generate_with_cache` with no run manager, so it emits
    no run at all. The result carries LangSmith *gateway metadata*, which the tracer promotes
    over the request-time identity (`tracers/core.py:_attach_gateway_metadata`) — the mechanism
    a gateway uses when it, too, only knows the real model once the response comes back.

    `_gen_info_and_msg_metadata` strips that key out of `generation_info` before it reaches
    `response_metadata`, so the caller's message is untouched by it (R1).
    """

    def _call_route(
        self,
        decision: RoutingDecision,
        messages: list[BaseMessage],
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> ChatResult:
        route = self.routes[decision.route]
        target, call_kwargs = self._target(decision, kwargs)
        bound_kwargs: dict[str, Any] = {}
        if isinstance(target, RunnableBinding):
            bound_kwargs = dict(target.kwargs)
        # No run manager, so the route opens no run of its own and there is no second
        # LLM run to bill. Reaching into `_generate_with_cache` keeps the route's cache,
        # rate limiter and response-metadata handling, which a bare `_generate` would skip.
        result = route._generate_with_cache(
            messages, stop=stop, run_manager=None, **bound_kwargs, **call_kwargs
        )
        identity = {
            **route._get_ls_params(),  # the identity the tracer should end up showing
            ROUTING_KEY: decision.as_dict(),
        }
        gateway = {
            "model": identity.get("ls_model_name"),
            "provider": identity.get("ls_provider"),
            ROUTING_KEY: decision.as_dict(),
        }
        for generation in result.generations:
            generation.generation_info = {
                **(generation.generation_info or {}),
                GATEWAY_METADATA_RESPONSE_KEY: gateway,
            }
            _record(cast("AIMessage", generation.message), decision)
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._call_route(self.decide(messages), messages, stop, kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._call_route(self.decide(messages), messages, stop, kwargs)
