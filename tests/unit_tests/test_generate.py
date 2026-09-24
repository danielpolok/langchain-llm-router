"""`generate()` and `agenerate()` route, record and cost exactly as `invoke` does.

The spike's design overrode only the public entry points a call *usually* takes, so these two
went down the base class's path: a model run of the router's own, opened around a `_generate`
that delegates — the same tokens billed twice, with nothing to say so. The
router now answers them by running `invoke` / `ainvoke` once per prompt, and this is what that
has to keep from the base `generate` (`chat_models.py:1592`): the prompts and their runs, the
arguments that name and tag those runs, `stop`, the `run` list on the result — and what it
knowingly does not (`ChatRouter.generate` says which).

Where the trace and the bill are concerned, `test_tracing.py` and `test_cost.py` parametrise over
these two along with every other convention; what is here is what only `generate` has: several
prompts to a call, and a result that is more than a message.
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from typing import Any, Literal, cast

import pytest
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.tracers.schemas import Run
from pydantic import ConfigDict, Field

from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    last_routing_decision,
    routing_decision,
)
from langchain_llm_router.decision import ROUTING_KEY
from tests.conventions import generated, prompts, respond
from tests.fakes import FakeChatModel, call_log
from tests.tracing import PricedCall, StartLog, model_runs, priced_calls

Generating = Literal["generate", "agenerate"]

GENERATING: tuple[Generating, ...] = ("generate", "agenerate")


class ByText(RoutingStrategy):
    """Routes to the route the request's text names."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route=request.text, reason=f"the request named {request.text!r}")


class RouteDown(Exception):
    """What a route's provider raises when it can't answer."""


FAILURE = RouteDown("the provider is down", 503)
DELAYED_FAILURE = RouteDown("the provider is slow and down", 504)

_MOMENT = 0.05
"""Seconds a `Slow` route takes. Only ever a floor under how long an answer takes, so it can
make a passing test slower and never make one fail: what it separates is a caller that waits
for the answers from one that does not."""

REFUSED = "no model call of its own"
"""What the router's `_generate` says when a call reaches it by going around the entry points."""


class Failing(FakeChatModel):
    """A route whose provider is down, and which counts the calls it took."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(dict(kwargs))
        raise FAILURE


class DelayedFailure(FakeChatModel):
    """A route whose provider is also down, but only finds out after a moment -- so a caller
    that raced completion order instead of prompt order would surface the *other* failure."""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(_MOMENT)
        raise DELAYED_FAILURE


class Slow(FakeChatModel):
    """A route that answers a moment after it is asked, so a call that fails meanwhile shows up
    before it — and a caller that gives up at the first failure shows up too."""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(_MOMENT)
        return self._generate(messages, stop, None, **kwargs)


class Watching(FakeChatModel):
    """A route that keeps what each call handed it: the last message's text, `stop`, kwargs."""

    seen: list[tuple[str, list[str] | None, dict[str, Any]]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append((messages[-1].text, stop, dict(kwargs)))
        return super()._generate(messages, stop, run_manager, **kwargs)


class Meeting(FakeChatModel):
    """A route whose async answer waits until `width` calls are waiting together."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    width: int
    opened: asyncio.Event = Field(default_factory=asyncio.Event)
    arrived: list[int] = Field(default_factory=list)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.arrived.append(1)
        if len(self.arrived) >= self.width:
            self.opened.set()
        await self.opened.wait()
        return self._generate(messages, stop, None, **kwargs)


def fake_routes(*names: str) -> dict[str, BaseChatModel]:
    """One fake route per name, each answering with its own name."""
    return {
        name: FakeChatModel(model_name=f"model-{name}", reply=f"{name} answer") for name in names
    }


def record_for(route: str) -> dict[str, Any]:
    """The record `ByText` produces for `route`."""
    return RoutingDecision(
        route=route, reason=f"the request named {route!r}", strategy="ByText"
    ).as_dict()


def shape(run: Run) -> tuple[str, str, list[Any]]:
    """A run and everything under it, as (type, name, children) — what a trace looks like."""
    return (run.run_type, run.name, [shape(child) for child in run.child_runs])


def chat_generations(result: LLMResult) -> list[list[ChatGeneration]]:
    """The candidates of each prompt, as chat generations — which is all a chat model makes."""
    answered = []
    for candidates in result.generations:
        assert all(isinstance(generation, ChatGeneration) for generation in candidates)
        answered.append(cast("list[ChatGeneration]", candidates))
    return answered


def only_generations(result: LLMResult) -> list[ChatGeneration]:
    """The one generation each prompt was answered with."""
    answered = chat_generations(result)
    assert [len(candidates) for candidates in answered] == [1] * len(answered)
    return [candidates[0] for candidates in answered]


def only_generation(result: LLMResult) -> ChatGeneration:
    """The one generation an `LLMResult` holds."""
    [[generation]] = chat_generations(result)
    return generation


# --- The same call as invoke ---


@pytest.mark.parametrize("convention", GENERATING)
async def test_generate_routes_records_and_costs_exactly_as_invoke_does(
    convention: Generating,
) -> None:
    """The same request through `invoke` and through `generate` gives the same
    decision record, the same usage totals, and the same trace — and no second LLM run, which
    is what the base `generate` path would have added."""
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=ByText()
    )
    invoked_starts, invoked_trace = StartLog(), RunCollectorCallbackHandler()
    generated_starts, generated_trace = StartLog(), RunCollectorCallbackHandler()

    with get_usage_metadata_callback() as invoked_usage:
        invoked = router.invoke("frontier", {"callbacks": [invoked_starts, invoked_trace]})
    with get_usage_metadata_callback() as generated_usage:
        answered = await respond(
            router, convention, "frontier", {"callbacks": [generated_starts, generated_trace]}
        )

    usage = invoked.usage_metadata
    assert usage is not None
    assert routing_decision(answered) == routing_decision(invoked)
    assert routing_decision(answered) == RoutingDecision.from_dict(record_for("frontier"))
    assert answered.usage_metadata == usage
    assert (
        generated_usage.usage_metadata == invoked_usage.usage_metadata == {"model-frontier": usage}
    )

    assert len(generated_starts.chat_models) == len(invoked_starts.chat_models) == 1
    assert generated_starts.llms == []
    assert len(model_runs(generated_trace.traced_runs)) == 1
    assert priced_calls(generated_trace.traced_runs) == priced_calls(invoked_trace.traced_runs)
    assert priced_calls(generated_trace.traced_runs) == [PricedCall("model-frontier", usage)]
    assert [shape(run) for run in generated_trace.traced_runs] == [
        shape(run) for run in invoked_trace.traced_runs
    ]


@pytest.mark.parametrize("convention", GENERATING)
async def test_a_generation_is_the_route_s_message_with_only_the_record_added(
    convention: Generating,
) -> None:
    """The answer is the route's own on the `generate` path: the message in the generation equals
    the one the route itself puts there — content, usage, tool calls, metadata — with `routing` the
    sole addition. (`id` names the route's own run, so it differs between two calls by design.)"""
    router = ChatRouter(routes=fake_routes("cheap", "frontier"), default_route="frontier")
    direct = fake_routes("frontier")["frontier"]

    through = only_generation(await generated(router, convention, prompts("hello")))
    straight = only_generation(await generated(direct, convention, prompts("hello")))

    fields = through.message.model_dump(exclude={"id"})
    record = fields["response_metadata"].pop(ROUTING_KEY)
    assert record == RoutingDecision(route="frontier", reason="no strategy configured").as_dict()
    assert fields == straight.message.model_dump(exclude={"id"})
    assert through.text == straight.text == "frontier answer"
    assert through.generation_info == straight.generation_info


@pytest.mark.parametrize("convention", GENERATING)
async def test_a_strategy_that_abstains_warns_once_per_prompt_and_records_the_fallback(
    convention: Generating,
) -> None:
    """On the `generate` path each prompt is decided on its own, so each
    that the strategy cannot decide falls back with one warning and a record that says so."""
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"),
        default_route="cheap",
        strategy=lambda request: None,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await generated(router, convention, prompts("one", "two"))

    routing = [warning for warning in caught if issubclass(warning.category, RoutingWarning)]
    assert [warning.category for warning in routing] == [FallbackWarning] * 2
    records = [routing_decision(generation.message) for generation in only_generations(result)]
    assert [record and (record.route, record.fallback) for record in records] == [
        ("cheap", True),
        ("cheap", True),
    ]


def test_generate_publishes_the_decision_for_last_routing_decision() -> None:
    """Every routed call publishes its record, and `generate` is one — after a call over
    several prompts the last one decided is what `last_routing_decision()` reads."""
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=ByText()
    )

    router.generate(prompts("frontier", "cheap"))

    assert last_routing_decision() == RoutingDecision.from_dict(record_for("cheap"))


# --- Several prompts to a call ---


@pytest.mark.parametrize("convention", GENERATING)
async def test_each_prompt_gets_its_own_decision_run_and_generation(
    convention: Generating,
) -> None:
    """A prompt is routed on its own text, so three prompts to two routes are three
    decisions, three router runs and three generations — in the order the prompts came, with
    `LLMResult.run` naming each prompt's router run, the top-level run the caller called."""
    routes = fake_routes("cheap", "frontier")
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())
    collector = RunCollectorCallbackHandler()

    result = await generated(
        router, convention, prompts("frontier", "cheap", "frontier"), {"callbacks": [collector]}
    )

    generations = only_generations(result)
    assert [generation.text for generation in generations] == [
        "frontier answer",
        "cheap answer",
        "frontier answer",
    ]
    assert [routing_decision(generation.message) for generation in generations] == [
        RoutingDecision.from_dict(record_for(name)) for name in ("frontier", "cheap", "frontier")
    ]
    assert [len(call_log(route)) for route in routes.values()] == [1, 2]

    assert result.run is not None
    runs = {run.id: run for run in collector.traced_runs}
    assert len(runs) == len({info.run_id for info in result.run}) == 3
    for generation, info in zip(generations, result.run, strict=True):
        run = runs[info.run_id]
        assert (run.run_type, run.name, run.parent_run_id) == ("chain", "ChatRouter", None)
        assert (run.outputs or {})[ROUTING_KEY] == generation.message.response_metadata[ROUTING_KEY]
        assert [child.run_type for child in run.child_runs] == ["chain", "llm"]
    assert result.llm_output == {}


@pytest.mark.parametrize("convention", GENERATING)
async def test_the_arguments_that_name_and_tag_a_run_reach_every_prompt_s_run(
    convention: Generating,
) -> None:
    """`callbacks`, `tags`, `metadata` and `run_name` apply to every prompt's run, as
    the base `generate` applies them to each of its model runs, and `run_id` names the first
    prompt's alone. The route's own runs get the tags and metadata — and their own name, as they
    do under `invoke`."""
    router = ChatRouter(routes=fake_routes("cheap", "frontier"), default_route="cheap")
    collector = RunCollectorCallbackHandler()
    run_id = uuid.uuid4()

    result = await generated(
        router,
        convention,
        prompts("one", "two"),
        {
            "callbacks": [collector],
            "tags": ["experiment"],
            "metadata": {"user": "ada"},
            "run_name": "routed generate",
            "run_id": run_id,
        },
    )

    assert result.run is not None
    assert result.run[0].run_id == run_id
    assert result.run[1].run_id != run_id
    assert {run.id for run in collector.traced_runs} == {info.run_id for info in result.run}
    for router_run in collector.traced_runs:
        (route_run,) = model_runs([router_run])
        assert router_run.name == "routed generate"
        assert route_run.name == "FakeChatModel"
        for run in (router_run, route_run):
            assert "experiment" in (run.tags or [])
            assert (run.extra or {})["metadata"]["user"] == "ada"


@pytest.mark.parametrize("convention", GENERATING)
async def test_stop_and_keyword_arguments_reach_the_route(convention: Generating) -> None:
    """`stop` and the caller's keyword arguments go to the route as they do
    from `invoke`, and the router adds none. The prompts reach it in the order they came."""
    route = Watching(model_name="model-only", reply="an answer")
    router = ChatRouter(routes={"only": route}, default_route="only")

    await generated(router, convention, prompts("first", "second"), stop=["END"], temperature=0.1)

    seen = sorted(route.seen) if convention == "agenerate" else route.seen
    assert seen == [
        ("first", ["END"], {"temperature": 0.1}),
        ("second", ["END"], {"temperature": 0.1}),
    ]


async def test_agenerate_runs_its_prompts_at_the_same_time() -> None:
    """As the base `agenerate` does, this overlaps its prompts — the route only answers once
    three of them are waiting on it, so prompts awaited one after another would never get past
    the first."""
    route = Meeting(model_name="model-only", reply="an answer", width=3)
    router = ChatRouter(routes={"only": route}, default_route="only")

    result = await asyncio.wait_for(router.agenerate(prompts("one", "two", "three")), timeout=10.0)

    assert [generation.text for generation in only_generations(result)] == ["an answer"] * 3
    assert len(route.arrived) == 3


@pytest.mark.parametrize("convention", GENERATING)
async def test_an_empty_generate_answers_as_a_chat_model_does(convention: Generating) -> None:
    """No prompts, no runs — the same empty result a chat model gives, `run` unset."""
    router = ChatRouter(routes=fake_routes("only"), default_route="only")
    bare = fake_routes("only")["only"]

    assert await generated(router, convention, []) == await generated(bare, convention, [])
    assert (await generated(router, convention, [])).run is None


# --- generate_prompt comes with it ---


async def test_generate_prompt_and_agenerate_prompt_are_routed_too() -> None:
    """The base class builds `generate_prompt` on `generate`, so the router's answers
    it with no code of its own — through the same routed pipeline, one model run each."""
    router = ChatRouter(
        routes=fake_routes("cheap", "frontier"), default_route="cheap", strategy=ByText()
    )
    for call in ("sync", "async"):
        starts = StartLog()
        values: list[PromptValue] = [StringPromptValue(text="frontier")]

        if call == "sync":
            result = router.generate_prompt(values, callbacks=[starts])
        else:
            result = await router.agenerate_prompt(values, callbacks=[starts])

        assert routing_decision(only_generation(result).message) == RoutingDecision.from_dict(
            record_for("frontier")
        )
        assert (len(starts.chat_models), len(starts.chains)) == (1, 2)


# --- Errors stay the route's ---


def test_a_route_s_error_stops_generate_and_surfaces_unchanged() -> None:
    """The route's own exception is what the caller gets, the failed prompt's router run
    closes as an error, and later prompts are never started — `generate` runs them in order."""
    routes: dict[str, BaseChatModel] = {"good": FakeChatModel(reply="good"), "bad": Failing()}
    router = ChatRouter(routes=routes, default_route="good", strategy=ByText())
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RouteDown) as raised:
            router.generate(prompts("good", "bad", "good"), callbacks=[collector])

    assert raised.value is FAILURE
    assert [warning for warning in caught if issubclass(warning.category, RoutingWarning)] == []
    good, bad = collector.traced_runs
    assert (good.error, (good.outputs or {})[ROUTING_KEY]) == (None, record_for("good"))
    assert bad.error is not None
    assert (len(call_log(routes["good"])), len(call_log(routes["bad"]))) == (1, 1)


async def test_agenerate_finishes_every_prompt_and_then_raises_the_route_s_error() -> None:
    """The prompts of an `agenerate` are in flight together, so the ones that can still answer
    do — their calls are paid for — and every router run closes, the failed one as an error. The
    route's own exception is then raised, unchanged."""
    routes: dict[str, BaseChatModel] = {"good": Slow(reply="good"), "bad": Failing()}
    router = ChatRouter(routes=routes, default_route="good", strategy=ByText())
    collector = RunCollectorCallbackHandler()

    with pytest.raises(RouteDown) as raised:
        await router.agenerate(prompts("good", "bad", "good"), callbacks=[collector])

    assert raised.value is FAILURE
    assert len(collector.traced_runs) == 3
    assert sorted(run.error is not None for run in collector.traced_runs) == [False, False, True]
    assert all(run.end_time is not None for run in collector.traced_runs)
    assert (len(call_log(routes["good"])), len(call_log(routes["bad"]))) == (2, 1)


async def test_agenerate_raises_the_first_prompt_s_failure_not_the_first_to_finish() -> None:
    """ "the first, in prompt order" (the docstring's own words) is not the same as "the first
    to finish" -- two prompts fail for different reasons, the *second* prompt's route fails
    immediately and the *first* prompt's route only after a moment, and `agenerate` still raises
    the first prompt's own exception. `asyncio.gather` keeps `outcomes` in submission order
    regardless of completion order, so this is what makes that true rather than incidental."""
    routes: dict[str, BaseChatModel] = {"slow_bad": DelayedFailure(), "fast_bad": Failing()}
    router = ChatRouter(routes=routes, default_route="slow_bad", strategy=ByText())

    with pytest.raises(RouteDown) as raised:
        await router.agenerate(prompts("slow_bad", "fast_bad"))

    assert raised.value is DELAYED_FAILURE


# --- The refusal that remains ---


@pytest.mark.parametrize("convention", GENERATING)
async def test_a_call_that_goes_around_the_routed_entry_points_is_refused_not_double_counted(
    convention: Generating,
) -> None:
    """The base class's own `generate` opens a model run for the router and calls `_generate`
    inside it. `_generate` refuses, so that path fails where someone sees it instead of billing
    the route's tokens a second time under the router's name — and no route is called."""
    routes = fake_routes("cheap", "frontier")
    router = ChatRouter(routes=routes, default_route="cheap", strategy=ByText())
    collector = RunCollectorCallbackHandler()

    if convention == "generate":
        with pytest.raises(NotImplementedError, match=REFUSED):
            BaseChatModel.generate(router, prompts("frontier"), callbacks=[collector])
    else:
        with pytest.raises(NotImplementedError, match=REFUSED):
            await BaseChatModel.agenerate(router, prompts("frontier"), callbacks=[collector])

    assert [len(call_log(route)) for route in routes.values()] == [0, 0]
    assert priced_calls(collector.traced_runs) == []
