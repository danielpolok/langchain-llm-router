"""T-119: response-cache correctness — the route owns caching, not the router (C10, D4).

`ChatRouter` delegates from `invoke` / `ainvoke` to the selected route's own `invoke` /
`ainvoke` rather than running `_generate_with_cache` itself (the module docstring's pipeline in
`router.py`), so LangChain's response cache applies structurally at the *route's* level: a
route's own `cache=` field, or the process-global cache `set_llm_cache` installs, is consulted
exactly where it would be on that route called bare — keyed by `_get_llm_string`
(`chat_models.py:1578`), the route's own serialized identity plus its call kwargs, looked up
before `_generate` runs (`:1899`). This module proves that structural claim empirically rather
than building any caching mechanism of its own (REQ-C10-1), and closes the two gaps D4 leaves
open: the router's own `cache=` must not silently no-op (REQ-C10-2), and a bound tool's
converted form must not carry a process-unstable repr into the route's own cache key
(REQ-C10-3). REQ-C10-4 checks that a cache hit still carries *this* call's decision record, not
whatever decision filled the cache.

`stream()` / `astream()` are not exercised here: `BaseChatModel.stream` calls `_stream` directly
and never reaches `_generate_with_cache` at all (`chat_models.py:727`, confirmed by reading the
installed source) — true of any chat model, not something the router changes, so there is
nothing route-cache-shaped to prove on that path.

`tests/fakes.py`'s routes now override `_identifying_params` with their own `model_name`
(T-119): the base fake, like the base `BaseChatModel`, defaults `_identifying_params` to `{}`
and is not `is_lc_serializable`, so two same-shaped fakes would otherwise be indistinguishable
to a cache lookup where two real provider routes never are (a real provider's `model` rides
along through its own `_identifying_params`, or through `is_lc_serializable`'s full
`dumpd(self)`).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import tool
from pydantic import ValidationError

from langchain_llm_router import ChatRouter, RoutingError
from langchain_llm_router.strategies.keyword import KeywordStrategy
from tests.fakes import FakeChatModel, ToolCallingFakeChatModel, call_log


@pytest.fixture(autouse=True)
def _clean_global_cache() -> Iterator[None]:
    """No test leaks a process-global cache to the next one (D4's `set_llm_cache` case).

    `langchain_core.globals` holds it in a module-level variable that pytest does not reset on
    its own, and it is the one piece of process-wide state this module's tests touch.
    """
    set_llm_cache(None)
    yield
    set_llm_cache(None)


def routing(message: AIMessage) -> dict[str, object]:
    """The decision record an answer carries, typed loosely — `response_metadata` is a dict."""
    return dict(message.response_metadata["routing"])


# --- REQ-C10-2: the router's own `cache=` is rejected, not silently ignored (D4) ---


def test_the_routers_own_cache_is_rejected_at_construction() -> None:
    """REQ-C10-2: `ChatRouter(cache=...)` raises, and the message points at per-route caching."""
    with pytest.raises(ValidationError) as caught:
        ChatRouter(routes={"cheap": FakeChatModel()}, default_route="cheap", cache=InMemoryCache())
    [error] = caught.value.errors()
    assert error["loc"] == ("cache",)
    reason = error.get("ctx", {}).get("error", error["type"])
    assert isinstance(reason, RoutingError)
    assert str(reason) == (
        "cache belongs on the route, not the router: ChatRouter delegates to each "
        "route's own invoke/stream rather than running _generate_with_cache itself, so its "
        "own cache would never be consulted — set cache= on each route instead."
    )


@pytest.mark.parametrize("cache", [True, False], ids=["true", "false"])
def test_the_routers_own_cache_is_rejected_for_the_boolean_forms_too(cache: bool) -> None:
    """REQ-C10-2: `cache=True` / `cache=False` would no-op exactly as an explicit cache would —
    `BaseChatModel.cache` accepts all three, so all three are rejected the same way."""
    with pytest.raises(ValidationError):
        ChatRouter(routes={"cheap": FakeChatModel()}, default_route="cheap", cache=cache)


def test_leaving_cache_unset_builds_fine() -> None:
    """The field's own default (`None`) is not what REQ-C10-2 rejects — only an explicit value
    is: pydantic does not validate an unset field's default (`validate_default` is off), and
    `None` on this field means "not set" either way, so a router built without `cache=` at all
    never reaches the validator."""
    router = ChatRouter(routes={"cheap": FakeChatModel()}, default_route="cheap")
    assert router.cache is None


# --- REQ-C10-1: a hit for the identical request, a miss for anything that changes the route ---


def test_an_identical_request_is_a_hit_under_a_global_cache() -> None:
    """REQ-C10-1: `set_llm_cache` (D4's global case) makes the second identical call skip the
    route's own `_generate` — the fake's call log stays at one entry."""
    set_llm_cache(InMemoryCache())
    cheap = FakeChatModel(model_name="cheap-1", reply="cheap answer")
    router = ChatRouter(routes={"cheap": cheap}, default_route="cheap")

    router.invoke("hello")
    router.invoke("hello")

    assert len(call_log(cheap)) == 1


async def test_an_identical_request_is_a_hit_under_a_global_cache_async() -> None:
    """REQ-C10-1, async: `ainvoke` goes through `_agenerate_with_cache` the same way."""
    set_llm_cache(InMemoryCache())
    cheap = FakeChatModel(model_name="cheap-1", reply="cheap answer")
    router = ChatRouter(routes={"cheap": cheap}, default_route="cheap")

    await router.ainvoke("hello")
    await router.ainvoke("hello")

    assert len(call_log(cheap)) == 1


def test_an_identical_request_is_a_hit_under_a_routes_own_cache() -> None:
    """REQ-C10-1, D4: a route's own `cache=` field is consulted exactly as `set_llm_cache`'s
    global cache is — D4's other in-scope case, no `set_llm_cache` involved at all."""
    cheap = FakeChatModel(model_name="cheap-1", reply="cheap answer", cache=InMemoryCache())
    router = ChatRouter(routes={"cheap": cheap}, default_route="cheap")

    router.invoke("hello")
    router.invoke("hello")

    assert len(call_log(cheap)) == 1


def test_forcing_the_same_prompt_to_a_different_route_is_a_miss() -> None:
    """REQ-C10-1: forced via `config={"configurable": {"route": ...}}` (T-116) is the clean way
    to change which route answers without a non-deterministic strategy. One shared global cache
    — so a false hit across routes would show up if the route weren't part of the key."""
    set_llm_cache(InMemoryCache())
    cheap = FakeChatModel(model_name="cheap-1", reply="cheap answer")
    frontier = FakeChatModel(model_name="frontier-1", reply="frontier answer")
    router = ChatRouter(routes={"cheap": cheap, "frontier": frontier}, default_route="cheap")

    router.invoke("hello")
    router.invoke("hello", config={"configurable": {"route": "frontier"}})

    assert len(call_log(cheap)) == 1
    assert len(call_log(frontier)) == 1


def test_a_changed_strategy_configuration_is_a_miss() -> None:
    """REQ-C10-1: two routers over the same routes and cache, differing only in which route
    their `KeywordStrategy` sends the same word to — a real (not forced) decision that changes
    under a changed configuration, exactly the case the requirement names."""
    set_llm_cache(InMemoryCache())
    cheap = FakeChatModel(model_name="cheap-1", reply="cheap answer")
    frontier = FakeChatModel(model_name="frontier-1", reply="frontier answer")
    routes: dict[str, BaseChatModel] = {"cheap": cheap, "frontier": frontier}
    to_frontier = ChatRouter(
        routes=routes, default_route="cheap", strategy=KeywordStrategy({"frontier": ["urgent"]})
    )
    to_cheap = ChatRouter(
        routes=routes, default_route="frontier", strategy=KeywordStrategy({"cheap": ["urgent"]})
    )

    to_frontier.invoke("urgent request")
    to_cheap.invoke("urgent request")

    assert len(call_log(frontier)) == 1
    assert len(call_log(cheap)) == 1


def test_different_bound_tools_is_a_miss() -> None:
    """REQ-C10-1: the route's own cache key includes its call kwargs, and a replayed tool
    binding is one (D4) — two different tool sets bound to the same router and route miss each
    other, and re-binding the first set again is a hit."""
    set_llm_cache(InMemoryCache())
    cheap = ToolCallingFakeChatModel(model_name="cheap-1", reply="cheap answer")
    router = ChatRouter(routes={"cheap": cheap}, default_route="cheap")

    @tool
    def get_weather(city: str) -> str:
        """Look up the weather in a city."""
        return "sunny"

    @tool
    def get_time(city: str) -> str:
        """Look up the time in a city."""
        return "noon"

    router.bind_tools([get_weather]).invoke("what's it like out?")
    router.bind_tools([get_time]).invoke("what's it like out?")
    router.bind_tools([get_weather]).invoke("what's it like out?")

    assert len(call_log(cheap)) == 2


# --- REQ-C10-3: a bound tool contributes nothing process-unstable to the cache key ---


def test_the_same_tools_built_twice_are_still_a_hit() -> None:
    """REQ-C10-3, the standard trick for "equal across two processes" inside one test process:
    two structurally-identical-but-distinct `@tool` objects (a fresh `StructuredTool` each
    time, exactly as a second process re-importing the same `@tool`-decorated function would
    build one) bind to an equal cache key — if a raw object identity or memory address leaked
    into it, this would be a miss every time, the spike's caveat."""
    set_llm_cache(InMemoryCache())
    cheap = ToolCallingFakeChatModel(model_name="cheap-1", reply="cheap answer")
    router = ChatRouter(routes={"cheap": cheap}, default_route="cheap")

    @tool
    def get_weather(city: str) -> str:
        """Look up the weather in a city."""
        return "sunny"

    first_definition = get_weather

    @tool  # type: ignore[no-redef]  # the second "process"'s own definition
    def get_weather(city: str) -> str:
        """Look up the weather in a city."""
        return "sunny"

    second_definition = get_weather
    assert first_definition is not second_definition

    router.bind_tools([first_definition]).invoke("what's it like out?")
    router.bind_tools([second_definition]).invoke("what's it like out?")

    assert len(call_log(cheap)) == 1


def test_a_plain_callable_tool_converts_to_something_with_no_object_repr_in_it() -> None:
    """REQ-C10-3: a bare function bound as a tool renders as `<function f at 0x...>` by
    default, which would change every process and miss every time (spike caveat) — but that
    repr never gets near the cache key. `bind_tools` on the *route itself* (public API, the
    same conversion `_tools.bound_route` replays) is what actually runs at call time; its
    result is what `_get_llm_string` folds in, and it is a plain list of dicts, not the raw
    function."""

    def get_weather(city: str) -> str:
        """Look up the weather in a city."""
        return "sunny"

    route = ToolCallingFakeChatModel(model_name="cheap-1")
    bound = route.bind_tools([get_weather])

    assert isinstance(bound, RunnableBinding)
    converted = repr(bound.kwargs["tools"])
    assert "0x" not in converted
    assert "<function" not in converted
    assert bound.kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather in a city.",
                "parameters": {
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "type": "object",
                },
            },
        }
    ]


# --- REQ-C10-4: a cache hit still carries the current call's own decision record ---


def test_a_cache_hit_carries_the_same_record_the_miss_that_filled_it_recorded() -> None:
    """REQ-C10-4: two identical calls take the identical decision (no strategy, same default
    route), so the hit's record and the miss's record are not just present but equal."""
    set_llm_cache(InMemoryCache())
    cheap = FakeChatModel(model_name="cheap-1", reply="cheap answer")
    router = ChatRouter(routes={"cheap": cheap}, default_route="cheap")

    miss = router.invoke("hello")
    hit = router.invoke("hello")

    assert len(call_log(cheap)) == 1
    assert routing(hit) == routing(miss)


def test_two_different_decisions_landing_on_the_same_route_each_keep_their_own_record() -> None:
    """REQ-C10-4's sharpest case: the cache key is the route's own identity and call kwargs —
    not the decision — so two *different* decisions that both settle on the same route (here:
    the unforced default, then the same route forced via T-116's runtime config) do reach the
    same cache entry. The second call is still a hit (the route's own `_generate` runs once),
    but its `response_metadata["routing"]` is its own decision, not the first's — `_with_record`
    (`router.py`) applies after the route's cache lookup returns, on every call, hit or miss."""
    set_llm_cache(InMemoryCache())
    cheap = FakeChatModel(model_name="cheap-1", reply="cheap answer")
    other = FakeChatModel(model_name="other-1", reply="other answer")
    router = ChatRouter(routes={"cheap": cheap, "other": other}, default_route="cheap")

    unforced = router.invoke("hello")
    forced = router.invoke("hello", config={"configurable": {"route": "cheap"}})

    assert len(call_log(cheap)) == 1  # one route call: the second was a cache hit
    assert routing(unforced)["forced"] is False
    assert routing(unforced)["reason"] == "no strategy configured"
    assert routing(forced)["forced"] is True
    assert routing(forced)["reason"] == "forced via runtime config"
    assert routing(unforced) != routing(forced)


def test_a_cache_hit_via_a_strategy_still_gets_its_own_strategy_named_on_the_record() -> None:
    """REQ-C10-4, one more shape of the same guarantee: a strategy-driven decision reaching a
    cache filled by an unrelated (forced) call to the same route still names the strategy that
    actually ran this call, not "forced via runtime config"."""
    set_llm_cache(InMemoryCache())
    cheap = FakeChatModel(model_name="cheap-1", reply="cheap answer")
    router = ChatRouter(
        routes={"cheap": cheap},
        default_route="cheap",
        strategy=KeywordStrategy({"cheap": ["urgent"]}),
    )

    forced = router.invoke("urgent request", config={"configurable": {"route": "cheap"}})
    via_strategy = router.invoke("urgent request")

    assert len(call_log(cheap)) == 1
    assert routing(forced)["forced"] is True
    assert routing(via_strategy)["forced"] is False
    assert routing(via_strategy)["strategy"] == "KeywordStrategy"
