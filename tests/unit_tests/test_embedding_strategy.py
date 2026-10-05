"""The opt-in embedding-similarity strategy: required arguments, tracing and cost, fallback.

Three threads:

- **Required arguments** — construction takes the embeddings instance and the threshold as required
  arguments, with no default for either (the module docstring explains why the threshold gets
  the same treatment as the embeddings instance): `test_construction_requires_*`.
- **Tracing** — every per-request `embed_query`/`aembed_query` call opens its own
  child run of the strategy's, recording the input length and a cost *estimate* (`Embeddings`
  reports no usage): `test_each_per_request_embedding_call_is_a_child_run_of_the_strategy_run`
  and its neighbours.
- **Fallback** — an embedding failure, an abstain (nothing clears the threshold, or nothing to
embed),
  and a route the router doesn't have all end the same way a built-in strategy's always have:
  the default route, one `FallbackWarning`, the cause recorded.

`DeterministicFakeEmbedding` (`langchain_core.embeddings`) embeds identical text identically and
different text apart, so a request that repeats an example verbatim scores that example exactly
`1.00` — deterministic enough to assert exact reasons and exact threshold behaviour without a
real provider. `CountingEmbeddings` and `FailingEmbeddings` wrap it to check the "embedded once"
promise and the failure path without needing a fake that does its own vector math.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from langchain_model_router import (
    ChatRouter,
    FallbackWarning,
    RoutingChoice,
    RoutingDecision,
    RoutingError,
    RoutingRequest,
    RoutingWarning,
    routing_decision,
)
from langchain_model_router._extraction import build_request
from langchain_model_router.strategies import embedding
from langchain_model_router.strategies.embedding import EmbeddingStrategy
from tests.conventions import Convention, respond
from tests.fakes import CountingEmbeddings, FakeChatModel
from tests.tracing import StartLog

EXAMPLES = {
    "coder": ("fix this stack trace", "refactor this function"),
    "support": ("reset my password", "cancel my subscription"),
}
ROUTES = ("coder", "support")


def make_request(
    text: str, *, routes: tuple[str, ...] = ROUTES, config: RunnableConfig | None = None
) -> RoutingRequest:
    """The request the router would build from a one-turn conversation."""
    request = build_request(
        [HumanMessage(text)],
        routes=routes,
        tools_bound=False,
        wants_full_context=False,
        config=config if config is not None else RunnableConfig(),
    )
    assert request is not None
    return request


def fake_routes() -> dict[str, BaseChatModel]:
    return {
        "coder": FakeChatModel(model_name="model-coder", reply="coder answer"),
        "support": FakeChatModel(model_name="model-support", reply="support answer"),
    }


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


FAILURE = ConnectionError("the embeddings endpoint is down")


class FailingEmbeddings(Embeddings):
    """Route-example embedding succeeds; every per-query embed raises (a network error, say)."""

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        raise FAILURE

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        raise FAILURE


# --- construction requires the embeddings instance, and a threshold, no defaults ---


def test_construction_requires_the_embeddings_instance() -> None:
    """No default — this strategy can never be enabled by accident."""
    with pytest.raises(TypeError):
        EmbeddingStrategy()  # type: ignore[call-arg]


def test_construction_requires_examples_too() -> None:
    with pytest.raises(TypeError):
        EmbeddingStrategy(DeterministicFakeEmbedding(size=8))  # type: ignore[call-arg]


def test_construction_requires_a_threshold_with_no_default() -> None:
    """The module doc's own extension of the no-defaults rule: no similarity score means the same
    thing across every embedding provider, so `threshold` is required like the embeddings
    instance."""
    with pytest.raises(TypeError):
        EmbeddingStrategy(DeterministicFakeEmbedding(size=8), EXAMPLES)  # type: ignore[call-arg]


# --- Configuration errors, at construction (never once per request) ---

BAD_EXAMPLES = [
    pytest.param(
        {},
        "examples is empty: an EmbeddingStrategy needs at least one route with examples",
        id="empty",
    ),
    pytest.param(
        [],
        "examples is list: an EmbeddingStrategy takes a mapping of route name to example "
        "utterances, such as {'coder': ['fix this stack trace']}",
        id="not-a-mapping",
    ),
    pytest.param(
        {"coder": []},
        "route 'coder' has no example utterances: a route needs at least one",
        id="no-examples",
    ),
    pytest.param(
        {"coder": "fix this"},
        "route 'coder' is given a single string, not a list: wrap it, as in "
        "{'coder': ['fix this']}",
        id="string-not-list",
    ),
    pytest.param(
        {" ": ["fix this"]}, "the route ' ' is blank: every route needs a name", id="blank-route"
    ),
    pytest.param(
        {"coder": ["", "ok"]},
        "route 'coder' has a blank or non-string example: ''",
        id="blank-example",
    ),
    pytest.param(
        {"coder": [None]},
        "route 'coder' has a blank or non-string example: None",
        id="non-string-example",
    ),
    pytest.param(
        {"coder": 42},
        "route 'coder' is given int: examples are a list of strings",
        id="not-a-sequence",
    ),
]


@pytest.mark.parametrize(("examples", "message"), BAD_EXAMPLES)
def test_a_route_mapping_that_could_never_decide_fails_at_construction(
    examples: Any, message: str
) -> None:
    """Everything judged without a request is judged here, once, not per request."""
    with pytest.raises(RoutingError, match=rf"^{re.escape(message)}$"):
        EmbeddingStrategy(DeterministicFakeEmbedding(size=8), examples, threshold=0.5)


@pytest.mark.parametrize(
    "threshold", [True, "0.5", None, [0.5]], ids=["bool", "str", "none", "list"]
)
def test_a_non_numeric_threshold_fails_at_construction(threshold: Any) -> None:
    with pytest.raises(RoutingError, match=r"^threshold must be a number, got .+$"):
        EmbeddingStrategy(DeterministicFakeEmbedding(size=8), EXAMPLES, threshold=threshold)


# --- Choosing a route ---


def test_the_route_whose_example_reads_closest_decides() -> None:
    """`DeterministicFakeEmbedding` embeds identical text identically: a request that repeats an
    example verbatim scores it 1.00 -- the maximum -- and that route decides."""
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=32), EXAMPLES, threshold=0.99)

    choice = strategy.decide(make_request("reset my password"))

    assert choice == RoutingChoice(
        route="support",
        reason="embedding similarity 1.00 >= 0.99 to 'reset my password' (route 'support')",
        messages_back=0,
    )


def test_a_route_s_score_is_the_best_of_its_examples_not_diluted_by_the_others() -> None:
    """Max aggregation (module doc): matching a route's *second* example verbatim still scores
    it 1.00, not averaged down by an unrelated first example."""
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=32), EXAMPLES, threshold=0.99)

    choice = strategy.decide(make_request("refactor this function"))

    assert choice == RoutingChoice(
        route="coder",
        reason="embedding similarity 1.00 >= 0.99 to 'refactor this function' (route 'coder')",
        messages_back=0,
    )


def test_a_score_that_clears_no_threshold_abstains() -> None:
    """1.0 is the ceiling of cosine similarity, so a threshold just above it can never be
    cleared -- the strategy abstains and the router falls back to the default route."""
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=32), EXAMPLES, threshold=1.01)

    assert strategy.decide(make_request("reset my password")) is None


def test_an_empty_request_abstains_without_embedding_anything() -> None:
    """Nothing to embed -- the same reasoning `HeuristicStrategy` uses for nothing to score
    -- checked before any embedding call, sync or async, is made."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=8))
    strategy = EmbeddingStrategy(embeddings, EXAMPLES, threshold=0.5)

    assert strategy.decide(make_request("   ")) is None

    assert (embeddings.document_calls, embeddings.query_calls) == (0, 0)


# --- A route examples names that the router doesn't have (the fallback path, `KeywordStrategy`'s
# precedent) ---


def test_a_route_named_in_examples_but_not_the_router_s_is_returned_anyway() -> None:
    """Matching happens regardless of what the router has -- the router reports a mismatch far
    better than this strategy could, the same precedent `KeywordStrategy` sets."""
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=8), EXAMPLES, threshold=0.99)

    choice = strategy.decide(make_request("reset my password", routes=("coder", "billing")))

    assert choice == RoutingChoice(
        route="support",
        reason="embedding similarity 1.00 >= 0.99 to 'reset my password' (route 'support')",
        messages_back=0,
    )


async def test_through_the_router_a_stray_route_falls_back_with_a_named_cause() -> None:
    # "billing" is in the router's routes, so `_check_it_can_decide` lets the request through;
    # "support" is not, so the winning route is still the one the router has to report on.
    examples = {**EXAMPLES, "billing": ("a billing example nothing else matches",)}
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=8), examples, threshold=0.99)
    router = ChatRouter(
        routes={"billing": FakeChatModel(model_name="model-billing", reply="billing answer")},
        default_route="billing",
        strategy=strategy,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke("reset my password")

    assert message.content == "billing answer"
    (warning,) = routing_warnings(caught)
    assert warning.category is FallbackWarning
    assert str(warning.message) == (
        "EmbeddingStrategy chose 'support', which is not one of the routes; falling back to "
        "the default route 'billing'"
    )


def test_no_route_in_examples_matching_the_router_s_raises_at_the_first_decide() -> None:
    """Same precedent as `KeywordStrategy`: a mapping naming none of the router's routes can
    never decide anything -- raised up front rather than abstaining silently forever."""
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=8), EXAMPLES, threshold=0.5)

    with pytest.raises(RoutingError, match="no route in examples is one this router has"):
        strategy.decide(make_request("reset my password", routes=("billing", "sales")))


async def test_through_the_router_that_unroutable_mapping_becomes_a_fallback_too() -> None:
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=8), EXAMPLES, threshold=0.5)
    router = ChatRouter(
        routes={"billing": FakeChatModel(model_name="model-billing", reply="billing answer")},
        default_route="billing",
        strategy=strategy,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke("reset my password")

    assert message.content == "billing answer"
    (warning,) = routing_warnings(caught)
    assert warning.category is FallbackWarning
    assert (
        "EmbeddingStrategy raised RoutingError: no route in examples is one this router has"
        in str(warning.message)
    )


# --- Route examples are embedded once, lazily, sync and async ---


def test_route_examples_are_embedded_once_not_per_request() -> None:
    """The module doc's promise: several sync decisions share one `embed_documents` call."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=8))
    strategy = EmbeddingStrategy(embeddings, EXAMPLES, threshold=-1.0)

    for _ in range(3):
        strategy.decide(make_request("reset my password"))

    assert embeddings.document_calls == 1
    assert embeddings.query_calls == 3


async def test_examples_embedded_via_aembed_documents_when_async_runs_first() -> None:
    """The async path never falls back to the blocking `embed_documents` on first use."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=8))
    strategy = EmbeddingStrategy(embeddings, EXAMPLES, threshold=-1.0)

    await strategy.adecide(make_request("reset my password"))
    await strategy.adecide(make_request("fix this stack trace"))

    assert (embeddings.document_calls, embeddings.adocument_calls) == (0, 1)
    assert embeddings.aquery_calls == 2


async def test_mixing_sync_and_async_still_embeds_the_routes_only_once() -> None:
    """Whichever path runs first fills the shared cache; the other reuses it (module doc)."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=8))
    strategy = EmbeddingStrategy(embeddings, EXAMPLES, threshold=-1.0)

    strategy.decide(make_request("reset my password"))
    await strategy.adecide(make_request("fix this stack trace"))
    await strategy.adecide(make_request("cancel my subscription"))

    assert (embeddings.document_calls, embeddings.adocument_calls) == (1, 0)
    assert (embeddings.query_calls, embeddings.aquery_calls) == (1, 2)


# --- Each per-request embedding call is its own child run of the strategy run ---


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_each_per_request_embedding_call_is_a_child_run_of_the_strategy_run(
    convention: Convention,
) -> None:
    """The acceptance criterion, sync and async: router run -> strategy run -> one 'embed_query'
    chain run holding the input length and a cost *estimate*; the route's own call
    stays where it always is, directly under the router run, untouched."""
    text = "fix this stack trace"
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=16), EXAMPLES, threshold=-1.0)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)
    starts, collector = StartLog(), RunCollectorCallbackHandler()

    await respond(router, convention, text, {"callbacks": [starts, collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    (embed_run,) = strategy_run.child_runs
    assert (strategy_run.run_type, embed_run.run_type) == ("chain", "chain")
    assert embed_run.name == "embed_query"
    assert embed_run.inputs == {"text": text}
    assert embed_run.outputs == {
        "input_length": len(text),
        "estimated_tokens": math.ceil(len(text) / 4),
        "cost_estimate_method": "chars / 4 (Embeddings reports no usage)",
    }
    assert (embed_run.child_runs, route_run.child_runs) == ([], [])

    (strategy_start,) = [start for start in starts.chains if start.name == "EmbeddingStrategy"]
    (embed_start,) = [start for start in starts.chains if start.name == "embed_query"]
    assert embed_start.parent_run_id == strategy_start.run_id
    assert embed_start.run_id == embed_run.id


async def test_the_route_example_embedding_is_not_traced_as_a_run_of_its_own() -> None:
    """The module doc's other half: only the per-request query is traced. The one-time route
    embedding is amortised setup work, not any one request's cost."""
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=16), EXAMPLES, threshold=-1.0)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)
    collector = RunCollectorCallbackHandler()

    router.invoke("fix this stack trace", config={"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, _route_run = router_run.child_runs
    assert [run.name for run in strategy_run.child_runs] == ["embed_query"]


# --- Embedding failure falls back to the default route ---


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_an_embedding_failure_falls_back_to_the_default_route(convention: Convention) -> None:
    """The strategy does not catch the `Embeddings` call's exception (module doc) -- the
    router's own machinery turns it into the default route and one `FallbackWarning`."""
    strategy = EmbeddingStrategy(
        FailingEmbeddings(DeterministicFakeEmbedding(size=8)), EXAMPLES, threshold=0.5
    )
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "fix this stack trace")

    assert message.content == "support answer"
    (warning,) = routing_warnings(caught)
    assert warning.category is FallbackWarning
    assert str(warning.message) == (
        "EmbeddingStrategy raised ConnectionError: the embeddings endpoint is down; falling "
        "back to the default route 'support'"
    )
    record = routing_decision(message)
    assert record == RoutingDecision(
        route="support",
        reason="EmbeddingStrategy raised ConnectionError: the embeddings endpoint is down; "
        "fell back to the default route",
        strategy="EmbeddingStrategy",
        fallback=True,
    )


def test_the_failed_embed_run_carries_the_error_and_the_route_still_answers() -> None:
    """The embedding run closes as an error too, not only the strategy run around it -- the
    traceback for whoever debugs a failed embedding call stays on the call that failed -- while
    the router's own run and the route's call complete normally."""
    strategy = EmbeddingStrategy(
        FailingEmbeddings(DeterministicFakeEmbedding(size=8)), EXAMPLES, threshold=0.5
    )
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)
    collector = RunCollectorCallbackHandler()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        router.invoke("fix this stack trace", config={"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    (embed_run,) = strategy_run.child_runs
    assert "ConnectionError: the embeddings endpoint is down" in (embed_run.error or "")
    assert "ConnectionError: the embeddings endpoint is down" in (strategy_run.error or "")
    assert (router_run.error, route_run.error) == (None, None)


# --- the built-in is exercised only through the public interface ---


async def test_through_the_router_an_embedding_strategy_is_an_ordinary_strategy() -> None:
    """Nothing about running it through `ChatRouter` is special-cased.

    `DeterministicFakeEmbedding`'s vectors carry no real semantics -- two different strings
    embed unrelated to each other, however similar they read -- so the request repeats a
    route's example verbatim, the same deterministic setup every other test here uses.
    """
    strategy = EmbeddingStrategy(DeterministicFakeEmbedding(size=16), EXAMPLES, threshold=0.5)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    message = router.invoke("refactor this function")

    assert message.content == "coder answer"
    record = routing_decision(message)
    assert record is not None
    assert record.route == "coder"
    assert record.fallback is False


# --- Follow-ups: the user's previous messages, with `lookback` ---

UNMATCHED = ("it still fails", "any idea why?", "please help")
"""Messages that match no example: `DeterministicFakeEmbedding` puts them nowhere near one."""


def conversation(*texts: str) -> RoutingRequest:
    """The last of `texts` as the current request, the others as the user's earlier messages."""
    *earlier, current = texts
    return replace(
        make_request(current),
        previous_requests=tuple(make_request(text) for text in reversed(earlier)),
    )


def strategy_with(lookback: int, embeddings: Embeddings | None = None) -> EmbeddingStrategy:
    return EmbeddingStrategy(
        embeddings or DeterministicFakeEmbedding(size=32),
        EXAMPLES,
        threshold=0.99,
        lookback=lookback,
    )


SUPPORT = "embedding similarity 1.00 >= 0.99 to 'reset my password' (route 'support')"
CODER = "embedding similarity 1.00 >= 0.99 to 'fix this stack trace' (route 'coder')"


@pytest.mark.parametrize("convention", ["decide", "adecide"])
async def test_a_follow_up_goes_where_the_message_before_it_went(convention: str) -> None:
    """A follow-up close to no example is routed by the message before it, and the reason says
    so, in the words every strategy uses."""
    strategy = strategy_with(1)
    request = conversation("reset my password", UNMATCHED[0])

    choice = strategy.decide(request) if convention == "decide" else await strategy.adecide(request)

    assert choice == RoutingChoice("support", f"{SUPPORT} (1 message back)", messages_back=1)


def test_the_newest_message_that_clears_the_threshold_decides() -> None:
    """Messages are tried newest first, so the topic the user moved to wins; and a current
    request that clears the bar wins outright, with the reason it has without lookback, before
    any earlier message is embedded."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=32))
    strategy = strategy_with(2, embeddings)

    assert strategy.decide(
        conversation("reset my password", "fix this stack trace", UNMATCHED[0])
    ) == RoutingChoice("coder", f"{CODER} (1 message back)", messages_back=1)
    embeddings.queries.clear()
    assert strategy.decide(
        conversation("fix this stack trace", "reset my password")
    ) == RoutingChoice("support", SUPPORT, messages_back=0)
    # The first request stopped at the message that decided, so this one is new; and the
    # earlier message isn't read at all, since the current one decides.
    assert embeddings.queries == ["reset my password"]


def test_a_current_request_that_decides_embeds_no_earlier_message() -> None:
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=32))

    strategy_with(2, embeddings).decide(conversation(*UNMATCHED[:2], "reset my password"))

    assert embeddings.queries == ["reset my password"]


def test_lookback_reaches_as_far_as_it_says_and_passes_over_messages_with_no_text() -> None:
    """A message with no text takes a place but has nothing to embed; up to `lookback` places,
    and no further."""
    request = conversation("reset my password", "   ", UNMATCHED[0])

    assert strategy_with(2).decide(request) == RoutingChoice(
        "support", f"{SUPPORT} (2 messages back)", messages_back=2
    )
    assert strategy_with(1).decide(request) is None
    assert strategy_with(0).decide(request) is None


def test_the_strategy_reads_its_own_lookback_whatever_the_request_carries() -> None:
    request = conversation("reset my password", *UNMATCHED[:2])

    assert strategy_with(1).decide(request) is None
    assert strategy_with(2).decide(request) == RoutingChoice(
        "support", f"{SUPPORT} (2 messages back)", messages_back=2
    )


@pytest.mark.parametrize("convention", ["decide", "adecide"])
async def test_each_message_is_embedded_once_however_many_turns_read_it(convention: str) -> None:
    """The cost bound: every earlier message was the current request on its own turn, and its
    outcome is remembered then, so a turn embeds only its own new message — and a message
    repeated as-is, as each step of an agent's tool loop repeats it, embeds nothing."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=32))
    strategy = strategy_with(2, embeddings)
    turns = [UNMATCHED[: n + 1] for n in range(3)] + [UNMATCHED]

    for turn in turns:
        request = conversation(*turn)
        if convention == "decide":
            assert strategy.decide(request) is None
        else:
            assert await strategy.adecide(request) is None

    assert embeddings.queries == list(UNMATCHED)
    assert embeddings.query_calls + embeddings.aquery_calls == len(UNMATCHED)


def test_without_lookback_nothing_is_remembered() -> None:
    """With `lookback=0` the strategy is what it was: every request is embedded."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=32))
    strategy = strategy_with(0, embeddings)

    for _ in range(2):
        strategy.decide(make_request(UNMATCHED[0]))

    assert embeddings.queries == [UNMATCHED[0]] * 2


def test_what_is_remembered_is_bounded_and_the_least_recently_read_goes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding, "_REMEMBERED", 2)
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=32))
    strategy = strategy_with(1, embeddings)
    first, second, third = UNMATCHED

    strategy.decide(make_request(first))
    strategy.decide(make_request(second))
    strategy.decide(make_request(first))  # read again: now the most recent
    strategy.decide(make_request(third))  # `second` is forgotten
    strategy.decide(conversation(second, first))

    assert embeddings.queries == [first, second, third, second]


def test_a_copy_made_by_with_lookback_shares_what_the_original_remembers() -> None:
    """Same embeddings, same examples: the copy has no reason to embed anything again."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=32))
    original = strategy_with(1, embeddings)
    original.decide(conversation(UNMATCHED[0], UNMATCHED[1]))

    copied = original.with_lookback(2)
    copied.decide(conversation(*UNMATCHED))

    assert embeddings.queries == [UNMATCHED[1], UNMATCHED[0], UNMATCHED[2]]
    assert embeddings.document_calls == 1


@pytest.mark.parametrize("convention", ["invoke", "ainvoke"])
async def test_a_remembered_message_makes_no_call_and_opens_no_run(
    convention: Convention,
) -> None:
    """Tracing is as it was: one `embed_query` run per embedding call. The follow-up's is the
    only one on its turn, since the message before it was embedded on its own turn."""
    strategy = strategy_with(1)
    router = ChatRouter(routes=fake_routes(), default_route="coder", strategy=strategy)
    collector = RunCollectorCallbackHandler()

    await respond(router, convention, "reset my password")
    message = await respond(
        router,
        convention,
        [HumanMessage("reset my password"), AIMessage("done"), HumanMessage(UNMATCHED[0])],
        {"callbacks": [collector]},
    )

    assert message.content == "support answer"
    (router_run,) = collector.traced_runs
    strategy_run, _route_run = router_run.child_runs
    assert [run.inputs for run in strategy_run.child_runs] == [{"text": UNMATCHED[0]}]


@pytest.mark.parametrize("lookback", [-1, True, 1.5, "2", None])
def test_lookback_must_be_a_count_of_messages(lookback: Any) -> None:
    with pytest.raises(
        RoutingError,
        match=(
            r"^lookback must be a non-negative integer, the number of the user's previous "
            rf"messages to read, got {re.escape(repr(lookback))}$"
        ),
    ):
        EmbeddingStrategy(
            DeterministicFakeEmbedding(size=8), EXAMPLES, threshold=0.5, lookback=lookback
        )
