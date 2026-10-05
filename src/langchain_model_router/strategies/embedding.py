r"""`EmbeddingStrategy`, opt-in because it makes calls: route on meaning, not words.

`KeywordStrategy` needs the right word; `EmbeddingStrategy` needs the right *idea* — a handful of
example requests per route, and the current request goes wherever it reads closest to. It makes
an embedding call every request, so — unlike the heuristic and keyword strategies — it is never
on by default: construction takes the application's own `Embeddings` instance, with no default.
No new dependency is added for it; whatever embeddings package the application
already has is what it is built from.

Setting one up
--------------
```python
from langchain_core.embeddings import Embeddings  # any provider's implementation

EmbeddingStrategy(
    my_embeddings,
    {
        "support": ["reset my password", "my order hasn't arrived", "cancel my subscription"],
        "coder": ["why does this stack trace happen", "refactor this function", "write a test"],
    },
    threshold=0.75,
)
```

**Examples are embedded once** (no repeated work: beyond the one embedding call each request
exists to make, nothing is recomputed): on the first `decide` or
`adecide`, in a single batched `embed_documents`/`aembed_documents` call, cached for the life of
the strategy. Not at construction, because construction has no async hook of its own — an
application building a router inside an async context would otherwise have to block the loop, or
`EmbeddingStrategy.__init__` would need to be async, which nothing in `RoutingStrategy` asks for
or could await. Lazy, one-time, on first use is the one point that is naturally both: the sync
path calls `embed_documents` directly, and the async path calls `aembed_documents` — which,
absent a provider override, is `Embeddings`' own default of running `embed_documents` in a
worker thread (`embeddings.py`'s `run_in_executor(None, ...)`), so the loop is never blocked
either way. A `threading.Lock` guards the cache: held for the sync embed, so two concurrent
`decide` calls (from worker threads) embed the routes' examples only once between them; released
before the async embed's `await` and re-acquired only to write the result, because holding a
plain lock across an `await` risks a second coroutine's blocking `acquire()` freezing the one
thread the event loop runs on. The async race that leaves — two requests, both first, both
embedding the routes before either finishes — computes the same vectors twice at worst, never an
inconsistent cache.

**Only the per-request query is traced.** `Embeddings` is a plain ABC: `embed_query` and
`embed_documents` take no `config`, so a call through it is invisible to any tracer unless the
strategy opens its own run around it — the same manual pattern `ChatRouter._start_run` uses
for its own run, read from `request.config`'s callback manager rather than a chat model's
`invoke`. Every per-request query embedding gets one, sync and async, holding the input length
and a cost *estimate* — `Embeddings` reports no token usage, so what it records is an estimate
from input length, not a measurement, with the method recorded alongside it. The one-time
route-example embedding does not get a run of its own: it is amortised construction-time work,
not any one request's cost, and whichever request happened to trigger it would otherwise carry
an arbitrary, non-reproducible charge that every later request escapes.

Deciding, and not deciding
---------------------------
The current request's text is embedded and compared against every route's example embeddings,
one **cosine similarity** each; a route's score is the **maximum** over its own examples — the
strategy asks "is this request close to any one thing this route handles", not "close to the
average of them", so a route with a deliberately varied example set is not penalised for its
variety. The route with the best score decides, *if* that score clears `threshold`; otherwise the
strategy abstains with `None` and the router falls back to the default route, exactly as
`HeuristicStrategy` does below its own bar.

**One global threshold, not one per route.** A per-route bar would need its own calibration
evidence per route before the strategy could be trusted at all, multiplying the calibration work by
the number of routes; a single bar is the simpler, more inspectable default, and nothing here stops
an application from subclassing for a per-route one if its evidence calls for it. Unlike
`HeuristicStrategy`'s signals, which are dimensionless by construction (`_ramp` always returns
`[0.0, 1.0]`), cosine similarity's *useful* range depends entirely on the embedding model — some
providers' vectors cluster closely and rarely score below 0.9 even for unrelated text, others
spread out — so there is no threshold this module could default to that would mean the same thing
for every provider. `threshold` is therefore required, with no default — the same call made
for the embeddings instance itself: a number that is silently wrong is worse than one the
caller has to supply.

**An empty request** — no text — has nothing to embed, and abstains for the same reason
`HeuristicStrategy` does with nothing to score: better the default route than a vector for
the empty string, which every embedding model answers *something* for and no provider documents
as meaningful.

**Embedding failure** is not caught here. `decide`/`adecide` let the `Embeddings` call's
exception propagate — the same shape `ConsultingStrategy` in the test suite relies on for a
classifier's model call — and the router's own `_conclude` is what turns a strategy that raised
into the default route with a `FallbackWarning` naming the cause. Catching it here would
only rebuild, one strategy at a time, machinery the router already has.

**A route named in `examples` that the router doesn't have** is handled the way `KeywordStrategy`
and `ConfigurableStrategy` handle a stray rule: matching happens regardless, and the router
reports the mismatch through `FallbackWarning` far better than this strategy could — a typo
costs only the requests that route's examples would have matched closest. The one case checked
up front, at the first `decide`, is a route mapping that names *none* of the router's routes: that
can never decide anything, and `RoutingError` says so before silently abstaining for the life of
the process, the same precedent.

Follow-ups: `lookback`
----------------------
With `lookback=N`, when the current request clears no `threshold`, the user's previous N
messages are tried in turn, newest first, and **the newest one that clears it decides**; the
reason ends with how far back it was, `(1 message back)`, the same words every built-in strategy
uses. A message with no text is passed over. A message that clears the bar is never overruled by
an older one, so a new topic takes over as soon as it is asked.

An earlier message costs an embedding only once. Each earlier message was the current request on
its own turn, so it has already been embedded, and embedding it again on every later turn would
multiply the per-request cost by up to N+1. So a strategy with a `lookback` **remembers the
outcome for the messages it has embedded recently** — the closest example, its route and the
similarity, not the vector — and a remembered message makes no call and opens no run. A turn
therefore makes one embedding call, for its own new message, and a message repeated as-is, as
every step of an agent's tool loop repeats it, makes none. At most N+1 calls a turn is the
bound, reached only when earlier messages have been forgotten (after a restart, or pushed out
by other conversations' messages), and a turn stops at the first message that clears the bar.

The two alternatives weighed, and why this one:

- **LangChain's `CacheBackedEmbeddings` with a `query_embedding_cache`** caches the vector under
  any `ByteStore`, and it still works here: it is an `Embeddings`, so an application can hand one
  to this strategy. It isn't the answer on its own. It lives in `langchain-classic`, which this
  package doesn't depend on; `langchain-core`'s `InMemoryByteStore` has no bound, so a long-running
  server would keep every message it has seen; and a hit inside it looks like a call from here,
  so every hit would be traced with a cost estimate for an embedding nobody paid for.
- **Remembering the vectors** would cost kilobytes a message (an embedding is hundreds to
  thousands of floats), where the outcome is three values, and the outcome is all that is ever
  read again: the examples and the similarity are fixed for the life of the strategy.

The outcomes are kept under a SHA-256 hash of the message's text, so the strategy keeps no user
text beyond the request. It holds the last `_REMEMBERED` distinct messages, the least recently
read forgotten first, shared by every conversation the strategy serves and guarded by a lock
like the example vectors. It is used only with a `lookback`: with `lookback=0` every request is
embedded, exactly as without one. A copy made by `with_lookback` shares it, and the example
vectors, with the original, since both are the same embeddings on the same examples.

**Only the user's words.** Setting `wants_full_context` on a subclass changes nothing here: the
model's answers are not embedded. The examples are requests, and an answer — long, and about
what the route said rather than what was asked — would be compared with them as if it were one.
It would also make the cost of a turn grow with the conversation. A subclass that wants the
transcript reads `request.messages` in its own `decide`.

No dependency beyond `langchain-core` is imported: the similarity itself is a few lines of
`math`, not `numpy`.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
)

from langchain_model_router.errors import RoutingError
from langchain_model_router.strategies._lookback import checked_lookback, considered, how_far_back
from langchain_model_router.strategy import RoutingChoice, RoutingRequest, RoutingStrategy

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.runnables import RunnableConfig

__all__ = ["EmbeddingStrategy"]

_EMBED_RUN_NAME = "embed_query"
"""The name every per-request embedding run carries in a trace."""

_CHARS_PER_TOKEN = 4
"""The rough English chars-per-token ratio several providers document for their own models —
close enough for an overhead *estimate*, never offered as a measurement."""

_REMEMBERED = 4096
"""How many messages' outcomes a strategy with a `lookback` keeps (module doc).

A few hundred bytes each, so about a megabyte in all."""

Vector = list[float]
_RouteVectors = dict[str, list[tuple[str, Vector]]]
"""Route name -> its examples, each paired with its embedding, in declaration order."""


class _Match(NamedTuple):
    """How one message compares with the examples: its closest example, and that route."""

    similarity: float
    route: str
    example: str


class _Memory:
    """What a strategy works out and keeps: the example vectors, and recent messages' matches.

    One object, so a copy made by `with_lookback` shares it with the original rather than
    working it all out again.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.route_vectors: _RouteVectors | None = None
        self.matches: OrderedDict[bytes, _Match] = OrderedDict()

    def recall(self, text: str) -> _Match | None:
        """The match remembered for `text`, now the most recently read, or `None`."""
        key = _key(text)
        with self.lock:
            match = self.matches.get(key)
            if match is not None:
                self.matches.move_to_end(key)
            return match

    def remember(self, text: str, match: _Match) -> None:
        """Keep `match` for `text`, forgetting the least recently read beyond `_REMEMBERED`."""
        key = _key(text)
        with self.lock:
            self.matches[key] = match
            self.matches.move_to_end(key)
            while len(self.matches) > _REMEMBERED:
                self.matches.popitem(last=False)


def _key(text: str) -> bytes:
    """What a message's match is kept under: a hash, so no user text is kept."""
    return hashlib.sha256(text.encode()).digest()


class EmbeddingStrategy(RoutingStrategy):
    """Routes on semantic similarity to example requests per route.

    The module docstring has the reasoning behind every choice below: examples embedded once
    and lazily, the per-request embedding call traced as its own run, maximum similarity per
    route, one global threshold with no default, an embedding failure left to propagate so
    the router's own fallback handles it, and, with `lookback`, the newest message that clears
    the threshold deciding, each message embedded once.

    Args:
        embeddings: The application's own `Embeddings` instance — there is no
            default, so this strategy can never be enabled by accident.
        examples: Each route's example utterances, at least one route and at least one example
            per route.
        threshold: The similarity a route's best-matching example must clear for this strategy
            to decide, in the embedding model's own units (cosine similarity — typically but not
            always `[-1.0, 1.0]`). No default: see the module docstring for why one number can't
            serve every embedding model.
        lookback: How many of the user's previous messages to try as well, newest first, when
            the current one clears no threshold. Defaults to `0`, the current request alone.
            Each costs an embedding call the first time it is read, and none after.

    Raises:
        RoutingError: for configuration that could never route — no routes, a route with no
            examples, a blank route or example, a non-numeric threshold — or a `lookback` that
            isn't a non-negative integer. Raised here, at construction, not once per request.

    Example:
        ```python
        ChatRouter(
            routes={"support": support_model, "coder": coder_model},
            default_route="support",
            strategy=EmbeddingStrategy(
                my_embeddings,
                {
                    "support": ["reset my password", "cancel my subscription"],
                    "coder": ["why does this stack trace happen", "refactor this function"],
                },
                threshold=0.75,
            ),
        )
        ```
    """

    def __init__(
        self,
        embeddings: Embeddings,
        examples: Mapping[str, Sequence[str]],
        *,
        threshold: float,
        lookback: int = 0,
    ) -> None:
        """Check the configuration and keep the embeddings; nothing is embedded yet.

        Args:
            embeddings: The application's own `Embeddings` instance.
            examples: Each route's example utterances.
            threshold: The similarity a route's best example must clear.
            lookback: How many of the user's previous messages to try as well.

        Raises:
            RoutingError: for configuration that could never route, or a `lookback` that isn't
                a non-negative integer.
        """
        self.embeddings = embeddings
        self.examples = _checked_examples(examples)
        self.threshold = _checked_threshold(threshold)
        self.lookback = checked_lookback(lookback)
        self._memory = _Memory()

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """The route whose closest example clears `threshold`, or `None` if none does.

        The newest message that clears it decides: the current request, and then, with
        `lookback`, the previous ones in turn. Thread-safe, as the interface requires: the only
        mutable state is what the strategy keeps, which a lock guards.

        Args:
            request: The current request; only its text is embedded, and with `lookback` its
                previous requests' text.

        Returns:
            The best route's choice if it clears `threshold`, otherwise `None`.

        Raises:
            RoutingError: if no route in `examples` is one the router has.
        """
        self._check_it_can_decide(request.routes)
        for distance, message in considered(request, self.lookback):
            if not message.text.strip():
                continue  # nothing to embed — better the default route than a guess
            match = self._match(message)
            if match.similarity >= self.threshold:
                return self._choice(match, distance)
        return None

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Async `decide`: a native implementation, as the interface asks of a strategy that calls.

        `embed_query`/`embed_documents` take no `config` for a thread's context to ride along
        on, so the config has to be threaded through explicitly here.

        Args:
            request: The current request; only its text is embedded.

        Returns:
            The best route's choice if it clears `threshold`, otherwise `None`.

        Raises:
            RoutingError: if no route in `examples` is one the router has.
        """
        self._check_it_can_decide(request.routes)
        for distance, message in considered(request, self.lookback):
            if not message.text.strip():
                continue
            match = await self._amatch(message)
            if match.similarity >= self.threshold:
                return self._choice(match, distance)
        return None

    def _match(self, message: RoutingRequest) -> _Match:
        """How `message` compares with the examples: remembered, or embedded now (module doc)."""
        remembered = self._memory.recall(message.text) if self.lookback else None
        if remembered is not None:
            return remembered
        route_vectors = self._ensure_route_vectors()
        vector = _traced_embed(message.config, message.text, self.embeddings.embed_query)
        match = _closest(vector, route_vectors)
        if self.lookback:
            self._memory.remember(message.text, match)
        return match

    async def _amatch(self, message: RoutingRequest) -> _Match:
        """Async `_match`."""
        remembered = self._memory.recall(message.text) if self.lookback else None
        if remembered is not None:
            return remembered
        route_vectors = await self._aensure_route_vectors()
        vector = await _atraced_embed(message.config, message.text, self.embeddings.aembed_query)
        match = _closest(vector, route_vectors)
        if self.lookback:
            self._memory.remember(message.text, match)
        return match

    def _choice(self, match: _Match, distance: int) -> RoutingChoice:
        """The choice `match` makes, saying how far back its message was."""
        reason = (
            f"embedding similarity {match.similarity:.2f} >= {self.threshold:.2f} to "
            f"{match.example!r} (route {match.route!r}){how_far_back(distance)}"
        )
        return RoutingChoice(route=match.route, reason=reason, messages_back=distance)

    def _check_it_can_decide(self, routes: tuple[str, ...]) -> None:
        """At least one route in `examples` is one the router has.

        The check only a request can make, same precedent as
        `KeywordStrategy`/`ConfigurableStrategy`.
        """
        if any(route in routes for route in self.examples):
            return
        msg = (
            f"no route in examples is one this router has: examples name {_names(self.examples)}; "
            f"the routes are {_names(routes)}"
        )
        raise RoutingError(msg)

    def _ensure_route_vectors(self) -> _RouteVectors:
        """The routes' example embeddings, computing them once on the first call (module doc)."""
        memory = self._memory
        vectors = memory.route_vectors
        if vectors is not None:
            return vectors
        with memory.lock:
            vectors = memory.route_vectors
            if vectors is None:
                vectors = _grouped(
                    self.examples, self.embeddings.embed_documents(_flattened(self.examples))
                )
                memory.route_vectors = vectors
            return vectors

    async def _aensure_route_vectors(self) -> _RouteVectors:
        """Async `_ensure_route_vectors`.

        The embed call itself runs outside the lock, so it never holds a plain `threading.Lock`
        across an `await` (module doc).
        """
        memory = self._memory
        vectors = memory.route_vectors
        if vectors is not None:
            return vectors
        computed = _grouped(
            self.examples, await self.embeddings.aembed_documents(_flattened(self.examples))
        )
        with memory.lock:
            if memory.route_vectors is None:
                memory.route_vectors = computed
            return memory.route_vectors


# --- Configuration checking (construction time only) ---


def _checked_examples(examples: Mapping[str, Sequence[str]]) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(examples, Mapping):
        msg = (
            f"examples is {type(examples).__name__}: an EmbeddingStrategy takes a mapping of "
            "route name to example utterances, such as {'coder': ['fix this stack trace']}"
        )
        raise RoutingError(msg)
    if not examples:
        msg = "examples is empty: an EmbeddingStrategy needs at least one route with examples"
        raise RoutingError(msg)
    checked: dict[str, tuple[str, ...]] = {}
    for route, utterances in examples.items():
        if not isinstance(route, str) or not route.strip():
            msg = f"the route {route!r} is blank: every route needs a name"
            raise RoutingError(msg)
        checked[route.strip()] = _checked_utterances(route, utterances)
    return MappingProxyType(checked)


def _checked_utterances(route: str, utterances: Sequence[str]) -> tuple[str, ...]:
    if isinstance(utterances, (str, bytes)):
        msg = (
            f"route {route!r} is given a single string, not a list: wrap it, as in "
            f"{{{route!r}: [{utterances!r}]}}"
        )
        raise RoutingError(msg)
    if not isinstance(utterances, Sequence):
        msg = (
            f"route {route!r} is given {type(utterances).__name__}: examples are a list of strings"
        )
        raise RoutingError(msg)
    listed = list(utterances)
    if not listed:
        msg = f"route {route!r} has no example utterances: a route needs at least one"
        raise RoutingError(msg)
    blank = [text for text in listed if not isinstance(text, str) or not text.strip()]
    if blank:
        msg = f"route {route!r} has a blank or non-string example: {blank[0]!r}"
        raise RoutingError(msg)
    return tuple(text.strip() for text in listed)


def _checked_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        msg = f"threshold must be a number, got {threshold!r}"
        raise RoutingError(msg)
    return float(threshold)


def _names(names: Sequence[str] | Mapping[str, object]) -> str:
    return ", ".join(repr(name) for name in names)


# --- Route-example embeddings ---


def _flattened(examples: Mapping[str, tuple[str, ...]]) -> list[str]:
    """Every example, in route-then-declaration order.

    What one batched `embed_documents` call takes, and the order `_grouped` splits its answer
    back up by.
    """
    return [text for texts in examples.values() for text in texts]


def _grouped(
    examples: Mapping[str, tuple[str, ...]], flat_vectors: Sequence[Vector]
) -> _RouteVectors:
    """`flat_vectors` — one `embed_documents` call's answer — split back up by route."""
    grouped: _RouteVectors = {}
    index = 0
    for route, texts in examples.items():
        grouped[route] = list(zip(texts, flat_vectors[index : index + len(texts)], strict=True))
        index += len(texts)
    return grouped


# --- Choosing a route ---


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """How alike two embedding vectors are, `-1.0` to `1.0` for most providers' vectors.

    `0.0` when either is the zero vector — undefined, strictly, but there is no route that is
    the empty vector's "closest" one, and `0.0` keeps the comparison total rather than raising
    on a provider's edge case.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _closest(vector: Vector, route_vectors: _RouteVectors) -> _Match:
    """The closest example overall, and its route: the first of equals, in declaration order."""
    return max(
        (
            _Match(_cosine_similarity(vector, example_vector), route, example_text)
            for route, examples in route_vectors.items()
            for example_text, example_vector in examples
        ),
        key=lambda match: match.similarity,
    )


# --- Tracing a per-request embedding call ---


def _embed_outputs(text: str) -> dict[str, object]:
    """What a per-request embedding run records: the input length and a cost estimate.

    It is an estimate because `Embeddings` reports no usage, so there is nothing to measure.
    """
    return {
        "input_length": len(text),
        "estimated_tokens": max(1, math.ceil(len(text) / _CHARS_PER_TOKEN)),
        "cost_estimate_method": f"chars / {_CHARS_PER_TOKEN} (Embeddings reports no usage)",
    }


def _start_embed_run(config: RunnableConfig, text: str) -> CallbackManagerForChainRun:
    """Open a run for one embedding call, nested under `config`'s callback manager.

    That manager is the strategy's run. It is the same manual pattern `ChatRouter._start_run`
    uses for its own, needed because `Embeddings.embed_query` takes no `config` of its own to
    do this for.
    """
    manager = CallbackManager.configure(
        config.get("callbacks"),
        None,
        False,
        config.get("tags"),
        None,
        config.get("metadata"),
        None,
    )
    return manager.on_chain_start(None, {"text": text}, name=_EMBED_RUN_NAME)


async def _astart_embed_run(config: RunnableConfig, text: str) -> AsyncCallbackManagerForChainRun:
    """Async `_start_embed_run`."""
    manager = AsyncCallbackManager.configure(
        config.get("callbacks"),
        None,
        False,
        config.get("tags"),
        None,
        config.get("metadata"),
        None,
    )
    return await manager.on_chain_start(None, {"text": text}, name=_EMBED_RUN_NAME)


def _traced_embed(
    config: RunnableConfig, text: str, embed_query: Callable[[str], Vector]
) -> Vector:
    """Call `embed_query(text)` as its own child run of the strategy's.

    The exception it might raise propagates unchanged, so the router's own machinery is what
    turns it into a default-route fallback (module doc) — this only makes sure the run closes
    either way.
    """
    run_manager = _start_embed_run(config, text)
    try:
        vector = embed_query(text)
    except BaseException as error:
        run_manager.on_chain_error(error)
        raise
    run_manager.on_chain_end(_embed_outputs(text))
    return vector


async def _atraced_embed(
    config: RunnableConfig, text: str, aembed_query: Callable[[str], Awaitable[Vector]]
) -> Vector:
    """Async `_traced_embed`."""
    run_manager = await _astart_embed_run(config, text)
    try:
        vector = await aembed_query(text)
    except BaseException as error:
        await run_manager.on_chain_error(error)
        raise
    await run_manager.on_chain_end(_embed_outputs(text))
    return vector
