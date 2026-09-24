# Strategies

A strategy applies your routing policy to one request and names a route. The router calls it once
per request (a [forced route](forced-routes.md) skips it) and uses the answer to pick where the
request goes. When it can't decide, the default route answers.

Every strategy here is the same small interface, so you can swap one for another without touching
anything else. The examples on this page share these routes:

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, routing_decision


def fake(name: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=cycle([AIMessage(f"({name})")]), name=name)


routes = {"small": fake("small"), "coder": fake("coder"), "frontier": fake("frontier")}


def route_of(router: ChatRouter, request: str) -> str:
    return routing_decision(router.invoke(request)).route
```

## Which one?

| Strategy | Decides by | Extra calls | Reach for it when |
| --- | --- | --- | --- |
| [`KeywordStrategy`](#keywordstrategy) | Whole words in the request | none | Domain routing: the topic is in the words |
| [`HeuristicStrategy`](#heuristicstrategy) | A difficulty score from length, code and more | none | Cost tiering: cheap for easy, strong for hard |
| [`ConfigurableStrategy`](#configurablestrategy) | Rules you combine | none | A few conditions of your own, no class to write |
| [`EmbeddingStrategy`](#embeddingstrategy) | Similarity to example requests | 1 embedding | The meaning matters, not the exact words |
| [`ClassifierStrategy`](#classifierstrategy) | A small model picks | 1 chat call | You can describe each route in plain language |
| [A custom strategy](#custom-strategies) | Anything | yours | You already have a classifier, or need state |

The first three cost nothing per request. The other two make a model call on every request, so
neither is on by default.

## `KeywordStrategy`

Matches whole words in the request's text against rules you write. The first rule that matches
wins, in the order you declare them. When nothing matches the strategy abstains and the default
route answers (with a warning — see [fallbacks](fallbacks-and-retries.md)).

```python
from langchain_llm_router import KeywordStrategy

router = ChatRouter(
    routes=routes,
    default_route="small",
    strategy=KeywordStrategy(
        {
            "coder": ["python", "regex", "stack trace"],
            "frontier": ["prove", "derive"],
        }
    ),
)
assert route_of(router, "Why does this Python stack trace happen?") == "coder"
assert route_of(router, "Prove this identity.") == "frontier"
```

Matching is case-insensitive and by whole word: `"python"` matches `"Python?"` but not
`"pythonic"`. A multi-word keyword like `"stack trace"` matches across any run of whitespace. For
anything a word list can't say — scripts with no word boundaries, or a pattern such as
`re.compile(r"regexe?s?")` — pass a compiled `re.Pattern` in place of a string.

Story: [domain routing](../stories/domain-routing.md).

## `HeuristicStrategy`

Scores each request from cheap signals — its length, whether it contains code, how many things it
asks for, whether it asks for reasoning, whether it carries anything but text — and the score picks
a tier. List the routes **cheapest first**.

```python
from langchain_llm_router import HeuristicStrategy

two_tiers = ChatRouter(
    routes=routes,
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),
)
assert route_of(two_tiers, "what's 2 + 2?") == "small"
assert (
    route_of(two_tiers, "Prove that there are infinitely many primes, and explain why.")
    == "frontier"
)

three_tiers = HeuristicStrategy("small", "coder", "frontier", thresholds=[1.0, 3.0])
```

`thresholds` is one shorter than the tier list, ascending. `weights=` scales an individual signal
(`0.0` silences it) and `signals=` replaces the whole set. The score can't see how hard a short
question really is — `"Is P = NP?"` scores low — so treat it as a cheap first cut, and see the
[benchmark](../../benchmark/README.md) for what it saved on one workload.

Story: [cost tiering](../stories/cost-tiering.md).

## `ConfigurableStrategy`

The middle level: build a strategy from rules without writing a class. A rule is
`Rule(route, when, priority=0, name=None)`, and the first rule whose condition holds wins.
Conditions compose from `keywords(...)`, `signal_at_least(signal, threshold)`, `modality(...)`,
`tools_bound()`, `predicate(func)` and `always()`, joined with `any_of`, `all_of` and `not_`.

```python
from langchain_llm_router import ConfigurableStrategy
from langchain_llm_router.strategies.configurable import Rule, any_of, keywords, signal_at_least
from langchain_llm_router.strategies.heuristic import code_signal, length_signal

long_or_code = any_of(
    signal_at_least(length_signal(20, 200), 0.5), signal_at_least(code_signal, 0.5)
)

router = ChatRouter(
    routes=routes,
    default_route="small",
    strategy=ConfigurableStrategy(
        [
            Rule("frontier", long_or_code, name="long or code-bearing"),
            Rule("coder", keywords("python", "regex"), name="code question"),
        ]
    ),
)
assert route_of(router, "Any regex tips?") == "coder"
```

## `EmbeddingStrategy`

Routes on similarity to a handful of example requests per route — the right *idea*, not the right
word. It calls your `Embeddings` model on every request, and there is no default model.

```python
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_llm_router import EmbeddingStrategy

router = ChatRouter(
    routes=routes,
    default_route="small",
    strategy=EmbeddingStrategy(
        DeterministicFakeEmbedding(size=16),  # your real Embeddings here
        {
            "coder": ["fix this traceback", "why does my function return None"],
            "frontier": ["prove this theorem", "derive the closed form"],
        },
        threshold=0.75,
    ),
)
decision = routing_decision(router.invoke("fix this traceback"))
assert decision.route == "coder"
```

A request whose best match is below `threshold` abstains, and the default route answers.

## `ClassifierStrategy`

Asks a chat model to choose a route from a plain-language description of each, using
`with_structured_output`. It calls that model on every request, and there is no default model.
Pass a small, cheap one.

```python skip
from langchain_llm_router import ClassifierStrategy

router = ChatRouter(
    routes=routes,
    default_route="small",
    strategy=ClassifierStrategy(
        small_model,  # your own BaseChatModel that supports structured output
        {
            "coder": "programming help: code, errors, tooling",
            "frontier": "maths and multi-step reasoning",
        },
    ),
)
```

The classifier's own model call is traced and costed under the strategy's run, separately from the
route's, so you can see what the routing decision itself cost.

## Custom strategies

For anything else — an existing classifier, per-request state, an async call, the whole
transcript — write your own. The common case needs no class: a plain function of a
`RoutingRequest` returning a route name (or a `RoutingChoice`, or `None` for "can't decide").

```python
from langchain_llm_router import RoutingChoice, RoutingRequest, RoutingStrategy


def pick(request: RoutingRequest) -> str | None:
    return "frontier" if len(request.text) > 200 else "small"


router = ChatRouter(routes=routes, default_route="small", strategy=pick)
assert route_of(router, "short") == "small"


class ByLanguage(RoutingStrategy):
    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        if "def " in request.text:
            return RoutingChoice("coder", "the request contains a function definition")
        return None


router = ChatRouter(routes=routes, default_route="small", strategy=ByLanguage())
assert route_of(router, "def f(): pass") == "coder"
```

A plain function must be synchronous and sees only the current request. Subclass
`RoutingStrategy` for an async strategy (`adecide`) or to see the whole transcript
(`wants_full_context = True`). The interface, every `RoutingRequest` field and the stability
promise are in the [strategy reference](../strategies.md).

Story: [a custom strategy](../stories/custom-strategy.md).

## What a strategy sees

By default a strategy sees only the **current request**: the most recent user message, its content
blocks and modalities, the route names and whether tools are bound. It never sees tool output,
system prompts or conversation length, because routing on those misroutes agent loops — after a
tool call the "last message" is a tool result, and a long conversation says nothing about how hard
the next question is. A strategy that truly needs more sets `wants_full_context = True`.
