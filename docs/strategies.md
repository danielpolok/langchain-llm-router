# Strategies

A **strategy** applies your routing policy to one request. `ChatRouter` calls it once per
request — except when the route is [forced](../README.md#forcing-a-route), which skips the
strategy entirely — and uses its answer to pick a route (R6).

R6 asks for one interface that works at three levels: ready-made strategies you configure and
nothing more, a configurable component for policies you assemble from rules, and a small
interface for fully custom code. All three are the same `RoutingStrategy` — a ready-made
strategy uses no private hooks the interface doesn't already offer (REQ-R6-1).

## The three levels

| Level | Use when | Start here |
| --- | --- | --- |
| **Ready-made** | Your policy fits a known shape: keyword rules, or a cheap difficulty score | [`KeywordStrategy`](#keywordstrategy), [`HeuristicStrategy`](#heuristicstrategy) |
| **Configured** | You need to combine a few conditions of your own, without writing a class | [`ConfigurableStrategy`](#configurablestrategy) |
| **Custom** | Anything else — an existing classifier, embeddings, per-request state | [Custom strategies](#custom-strategies) |

All five built-ins are exported from `langchain_llm_router` directly.

### `KeywordStrategy`

Matches whole words in the current request's text against rules you write (R7 — no model or API
call). The first rule that matches wins, in declaration order; when nothing matches, the
strategy abstains and the default route answers (R9).

```python
from langchain_llm_router import KeywordStrategy

KeywordStrategy(
    {
        "coder": ["python", "regex", "stack trace"],  # a compiled re.Pattern works too
        "frontier": ["prove", "derive"],
    }
)
```

Matching is whole-word and case-insensitive (`"python"` matches `"Python?"`, not `"pythonic"`); a
multi-word keyword like `"stack trace"` matches across any run of whitespace. Pass a compiled
`re.Pattern` for anything a word list can't express — scripts with no word boundaries (Chinese,
Japanese, Thai), or a pattern like `re.compile(r"regexe?s?")`. See
[`examples/domain_routing.py`](../examples/domain_routing.py) and the module docstring in
`langchain_llm_router.strategies.keyword` for the full matching rules.

### `HeuristicStrategy`

Scores each request from signals computable on the spot — length, code, how many things it asks
for, whether it asks for reasoning, whether it carries anything but text — and the score picks a
tier (R7 — no model or API call). This is the §4 "cost tiering" use case.

```python
from langchain_llm_router import HeuristicStrategy

HeuristicStrategy("small", "frontier")  # two tiers, the default threshold
HeuristicStrategy("small", "mid", "frontier", thresholds=[1.0, 3.0])  # three tiers
```

Tiers are cheapest first; `thresholds` is one shorter than the tier list (ascending). `weights=`
scales an individual signal (`0.0` silences it) and `signals=` replaces the whole set — both
override module-level defaults that T-140's benchmark tunes, so a difficulty bar that feels wrong
today may simply not be tuned yet. See
[`examples/cost_tiering.py`](../examples/cost_tiering.py) and
`langchain_llm_router.strategies.heuristic` for the signal table and what the score can't see.

### `ConfigurableStrategy`

R6's middle level: a strategy built from `Rule`s, without writing a class. Both of REQ-R6-3's use
cases — cost tiering and domain routing — are configuration here.

```python
from langchain_llm_router import ConfigurableStrategy
from langchain_llm_router.strategies.configurable import Rule, any_of, keywords, signal_at_least
from langchain_llm_router.strategies.heuristic import code_signal, length_signal

long_or_code = any_of(signal_at_least(length_signal(20, 200), 0.5), code_signal)

ConfigurableStrategy(
    [
        Rule("frontier", long_or_code, name="long or code-bearing"),
        Rule("coder", keywords("python", "regex", "stack trace"), name="code"),
    ]
)
```

A `Rule` is `Rule(route, when, priority=0, name=None)`; the first rule (by declaration order,
then `priority`) whose condition holds wins. Conditions compose from `keywords(...)`,
`signal_at_least(signal, threshold)`, `modality(...)`, `tools_bound()`, `predicate(func)` and
`always()`, combined with `any_of`, `all_of` and `not_` — no other operators, no nesting syntax.
See `langchain_llm_router.strategies.configurable` for the full condition table and what each
one inherits from the request.

### Opt-in strategies: `EmbeddingStrategy`, `ClassifierStrategy`

Both make an extra call every request (an embedding call, or a chat model call), so neither is
ever on by default (R7, REQ-R7-2) — construction takes your own `Embeddings` or `BaseChatModel`
instance, with no default.

```python
from langchain_llm_router import EmbeddingStrategy

EmbeddingStrategy(
    my_embeddings,
    {"support": ["reset my password", "cancel my subscription"], "coder": ["fix this traceback"]},
    threshold=0.75,
)
```

```python
from langchain_llm_router import ClassifierStrategy

ClassifierStrategy(
    my_small_model,
    {"support": "account issues: billing, cancellations", "coder": "programming help"},
)
```

`EmbeddingStrategy` routes on cosine similarity to a handful of example requests per route — the
right *idea*, not the right word. `ClassifierStrategy` asks a chat model to pick a route from a
plain-language description of each one, through `with_structured_output`, the same mechanism the
router itself uses for tool calls. `ClassifierStrategy` is also this project's own template for
plugging in "an existing classifier" (§4): start from
`langchain_llm_router/strategies/classifier.py` and change only the prompt and the model.

## Custom strategies

For anything the built-ins don't cover — an existing classifier, per-request state, an async
call, seeing the whole transcript — write your own. The interface is deliberately small:

```python
from langchain_llm_router import RoutingChoice, RoutingRequest, RoutingStrategy


class MyStrategy(RoutingStrategy):
    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        # your policy; None sends the request to the default route (R9)
        ...
```

The common case needs no subclass at all: a plain function is coerced into a strategy
(REQ-R6-2).

```python
def pick(request: RoutingRequest) -> RoutingChoice | str | None:
    label = my_existing_classifier(request.text)
    return label  # a bare route name works — the router writes the reason for you


ChatRouter(routes=..., default_route=..., strategy=pick)
```

A plain function must be synchronous and sees only the current request's text; for an async
strategy, or the whole transcript, subclass `RoutingStrategy` directly. See
[`examples/custom_strategy.py`](../examples/custom_strategy.py) for a worked version of both.

## The interface reference

### `RoutingRequest`

What a strategy decides on — a frozen dataclass built fresh for every request:

| Field | Type | Holds |
| --- | --- | --- |
| `text` | `str` | The current request's text (R4) — the most recent `HumanMessage`, ignoring trailing AI and tool messages. Never tool output, system prompts, or conversation length. |
| `content_blocks` | `list[ContentBlock]` | The current request's content, as LangChain defines content blocks (C7). |
| `modalities` | `frozenset[str]` | What the request carries: `"text"`, `"image"`, `"audio"`, `"video"`, `"file"`, `"other"`. |
| `routes` | `tuple[str, ...]` | The router's route names, in declaration order. |
| `tools_bound` | `bool` | Whether tools or structured output are bound to this call. |
| `messages` | `list[BaseMessage] \| None` | The whole transcript — `None` unless the strategy sets `wants_full_context = True` (R4). |
| `config` | `RunnableConfig` | The strategy run's own child config (D9) — pass it to any model or embeddings call your strategy makes, so that call is traced and costed under the strategy's run, not the router's or the route's. Excluded from equality and repr: it identifies a run, not a request. |

By default a strategy sees only the current request — R4 exists because routing on tool output,
system prompts or conversation length misroutes agent loops. Set `wants_full_context = True` on
your `RoutingStrategy` subclass to receive the whole transcript when you genuinely need it.

### `RoutingChoice`

A strategy's answer: `RoutingChoice(route: str, reason: str)`. The reason rides on the trace and
under `response_metadata["routing"]`, for a human reading either one.

### `RoutingStrategy`

```python
class RoutingStrategy(ABC):
    wants_full_context: ClassVar[bool] = False

    @abstractmethod
    def decide(self, request: RoutingRequest) -> RoutingChoice | None: ...

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None: ...
```

`adecide` defaults to running `decide` in a worker thread, so `decide` must be thread-safe (one
strategy instance can serve concurrent requests) — and a strategy that calls a model overrides
`adecide` with a native async implementation, passing `request.config` along so the call nests
under the strategy's own run (D9). `ClassifierStrategy` is a worked example of that override.

`None` from either method — or a raised exception, or a route name the router doesn't have —
means "can't decide": the router falls back to the default route, with a `FallbackWarning` and
the reason recorded (R9). See [`decision-record.md`](decision-record.md) for the full fallback,
warning and error reference.

### `RoutingCallable`

`Callable[[RoutingRequest], RoutingChoice | str | None]` — what a plain function passed as
`strategy=` must look like. It is coerced into a `RoutingStrategy` at construction (REQ-R6-2).

## Stability promise (REQ-R6-4)

The interface is the four names above — `RoutingStrategy`, `RoutingRequest`, `RoutingChoice` and
`RoutingCallable` — their members and what each means. Until 1.0, the minor version stands in
for the major one:

- **Minor release (compatible):** a new `RoutingRequest` field, added last with a default; a new
  optional `RoutingStrategy` member whose default keeps today's behaviour, as
  `wants_full_context` does; new values in `modalities` as LangChain adds content-block types;
  new wording in the reasons the package writes; a widened type that keeps accepting everything
  it accepts today.
- **Major release (breaking):** removing or renaming anything above, or retyping it so that code
  which type-checks today no longer does; a new abstract method; changing what `None` or a bare
  route name means; changing the signature of `decide` or `adecide`; changing
  `wants_full_context`'s default.

Everything else in `langchain_llm_router.strategy` — helper functions, the underscore names — is
the router's own internals and may change in any release. The same promise, and the same
"everything else is internal" rule, applies to each built-in strategy's own constructor and
public fields; private helpers in `strategies/*.py` (leading underscore) are not part of the
API.
