# Routing strategies

A **strategy** is your routing policy, written as code. For each request it names the route that
should answer and says why. If it can't tell, it abstains, and the router's default route answers.

A strategy sees the **current request**: the text of the user's latest message, any images or
files attached to it, and whether tools are bound. By default it doesn't see tool output, the
system prompt or the rest of the conversation. That keeps an agent's tool loop, or a long chat,
from changing where a simple question goes.

| Strategy | Decides by | Extra cost per request |
| --- | --- | --- |
| [`HeuristicStrategy`](#heuristicstrategy-route-on-difficulty) | A difficulty score | None |
| [`KeywordStrategy`](#keywordstrategy-route-on-words) | Words in the request | None |
| [`ConfigurableStrategy`](#configurablestrategy-combine-rules) | Rules you combine | None |
| [`EmbeddingStrategy`](#embeddingstrategy-route-on-meaning) | Similarity to example requests | One embedding call |
| [`ClassifierStrategy`](#classifierstrategy-let-a-small-model-choose) | A small model's choice | One small-model call |
| [Your own](#your-own-strategy) | Your code | Whatever your code costs |

## Setup

Every example on this page uses these two models and a small helper. The helper sends one
question and prints the route that answered it, with the reason.

```python
from langchain.chat_models import init_chat_model

from langchain_llm_router import ChatRouter, routing_decision

small = init_chat_model("google_genai:gemini-3.5-flash-lite")
frontier = init_chat_model("google_genai:gemini-3.8-flash")


def ask(router, question):
    decision = routing_decision(router.invoke(question))
    print(f"{decision.route:<8} {decision.reason}")
```

## `HeuristicStrategy`: route on difficulty

This strategy sends easy requests to a cheap model and hard ones to a strong model. It judges
difficulty from signals it can compute instantly, without calling a model, and adds them up into
a **difficulty score**.

```python
from langchain_llm_router import HeuristicStrategy

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),
)

ask(router, "What's the capital of France?")
ask(router, "What is a monad? How is it different from a functor?")
ask(router, "Why does this fail?\n```python\nprint({}['x'])\n```\nKeyError: 'x'")
```

```text
small    difficulty 0.00 < 1.00 (no signal fired)
small    difficulty 0.50 < 1.00 (parts 0.50)
frontier difficulty 1.50 >= 1.00 (code 1.00, analysis 0.50)
```

### How the score works

Each signal gives a strength from 0 (not present) to 1 (fully present):

| Signal | Looks for | 0 when | 1 when |
| --- | --- | --- | --- |
| `length` | How many words the request has | 20 words or fewer | 200 words or more |
| `code` | Code: a ```` ``` ```` fence, code syntax (`def f(`, `SELECT … FROM`, `=>`), an error traceback | None of them | Two of the three kinds |
| `parts` | How many things it asks: question marks or list items | One | Three or more |
| `analysis` | Reasoning words: *compare, analyse, explain why, trade-off, design, debug, prove, optimise, refactor, root cause, why does …* | None | Two different ones |
| `modalities` | Anything but text: an image, audio, video or a file | Text only | Anything else |

Between those bounds a signal rises linearly. For example, 110 words gives a `length` of 0.5, and
two questions give a `parts` of 0.5.

The **score** is the sum of the signals, each multiplied by its weight. Every weight defaults to
1, so the score runs from 0 to 5. The **threshold** decides the tier. A score below it goes to the
first tier, and a score at or above it goes to the second. The default threshold is **1.0**: one
signal fully present, or two half present, is enough for the frontier model.

In the examples above:

- The first question shows no signal, so it scores 0.00.
- The second asks two questions, so `parts` is 0.5. That alone isn't enough.
- The third has two kinds of code marker, a fence and a traceback, so `code` is 1.0. "Why does"
  is one reasoning phrase, which adds 0.5. The total is 1.50.

The reason always shows the score, the threshold and the signals that contributed, so a trace
tells you exactly why a request went where it did.

### Tuning it

**Raise or lower the bar.** A higher threshold sends fewer requests to the frontier model:

```python
stricter = HeuristicStrategy("small", "frontier", thresholds=[2.0])
```

**Add a middle tier.** Give one threshold per step, in ascending order. A score below 1.0 goes to
`small`, a score from 1.0 up to 3.0 goes to `mid`, and 3.0 or more goes to `frontier`:

```python
three_tiers = HeuristicStrategy("small", "mid", "frontier", thresholds=[1.0, 3.0])
```

**Change what counts.** Use `weights=` to make one signal count for more or less. A weight of `0`
turns a signal off:

```python
code_heavy = HeuristicStrategy("small", "frontier", weights={"code": 2.0, "length": 0.5})
```

**Add your own signal.** A signal is any function that takes the request and returns a number
from 0 to 1. Add it to the default set with `signals=`:

```python
from langchain_llm_router.strategies.heuristic import DEFAULT_SIGNALS


def mentions_money(request):
    return 1.0 if "$" in request.text or "invoice" in request.text.lower() else 0.0


router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy(
        "small", "frontier", signals={**DEFAULT_SIGNALS, "money": mentions_money}
    ),
)

ask(router, "Check this invoice: 3 hours at $40 comes to $150.")
```

```text
frontier difficulty 1.00 >= 1.00 (money 1.00)
```

> [!NOTE]
> **What the score can't see.** `length` counts words separated by spaces, and the reasoning
> words are English. A hard question in Chinese, Japanese or Thai can score 0. Length is also
> only a stand-in for difficulty: a long pasted log can score high, and a short but deep question
> can score low. If that matches your traffic, replace a signal or try a strategy that reads
> meaning, such as [embeddings](#embeddingstrategy-route-on-meaning) or a
> [classifier](#classifierstrategy-let-a-small-model-choose).

## `KeywordStrategy`: route on words

This strategy routes a topic to the model that suits it, using words that give the topic away. In
the examples below, programming questions go to the `coder` route and everything else goes to
`general`.

```python
from langchain_llm_router import KeywordStrategy

router = ChatRouter(
    routes={"general": small, "coder": frontier},
    default_route="general",
    strategy=KeywordStrategy({"coder": ["python", "sql", "regex", "stack trace"]}),
)

ask(router, "Write a regex that matches an email address")
ask(router, "Suggest a name for my cat")
```

```text
coder    matched keyword 'regex'
general  KeywordStrategy could not decide; fell back to the default route
FallbackWarning: KeywordStrategy could not decide; falling back to the default route 'general'
```

How it matches:

- **Whole words, ignoring case.** `"python"` matches "Python?" but not "pythonic". A phrase such as
  `"stack trace"` matches across any spaces or line breaks.
- **The first match wins.** Rules are tried route by route, then keyword by keyword, in the order
  you wrote them. Put specific keywords before general ones.
- **Regular expressions are opt-in.** Pass a compiled pattern instead of a string, for example
  `re.compile(r"\bdef \w+\(")`. It is searched exactly as compiled.
- **No match means the strategy abstains.** The default route answers and the router issues a
  `FallbackWarning`, so a request that nothing matched never goes unnoticed. If "everything else
  goes to the default" is part of your policy, write it as an explicit rule with
  [`ConfigurableStrategy`](#configurablestrategy-combine-rules) and `always()`.

## `ConfigurableStrategy`: combine rules

This strategy suits a policy with several conditions, such as "attachments or legal questions go
to the frontier model, and everything else goes to the small one". Each `Rule` names a route, a
condition and optionally a name for the reason:

```python
from langchain_llm_router import ConfigurableStrategy
from langchain_llm_router.strategies.configurable import (
    Rule,
    always,
    any_of,
    keywords,
    modality,
    signal_at_least,
)
from langchain_llm_router.strategies.heuristic import code_signal

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=ConfigurableStrategy(
        [
            Rule("frontier", modality("image", "file"), name="has an attachment"),
            Rule(
                "frontier",
                any_of(keywords("contract", "lawsuit"), signal_at_least(code_signal, 0.5)),
                name="legal or code",
            ),
            Rule("small", always(), name="everything else"),
        ]
    ),
)

ask(router, "Is this contract clause enforceable: the tenant waives all repair rights?")
ask(router, "Give me three name ideas for a bakery.")
```

```text
frontier rule 'legal or code' matched: keyword 'contract'
small    rule 'everything else' matched: always
```

The conditions you can use:

| Condition | Holds when |
| --- | --- |
| `keywords("python", "sql")` | Any of the words appears, matched as `KeywordStrategy` does |
| `signal_at_least(code_signal, 0.5)` | A [heuristic signal](#how-the-score-works), or your own, scores at least that much |
| `modality("image", "audio")` | The request carries any of those kinds of content |
| `tools_bound()` | Tools or structured output are bound to the call |
| `predicate(func)` | `func(request)` returns `True`, for anything else |
| `always()` | Always. Use it for a final "everything else" rule |
| `any_of(...)`, `all_of(...)`, `not_(...)` | Combine the conditions above |

Rules are tried from the highest `priority=` to the lowest. The default priority is `0`, so
without priorities the first matching rule wins. Mistakes that can't work are caught when the
strategy is built. For example, a rule that can never fire because an earlier rule always matches
first is rejected at construction instead of failing silently later.

## `EmbeddingStrategy`: route on meaning

Use this when a topic has no telltale words. "My loop never terminates" is a programming question,
but it contains none of the keywords above. Give each route a few example requests, and each new
request goes to the route whose example it is closest to in meaning.

Any LangChain [embeddings model](https://docs.langchain.com/oss/python/integrations/embeddings)
works. This example uses Gemini's:

```python
from langchain.embeddings import init_embeddings

from langchain_llm_router import EmbeddingStrategy

embeddings = init_embeddings("google_genai:gemini-embedding-001")

router = ChatRouter(
    routes={"general": small, "coder": frontier},
    default_route="general",
    strategy=EmbeddingStrategy(
        embeddings,
        {
            "coder": [
                "Why is my Python function raising a KeyError?",
                "Refactor this SQL query so it runs faster",
                "Write a unit test for this class",
            ],
        },
        threshold=0.6,
    ),
)

ask(router, "My loop never terminates, what's wrong with it?")
ask(router, "Recommend a good book about the Roman Empire")
```

```text
coder    embedding similarity 0.64 >= 0.60 to 'Why is my Python function raising a KeyError?' (route 'coder')
general  EmbeddingStrategy could not decide; fell back to the default route
FallbackWarning: EmbeddingStrategy could not decide; falling back to the default route 'general'
```

How it decides:

1. **The examples are embedded once**, in one batched call on first use, and kept for the life of
   the strategy.
2. **Each request is embedded**, which is one embedding call per request.
3. **It is compared with every example** by cosine similarity. The route of the closest example
   wins.
4. **It routes only if the similarity reaches `threshold`.** Otherwise it abstains and the default
   route answers.

**Choosing the threshold.** `threshold` has no default because similarity values depend on the
embedding model. Some models rarely score unrelated text below 0.9, and others spread scores much
wider. To calibrate, run requests that are typical of each route and read the scores in the
reasons. With `gemini-embedding-001`, related requests scored about 0.62 to 0.64 and unrelated
ones about 0.55, so 0.6 separates them. More varied examples per route help more than a finely
tuned threshold.

## `ClassifierStrategy`: let a small model choose

Describe each route in plain words, and a chat model reads the request and picks one. This is the
most accurate way to separate subtle topics. It costs one extra model call per request, so give it
a small, fast model:

```python
from langchain_llm_router import ClassifierStrategy

router = ChatRouter(
    routes={"general": small, "coder": frontier},
    default_route="general",
    strategy=ClassifierStrategy(
        small,
        {
            "coder": "programming: code, bugs, errors, SQL, tooling",
            "general": "anything that isn't programming",
        },
    ),
)

ask(router, "My loop never terminates, what's wrong with it?")
ask(router, "Recommend a good book about the Roman Empire")
```

```text
coder    the classifier chose 'coder': programming: code, bugs, errors, SQL, tooling
general  the classifier chose 'general': anything that isn't programming
```

The model gets a prompt listing each route and its description, followed by the request. It must
answer through structured output with one of the route names, so it can't invent a route. If
its answer doesn't parse, or the call fails, the strategy abstains and the default route answers.
The classifier's call appears in your trace and in token usage, under the strategy.

> [!TIP]
> On [our benchmark](../benchmark/README.md), the classifier picked the intended tier for 31 of 32
> requests, but still saved less than the free heuristic, because its own call is billed too.
> Reach for it when the free strategies measurably misroute your traffic.

## Your own strategy

Your policy may depend on your own data, such as a customer's plan, a feature flag or a classifier
you already run. In that case, write the strategy yourself.

### A function

Any function that takes the request and returns a route name (or `None` to abstain) is a
strategy. The request's `config` carries the runtime config of the call, so a function can route
on anything you pass in:

```python
def by_plan(request):
    plan = request.config.get("configurable", {}).get("plan", "free")
    return "frontier" if plan == "pro" else "small"


router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=by_plan,
)

response = router.invoke("Hi there!", config={"configurable": {"plan": "pro"}})
print(routing_decision(response).reason)
```

```text
by_plan chose 'frontier'
```

Return a `RoutingChoice(route, reason)` instead of a bare name to write the reason yourself. A
function must be synchronous. The type it must match is `RoutingCallable`.

### A strategy class

Subclass `RoutingStrategy` when you need state, async support or the whole conversation. This one
keeps a conversation on the route that answered its previous turn. That way a follow-up question
stays with the model that has the context, and keeps the provider's prompt cache warm:

```python
from langchain_llm_router import RoutingChoice, RoutingStrategy


class Sticky(RoutingStrategy):
    """Keep a conversation on the route that answered it last."""

    wants_full_context = True  # receive the whole conversation in request.messages

    def __init__(self, first_turn):
        self.first_turn = first_turn

    def decide(self, request):
        for message in reversed(request.messages):
            decision = routing_decision(message)  # None unless the router answered it
            if decision is not None:
                return RoutingChoice(decision.route, f"conversation is on {decision.route!r}")
        return self.first_turn.decide(request)


router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=Sticky(HeuristicStrategy("small", "frontier")),
)

conversation = [("user", "Compare the trade-offs of B-trees and LSM trees for databases.")]
first = router.invoke(conversation)
conversation += [first, ("user", "Thanks! Which one does SQLite use?")]
second = router.invoke(conversation)

print(routing_decision(first).reason)
print(routing_decision(second).reason)
```

```text
difficulty 1.00 >= 1.00 (analysis 1.00)
conversation is on 'frontier'
```

On its own, "Which one does SQLite use?" scores 0 and would go to the small model.

`decide` returns a `RoutingChoice` or `None`. A few rules apply:

- **Keep `decide` thread-safe.** The router may call it from several threads at once.
  `adecide`, the async form, runs `decide` in a worker thread by default.
- **Pass `request.config` on.** If your strategy calls a model itself, override `adecide` with a
  native async version, and pass `request.config` to the call so it is traced and costed under
  the strategy. `ClassifierStrategy` does exactly this.
- **Failures fall back.** If the strategy raises, or names a route the router doesn't have, the
  default route answers and a `FallbackWarning` says why.

### What a strategy sees

`decide` receives a `RoutingRequest`:

| Field | Holds |
| --- | --- |
| `text` | The current request's text: the user's latest message, never tool output |
| `content_blocks` | The current request's content, as LangChain content blocks |
| `modalities` | What it carries: `"text"`, `"image"`, `"audio"`, `"video"`, `"file"`, `"other"` |
| `routes` | The router's route names, in the order they were declared |
| `tools_bound` | Whether tools or structured output are bound to the call |
| `messages` | The whole conversation, only if the strategy sets `wants_full_context = True` |
| `config` | The call's runtime config. Pass it to any model your strategy calls |

### Stability promise

`RoutingStrategy`, `RoutingRequest`, `RoutingChoice` and `RoutingCallable` are a public interface
with a compatibility promise. Until 1.0, the minor version plays the role of the major one:

- **Minor releases stay compatible.** They may add a `RoutingRequest` field (last, with a
  default), an optional `RoutingStrategy` member whose default keeps today's behaviour, new
  `modalities` values, and new wording in the built-in strategies' reasons.
- **Anything breaking needs a major release.** That covers removing, renaming or retyping any of
  the above, adding an abstract method, changing what `None` or a bare route name means, or
  changing `decide` / `adecide` or the default of `wants_full_context`.

The same promise covers each built-in strategy's constructor and public attributes. Names that
start with an underscore are internal and may change in any release.
