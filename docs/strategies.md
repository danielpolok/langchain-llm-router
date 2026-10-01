# Routing strategies

A **strategy** is your routing policy, written as code. For each request it names the route that
should answer and says why. If it can't tell, it abstains, and the router's default route answers.

A strategy sees the **current request**: the text of the user's latest message, any images or
files attached to it, and whether tools are bound. By default it doesn't see tool output, the
system prompt or the rest of the conversation. That keeps an agent's tool loop, or a long chat,
from changing where a simple question goes. To route a short follow-up in the light of what the
user asked before, give the strategy [`lookback`](#follow-up-questions-lookback).

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

**Keep follow-ups on the stronger model.** "Thanks! Which one does SQLite use?" scores 0 on its
own. With `lookback=2`, the strategy scores the user's two previous messages too, and the
hardest one decides. See [Follow-up questions](#follow-up-questions-lookback).

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
- **Follow-ups can match on earlier messages.** With `lookback=2`, a request with no match of
  its own is matched on the user's two previous messages, and the newest match decides. See
  [Follow-up questions](#follow-up-questions-lookback).

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

With `lookback=2`, the rules are also tried on the user's two previous messages, and the newest
message that a rule matches decides. See [Follow-up questions](#follow-up-questions-lookback).

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

With `lookback=2`, a request that clears no threshold is routed by the newest of the user's two
previous messages that does. See [Follow-up questions](#follow-up-questions-lookback).

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

With `lookback=2`, the prompt also shows the user's two previous messages, so the model reads a
follow-up in their light. See [Follow-up questions](#follow-up-questions-lookback), which also
covers caching the classifier's answers.

## Follow-up questions: `lookback`

People rarely repeat the topic in a follow-up. "Thanks! Which one does SQLite use?" looks easy,
although it continues a question that needed the frontier model, so on its own it goes to the
small one. Give the strategy `lookback=N`, and it reads the user's previous N messages as well as
the current one.

The examples in this section send each question as part of one conversation, with the answers
before it, the way a chat app does:

```python
def chat(router, *questions):
    conversation = []
    for question in questions:
        conversation.append(("user", question))
        response = router.invoke(conversation)
        conversation.append(response)
        decision = routing_decision(response)
        print(f"{decision.route:<8} {decision.reason}")


router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier", lookback=2),
)

chat(
    router,
    "Compare the trade-offs of B-trees and LSM trees for databases.",
    "Thanks! Which one does SQLite use?",
    "And Postgres?",
    "Suggest a name for my cat",
)
```

```text
frontier difficulty 1.00 >= 1.00 (analysis 1.00)
frontier difficulty 1.00 >= 1.00 (analysis 1.00) (1 message back)
frontier difficulty 1.00 >= 1.00 (analysis 1.00) (2 messages back)
small    difficulty 0.00 < 1.00 (no signal fired)
```

The two follow-ups stayed on the frontier model, and each reason says which message decided and
how far back it was. The decision record carries the distance as a number too, `messages_back`
(`0` for the current message), so a dashboard can count the follow-ups an earlier message decided
without reading the reason. By the fourth question the hard one was three messages back, out of reach,
so the conversation came back down to the small model.

Each strategy reads the extra messages its own way:

| Strategy | With `lookback=N` |
| --- | --- |
| `HeuristicStrategy` | The hardest of the messages decides. A conversation moves up at once, and back down after N easier messages. |
| `KeywordStrategy` | The newest message with a keyword match decides. |
| `ConfigurableStrategy` | The newest message that a rule matches decides. An `always()` rule applies only when none of them matched. |
| `EmbeddingStrategy` | The newest message that clears `threshold` decides. |
| `ClassifierStrategy` | The previous messages go into the prompt as context, and the model classifies the latest one in their light. |

With `KeywordStrategy`, a follow-up with no keyword of its own goes where the question before it
went:

```python
router = ChatRouter(
    routes={"general": small, "coder": frontier},
    default_route="general",
    strategy=KeywordStrategy({"coder": ["python", "sql", "regex"]}, lookback=2),
)

chat(
    router,
    "Write a regex that matches an email address",
    "Now make it reject addresses with a plus sign",
)
```

```text
coder    matched keyword 'regex'
coder    matched keyword 'regex' (1 message back)
```

With `ConfigurableStrategy`, the rules are tried on each message in turn, newest first. The final
`always()` rule waits until no message matched, so it doesn't answer the follow-up before the
legal rule has seen the question it follows:

```python
router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=ConfigurableStrategy(
        [
            Rule("frontier", keywords("contract", "clause", "liability"), name="legal"),
            Rule("small", always(), name="everything else"),
        ],
        lookback=2,
    ),
)

chat(
    router,
    "Review this contract clause on limitation of liability",
    "Is it enforceable in Germany?",
)
```

```text
frontier rule 'legal' matched: keyword 'contract'
frontier rule 'legal' matched: keyword 'contract' (1 message back)
```

With `EmbeddingStrategy`, a follow-up that is close to no example goes where the question before
it went. Set the threshold above where a vague message lands first. Against these examples,
`gemini-embedding-001` scores a short message that says nothing about programming at about 0.6,
so at the 0.6 used above a follow-up clears it by chance and lookback never gets a turn. Here the
threshold is 0.7:

```python
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
        threshold=0.7,
        lookback=2,
    ),
)

chat(
    router,
    "My Python function raises a KeyError when it reads the config file",
    "Alright, and how would you explain that to a child?",
    "Great, what should I cook for dinner tonight?",
)
```

```text
coder    embedding similarity 0.79 >= 0.70 to 'Why is my Python function raising a KeyError?' (route 'coder')
coder    embedding similarity 0.79 >= 0.70 to 'Why is my Python function raising a KeyError?' (route 'coder') (1 message back)
coder    embedding similarity 0.79 >= 0.70 to 'Why is my Python function raising a KeyError?' (route 'coder') (2 messages back)
```

The follow-up stayed with the programming question, and so did the question about dinner: it
matches no example either, and to an embedding a message that matches nothing looks the same
whether it continues the topic or leaves it.

`ClassifierStrategy` can tell the two apart, because the model reads the earlier messages with
the new one. "Still the same. Any other ideas?" on its own goes to `general`. After a
programming question, it goes to `coder`:

```python
router = ChatRouter(
    routes={"general": small, "coder": frontier},
    default_route="general",
    strategy=ClassifierStrategy(
        small,
        {
            "coder": "programming: code, bugs, errors, SQL, tooling",
            "general": "anything that isn't programming",
        },
        lookback=2,
    ),
)

chat(
    router,
    "My deploy script fails with KeyError: 'user'",
    "Still the same. Any other ideas?",
    "Thanks! What should I cook for dinner tonight?",
)
```

```text
coder    the classifier chose 'coder': programming: code, bugs, errors, SQL, tooling
coder    the classifier chose 'coder': programming: code, bugs, errors, SQL, tooling
general  the classifier chose 'general': anything that isn't programming
```

The prompt lists the earlier messages, oldest first, as context, and asks the model to classify
only the latest one. That is the message it classifies, so the reason has no "messages back",
and `messages_back` is `0`. The model
is not told which route answered before, so one misrouted turn doesn't pull the next ones after
it. It also doesn't see the model's answers, which are long and would make every classification
cost more.

What to know before you turn it on:

- **Only the user's messages count.** The model's answers, tool output and the system prompt
  never take one of the N places. Inside an agent's tool loop, the earlier messages are the ones
  before the question that started the loop.
- **What it costs depends on the strategy.** The first three make no call, so it costs nothing.
  `EmbeddingStrategy` embeds each message once: it remembers the result for the last few thousand
  messages, so a turn embeds only its own new message, and at most N+1 when earlier ones have been
  forgotten, such as after a restart. `ClassifierStrategy` makes one call as before, with up to N
  more messages in the prompt.
- **Keep N small.** Two or three is usually enough, since a follow-up leans on the last message
  or two. A larger N also holds a topic longer after the user has moved on: a question about
  something else still goes to `coder` while a keyword match is within N messages.
- **The newest match still wins.** Lookback helps a message with no signal of its own. The rule
  and embedding strategies can't correct one with a misleading signal: in a legal conversation,
  "Which article of the civil code covers this?" matches `code`. The classifier reads it in the
  light of the legal question before it.
- **Say "everything else" with `always()`.** In `ConfigurableStrategy`, a `not_(...)` rule holds
  for the follow-up itself, so it decides before lookback reaches the question that set the
  topic.

**Caching the classifier.** Give the classifier model LangChain's exact cache, and the same
prompt is classified only once:

```python
from langchain_core.caches import InMemoryCache

classifier = init_chat_model("google_genai:gemini-3.5-flash-lite", cache=InMemoryCache())
```

Pass `classifier` to `ClassifierStrategy` in place of `small`. Inside an agent's tool loop, every
step shows the classifier the same messages, since tool calls and results take none of the
places, so a loop makes one classifier call instead of one per step, and its route can't change
halfway through. Don't give the classifier a semantic cache, which answers a request with
whatever a similar earlier one got. That turns the classifier into an embedding lookup against
past requests, without a threshold or a reason you can read.

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

Subclass `RoutingStrategy` when you need state, async support, the user's earlier messages or the
whole conversation. This one keeps a conversation on the route that answered its previous turn.
That way a follow-up question stays with the model that has the context, and keeps the provider's
prompt cache warm:

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
[`lookback`](#follow-up-questions-lookback) keeps it on the frontier model too, by reading the
question before it. `Sticky` differs in never moving a conversation once it has a route.

`decide` returns a `RoutingChoice` or `None`. A few rules apply:

- **Keep `decide` thread-safe.** The router may call it from several threads at once.
  `adecide`, the async form, runs `decide` in a worker thread by default.
- **Pass `request.config` on.** If your strategy calls a model itself, override `adecide` with a
  native async version, and pass `request.config` to the call so it is traced and costed under
  the strategy. `ClassifierStrategy` does exactly this.
- **Failures fall back.** If the strategy raises, or names a route the router doesn't have, the
  default route answers and a `FallbackWarning` says why.
- **Set `lookback` to read earlier messages.** With `lookback = 2` on the class, or
  `self.lookback = 2` in `__init__`, `request.previous_requests` holds up to two of the user's
  previous messages, newest first. Each is the `RoutingRequest` the router read when that
  message arrived, so you can hand it to another strategy's `decide`. Its `messages` end at that
  message; the whole conversation is on the current request. A plain function always sees the
  current request alone.
- **Pass `lookback` on when you wrap a strategy.** The router hands over as many previous
  messages as the strategy you give it asks for, and a built-in strategy reads at most its own
  `lookback`. A strategy that hands the request to another one, as `Sticky` does, sets
  `self.lookback = inner.lookback` so the two agree. To read a different number, it keeps
  `inner.with_lookback(n)` instead, a copy that reads `n` messages, and takes the copy's
  `lookback`. The strategy you passed in is left as it was.
- **Say which message decided.** A strategy that reads earlier messages returns
  `RoutingChoice(route, reason, messages_back=n)`, where `n` is how many of the user's messages
  back the deciding one was, and the record carries it. A wrapper hands back the inner
  strategy's choice as it is, or changes it with `dataclasses.replace`, so the number survives.

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
| `previous_requests` | Up to `lookback` of the user's previous messages, newest first, each the `RoutingRequest` read when that message arrived. Empty unless the strategy sets `lookback` |

### Stability promise

`RoutingStrategy`, `RoutingRequest`, `RoutingChoice` and `RoutingCallable` are a public interface
with a compatibility promise. Until 1.0, the minor version plays the role of the major one:

- **Minor releases stay compatible.** They may add a `RoutingRequest` or `RoutingChoice` field
  (last, with a default, as `previous_requests` and `messages_back` were), an optional
  `RoutingStrategy` member whose default keeps today's behaviour (as `wants_full_context`,
  `lookback` and `with_lookback` are), new `modalities` values, and new wording in the built-in
  strategies' reasons.
- **Anything breaking needs a major release.** That covers removing, renaming or retyping any of
  the above, adding an abstract method, changing what `None` or a bare route name means, changing
  `decide` / `adecide`, changing the default of `wants_full_context` or `lookback`, or changing
  what `lookback` or `messages_back` counts or the order of `previous_requests`.

The same promise covers each built-in strategy's constructor and public attributes. Names that
start with an underscore are internal and may change in any release.
