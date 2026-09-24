# Get started

By the end of this page you'll have a router with two routes, know how to pick a strategy, read
why a request went where it did, and be able to test it without a network call.

## 1. Install

```bash
pip install langchain-llm-router
```

## 2. Build your routes

A route is any LangChain chat model. In an application they come from `init_chat_model` or a
provider class; each needs whatever credentials its provider needs, and the router needs none of
its own.

```python skip
from langchain.chat_models import init_chat_model

small = init_chat_model("ollama:qwen3:8b")
frontier = init_chat_model("google_genai:gemini-3-flash-preview")
```

To follow along offline, use fakes from `langchain-core` instead. `cycle` makes a fake answer
forever rather than run out after one reply.

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

small = GenericFakeChatModel(messages=cycle([AIMessage("2 + 2 is 4.")]), name="small")
frontier = GenericFakeChatModel(
    messages=cycle([AIMessage("Suppose there were finitely many primes...")]), name="frontier"
)
```

## 3. Choose a strategy

The strategy is your routing policy. Start with the simplest one that fits:

| Your policy | Use | Extra model calls |
| --- | --- | --- |
| "Requests that mention X go to route R" | `KeywordStrategy` | none |
| "Short and simple to a cheap model, long or hard to a strong one" | `HeuristicStrategy` | none |
| "A few conditions of mine, combined" | `ConfigurableStrategy` | none |
| "The request means something like these examples" | `EmbeddingStrategy` | one embedding call |
| "Ask a small model which route fits" | `ClassifierStrategy` | one chat call |
| "I already have a classifier" | a plain function | yours |

For two tiers by difficulty, `HeuristicStrategy("small", "frontier")` lists the routes cheapest
first:

```python
from langchain_llm_router import ChatRouter, HeuristicStrategy

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),
)
```

Each strategy is described on [its own page](capabilities/strategies.md).

## 4. Call it like any chat model

```python
response = router.invoke("Prove that there are infinitely many primes, and explain why it works.")
print(response.text)
```

`response` is the selected route's own `AIMessage`: its content, tool calls and usage metadata,
untouched. The router adds one thing — the routing decision — under
`response_metadata["routing"]`.

## 5. Read the decision

```python
from langchain_llm_router import routing_decision

decision = routing_decision(response)
print(decision.route)  # which route ran
print(decision.reason)  # why, in words
assert decision.route == "frontier"
```

`routing_decision(response)` returns a [`RoutingDecision`](decision-record.md#the-decision-record):
the route, the reason, the strategy's name and whether the answer came from a fallback, a forced
route or a diversion. The same record is in your traces.

Try the easy question and compare:

```python
easy = routing_decision(router.invoke("what's 2 + 2?"))
print(easy.route, "-", easy.reason)
assert easy.route == "small"
```

## 6. Test it offline

Fakes are how this project tests itself, and how you can test yours. A strategy is deterministic
for a given request, so a test can assert which route ran:

```python
def test_arithmetic_stays_on_the_small_model() -> None:
    reply = router.invoke("what's 2 + 2?")
    assert routing_decision(reply).route == "small"


test_arithmetic_stays_on_the_small_model()
```

To pin one call to a route regardless of the strategy — for example to assert what each model
answers — see [forced routes](capabilities/forced-routes.md).

## When it doesn't decide

If the strategy raises, abstains or names a route you don't have, the default route answers and a
`FallbackWarning` says why. Nothing is lost silently. See
[fallbacks and retries](capabilities/fallbacks-and-retries.md).

## Next

- [Strategies](capabilities/strategies.md) — every built-in strategy in one place.
- [User stories](index.md#where-to-go-next) — worked examples for the common goals.
- [Tools and structured output](capabilities/tools-and-structured-output.md) — before you use the
  router in an agent.
