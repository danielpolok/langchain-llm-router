# Fallbacks and retries

There are two different things that can go wrong, and the router treats them differently.

| What failed | Who handles it |
| --- | --- |
| The **strategy** couldn't decide | The router: the default route answers, with a warning |
| The **route** raised (network error, rate limit, provider outage) | You, with LangChain's `with_retry` and `with_fallbacks` |

## When the strategy can't decide

The router always answers. If the strategy raises, returns `None` or names a route that doesn't
exist, the default route serves the request. Each case issues a `FallbackWarning` and sets
`fallback=True` on the [decision record](../decision-record.md), with the cause in `reason`.

```python
import warnings
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    KeywordStrategy,
    routing_decision,
)


def fake(name: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=cycle([AIMessage(f"({name})")]), name=name)


router = ChatRouter(
    routes={"small": fake("small"), "coder": fake("coder")},
    default_route="small",
    strategy=KeywordStrategy({"coder": ["python"]}),
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    response = router.invoke("What's the capital of Poland?")  # no keyword matches

assert caught[0].category is FallbackWarning
decision = routing_decision(response)
print(decision.fallback, decision.reason)
assert decision.fallback and decision.route == "small"
```

Choosing the default route deliberately when nothing matches is normal for a keyword strategy. If
the warning is noise for you, filter it with `warnings.filterwarnings("ignore", category=FallbackWarning)`;
the record still says what happened.

## When the route fails

The router does not absorb a route's own failure. If the selected model raises, the exception
propagates unchanged and the router's run ends as an error — you see the provider's error, not a
wrapped one. Retries and fallbacks belong to LangChain, so use its mechanisms on the router.

`with_retry` re-runs a failed call. Because each attempt goes back through the router, the strategy
decides again:

```python
resilient = router.with_retry(stop_after_attempt=3)
assert routing_decision(resilient.invoke("Fix my python.")).route == "coder"
```

`with_fallbacks` tries other runnables if the router raises — for example a different provider's
model, or a second router:

```python
backup = fake("backup")
guarded = router.with_fallbacks([backup])
assert guarded.invoke("Fix my python.").text == "(coder)"
```

Retries **on a route** are also legitimate: configure them in the provider's client
(`max_retries=`), which every provider integration offers. What you cannot do is wrap a route in
a runnable — `routes={"small": model.with_retry()}` is rejected at construction, because the router
needs to ask a route what it can do (tool support, its profile, its cache), and a wrapper answers
none of that.

## Summary

- Strategy problems degrade to the default route, visibly.
- Route problems surface as they are, and you decide what to do with LangChain's tools.
- A forced route that can't be used raises by default; see [forced routes](forced-routes.md).
