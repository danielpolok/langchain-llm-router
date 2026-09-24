# Response caching

LangChain's response cache works with the router, and **the route owns it**. The router delegates
each call to the selected route's own `invoke`, so the cache is consulted exactly where it would be
if you'd called that route directly — keyed by that route's own identity and call settings.

```python
from itertools import cycle

from langchain_core.caches import InMemoryCache
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, routing_decision

cache = InMemoryCache()
cached = GenericFakeChatModel(messages=cycle([AIMessage("an answer")]), name="cached", cache=cache)
router = ChatRouter(routes={"cached": cached}, default_route="cached")

first = router.invoke("the same question")
second = router.invoke("the same question")
print(len(cache._cache))  # one entry: the second call was a cache hit
assert len(cache._cache) == 1
```

## Where to set the cache

- **On each route:** `cache=True`, or a `BaseCache`, in the route's constructor.
- **Process-wide:** `set_llm_cache(InMemoryCache())` from `langchain_core.globals`. Every route
  without its own setting uses it.

Setting `cache=` on the **router** itself is rejected at construction with an error that says why:
the router never runs the cache lookup, so the setting would silently do nothing.

## A cache hit still tells the truth

A cache hit returns the route's stored message, and the router stamps it with the decision made for
*this* call — the strategy still runs on every request — not the record that was current when the
entry was written.

## Streaming

LangChain's `stream` doesn't consult the response cache on any chat model, and the router doesn't
change that.

## Caching across routes

Two routes with separate caches don't share entries: a question answered on `small` is a miss on
`frontier`. Provider-side *prompt* caching has a related caveat, since it is per model; see
[scope](../scope.md#the-prompt-caching-caveat).
