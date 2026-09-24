# Forced routes and runtime config

Sometimes the strategy shouldn't decide: you're comparing models on the same traffic, reproducing a
bug on one model, or pinning a session to one route. Runtime config forces a route for one call and
skips the strategy entirely.

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision


def fake(name: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=cycle([AIMessage(f"({name})")]), name=name)


router = ChatRouter(
    routes={"small": fake("small"), "frontier": fake("frontier")},
    default_route="small",
    strategy=KeywordStrategy({"frontier": ["prove"]}),
)
```

## Force one call

```python
response = router.invoke("hello", config={"configurable": {"route": "frontier"}})
decision = routing_decision(response)
print(decision.route, decision.forced, decision.reason)
assert decision.route == "frontier" and decision.forced
```

The decision record says `forced=True`, and `strategy` is `None`: no strategy ran.

## Force every call on a handle

`with_config` fixes the route for everything called through the result, which is the way to pin
an agent or chain step:

```python
pinned = router.with_config(configurable={"route": "small"})
assert routing_decision(pinned.invoke("Prove it.")).route == "small"
```

The `"route"` key is declared in the router's `config_specs`, so it shows in config-schema
introspection and works through `bind_tools` and `with_structured_output` results as well.

## A forced route is never silently swapped

If the route you force doesn't exist, or can't use the tools bound to the call, the default is to
**raise** `ForcedRouteError`. A comparison that quietly ran on the wrong model is worse than none.

```python
from langchain_llm_router import ForcedRouteError

try:
    router.invoke("hello", config={"configurable": {"route": "nope"}})
except ForcedRouteError as error:
    print(error)
```

Build the router with `on_unavailable_forced_route="fallback"` to get the ordinary behaviour
instead: the default route answers, a `ForcedRouteWarning` fires and the reason is recorded.

```python
import warnings

from langchain_llm_router import ForcedRouteWarning

lenient = ChatRouter(
    routes={"small": fake("small"), "frontier": fake("frontier")},
    default_route="small",
    on_unavailable_forced_route="fallback",
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    response = lenient.invoke("hello", config={"configurable": {"route": "nope"}})
assert caught[0].category is ForcedRouteWarning
assert routing_decision(response).route == "small"
```

## Uses

- **Experiments.** Send the same request down two routes and compare — see
  [experimentation](../stories/experimentation.md).
- **Session pinning.** Provider prompt caching is per model, so keeping a whole conversation on one
  route keeps its cached prefix; see [the caveat](../scope.md#the-prompt-caching-caveat).
- **Debugging.** Reproduce a problem on one route regardless of what the strategy would choose.

The other configuration keys on a call are LangChain's own (`tags`, `metadata`, `callbacks`) and
work as on any chat model. The full error and warning reference is in
[the decision record](../decision-record.md#forced-routes).
