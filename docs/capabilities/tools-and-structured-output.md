# Tools and structured output

`bind_tools` and `with_structured_output` work on the router as on any chat model. The router
replays the binding onto whichever route it picks, so a request that lands on a route gets that
route's own tool-calling, in that provider's own format.

The one wrinkle is routes that **can't** use tools. This page covers what the router does about
them.

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's sunny in {city}."


class ToolCapableFake(GenericFakeChatModel):
    """A fake that can bind tools, as any real tool-calling model can."""

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=list(tools), **kwargs)


call = AIMessage("", tool_calls=[{"name": "get_weather", "args": {"city": "Warsaw"}, "id": "1"}])
small = GenericFakeChatModel(messages=cycle([AIMessage("hi")]), name="small")  # no tool support
frontier = ToolCapableFake(messages=cycle([call]), name="frontier")

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=KeywordStrategy({"frontier": ["weather"]}),
)
```

## Binding tools

```python
import warnings

from langchain_llm_router import ToolSupportWarning

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with_tools = router.bind_tools([get_weather])

assert caught[0].category is ToolSupportWarning
print(caught[0].message)  # routes that can't use tools: 'small'; ...

answer = with_tools.invoke("What's the weather in Warsaw?")
print(answer.tool_calls)
assert routing_decision(answer).route == "frontier"
```

Binding tells you up front, with a `ToolSupportWarning`, which routes can't use the tools. It does
not fail — the routes that *can* still serve tool requests.

If **no** route can use tools, binding raises `NoToolCapableRouteError`, because there would be
nothing left to route to.

## What happens to a request the strategy sends to a tool-less route

It is **diverted**: the request goes to the default route if that one can use tools, otherwise to
the first tool-capable route in declaration order. Each diversion issues a `ToolSupportWarning`
and the decision record's `diverted_from` names the route the strategy actually picked.

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    diverted = with_tools.invoke("Hello there")  # the strategy abstains; default is 'small'
print([w.category.__name__ for w in caught])  # FallbackWarning, then ToolSupportWarning

decision = routing_decision(diverted)
print(decision.route, "<- diverted from", decision.diverted_from)
assert decision.route == "frontier" and decision.diverted_from == "small"
```

Diversion doesn't re-run the strategy: that would spend another call under the model-calling
strategies and could loop.

## How the router knows what a route can do

1. `tool_support_overrides={"route": True | False}` on the router always wins.
2. Otherwise a route's `profile["tool_calling"]`, when it reports one.
3. Otherwise whether the route's class overrides `BaseChatModel.bind_tools`, since the base
   implementation only fails when called.

```python
router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="frontier",
    tool_support_overrides={"small": False},
)
```

## Structured output

`with_structured_output` follows the same rules — it counts as tool binding, so a route that can't
use tools is skipped in the same way.

```python
from pydantic import BaseModel


class Forecast(BaseModel):
    city: str


with warnings.catch_warnings():
    warnings.simplefilter("ignore", ToolSupportWarning)
    structured = router.with_structured_output(Forecast, include_raw=True)
```

Pass `include_raw=True` when you want the decision: the raw message carries it, so
`routing_decision(result["raw"])` works. Without it you get only the parsed object, which has
nowhere to carry a record; for that one path read it back with
[`last_routing_decision()`](../decision-record.md#reading-it-back).

## The router's own profile

`router.profile` reports what **every** route can do — the intersection of the routes' profiles —
so `create_agent` can pick a working structured-output strategy for the router. It is `None` when
any route reports no profile, and an explicit `ChatRouter(..., profile=...)` wins.
