# Using the router

`ChatRouter` is a LangChain chat model, so everything you already do with a chat model works on it
unchanged. This page covers what is specific to routing: reading the decision, tools, agents,
forcing a route, and seeing what each route cost.

## Setup

The examples on this page share one router:

```python
from langchain.chat_models import init_chat_model

from langchain_llm_router import ChatRouter, HeuristicStrategy, routing_decision

router = ChatRouter(
    routes={
        "small": init_chat_model("google_genai:gemini-3.5-flash-lite"),
        "frontier": init_chat_model("google_genai:gemini-3.8-flash"),
    },
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),
)
```

## Reading the decision

Every response carries a record of which route answered and why:

```python
response = router.invoke("Compare the trade-offs of REST and GraphQL.")
print(routing_decision(response))
```

```text
RoutingDecision(route='frontier', reason='difficulty 1.00 >= 1.00 (analysis 1.00)', strategy='HeuristicStrategy', fallback=False, forced=False, diverted_from=None)
```

| Field | Meaning |
| --- | --- |
| `route` | The route that answered |
| `reason` | Why, in words |
| `strategy` | The strategy that decided, or `None` when the route was forced |
| `fallback` | `True` if the default route answered because the strategy couldn't decide |
| `forced` | `True` if the route was [forced through runtime config](#forcing-a-route) |
| `diverted_from` | The route the strategy chose, if that route couldn't use the [bound tools](#tools-and-structured-output) |

`routing_decision(response)` reads the record from `response.response_metadata["routing"]` and
returns it as a `RoutingDecision`. For a message the router didn't answer, it returns `None`. The
same record appears in your [trace](#tracing-and-cost).

## Streaming, batching and async

These work exactly as on any chat model. When streaming, the decision arrives on one of the
chunks. Add the chunks together, as LangChain does, and read it from the result:

```python
full = None
for chunk in router.stream("Write a haiku about trains."):
    print(chunk.text, end="")
    full = chunk if full is None else full + chunk

print("\n->", routing_decision(full).route)
```

```text
Iron tracks gleam bright,
Chugging through the morning mist,
Rushing toward tomorrow.
-> small
```

Every request in a batch is routed on its own:

```python
responses = router.batch(["What's 2 + 2?", "Design a URL shortener and explain the trade-offs."])
print([routing_decision(r).route for r in responses])
```

```text
['small', 'frontier']
```

`ainvoke`, `astream`, `abatch` and `astream_events` work the same way:

```python
response = await router.ainvoke("What's 2 + 2?")
print(response.text)
```

```text
2 + 2 = 4
```

## Tools and structured output

`bind_tools` works as it does on any chat model. The router passes the tools to whichever route it
picks, so each model calls them in its own provider's format:

```python
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."


response = router.bind_tools([get_weather]).invoke("What's the weather in Paris?")
print(response.tool_calls)
```

```text
[{'name': 'get_weather', 'args': {'city': 'Paris'}, 'id': 'call_200342', 'type': 'tool_call'}]
```

**Routes that can't use tools** are skipped rather than failing the request. When you bind tools,
a `ToolSupportWarning` names those routes. A request the strategy sends to one of them goes to the
default route instead, or to the first route that can use tools if the default can't. The decision
records the skipped route in `diverted_from`. If no route can use tools, `bind_tools` raises
`NoToolCapableRouteError`.

The router detects tool support from each model's `profile`. If a model reports it wrongly, correct
it with `ChatRouter(..., tool_support_overrides={"local": False})`.

**Structured output** works the same way. A parsed object has nowhere to carry the decision, so
read it with `last_routing_decision()`, which returns the decision of the last routed call made in
the current thread or task:

```python
from pydantic import BaseModel

from langchain_llm_router import last_routing_decision


class Location(BaseModel):
    city: str
    country: str


print(router.with_structured_output(Location).invoke("Where is the Eiffel Tower?"))
print(last_routing_decision().route)
```

```text
city='Paris' country='France'
small
```

With `include_raw=True`, the raw message is returned as well, and `routing_decision()` reads the
decision from it.

## Inside an agent

Pass the router to `create_agent` as its model. Every model call in the agent loop is routed on
the user's request, so tool results don't move the conversation to a different model partway
through:

```python
from langchain.agents import create_agent

agent = create_agent(router, tools=[get_weather])
result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris and Rome?"}]})

for message in result["messages"]:
    if message.type == "ai":
        calls = [call["name"] for call in message.tool_calls]
        print(routing_decision(message).route, calls or message.text)
```

```text
small ['get_weather', 'get_weather']
small The weather in Paris is sunny, and it's also sunny in Rome.
```

The router also works in LangGraph nodes and chains, anywhere a chat model goes. To choose a
model from agent state, such as the number of steps taken, use
[`@wrap_model_call` middleware](https://docs.langchain.com/oss/python/langchain/middleware). It
can use the router as its model, as
[`examples/agent_middleware.py`](../examples/agent_middleware.py) shows.

## Forcing a route

Set `route` in the runtime config to skip the strategy and use a specific route:

```python
response = router.invoke("What's 2 + 2?", config={"configurable": {"route": "frontier"}})
print(routing_decision(response))
```

```text
RoutingDecision(route='frontier', reason='forced via runtime config', strategy=None, fallback=False, forced=True, diverted_from=None)
```

Use `router.with_config(configurable={"route": "frontier"})` to pin a route for a whole session.
A forced route is never silently swapped. If it doesn't exist, or can't use the bound tools, the
call raises `ForcedRouteError`. To fall back to the default route instead, build the router with
`on_unavailable_forced_route="fallback"`, and you get a `ForcedRouteWarning` in place of the error.

**A/B tests.** Forcing is how you compare models on real traffic. Assign each user to an arm with
a stable hash, force that arm's route, and tag the trace with the arm:

```python
import zlib


def arm_for(user_id):
    return "frontier" if zlib.crc32(user_id.encode()) % 2 else "small"


arm = arm_for("user-42")
response = router.invoke(
    "Summarise Hamlet in one sentence.",
    config={"configurable": {"route": arm}, "metadata": {"experiment": "tiers", "arm": arm}},
)
print(arm, routing_decision(response).forced)
```

```text
frontier True
```

In LangSmith, filter by the `experiment` metadata to compare the arms' quality, latency and cost.
For experiments only, a router without a `strategy` always uses the default route unless one is
forced.

## Tracing and cost

Turn on [LangSmith](https://docs.langchain.com/langsmith) as usual. The router needs no setup of
its own:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="..."
```

Each routed call is one trace. The decision and the real model call are both visible:

```text
ChatRouter                  chain   the decision is in its outputs
├── HeuristicStrategy       chain   how the strategy decided
└── ChatGoogleGenerativeAI  llm     the model that answered, with its tokens and cost
```

The router's own run is a chain, not a model run, so tokens are counted once, against the model
that actually answered. If a strategy calls a model itself, as `ClassifierStrategy` and
`EmbeddingStrategy` do, that call appears under the strategy's run with its own cost. You can then
see what deciding cost separately from what answering cost.

Without LangSmith, LangChain's usage callback shows the tokens used by each model:

```python
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as usage:
    router.invoke("What's the capital of France?")
    router.invoke("Compare the trade-offs of REST and GraphQL.")

for model, tokens in usage.usage_metadata.items():
    print(f"{model:<22} {tokens['total_tokens']:>5} tokens")
```

```text
gemini-3.5-flash-lite     16 tokens
gemini-3.8-flash        2567 tokens
```

## Retries and fallbacks

The router doesn't retry or switch models when a call fails. The chosen model's error reaches you
unchanged, so LangChain's usual tools apply to the router as a whole:

```python
backup = init_chat_model("google_genai:gemini-3.1-flash-lite")
resilient = router.with_retry(stop_after_attempt=3).with_fallbacks([backup])
```

A route must be a plain chat model, so `small.with_retry()` is rejected as a route. For retries on
a single route, set them on the model itself, for example `max_retries=` on the provider class.

A strategy that fails is different: the request still gets an answer. If a strategy raises, the
default route answers and a `FallbackWarning` gives the cause.

## Caching

A response cache belongs on the **routes**. Pass one to each model you want cached, for example
`init_chat_model(..., cache=InMemoryCache())`, or set one for the whole process with
`set_llm_cache()`. Each route caches its own answers, so an answer cached for one model is never
returned for another. The router rejects a `cache=` of its own, because it would never be used.
A cached answer still carries the decision made for the current call.

## Warnings and errors

The router warns when it answers differently from what you configured, and raises only when it
can't answer as asked.

| Name | Kind | When |
| --- | --- | --- |
| `FallbackWarning` | warning | The strategy abstained, failed, or named a route that doesn't exist, so the default route answered |
| `ToolSupportWarning` | warning | Some routes can't use the tools you bound, so they will be skipped for tool calls |
| `ForcedRouteWarning` | warning | A forced route couldn't be used, and `on_unavailable_forced_route="fallback"` sent the request to the default route |
| `RoutingWarning` | warning | The base class of the three above |
| `RoutingError` | error | The router or a strategy is misconfigured, reported when it is built. It is a `ValueError` |
| `NoToolCapableRouteError` | error | Tools are bound but no route can use them |
| `ForcedRouteError` | error | A forced route doesn't exist or can't use the bound tools |

Warnings use Python's `warnings` module. To make an unexpected fallback fail your tests, for
example, turn it into an error:

```python
import warnings

from langchain_llm_router import FallbackWarning

warnings.filterwarnings("error", category=FallbackWarning)
```
