# As an agent's model

**Goal:** run an agent on the router, so each model call the agent makes goes to the right model
for the request.

## The approach

`ChatRouter` is a chat model, so pass it to `create_agent` like one.

```python skip
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_llm_router import ChatRouter, KeywordStrategy


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."


router = ChatRouter(
    routes={"small": small_model, "frontier": frontier_model},  # your own chat models
    default_route="small",
    strategy=KeywordStrategy({"small": ["weather"], "frontier": ["urgent"]}),
)

agent = create_agent(model=router, tools=[get_weather])
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Warsaw?"}]})
```

This block is fenced `skip` because it needs two tool-calling models. The complete, offline version
uses a scripted fake in place of each route and is run in CI:

```bash
uv run python examples/agent_backbone.py
```

See [`examples/agent_backbone.py`](../../examples/agent_backbone.py).

## What to look at

- **Every model call in the loop goes through the same routing.** The agent's tool round-trip —
  model, tool, model — is two routed calls.
- **Routing is on the current request, not the tool output.** After the tool runs, the "last
  message" is a tool result; the router still routes on the user's request, so a tool result never
  drags the follow-up call onto the wrong model. See
  [what a strategy sees](../capabilities/strategies.md#what-a-strategy-sees).
- **Tools.** The agent binds its tools to the router, and the router replays that binding onto the
  route it picks. Routes that can't use tools are skipped with a warning, and requests are diverted
  to one that can. See [tools and structured output](../capabilities/tools-and-structured-output.md).
- **The final message carries the decision:** `routing_decision(result["messages"][-1])`.

## Router or middleware?

If the decision needs agent **state** — which iteration this is, a running tool-call count, what an
earlier middleware stored — use LangChain's
[`@wrap_model_call` middleware](https://docs.langchain.com/oss/python/langchain/middleware)
instead; the router deliberately never sees that. The two combine: the router as the agent's model,
with middleware observing it. [`examples/agent_middleware.py`](../../examples/agent_middleware.py)
runs that combination, and [scope](../scope.md#when-to-use-agent-middleware-instead) compares the
two.
