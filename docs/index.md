# langchain-llm-router

`ChatRouter` is a LangChain chat model that, for every request, picks one of several chat models
you already have and returns that model's answer unchanged — plus a record of which one it picked
and why.

Easy requests belong on cheap models and hard ones on frontier models. Routing services do that
outside your application: another hop, another system to run, and a decision your traces can't
see. `ChatRouter` does it in your process, as a chat model, so it goes anywhere a chat model goes:
a chain, an agent, a LangGraph node.

## Install

```bash
pip install langchain-llm-router
# or
uv add langchain-llm-router
```

The package needs only `langchain-core`. You bring the chat models — install whichever provider
packages your routes use.

## A first router

Two routes and a keyword rule. The routes here are offline fakes so the page runs anywhere; swap in
`init_chat_model("provider:model")` for each and nothing else changes.

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision

small = GenericFakeChatModel(messages=cycle([AIMessage("Warsaw.")]), name="small")
coder = GenericFakeChatModel(messages=cycle([AIMessage("Use a dict.")]), name="coder")

router = ChatRouter(
    routes={"small": small, "coder": coder},
    default_route="small",
    strategy=KeywordStrategy({"coder": ["python", "stack trace"]}),
)

response = router.invoke("Why does this Python stack trace happen?")
print(response.text)  # "Use a dict." — the coder route's own answer
decision = routing_decision(response)
print(decision.route, "-", decision.reason)
assert decision.route == "coder"
```

Three ideas carry the rest of the documentation:

- A **route** is an ordinary chat model with a name. Routes are used exactly as you built them.
- A **strategy** applies your routing policy to one request and names a route. Ready-made
  strategies cover the common shapes; a plain function covers everything else.
- The **default route** is mandatory. When the strategy can't decide, the default route answers,
  with a warning and the reason recorded — the router always answers.

## Where to go next

**Start here**

- [Get started](get-started.md) — a first router step by step, choosing a strategy, reading the
  decision and testing offline.

**Capabilities**

- [Strategies](capabilities/strategies.md) — keyword, heuristic, configurable, embedding,
  classifier and custom, at a glance.
- [Tools and structured output](capabilities/tools-and-structured-output.md) — `bind_tools`,
  `with_structured_output` and which routes are skipped, and why.
- [Calling conventions](capabilities/calling-conventions.md) — `invoke`, `stream`, `batch` and
  their async forms.
- [Forced routes and runtime config](capabilities/forced-routes.md) — pin one call to one route.
- [Fallbacks and retries](capabilities/fallbacks-and-retries.md) — `with_retry`,
  `with_fallbacks`, and what the router does when a strategy can't decide.
- [Response caching](capabilities/caching.md) — the route owns the cache.
- [Tracing and cost](capabilities/tracing-and-cost.md) — one trace, priced once.
- [The decision record, warnings and errors](decision-record.md) — every field, every warning,
  every error.

**User stories** — each starts from a goal and runs an example that is tested in CI.

- [Cost tiering](stories/cost-tiering.md) — simple requests to a small model, hard ones to a
  frontier model.
- [Domain routing](stories/domain-routing.md) — code questions to a code-strong model.
- [Experimentation and A/B comparison](stories/experimentation.md) — run the same traffic through
  different models.
- [A custom strategy](stories/custom-strategy.md) — plug an existing classifier in.
- [As an agent's model](stories/agent-model.md) — the router behind `create_agent`.

**Reference and background**

- [Strategy reference](strategies.md) — the strategy interface field by field and its stability
  promise.
- [Scope](scope.md) — when to use agent middleware instead, what the router deliberately doesn't
  do, and the prompt-caching caveat.
- [Design notes](design.md) — why the router is built the way it is.
- [Examples](../examples/README.md) — one runnable script per use case.
- [Benchmark](../benchmark/README.md) — what routing saved on one workload, and how to rerun it.
