# Experimentation and A/B comparison

**Goal:** find out which model is better for your traffic by running the same requests through
different ones — and, when you do, know exactly which model produced each answer.

## The approach

Runtime config forces a route for one call and skips the strategy. Send the same request to each
route and compare.

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import (
    ChatRouter,
    ForcedRouteError,
    KeywordStrategy,
    routing_decision,
)

model_a = GenericFakeChatModel(
    messages=cycle([AIMessage("(model A) Warsaw is the capital of Poland.")]), name="model-a"
)
model_b = GenericFakeChatModel(
    messages=cycle([AIMessage("(model B) Warsaw, Poland.")]), name="model-b"
)

router = ChatRouter(
    routes={"model-a": model_a, "model-b": model_b},
    default_route="model-a",
    strategy=KeywordStrategy({"model-b": ["urgent"]}),
)

question = "What's the capital of Poland?"
for forced in ["model-a", "model-b"]:
    response = router.invoke(question, config={"configurable": {"route": forced}})
    decision = routing_decision(response)
    assert decision.forced
    print(f"forced {forced!r} -> {decision.route}: {response.text}")

try:
    router.invoke(question, config={"configurable": {"route": "model-c"}})
except ForcedRouteError as error:
    print(f"an unknown route raises: {error}")
```

## Run the full example

[`examples/experimentation.py`](../../examples/experimentation.py) is the same story as a script,
tested in CI:

```bash
uv run python examples/experimentation.py
```

## What to look at

- **`decision.forced`** is `True` and `decision.strategy` is `None`: nothing but your config chose
  the route.
- **Unknown routes raise.** A forced route that doesn't exist, or can't use the tools bound to the
  call, raises `ForcedRouteError` instead of falling back — a comparison that quietly ran on the
  wrong model is worse than none. Set `on_unavailable_forced_route="fallback"` if you'd prefer a
  warning and the default route; see [forced routes](../capabilities/forced-routes.md).

## Splitting traffic

Forcing is per call, so you choose the split. A simple A/B assignment:

```python
import random


def assign(user_id: str) -> str:
    return random.Random(user_id).choice(["model-a", "model-b"])  # stable per user


response = router.invoke(question, config={"configurable": {"route": assign("user-42")}})
print(routing_decision(response).route)
```

Tag the call so the split is visible in traces:

```python
response = router.invoke(
    question,
    config={"configurable": {"route": "model-b"}, "tags": ["experiment:capital-q"]},
)
```

Then compare in LangSmith by tag and by `routing.route`; cost is attributed to the model that
ran, so the comparison prices itself. See [tracing and cost](../capabilities/tracing-and-cost.md).

## Variations

- Comparing a *policy* rather than models: build two routers with different strategies and send
  the same traffic through both.
