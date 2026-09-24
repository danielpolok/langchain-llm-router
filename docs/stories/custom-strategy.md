# A custom strategy

**Goal:** you already have something that decides — a trained classifier, a rules engine, a call to
another service — and you want its answer to pick the model.

## The approach

A plain function of a `RoutingRequest` is a strategy. It returns a route name, a `RoutingChoice`
(a route plus a reason for the trace), or `None` for "can't decide".

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, RoutingChoice, RoutingRequest, routing_decision


def existing_classifier(text: str) -> str:
    """Stands in for your own classifier — a trained model, a rules engine, a REST call."""
    return "urgent" if "asap" in text.lower() or "urgent" in text.lower() else "normal"


def pick_route(request: RoutingRequest) -> RoutingChoice:
    label = existing_classifier(request.text)
    return RoutingChoice(route=label, reason=f"existing_classifier labelled it {label!r}")


urgent = GenericFakeChatModel(messages=cycle([AIMessage("Escalating now.")]), name="urgent")
normal = GenericFakeChatModel(messages=cycle([AIMessage("I'll get to it today.")]), name="normal")

router = ChatRouter(
    routes={"urgent": urgent, "normal": normal},
    default_route="normal",
    strategy=pick_route,
)

for question in ["The prod database is down, need this ASAP.", "Any tips for a Friday demo?"]:
    decision = routing_decision(router.invoke(question))
    print(f"{question!r} -> {decision.route} ({decision.reason})")
```

Returning the bare label works too — the router writes a generic reason. Returning a
`RoutingChoice` costs one line and makes the trace say *why*.

## Run the full example

[`examples/custom_strategy.py`](../../examples/custom_strategy.py) is the same story as a script,
tested in CI:

```bash
uv run python examples/custom_strategy.py
```

## What to look at

- **Fallbacks.** If your function raises, returns `None` or returns a name that isn't a route, the
  default route answers and a `FallbackWarning` says which. A bug in your classifier never becomes
  an outage.
- **What `request` holds.** `text`, `content_blocks`, `modalities`, `routes` and `tools_bound` —
  the current request only. Full table in the [strategy reference](../strategies.md#routingrequest).

## When a function isn't enough

A plain function must be synchronous and sees only the current request. Subclass `RoutingStrategy`
when you need more:

```python
from langchain_llm_router import RoutingStrategy


class Contextual(RoutingStrategy):
    wants_full_context = True  # request.messages holds the whole transcript

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        long_conversation = request.messages is not None and len(request.messages) > 20
        return RoutingChoice("urgent", "a long conversation") if long_conversation else None

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        return self.decide(request)  # or await your own async model call here
```

If your strategy calls a model or service, override `adecide` with a native async version and pass
`request.config` to the call, so it is traced and costed under the strategy's own run. The
[strategy reference](../strategies.md) has the full interface and its stability promise.
