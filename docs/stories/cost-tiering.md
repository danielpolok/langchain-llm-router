# Cost tiering

**Goal:** stop paying frontier prices for questions a small model answers just as well. Most
traffic is easy; a minority is hard. Send each request to the cheapest model that can handle it.

## The approach

Use `HeuristicStrategy`. It scores each request from signals it can compute on the spot — length,
code, how many things it asks for, whether it asks for reasoning, whether it carries anything but
text — with no extra model call, and the score picks a tier. List the routes cheapest first.

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, HeuristicStrategy, routing_decision

small = GenericFakeChatModel(messages=cycle([AIMessage("2 + 2 is 4.")]), name="small")
frontier = GenericFakeChatModel(
    messages=cycle([AIMessage("A proof by contradiction: suppose there are finitely many...")]),
    name="frontier",
)

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),
)

for question in [
    "what's 2 + 2?",
    "Prove that there are infinitely many primes, and explain why the proof works.",
]:
    decision = routing_decision(router.invoke(question))
    print(f"{question!r} -> {decision.route} ({decision.reason})")
```

The default route is where a request goes if the strategy can't score it. Make it the cheap one
unless a wrong answer costs more than the extra spend.

## Run the full example

[`examples/cost_tiering.py`](../../examples/cost_tiering.py) is the same story as a script, tested
in CI:

```bash
uv run python examples/cost_tiering.py
```

## What to look at

- **`decision.reason`** says which signals pushed the score up. If easy questions are landing on
  the frontier model, the reason shows why.
- **Thresholds and weights.** `HeuristicStrategy("small", "mid", "frontier", thresholds=[1.0, 3.0])`
  adds a middle tier; `weights=` turns a signal up or down (`0.0` silences it).
- **Cost in the trace.** Token usage is attributed to the model that ran, so LangSmith shows the
  actual split; see [tracing and cost](../capabilities/tracing-and-cost.md).

## Limits to know about

A length-and-shape score can't see that a short question is hard (`"Is P = NP?"`), so start with a
conservative threshold and check the routed traffic before trusting it. The
[benchmark](../../benchmark/README.md) measured 42–50% lower cost at no loss of quality on one
32-item workload with one model pair — a shape to expect, not a guarantee for your traffic.

Provider prompt caching is per model, so switching routes mid-conversation loses the cached prefix;
see [scope](../scope.md#the-prompt-caching-caveat). To fix a conversation to one route, use a
[forced route](../capabilities/forced-routes.md).

## Variations

- Rules of your own instead of a score: [`ConfigurableStrategy`](../capabilities/strategies.md#configurablestrategy).
- Let a small model judge difficulty: [`ClassifierStrategy`](../capabilities/strategies.md#classifierstrategy).
