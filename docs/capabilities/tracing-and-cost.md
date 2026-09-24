# Tracing and cost

A routed call produces one trace in which you can see both the routing decision and the real model
call, and its token cost is counted exactly once, against the model that actually ran.

## What a trace contains

```text
ChatRouter                 (chain run)   ← the routing decision is in its outputs
├── KeywordStrategy        (chain run)   ← the strategy's own run; its output is the decision
└── big-model              (llm run)     ← the selected route's call; its metadata carries the decision
```

The router's own run is a **chain** run, never a model run. That matters because cost tooling
prices model runs: LangSmith prices each LLM run from its usage, and LangChain's
`UsageMetadataCallbackHandler` adds up the usage of each. If the router recorded a model run of its
own, the same tokens would be counted twice and nothing would error — the number would just be
wrong. With a chain run, the route's call is the only model run, so:

- token usage is attributed to the model that ran (`big-model`, not "the router"),
- the real call still appears in the trace with its own latency, inputs and outputs, and
- the decision is visible three times: on the strategy's run, in the router run's outputs and in
  the route run's metadata.

A strategy that calls a model itself (`ClassifierStrategy`, `EmbeddingStrategy`) has that call
traced under the strategy's run and costed to *its* model, so you can see what deciding cost
separately from what answering cost.

## Turning on LangSmith

Set the usual environment variables; the router needs nothing else.

```python skip
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."  # your key
```

## Measuring cost without LangSmith

`get_usage_metadata_callback` sums usage per model, and shows the once-only counting:

```python
from itertools import cycle

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, KeywordStrategy


def priced(name: str, input_tokens: int) -> GenericFakeChatModel:
    reply = AIMessage(
        "ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 5,
            "total_tokens": input_tokens + 5,
        },
        response_metadata={"model_name": name},
    )
    return GenericFakeChatModel(messages=cycle([reply]), name=name)


router = ChatRouter(
    routes={"small": priced("small-model", 10), "big": priced("big-model", 100)},
    default_route="small",
    strategy=KeywordStrategy({"big": ["hard"], "small": ["easy"]}),
)

with get_usage_metadata_callback() as usage:
    router.invoke("a hard question")
    router.invoke("an easy question")

print(usage.usage_metadata)
assert usage.usage_metadata["big-model"]["input_tokens"] == 100
assert usage.usage_metadata["small-model"]["input_tokens"] == 10
```

Each model shows up once, with its own tokens — the total is what the provider billed.

## Reading the decision from a trace

The record is under the `routing` key of the router run's outputs and of the route run's
metadata, in the same shape as `response_metadata["routing"]`; filter on `routing.route` or
`routing.fallback` in LangSmith to find, say, every request that fell back to the default route.
See [the decision record](../decision-record.md) for every field.

## Tags, metadata and callbacks

Config you pass on the call — `tags`, `metadata`, `callbacks`, `run_name` — is honoured by the
router and passed on to the route, as with any chat model.
