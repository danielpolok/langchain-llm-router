# langchain-model-router

[![PyPI](https://img.shields.io/pypi/v/langchain-model-router)](https://pypi.org/project/langchain-model-router/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-model-router)](https://pypi.org/project/langchain-model-router/)
[![CI](https://github.com/danielpolok/langchain-model-router/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/danielpolok/langchain-model-router/actions/workflows/ci.yml)
[![LangChain](https://img.shields.io/badge/langchain--core-%E2%89%A51.2.21%2C%20%3C2-1c3c3c)](https://docs.langchain.com/oss/python/langchain/overview)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/danielpolok/langchain-model-router/blob/main/LICENSE)

**Send each request to the right model, by rules you control, from one LangChain chat model.**

*A community package, not affiliated with or endorsed by LangChain.*

`ChatRouter` is a drop-in LangChain chat model. You give it a few named models and a **strategy**.
For each request, the strategy picks one of the models. The router returns that model's own
response, along with a note of which model answered and why.

What "the right model" means is up to you:

- **Difficulty:** a fast model for easy requests and a frontier model for hard ones.
- **Topic:** code questions to a coding model, and legal questions to a model tuned for them.
- **Content:** requests with images or documents to a model that can read them.
- **Data:** requests that touch sensitive data to a model you host yourself.
- **Customer or experiment:** each plan or A/B test group to its own model.

Whatever the policy, the router behaves the same way:

- **Nothing else to change.** It is a chat model, so `invoke`, `stream`, `batch`, tools, structured
  output, agents and LangGraph all work unchanged.
- **Routing you can read.** Every response says which model answered and why, and your
  LangSmith trace shows the same.
- **Always answers.** If the strategy can't decide, the default model answers and you get a
  warning.
- **Nothing extra to run.** No proxy, no service and no API key of its own. It needs only
  `langchain-core`.

## Get started

```bash
pip install langchain-model-router
```

It needs Python 3.10 or later and `langchain-core` 1.2.21 or later within 1.x (`>=1.2.21,<2`).

A route can be any LangChain chat model (a `BaseChatModel`) from any
[provider](https://docs.langchain.com/oss/python/integrations/chat), hosted or local. This first
router has two routes, a small model and a frontier one, and sends each question to one of them
depending on how hard it looks.

```python
from langchain.chat_models import init_chat_model

from langchain_model_router import ChatRouter, HeuristicStrategy, routing_decision

router = ChatRouter(
    routes={
        "small": init_chat_model("google_genai:gemini-3.5-flash-lite"),
        "frontier": init_chat_model("google_genai:gemini-3.8-flash"),
    },
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),  # cheapest first
)

for question in [
    "What's the capital of France?",
    "Compare the trade-offs of quicksort and mergesort on linked lists.",
]:
    response = router.invoke(question)
    print(f"{routing_decision(response).route:<8} {question}")
```

```text
small    What's the capital of France?
frontier Compare the trade-offs of quicksort and mergesort on linked lists.
```

The simple fact question went to the small model. Comparing trade-offs takes reasoning, so that
question went to the frontier model.
[Routing strategies](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md#heuristicstrategy-route-on-difficulty)
explains how the difficulty is judged.

`response` is the chosen model's own answer, unchanged, so you read its text, tool calls and token
usage as usual. `routing_decision(response)` tells you which route answered and why.

## How it works

<img alt="A request enters ChatRouter. Its strategy picks one of the routes (a small model, a frontier model or a code model), or the default route if it can't decide. The chosen model's answer comes back with which route answered and why." src="https://raw.githubusercontent.com/danielpolok/langchain-model-router/main/docs/images/how-it-works.svg" width="820">

1. The strategy looks at the **current request**, meaning the user's latest message. It ignores
   tool output and the rest of the conversation, so an agent's tool loop can't change the route.
2. It names a route and gives a reason, or it abstains. If it abstains, the default route answers.
3. The router calls that one model and returns its response, with the routing decision attached.

## Choose a strategy

| Strategy | Decides by | Extra cost per request | Good for |
| --- | --- | --- | --- |
| [`HeuristicStrategy`](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md#heuristicstrategy-route-on-difficulty) | A difficulty score: length, code, several questions, reasoning words, images | None | Cheap model for easy requests, strong model for hard ones |
| [`KeywordStrategy`](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md#keywordstrategy-route-on-words) | Words in the request | None | Topics with telltale words, such as "SQL" or "invoice" |
| [`ConfigurableStrategy`](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md#configurablestrategy-combine-rules) | Your rules, combining keywords, scores, media and tools | None | A policy with several conditions |
| [`EmbeddingStrategy`](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md#embeddingstrategy-route-on-meaning) | Similarity to example requests | One embedding call | Topics without telltale words |
| [`ClassifierStrategy`](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md#classifierstrategy-let-a-small-model-choose) | A small model reads route descriptions and picks one | One small-model call | Subtle distinctions, when accuracy matters most |
| [Your own function](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md#your-own-strategy) | Your code | Whatever your code costs | Business rules, an existing classifier |

Start with a strategy that makes no extra call. Reach for embeddings or a classifier when words and
rules can't tell your routes apart. They read meaning, but they add a call to every request.

## Documentation

- **[Routing strategies](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md)**
  covers how each strategy decides, with examples, tuning and how to write your own.
- **[Using the router](https://github.com/danielpolok/langchain-model-router/blob/main/docs/guide.md)**
  covers reading the decision, streaming, tools, structured output, agents, forcing a route for
  A/B tests, tracing and cost, retries and caching.
- **[Examples](https://github.com/danielpolok/langchain-model-router/blob/main/examples/README.md)**
  has one runnable script per use case, each a small real-world scenario on the same models as
  these pages.
- **[Benchmark](https://github.com/danielpolok/langchain-model-router/blob/main/benchmark/README.md)**
  compares the strategies on cost and answer quality over a mixed workload, and shows how to rerun
  it with your own models.
- **[Design notes](https://github.com/danielpolok/langchain-model-router/blob/main/docs/design.md)**
  explain why the router is built the way it is. Read these before contributing.
- **[Changelog](https://github.com/danielpolok/langchain-model-router/blob/main/CHANGELOG.md)**
  lists what changed in each release.

## Scope

The router picks one model **before** the call, using a policy you define. It doesn't learn from
traffic, retry a weak answer on a stronger model, spend against a budget, or run as a proxy.
Inside `create_agent`, a choice that depends on agent state belongs in
[`@wrap_model_call` middleware](https://docs.langchain.com/oss/python/langchain/middleware), which
can use the router as its model.

> **Provider prompt caching.** Moving a conversation between models loses the prompt prefix that
> the provider has cached. For long conversations, you can
> [keep each conversation on one route](https://github.com/danielpolok/langchain-model-router/blob/main/docs/strategies.md#a-strategy-class).

## Contributing

Bug reports, ideas and pull requests are welcome. Development uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
make test   # offline unit tests
make lint   # ruff and mypy
```

[CONTRIBUTING.md](https://github.com/danielpolok/langchain-model-router/blob/main/CONTRIBUTING.md)
explains how to propose a change, and
[AGENTS.md](https://github.com/danielpolok/langchain-model-router/blob/main/AGENTS.md)
describes the repository layout and conventions. Open work is in
[GitHub Issues](https://github.com/danielpolok/langchain-model-router/issues). To report a security
problem, follow [SECURITY.md](https://github.com/danielpolok/langchain-model-router/blob/main/SECURITY.md).

## License

[MIT](https://github.com/danielpolok/langchain-model-router/blob/main/LICENSE)
