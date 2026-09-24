# langchain-llm-router

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![LangChain](https://img.shields.io/badge/langchain--core-1.x-1c3c3c)](https://docs.langchain.com/oss/python/langchain/overview)

In-process model routing for LangChain: `ChatRouter` is a chat model that, per request, picks one of
several candidate chat models and returns that model's response — with a record of which one ran
and why.

> [!NOTE]
> **Not on PyPI yet.** The router, its strategies and its benchmark are implemented and merged,
> but nothing has been released, so nothing here is a compatibility guarantee until the first
> release.

## Install

Once published:

```bash
pip install langchain-llm-router
```

## Quickstart

Give the router named routes — ordinary chat models — and a strategy that applies your routing
policy. It is a chat model itself, so it goes anywhere one does: chains, agents, LangGraph nodes.

```python skip
from langchain.chat_models import init_chat_model

from langchain_llm_router import ChatRouter, HeuristicStrategy, routing_decision

router = ChatRouter(
    routes={
        "small": init_chat_model("ollama:qwen3:8b"),
        "frontier": init_chat_model("google_genai:gemini-3-flash-preview"),
    },
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),  # cheap first
)

response = router.invoke("Prove that there are infinitely many primes.")
print(response.text)  # the selected model's own answer
print(routing_decision(response))  # which route ran, and why
```

The response is the selected model's own `AIMessage`, with the routing decision added under
`response_metadata["routing"]`. To try it without a provider, use the offline fakes in
[Get started](docs/get-started.md).

## Why

- **Works with LangChain, not around it.** `invoke`, `stream`, `batch`, the async forms,
  `bind_tools`, `with_structured_output`, `with_retry` and `with_fallbacks` behave as on any chat
  model.
- **One trace, priced once.** LangSmith shows the routing decision wrapping the real model call,
  and token cost is attributed to the model that ran.
- **Always answers.** The default route is mandatory; when a strategy can't decide, the request
  goes there, with a warning and the reason recorded.
- **No extra infrastructure.** No proxy, no service, no credentials of its own; it depends only on
  `langchain-core`.

## Documentation

- [Documentation index](docs/index.md) — start here.
- [Get started](docs/get-started.md) — a first router, choosing a strategy, reading the decision.
- [Capabilities](docs/index.md#where-to-go-next) — strategies, tools and structured output,
  streaming and batch, forced routes, fallbacks, caching, tracing and cost.
- [User stories](docs/index.md#where-to-go-next) — cost tiering, domain routing, experimentation,
  a custom strategy, and use as an agent's model.
- [Examples](examples/README.md) — one runnable, offline script per use case.
- [Benchmark](benchmark/README.md) — what routing saved on one workload, and how to rerun it.
- [Scope](docs/scope.md) — what the router doesn't do, and when agent middleware fits better.

## Contributing

Development uses [uv](https://docs.astral.sh/uv/) (Python 3.12; the package supports 3.10+):

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

See [AGENTS.md](AGENTS.md) for the repository layout and conventions,
[docs/design.md](docs/design.md) for why the code is shaped the way it is, and
[GitHub Issues](https://github.com/danielpolok/langchain-llm-router/issues) for the open work.

## License

[MIT](LICENSE)
