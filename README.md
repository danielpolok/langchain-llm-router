# langchain-llm-router

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![LangChain](https://img.shields.io/badge/langchain--core-1.x-1c3c3c)](https://docs.langchain.com/oss/python/langchain/overview)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)](#-project-status)

In-process model routing for LangChain: `ChatRouter` is a chat model that, per request, picks one of
several candidate chat models and returns that model's response.

> [!WARNING]
> **Pre-alpha — nothing to install yet.** The design is settled and its riskiest parts have been
> proven in a spike, but the package ships no implementation and is not on PyPI. The examples below
> show the **planned** API; names can still change until the v1 requirements are written. See
> [Project status](#-project-status).

## Quick Install

Once published:

```bash
pip install langchain-llm-router
```

## 🤔 What is this?

Easy requests belong on cheap models, hard ones on frontier models. Routing services and proxies do
this outside your application — another hop, another system to run, and a decision your traces
can't see. LangChain's `@wrap_model_call` middleware does it in-process, but only inside
`create_agent`.

`ChatRouter` is a `BaseChatModel`, so it goes anywhere a chat model goes: chains, agents, LangGraph
nodes. You give it named **routes** (ordinary chat models you already build) and a **strategy**
(your routing policy). For each request the strategy picks a route; the router calls it and hands
back its response unchanged, plus a record of which route was taken and why.

- **Works with LangChain, not around it** — `invoke`, `stream`, `batch` and their async forms,
  `bind_tools`, `with_structured_output`, `with_config`, `with_retry` and `with_fallbacks` behave
  as on any chat model.
- **One trace, priced once** — LangSmith shows the routing decision wrapping the real model call,
  and token cost is attributed to the model that actually ran.
- **Always answers** — a default route is mandatory; when the strategy fails or can't decide, the
  request goes there, with a warning and the reason recorded.
- **No extra infrastructure** — no proxy, no service, no credentials; depends only on
  `langchain-core`.

## Overview

### Integration details

| Class | Package | Depends on | Version |
| :--- | :--- | :--- | :--- |
| `ChatRouter` | `langchain-llm-router` | `langchain-core` 1.x | unreleased |

### Model features

`ChatRouter` has no capabilities of its own: each feature is whatever the **selected route**
supports.

| Tool calling | Structured output | Multimodal input | Token-level streaming | Native async | Token usage |
| :---: | :---: | :---: | :---: | :---: | :---: |
| via route | via route | via route | via route | via route | via route |

When tools or structured output are bound, routes that can't use tools are skipped with a warning;
if none can, binding raises an error.

## Instantiation

Routes are any LangChain chat models. The strategy decides which one serves a request — by default
it sees the user's **current request**, not tool output, system prompts or conversation length, so
it doesn't misroute agent loops.

```python
from langchain.chat_models import init_chat_model
from langchain_llm_router import ChatRouter

router = ChatRouter(
    routes={
        "small": init_chat_model("ollama:qwen3:8b"),
        "frontier": init_chat_model("google_genai:gemini-3-flash-preview"),
    },
    default_route="small",
    strategy=...,  # a ready-made strategy, a configured one, or your own
)
```

Strategies come at three levels: ready-made heuristic strategies (keyword, heuristic) that make no
extra model calls; a configurable component for defining your own policy; and a small interface for
fully custom code, such as an existing classifier. Embedding- and LLM-classifier strategies, which
do make calls, are explicit opt-ins.

## Invocation

```python
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Prove that there are infinitely many primes."),
]
response = router.invoke(messages)

response.text
response.response_metadata["routing"]  # {"route": "frontier", "reason": "..."}
```

The response is the selected model's own `AIMessage` — content, tool calls, usage metadata — with
the routing decision added.

## Chaining

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{input}"),
    ]
)

chain = prompt | router
chain.invoke({"input_language": "English", "output_language": "German", "input": "I love programming."})
```

## Agents

Use the router as the model behind an agent:

```python
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's sunny in {city}."


agent = create_agent(model=router, tools=[get_weather])
agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Warsaw?"}]})
```

To choose a model from agent **state** inside `create_agent`, use
[`@wrap_model_call` middleware](https://docs.langchain.com/oss/python/langchain/middleware) instead —
the two are complementary.

## Forcing a route

Pin one call to a route through runtime config, e.g. to compare models on the same traffic. A forced
route is never silently swapped: if it doesn't exist or can't serve the request, the call errors by
default (falling back is an opt-in setting).

```python
router.invoke(messages, config={"configurable": {"route": "frontier"}})
```

## What it doesn't do

- Learn from traffic (bandits, LLM-as-judge feedback, retraining).
- Escalate after generation (cascades).
- Optimise against a cost budget — your policy decides.
- Run as a hosted service, proxy or gateway.

One caveat: provider-side prompt caching is per model, so routing consecutive turns of a conversation
to different models loses the cached prefix.

## 📖 Documentation

- [PRD.md](PRD.md) — what the router is and the principles it holds to (C1–C10, R1–R11).
- [docs/spike-findings.md](docs/spike-findings.md) — what the v0 spike proved and the caveats carried
  into v1.
- [tasks/](tasks/README.md) — the work breakdown.
- [LangChain docs](https://docs.langchain.com/oss/python/langchain/models) — chat models, tools,
  structured output and middleware.

## 🚧 Project status

**v0 is complete.** A spike tested the two riskiest principles against Gemini (cloud), Ollama
(local) and a live LangSmith trace:

- Tool binding and structured output work through the router — with caveats.
- Cost is counted exactly once while the real model call stays in the trace, once the router's own
  run is a chain run rather than a model run.

**Next:** the v1 requirements ([T-101](tasks/T-101-v1-requirements.md)), then the core router,
built-in strategies, a cost/quality benchmark and a PyPI release.

## 💁 Contributing

Development uses [uv](https://docs.astral.sh/uv/) (Python 3.12; the package supports 3.10+).

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Tests that call real providers skip unless their credentials or server are available:
`GEMINI_API_KEY` for Gemini, a reachable local Ollama server for Ollama (a `.env` file is loaded).
Offline tests use fake chat models.

Work is tracked in [tasks/](tasks/README.md); reference principles by their PRD ID (e.g. `R4`, `C10`)
in code, tests and commits.

## License

[MIT](LICENSE)
