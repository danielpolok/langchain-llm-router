# Development guidelines for `langchain-llm-router`

Guidance for anyone — or any coding agent — working in this repository. For the user-facing
picture, read [README.md](README.md); for why the code is shaped the way it is, read
[docs/design.md](docs/design.md) before changing behaviour it describes.

## What this is

`ChatRouter` (PyPI `langchain-llm-router`, MIT) is a LangChain 1.x chat model (`BaseChatModel`,
`langchain-core` 1.x) that, per request, picks one of several named candidate chat models and
returns that model's response unchanged plus a record of the routing decision. The routing
*policy* lives in the application's code; a *strategy* applies it.

**Work with LangChain's abstractions, not against them.** Anything that works on a chat model must
work identically on the router. Errors, retries and fallbacks stay LangChain's job (`with_retry`,
`with_fallbacks`); the selected model's errors surface unchanged.

Explicit non-goals: learning from traffic, cascades or escalation, cost-budget optimisation, a
hosted proxy, and replacing `@wrap_model_call` agent middleware. See
[docs/scope.md](docs/scope.md).

## Layout

| Path | Holds |
| --- | --- |
| `src/langchain_llm_router/router.py` | `ChatRouter` — the pipeline, and every calling convention |
| `.../strategy.py` | the strategy interface (`RoutingStrategy`, `RoutingRequest`, `RoutingChoice`) and its stability promise |
| `.../_extraction.py` | current-request extraction |
| `.../decision.py` | the decision record, and `last_routing_decision()` |
| `.../errors.py` | the warning and error hierarchy |
| `.../_tools.py`, `_profile.py` | tool binding replay, capability detection, `StructuredRouter`, the router's own profile |
| `.../strategies/` | the built-in strategies: `keyword`, `heuristic`, `configurable`, and the opt-in `embedding`, `classifier` |
| `tests/unit_tests/` | offline tests, using fake routes from `tests/fakes.py` |
| `tests/integration_tests/` | tests against real providers |
| `tests/tracing.py` | walks collected runs and prices them the way LangSmith would |
| `tests/conventions.py` | parametrises tests over the calling conventions (`invoke`, `stream`, `batch`, async…) |
| `examples/` | one runnable, offline script per use case; `tests/unit_tests/test_examples.py` runs each in CI |
| `benchmark/` | the cost/quality benchmark — a versioned harness and dataset; see its README |
| `docs/` | user reference (`strategies.md`, `decision-record.md`, `scope.md`) and `design.md` |

## Commands

`uv` manages the environment (Python 3.12 for development; the package supports 3.10 and later).

| Task | Command |
| --- | --- |
| Install | `uv sync` (dev and provider dependency groups) |
| Test | `uv run pytest` |
| One test | `uv run pytest tests/unit_tests/test_package.py::test_package_exports_the_pinned_api` |
| Lint | `uv run ruff check . && uv run ruff format --check .` |
| Type-check | `uv run mypy` |

Set `LANGSMITH_TRACING=false` for offline runs, or every test tries to trace.

Tests that call real providers are marked `@pytest.mark.requires_env("GEMINI_API_KEY", ...)` or
`@pytest.mark.requires_ollama`, and skip when a named variable is unset or the local Ollama server
is unreachable (hooks in the root `conftest.py`, which also loads `.env`). The real providers are
Gemini (cloud, `google_genai:gemini-3-flash-preview`) and Ollama (local, `ollama:qwen3:8b`); set
`LLM_ROUTER_GEMINI_MODEL` / `LLM_ROUTER_OLLAMA_MODEL` to override. Offline tests use fake models
such as `GenericFakeChatModel`.

## Conventions

- **Dependencies:** the package imports only the standard library and `langchain-core`, and only
  its public names, never private ones. `langchain`, `langgraph`, the provider packages and
  `python-dotenv` are development dependencies and are never imported by `src/`.
- **Comments and docstrings** say what the code does and why, in plain language. Don't refer to
  internal document, requirement or task numbers — write the behaviour out.
- **Public API stability:** the strategy interface carries a stability promise
  ([docs/strategies.md](docs/strategies.md#stability-promise)). Changing a dataclass field, its
  order or a method signature needs a version bump the promise names.
- **Tests** exercise behaviour through the public API and every calling convention
  (`tests/conventions.py`), not internals. A test that pins a decision from `docs/design.md`
  should say so in its docstring.
- **Cost and tracing:** the router's own run is a chain run, never a model run, so tokens are
  counted once. Any change to how the router calls a route must keep
  `tests/unit_tests/test_cost.py` and `test_tracing.py` passing.

## Current LangChain APIs

`.mcp.json` configures the `docs-langchain` MCP server (docs.langchain.com). Use it to check
current LangChain 1.x behaviour — `BaseChatModel` internals, callbacks, `bind_tools`,
`with_structured_output`, `configurable_fields`, caching — rather than relying on memory, since the
pre-1.0 APIs differ.
