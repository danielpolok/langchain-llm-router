# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

v1 is being built on the integration branch `v1`, one reviewed task at a time; `v1` merges into
`main` as a single pull request once the tasks are done, so `main` still holds only the spike and
the requirements. Status is on the [v1 Project board](https://github.com/users/danielpolok/projects/2),
not here.

The repo holds `PRD.md` (principles), `docs/v1-requirements.md` (the numbered requirements v1
builds to), `tasks/` (tracking rules and closed tasks — open work is in GitHub Issues),
`src/langchain_llm_router/` (the package), `spike/` (throwaway spike code and its tests — never
import it from `src/` or `tests/`) and `benchmark/` (T-140's cost/quality benchmark — a real,
versioned harness and dataset, not throwaway; see `benchmark/README.md`).

The package:

| Module | Holds |
| --- | --- |
| `router.py` | `ChatRouter` — the pipeline, and every calling convention |
| `strategy.py` | the strategy interface (`RoutingStrategy`, `RoutingRequest`, `RoutingChoice`) and its stability promise |
| `_extraction.py` | current-request extraction (R4, C7) |
| `decision.py` | the decision record, and `last_routing_decision()` |
| `errors.py` | the warning and error hierarchy |
| `_tools.py` | tool binding replay, capability detection (D5), `StructuredRouter` |
| `strategies/` | the built-in strategies: `keyword`, `heuristic`, `configurable` |

Tests mirror it: `tests/unit_tests` (offline, fake routes from `tests/fakes.py`; `tests/tracing.py`
walks collected runs and prices them the way LangSmith would; `tests/conventions.py` parametrises
over the calling conventions), `tests/integration_tests` (real providers) and
`spike/tests`.

Decisions made while building are recorded where they belong, in the repo: D3 (how the decision
reaches a caller who only holds a parsed object), REQ-C6-2 (a wrapped route is refused),
REQ-C2-1 (the beta v3 streaming protocol is out of scope), REQ-C3-2 and D5 (what a bound router
leaves for LangChain to read) and REQ-C2-2 (`generate()`/`agenerate()` scoped to message-level
parity) were each amended after a task measured what the original wording would have done. Check
`docs/v1-requirements.md` before assuming a requirement reads as it did in an older issue.

## Commands

uv manages the environment (Python 3.12 for development; the package supports ≥3.10).

- Install: `uv sync` (dev + provider dependency groups)
- Test: `uv run pytest`
- Single test: `uv run pytest tests/unit_tests/test_package.py::test_package_exports_the_pinned_api`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Type-check: `uv run mypy`

Tests that call real providers are marked `@pytest.mark.requires_env("GEMINI_API_KEY", ...)` or
`@pytest.mark.requires_ollama`, and skip when a named variable is unset or the local Ollama
server is unreachable (hooks in the root `conftest.py`, which also loads `.env`). Set
`LANGSMITH_TRACING=false` for offline runs, or every test tries to trace. Real providers
are Gemini (cloud, `google_genai:gemini-3-flash-preview`) and Ollama (local, `ollama:qwen3:8b`)
— set via `LLM_ROUTER_GEMINI_MODEL` / `LLM_ROUTER_OLLAMA_MODEL` to override. Offline tests use
fake models (`GenericFakeChatModel`). Layout: `tests/unit_tests`, `tests/integration_tests`,
`spike/tests`, `benchmark/tests`.

## Spike

The spike (T-001–T-005) is **built and written up** in `docs/spike-findings.md`. It tested the
two riskiest principles:
- **C3** — `bind_tools` and `with_structured_output` work through the router.
- **R3** — cost is counted exactly once, attributed to the model that actually ran, while that
  model's call still appears as a nested run in LangSmith traces (C5).

Neither failed. **C3 holds with caveats**, and **R3 and C5 both hold** once the router's own run
is a chain run rather than a model run — the one design decision the spike forced, now in PRD
§11. Read `docs/spike-findings.md` before starting v1 work: it lists the caveats against the task
that answers each. Don't re-debate settled decisions in PRD §11.

**v0 is closed.** The real-provider run (T-003, Gemini + Ollama) and the LangSmith trace (T-004)
have both run. **T-101 is done too:** `docs/v1-requirements.md` holds the public API, the decision
record schema, and all 21 principles as numbered `REQ-` requirements with a testable check each.
Read it before any v1 code — it, not the spike, is what T-110 onward build. The nine rules it
settled are D1–D9 in PRD §11; don't re-decide them.

## What is being built

`ChatRouter` (PyPI `langchain-llm-router`, MIT): a LangChain 1.x chat model (`BaseChatModel`, `langchain-core` 1.x) that, per
request, picks one of several named candidate chat models and returns that model's response
unchanged plus a record of the routing decision. The routing *policy* lives in the application's
code; a *strategy* applies it.

Core constraint: **work with LangChain's abstractions, not against them.** Anything that works on
a chat model must work identically on the router. PRD §3 lists the principles as numbered IDs
(C1–C10 for LangChain consistency, R1–R11 for routing behaviour) — reference them by ID in code,
tests and commits. Non-obvious ones that shape the design:

- **R4** — route on the user's *current request* by default, not tool output, system prompts or
  conversation length (these misroute agent loops).
- **R7/R8** — built-in heuristic strategies make no extra model/API calls; no dependencies beyond
  `langchain-core`; no proxy or external service.
- **R9** — a default route is mandatory; strategy failure/indecision falls back to it with a
  warning and recorded reason.
- **R10** — when tools are bound, routes that can't use tools are skipped (with warnings); if none
  can, binding is an error. Structured output counts as tool binding.
- **R11** — a route forced via runtime config (C4) is never silently swapped; errors by default,
  fallback is opt-in.
- **C6** — errors, retries and fallbacks stay LangChain's job (`with_retry`, `with_fallbacks`);
  the selected model's errors surface unchanged.
- **C10** — LangChain's response cache must never return an answer cached for a different route
  or configuration.

Explicit non-goals (PRD §5): learning from traffic, cascades/escalation, cost-budget
optimisation, hosted proxy, replacing `@wrap_model_call` agent middleware.

## Tasks

Open tasks are **GitHub issues** in milestone `v1`, titled `T-NNN · Title`; `tasks/README.md` holds
the rules, a task-ID → issue lookup and the archive of closed tasks (T-001–T-005, T-101), which stay
as files. Status is the *Status* field on the [v1 Project board](https://github.com/users/danielpolok/projects/2)
(Todo / In Progress / Done; closing the issue moves it to Done), and dependencies are native
*blocked by* links. Pick the lowest-numbered issue that is open, unassigned and unblocked:

```bash
gh issue list --milestone v1 --search "-is:blocked no:assignee" --json number,title
```

Read the task with `gh issue view <N>`, then assign yourself and move it to *In Progress* — the
exact commands are under *Starting it* in `tasks/README.md` — and put `Closes #<N>` in the PR. Each task names the `REQ-` requirements it owns; they are the
acceptance criteria. Requirements change in `docs/v1-requirements.md` through a PR, never in an
issue thread; update the affected issues once it merges.

## Tooling

`.mcp.json` configures the `docs-langchain` MCP server (docs.langchain.com). Use it to check
current LangChain 1.x APIs — `BaseChatModel` internals, callbacks, `bind_tools`,
`with_structured_output`, `configurable_fields`, caching — rather than relying on memory, since
the pre-1.0 APIs differ.
