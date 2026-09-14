---
id: T-001
title: Project scaffold
phase: v0
status: done
principles: [C9, R8]
depends_on: []
prd: ["§3", "§6"]
---

# T-001 · Project scaffold

## Goal

A minimal Python package for the spike (T-002–T-004) to live in, with tooling that v1 keeps.

## Scope

- `pyproject.toml` with a `src/` layout. Choose the environment/build tool (e.g. uv) — this
  decides how the repo is run.
- Runtime dependency: `langchain-core` 1.x **only** (R8, C9). Set the minimum 1.x version from
  the APIs actually used (model profiles, for example, need a 1.1+ release).
- Dev dependencies: pytest (+ an async plugin), ruff, mypy, `langchain-tests`. Provider packages
  (e.g. `langchain-openai`, `langchain-anthropic`) and `langchain` / `langgraph` for compatibility
  tests go in dev or optional extras, never runtime.
- Test layout `tests/unit_tests` (offline, fake models) and `tests/integration_tests` (real
  providers) — the layout `langchain-tests` expects.

## Acceptance criteria

- [x] Tests, lint and type-check run clean on the empty package.
- [x] Installing the package pulls in no runtime dependency other than `langchain-core` and its
      transitive dependencies.
- [x] Integration tests are skipped, not failed, when provider API keys are absent.
- [x] CLAUDE.md gains a **Commands** section: install, test, single test, lint, type-check.

## Outcome

- **uv** manages the environment and builds (`uv_build` backend); Python 3.12 for development,
  `requires-python >= 3.10` (langchain-core 1.6's floor).
- Runtime dependency `langchain-core>=1.1,<2` — 1.1 is the first release with model profiles.
  `uv tree --no-default-groups` shows only langchain-core and its transitive dependencies, and
  `tests/unit_tests/test_package.py` asserts it.
- Dependency groups: `dev` (pytest, pytest-asyncio, ruff, mypy, langchain-tests, langchain,
  langgraph) and `providers` (langchain-openai, langchain-anthropic); both install by default.
- `@pytest.mark.requires_env(...)` (root `conftest.py`) skips tests whose provider keys are unset.
- mypy runs strict against the dev interpreter. Pinning `python_version = "3.10"` breaks on
  numpy's 3.12-only stub syntax, which arrives via the dev dependencies.
