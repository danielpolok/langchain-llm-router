---
id: T-116
title: Forced routes via runtime config
phase: v1
status: todo
principles: [C4, R11, R2]
depends_on: [T-110, T-114, T-115]
prd: ["§3.1", "§3.2", "§4", "§11"]
---

# T-116 · Forced routes via runtime config

## Goal

Force a specific route for one call through LangChain's normal runtime configuration (C4) — the
"experimentation" use case (§4) — and never silently swap it (R11).

Requirements: **REQ-C4-1, REQ-C4-2, REQ-R11-1…3**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- The configurable key is `route` (D7), declared through `config_specs` as a
  `ConfigurableFieldSpec` (`runnables/utils.py:655`), so `config={"configurable": {"route": …}}`,
  `with_config` and config-schema introspection all work. Other runtime configuration — tags,
  metadata, callbacks, run name, concurrency — keeps reaching the route unchanged.
- **A forced route skips the strategy entirely (D2).** It is not a suggestion, and running a
  strategy whose answer is discarded costs a call under T-133 / T-134.
- If the forced route doesn't exist or can't use the bound tools: `ForcedRouteError` by default;
  `on_unavailable_forced_route="fallback"` switches to the R9 / R10 path with one warning.
- Record `forced=True`, and `fallback=True` with a reason when it gave way.

## Acceptance criteria

- [ ] The forced route is used even when the strategy would choose otherwise, and the strategy's
      `decide` is never called.
- [ ] Unknown and tool-incompatible forced routes: error by default; fallback plus exactly one
      warning under the setting.
- [ ] Runtime configuration other than `route` is observed on the route's run.
