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

## Scope

- Expose the forced route through standard mechanisms (`configurable_fields` plus `with_config` or
  `config={"configurable": ...}`); other runtime configuration keeps working as usual.
- If the forced route doesn't exist or can't use the bound tools: explicit error by default; a
  setting switches to falling back per R9/R10 with a warning.
- Record forced routes and fallbacks in the decision record.

## Acceptance criteria

- [ ] The forced route is used even when the strategy would choose otherwise (whether the strategy
      still runs is a T-101 decision).
- [ ] Unknown and tool-incompatible forced routes: error by default; fallback plus warning with the
      setting.
