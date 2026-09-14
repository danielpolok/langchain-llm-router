---
id: T-150
title: Documentation
phase: v1
status: todo
principles: [C8, R6]
depends_on: [T-120, T-130, T-131, T-132]
prd: ["§4", "§5", "§8", "§9"]
---

# T-150 · Documentation

## Goal

Enough documentation for another LangChain developer to adopt the router without reading its
source.

## Scope

- Quickstart.
- One runnable example per use case (§4): cost tiering, domain routing, agent backbone, custom
  strategy, experimentation (forced routes).
- The three strategy levels (R6) and a reference for the strategy interface.
- The decision record, warnings and errors (R2, R9–R11).
- When to use agent middleware instead (C8); non-goals (§5); the prompt-caching caveat (§8).

## Acceptance criteria

- [ ] Every example runs in CI (offline with fakes where possible).
