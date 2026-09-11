---
id: T-120
title: LangChain compatibility suite
phase: v1
status: todo
principles: [C1, C8, C9]
depends_on: [T-113, T-115, T-116, T-117, T-118, T-119]
prd: ["§3.1", "§4"]
---

# T-120 · LangChain compatibility suite

## Goal

Prove "it is a chat model" (C1): anything that works with a LangChain chat model works the same
way with the router.

## Scope

- `langchain-tests` standard suites: `ChatModelUnitTests` offline, `ChatModelIntegrationTests`
  with real routes.
- Placement tests: LCEL chain, `create_agent(model=router)` (the "agent backbone" use case, §4),
  LangGraph node, message-history wrapper.
- C8: coexists with `@wrap_model_call` middleware inside `create_agent`.
- C9: CI matrix on the minimum supported and the latest `langchain-core` 1.x.

## Acceptance criteria

- [ ] The standard unit suite passes; every skipped standard test has a documented reason.
- [ ] Each placement test passes offline with fake routes.
- [ ] CI runs the version matrix.
