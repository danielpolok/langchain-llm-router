---
id: T-111
title: Strategy interface
phase: v1
status: todo
principles: [R6, R4, R7]
depends_on: [T-101]
prd: ["§3.2", "§4", "§11"]
---

# T-111 · Strategy interface

## Goal

The one small, stable interface every strategy implements — ready-made, configured or custom
(R6).

## Scope

- Input: the extracted current request (T-112), plus wider context only when the strategy opts in
  (R4); the available route names.
- Output: a route name and a reason — or an explicit "can't decide" (→ default route, R9).
- Sync and async forms, so strategies that call models (T-133, T-134) don't block async callers.
- Public API: document it and state its stability promise.

## Acceptance criteria

- [ ] A custom strategy takes a few lines of user code (the "custom strategy" use case, §4).
- [ ] The built-in strategies (T-130–T-134) implement the same interface with no private hooks.
- [ ] Sync and async paths are tested.
