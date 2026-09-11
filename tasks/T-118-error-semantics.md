---
id: T-118
title: Error semantics — leave retries and fallbacks to LangChain
phase: v1
status: todo
principles: [C6, R9]
depends_on: [T-110]
prd: ["§3.1", "§3.2"]
---

# T-118 · Error semantics — leave retries and fallbacks to LangChain

## Goal

The selected model's errors surface unchanged, so `with_retry` and `with_fallbacks` behave on the
router exactly as on a model (C6). Strategy errors are the only ones the router absorbs (→ default
route, R9).

## Scope

- No router-level retries or model fallbacks.
- Exceptions from a route propagate with their original type and details.
- A clear line between strategy failure (absorbed, warned, recorded) and model failure
  (propagated).

## Acceptance criteria

- [ ] Route exceptions reach the caller as the same exception the route raised.
- [ ] `router.with_retry(...)` and `router.with_fallbacks([...])` work, as do routes that are
      themselves wrapped with retries or fallbacks.
