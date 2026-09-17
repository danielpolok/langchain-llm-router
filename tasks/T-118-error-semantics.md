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

Requirements: **REQ-C6-1, REQ-C6-2, REQ-C6-3, REQ-R9-3**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- No router-level retries or model fallbacks.
- Exceptions from a route propagate with their original type and details, and close the router's
  chain run through `on_chain_error` — no dangling open run in the trace.
- A clear line between strategy failure (absorbed, warned, recorded) and model failure
  (propagated). The R9 fallback must never swallow a route's error.

## Acceptance criteria

- [ ] Route exceptions reach the caller as the same exception the route raised, and the route was
      called once.
- [ ] `router.with_retry(...)` and `router.with_fallbacks([...])` work, as do routes that are
      themselves wrapped with retries or fallbacks.
- [ ] The tracer records `on_chain_error` on the router's run when a route fails.
