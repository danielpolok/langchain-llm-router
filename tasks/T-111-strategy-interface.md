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

The one small, stable interface every strategy implements — ready-made, configured or custom (R6).

Requirements: **REQ-R6-1, REQ-R6-2, REQ-R4-3**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- `RoutingStrategy`, `RoutingRequest`, `RoutingChoice` and `RoutingCallable` exactly as pinned in
  the requirements' *Public API* section (D6). No private hooks: whatever the built-ins need is on
  the public interface.
- `decide` returns a `RoutingChoice` or `None` for "can't decide" (→ default route, R9).
- `adecide` defaults to running `decide` in an executor, so a synchronous custom strategy never
  blocks an async caller (C2); strategies that call models override it properly (T-133, T-134).
- Wider context only when the strategy sets `wants_full_context` — otherwise `messages` is `None`
  (R4).
- A plain callable passed as `strategy=` is coerced to a `RoutingStrategy` at construction, and a
  bare route-name string return is accepted.
- Document the interface and state its stability promise (the promise is enforced at T-151).

## Acceptance criteria

- [ ] A custom strategy is five lines or fewer in a test (the "custom strategy" use case, §4), both
      as a function and as a subclass.
- [ ] The built-in strategies (T-130–T-134) implement this interface with no private hooks.
- [ ] Sync and async paths are both tested, including the executor default.
- [ ] `messages` is `None` unless the strategy opts in.
