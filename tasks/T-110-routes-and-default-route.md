---
id: T-110
title: Named routes, the default route and the public API
phase: v1
status: todo
principles: [R5, R9, R2]
depends_on: [T-101]
prd: ["§3.2", "§11"]
---

# T-110 · Named routes, the default route and the public API

## Goal

Any number of named routes, each an ordinary chat model the application already builds, plus a
default route that is always configured — behind the public surface the rest of v1 builds on.

Requirements: **REQ-R5-1, REQ-R5-2, REQ-R9-1, REQ-R9-2**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- The `ChatRouter` constructor exactly as pinned in the requirements' *Public API* section:
  `routes`, `default_route`, `strategy`, `on_unavailable_forced_route`, `tool_support_overrides`.
  Declaration order of `routes` is significant (D1) — keep the mapping ordered.
- The package's exports and the `RoutingDecision` / warning / error types. The rest of v1 imports
  them from here.
- Validate at construction: at least one route, the default route names one of them, names unique.
  A `ValueError` raised in a pydantic validator surfaces as `ValidationError` — tests assert that.
- Whenever the strategy can't decide, raises, or returns an unknown route: default route, one
  `FallbackWarning`, reason recorded (R9, R2).
- Routes are used as given — never reconfigured or mutated.

## Acceptance criteria

- [ ] Construction without a default route, with an unknown default, or with no routes fails, and
      the message names the offender.
- [ ] Each fallback path (abstained, raised, unknown route) is tested: correct route, exactly one
      warning, `fallback=True` and a reason naming the cause.
- [ ] Works with one, two and many routes; a route object is identical (`is`) before and after.
