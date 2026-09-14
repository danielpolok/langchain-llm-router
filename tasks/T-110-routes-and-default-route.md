---
id: T-110
title: Named routes and the mandatory default route
phase: v1
status: todo
principles: [R5, R9, R2]
depends_on: [T-101]
prd: ["§3.2", "§11"]
---

# T-110 · Named routes and the mandatory default route

## Goal

Any number of named routes, each an ordinary chat model the application already builds, plus a
default route that is always configured.

## Scope

- Declare routes by name; validate at construction (at least one route, default present, names
  unique).
- Whenever the strategy can't decide, raises, or returns an unknown route, use the default route,
  emit a warning and record the reason (R9, R2).
- Routes are used as given — the router doesn't reconfigure them.

## Acceptance criteria

- [ ] Construction without a default route is an error.
- [ ] Each fallback path (undecided, exception, unknown route) is tested: correct route, one
      warning, reason in the decision record.
- [ ] Works with one, two and many routes.
