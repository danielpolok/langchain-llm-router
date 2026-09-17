---
id: T-121
title: Capability reporting — the router's own profile
phase: v1
status: todo
principles: [C3, C1, R10]
depends_on: [T-110, T-115]
prd: ["§3.1", "§11"]
---

# T-121 · Capability reporting — the router's own profile

## Goal

The router answers for capabilities it does not own. `create_agent` reads `model.profile` to pick
a structured-output strategy (`_supports_provider_strategy`, `langchain/agents/factory.py:560`),
so a router that claimed a capability only some routes have would have a strategy chosen for it
that a route cannot serve.

Requirement: **REQ-C3-4** ([v1 requirements](../docs/v1-requirements.md)).

## Scope

- Override `_resolve_model_profile()` (`chat_models.py:398`) to report the **intersection** of the
  routes' profiles: booleans AND-ed, ints reduced to the minimum, equal values kept, differing
  values dropped, `None` when any route reports no profile.
- `profile` is beta and its keys may change; unknown keys must not crash the reduction
  (`_warn_unknown_profile_keys`, `chat_models.py:443`).
- An explicitly supplied `profile=` on the router wins, as it does on any chat model.
- T-115 reads `profile["tool_calling"]` per route for capability detection (D5); this task is the
  other direction — what the router reports about itself.

## Acceptance criteria

- [ ] Intersection is tested per value kind: booleans, ints, equal values, differing values, a
      route with no profile.
- [ ] `create_agent(model=router, response_format=…)` over routes with mixed profiles settles on a
      strategy every route can serve, and the structured response comes back.
- [ ] A router with one route reports that route's profile unchanged.

## Notes

The spike wrote a working version of this reduction (`spike/router.py:134`), confirmed in T-003's
verdict. Evidence, not code to import.
