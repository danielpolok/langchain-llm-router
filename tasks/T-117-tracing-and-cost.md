---
id: T-117
title: Tracing and cost attribution
phase: v1
status: todo
principles: [C5, R3]
depends_on: [T-113, T-114]
prd: ["§3.1", "§3.2"]
---

# T-117 · Tracing and cost attribution

## Goal

Build the design chosen in T-004: traces show the routing decision and the real model call in
LangSmith's normal shape (C5), and cost is counted once and attributed to the model that ran (R3).

## Scope

- Nest the route's run under the router's run for every calling convention.
- Place usage and `ls_*` metadata as decided in T-004.
- Extra calls made by opt-in strategies (T-133, T-134) are traced and costed under their own
  models — never folded into the router or the answering model.

## Acceptance criteria

- [ ] `UsageMetadataCallbackHandler` totals equal the sum of the routes' direct usage across
      invoke, stream and batch.
- [ ] A LangSmith trace per convention shows the decision and the real call, with cost equal to
      one call of the selected model (plus any strategy calls).
