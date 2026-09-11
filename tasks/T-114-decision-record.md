---
id: T-114
title: Routing decision record
phase: v1
status: todo
principles: [R1, R2]
depends_on: [T-110]
prd: ["§3.2"]
---

# T-114 · Routing decision record

## Goal

Every response carries the selected model's output untouched, plus a record of which route was
taken and why.

## Scope

- Schema from T-101: route name, reason, strategy, and flags for fallback (R9), tool diversion
  (R10) and forced route (R11).
- Placement on the returned message (e.g. `response_metadata`) and on the trace (C5).
- R1: content, tool calls, `usage_metadata`, `response_metadata`, ids — nothing dropped or renamed.

## Acceptance criteria

- [ ] A test compares router output with the route's direct output field by field; only the added
      decision record differs.
- [ ] The record is present for invoke, stream, batch and every fallback, diversion and forced
      path.
- [ ] The record is visible in LangSmith on the router's run.
