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

Requirements: **REQ-R1-1, REQ-R2-1, REQ-R2-2, REQ-R2-4**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- The six-field `RoutingDecision` schema (D8): `route`, `reason`, `strategy`, `fallback`, `forced`,
  `diverted_from`. It rides under `response_metadata["routing"]` and, as the same dict, in the
  router's chain-run metadata (C5).
- `routing_decision(message)` reads it back off a response.
- Structured output (D3): the parsed object has nowhere to carry a record, so the chain run always
  has it, `include_raw=True` puts it on the raw message, and `last_routing_decision()` — a context
  variable set per call — covers the parsed-only path. Document the limit.
- R1: content, tool calls, `usage_metadata`, `response_metadata`, ids — nothing dropped or renamed.

## Acceptance criteria

- [ ] A test compares router output with the route's direct output field by field; only the added
      decision record differs.
- [ ] The record is present for invoke, stream, batch and every fallback, diversion and forced
      path.
- [ ] The record is visible on the router's run in LangSmith.
- [ ] Structured output: `include_raw=True` carries it, and `last_routing_decision()` returns the
      same record after a parsed-only call.
