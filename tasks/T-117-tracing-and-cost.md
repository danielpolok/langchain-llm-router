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

Requirements: **REQ-C5-1, REQ-C5-2, REQ-C5-4, REQ-C2-2, REQ-R3-1, REQ-R3-2**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- The router's own run is a **chain** run carrying the decision in its metadata; the route's call,
  nested inside it, is the only LLM run. One model run means one bill (R3) and the real call stays
  traced (C5).
- Cover every calling convention, **including `generate()` / `agenerate()`** — the spike's design
  overrode only the public entry points, so these still took the base path and would double count
  (spike caveat 1).
- Place usage and `ls_*` metadata as decided in T-004. If the router's own run ever needs the
  selected model's identity, `_attach_gateway_metadata` (`tracers/core.py:352`) is the supported
  mechanism — it exists because a gateway, too, only learns the real model from the response.
- Extra calls made by opt-in strategies (T-133, T-134) are traced and costed under their own
  models — never folded into the router or the answering model.

## Acceptance criteria

- [ ] `UsageMetadataCallbackHandler` totals equal the sum of the routes' direct usage across
      invoke, stream, batch **and `generate`**.
- [ ] The run-tree assertion (one chain run, exactly one nested LLM run) is parametrised over
      every convention.
- [ ] A LangSmith trace per convention shows the decision and the real call, with cost equal to
      one call of the selected model (plus any strategy calls, as their own runs).
