---
id: T-004
title: "Spike: cost counted once, real call still traced (R3 vs C5)"
phase: v0
status: todo
principles: [R3, C5, R1]
depends_on: [T-002]
prd: ["§6", "§8"]
---

# T-004 · Spike: cost counted once, real call still traced (R3 vs C5)

## Goal

Show that cost is counted exactly once and attributed to the model that ran (R3), while that
model's call still appears as a nested run in the trace (C5). If both can't hold, decide which
gives way (§8).

## The tension

The router and the selected route are both chat models, so both emit an LLM run. If both runs
carry `usage_metadata`, usage is summed twice — by `UsageMetadataCallbackHandler` and by
LangSmith, which prices each LLM run from its `usage_metadata` plus `ls_provider` /
`ls_model_name`. Stripping usage from the router's output conflicts with R1 (nothing dropped from
the response); hiding the inner run conflicts with C5.

## Scope

First confirm the double count with the naive design from T-002. Then try candidate designs and
score each against R1, R3 and C5, for example:

- the nested route run carries usage; the router's own run reports none — can that be done
  without changing the message the caller receives?
- no nested run; the router's run takes on the selected model's identity (`ls_*` params);
- anything else the callback and tracing APIs allow.

## Acceptance criteria

- [ ] `get_usage_metadata_callback()` reports usage once, keyed by the selected model's name —
      for both `invoke` and `stream`.
- [ ] A LangSmith trace shows the routing decision and the real model call, and the trace cost
      equals one call of the selected model.
- [ ] The caller's `AIMessage.usage_metadata` is identical to the route's (R1) — or the conflict
      is documented.
- [ ] Written verdict in this file: **both hold**, or **which gives way and why**, recorded in
      PRD §11.
