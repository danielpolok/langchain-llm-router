---
id: T-004
title: "Spike: cost counted once, real call still traced (R3 vs C5)"
phase: v0
status: done
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

- [x] `get_usage_metadata_callback()` reports usage once, keyed by the selected model's name —
      for both `invoke` and `stream`. (Design A, `spike/tests/test_cost_and_tracing.py`.)
- [x] A LangSmith trace shows the routing decision and the real model call, and the trace cost
      equals one call of the selected model. Confirmed live (project `llm-router`, run
      `01a0a161-…`): one `chain` run (`DelegatingRouterChatModel`) with one nested `llm` run
      (`ChatGoogleGenerativeAI`, `ls_model_name: gemini-3-flash-preview`) carrying
      `routing: {route: gemini, reason: 'no strategy configured'}` and `usage_metadata`; both
      runs report the same `total_cost` ($0.0001745) because there is only one billed call.
- [x] The caller's `AIMessage.usage_metadata` is identical to the route's (R1).
- [x] Written verdict in this file: **both hold**, or **which gives way and why**, recorded in
      PRD §11.

## Verdict — both hold, once the router stops being a model run (confirmed against LangSmith)

R3 and C5 do not conflict. They only appear to while the router emits a *model* run of its own:
two model runs mean the same tokens are billed twice. The fix is that the router's own run is a
**chain** run that delegates, leaving the selected route's call as the only model run in the
trace.

| Design | R1 response untouched | R3 counted once | C5 real call traced |
| --- | --- | --- | --- |
| Naive (T-002) — router and route both emit a model run | yes | **no** — 16 tokens billed for an 8-token call, invoke and stream alike | **no** — the streamed route call is missing from the trace entirely |
| **A — the router is a chain run that delegates** | yes | **yes** — invoke, ainvoke and stream | **yes** — nested under a run whose metadata carries the decision |
| B — one model run, relabelled through gateway metadata | yes | yes for invoke; streaming not implemented | partly — one run wearing the model's identity, no separate call |

**Chosen: A** (`DelegatingRouterChatModel` in `spike/designs.py`).

- It needs no private API. A chain run with child callbacks is how `RunnableWithFallbacks` and
  LangChain's own dynamic model (`_ConfigurableModel`, behind
  `init_chat_model(configurable_fields=...)`) already delegate. B reaches into
  `_generate_with_cache` and the tracer's gateway key.
- It closes T-002's streaming gap as a side effect: a *chain* run manager has `get_child()`,
  which an LLM run manager does not, so the route's streamed run finally nests.
- It shows the decision **and** the call as separate runs; B collapses them into one.

**Worth keeping from B:** `_attach_gateway_metadata` (`tracers/core.py:352`) promotes a
response-time model identity over the request-time one — the mechanism a gateway uses because
it, too, only learns the real model from the response. If v1 ever needs the router's own run to
be priced as the selected model, that is the supported way to do it.

### Caveats for v1

1. **A overrides the public entry points** (`invoke`, `ainvoke`, `stream`, `astream`).
   `generate()` / `agenerate()` still take the base path and would double count. v1 must cover
   them too (T-117).
2. **The router's run is a chain run**, so anything that looks for a chat-model run *around* the
   router — LangGraph's `stream_mode="messages"` registers on `on_chat_model_start` — now sees
   only the route's run. That is the wanted behaviour (one stream of tokens, not two), but it
   should be confirmed against a real graph (T-113, T-117).
3. **The naive design fails quietly.** Its router run carries usage but no `ls_model_name`, so
   LangSmith would likely price it at zero while still double counting the tokens — wrong in a
   way that does not show up as an error. Exactly what R3 exists to prevent.
