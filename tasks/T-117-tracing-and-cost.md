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

Requirements: **REQ-C5-1, REQ-C5-2, REQ-C5-4, REQ-C5-5, REQ-C2-2, REQ-R3-1, REQ-R3-2**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- The router's own run is a **chain** run; the route's call, nested inside it, is the only LLM run
  outside the strategy run. One model run per model called means one bill each (R3), and the real
  call stays traced (C5).
- **Order of operations (D9): open the router run, then decide, then call the route.** The spike
  decided first and opened its run afterwards, so a strategy's model call had no parent to nest
  under. Now the strategy runs in a child run of its own, named after the strategy class, whose
  output is the decision. `decide` / `adecide` run inside `set_config_context` with that run's
  child config (sync: the returned context's `run`; async: `coro_with_context`), and the same
  config goes to the strategy as `RoutingRequest.config`. No strategy run when there's no strategy
  or the route is forced.
- The decision can't ride in the router run's start metadata any more: it goes on the strategy
  run's output, the router run's output (`on_chain_end`) and — through `patch_config` metadata —
  the route run's metadata.
- Cover every calling convention, **including `generate()` / `agenerate()`** — the spike's design
  overrode only the public entry points, so these still took the base path and would double count
  (spike caveat 1).
- Place usage and `ls_*` metadata as decided in T-004. If the router's own run ever needs the
  selected model's identity, `_attach_gateway_metadata` (`tracers/core.py:352`) is the supported
  mechanism — it exists because a gateway, too, only learns the real model from the response.
- Extra calls made by opt-in strategies (T-133, T-134) are traced and costed under their own
  models, nested in the strategy run — never folded into the router or the answering model.
- Python 3.10: asyncio tasks can't take a `context` there, so an async call propagates callbacks
  only if handed its config (LangChain's docs). Test the nesting on 3.10 in CI, both with a
  strategy that passes `request.config` (must nest) and one that doesn't (documented limit).

## Acceptance criteria

- [ ] `UsageMetadataCallbackHandler` totals equal the sum of the routes' direct usage across
      invoke, stream, batch **and `generate`**.
- [ ] The run-tree assertion (one chain run, exactly one nested LLM run) is parametrised over
      every convention.
- [ ] With a strategy that calls a fake chat model: router run → strategy run → strategy's LLM
      run, and router run → route's LLM run — sync and async, on Python 3.10 and 3.12.
- [ ] `UsageMetadataCallbackHandler` with that strategy reports the strategy's model and the
      route's model separately, each counted once.
- [ ] A LangSmith trace per convention shows the decision and the real call, with cost equal to
      one call of the selected model (plus any strategy calls, as their own runs).
