---
id: T-002
title: "Spike: minimal router skeleton"
phase: v0
status: done
principles: [C1, C2, R1, R2, R5, R9]
depends_on: [T-001]
prd: ["§6"]
---

# T-002 · Spike: minimal router skeleton

## Goal

The smallest `RouterChatModel` that T-003 and T-004 can test against. Spike quality: it answers
questions, it is not v1.

## Scope

- A `BaseChatModel` subclass holding named routes (any number, R5), a default route (R9) and a
  trivial strategy — a plain callable from messages to a route name is enough.
- Delegate `_generate`, `_agenerate`, `_stream` and `_astream` to the selected route; `batch`
  comes from the Runnable defaults (C2).
- Attach a minimal decision record (route, reason) to the response (R1, R2) — the shape is not
  final.
- Keep spike code isolated (e.g. a `spike/` package or a branch) so v1 starts from requirements,
  not from the spike.

## Acceptance criteria

- [x] With two fake routes, `invoke`, `ainvoke`, `stream`, `astream` and `batch` all return the
      selected route's output unchanged.
- [x] A strategy that raises or returns an unknown route falls back to the default route with a
      warning.
- [x] The decision record is readable from the returned message.

## Notes

- Delegation must hand the route the child callback manager so its run nests under the router's
  run — T-004 depends on this.

## Outcome

`spike/router.py` (`SpikeRouterChatModel`), fakes in `spike/fakes.py`, tests in
`spike/tests/test_skeleton.py` — 20 offline tests.

**Deviation.** The routes are a local `FakeChatModel`, not `GenericFakeChatModel`: the core fake
drops `usage_metadata` and `response_metadata` when it streams (it re-splits the content of
`_generate`), which T-004 needs, and no core fake implements `bind_tools`, which T-003 needs.

**Findings for T-003 and T-004:**

1. **An LLM run manager cannot make a child.** `CallbackManagerForLLMRun` is not a
   `ParentRunManager`, so it has no `get_child()` — LangChain does not expect an LLM run to have
   children. The router reproduces `ParentRunManager.get_child` by hand
   (`_child_callbacks`), and the route's run then nests correctly under `invoke`/`ainvoke`.
2. **Streaming cannot nest at all.** `BaseChatModel.stream` and `astream` call `_stream`/
   `_astream` *without* a run manager (`chat_models.py:794`, `:927`), and so does the streaming
   branch of `_generate_with_cache` (`:1981`, `:2136`) — which is what runs when anything
   streams the application around the router, e.g. a LangGraph node. Only the v2 protocol path
   passes one (`:687`, `:718`). So under the naive design the route's streamed call has no
   callbacks: it is invisible to the caller's handlers and to LangSmith (C5). T-004 weighs
   designs that close this; it is a gap in this design, not a decision.
3. **The decision record can ride on only one chunk.** `merge_dicts` concatenates string values
   that repeat across chunks, so a record on every chunk aggregates to `"frontierfrontier…"`.
4. The record is added to the route's own message in place, so the *nested* run's traced output
   does not carry it — that callback has already fired when the router adds it.
5. The router's `_get_llm_string` (cache key) does not include which route was selected, so a
   router-level cache would answer across routes. C10's problem, left to T-119.
