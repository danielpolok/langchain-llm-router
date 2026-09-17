---
id: T-113
title: Sync, async, streaming and batch
phase: v1
status: todo
principles: [C2, R1]
depends_on: [T-110, T-111]
prd: ["§3.1"]
---

# T-113 · Sync, async, streaming and batch

## Goal

Every Runnable calling convention works on the router, with no new ways of calling it (C2).

Requirements: **REQ-C2-1, REQ-C2-3, REQ-C2-4, REQ-R2-3, REQ-C5-3**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- `invoke` / `ainvoke`, `stream` / `astream`, `batch` / `abatch`, `astream_events`. These are the
  public entry points the chosen design overrides (T-004); `generate()` / `agenerate()` are
  T-117's.
- Streaming: the route is decided before the first chunk; chunks come from the selected route
  unchanged; exactly **one** chunk carries the decision record (D8 — `merge_dicts` concatenates a
  string repeated across chunks).
- The selected route's own streaming behaviour is preserved: a route with no native streaming
  still yields a single chunk equal to its full message.
- Async: both the strategy (`adecide`) and the route are awaited; neither blocks the event loop.
- Confirm LangGraph's `stream_mode="messages"` against a real graph — the router's run being a
  *chain* run changes what registers on `on_chat_model_start` (spike caveat 2). One token stream
  is the wanted result.

## Acceptance criteria

- [ ] Each convention tested with fake routes; merged stream output equals the `invoke` output.
- [ ] Async paths never block: a sleeping `adecide` does not stall a concurrent task.
- [ ] Merging every streamed chunk yields exactly one decision record.
- [ ] A real LangGraph node streams the route's tokens once under `stream_mode="messages"`.
