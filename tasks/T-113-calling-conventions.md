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

## Scope

- `invoke` / `ainvoke`, `stream` / `astream`, `batch` / `abatch`, `astream_events`.
- Streaming: the route is decided before the first chunk; chunks come from the selected route
  unchanged; the decision record reaches the streamed result (R2) — define which chunk carries it.
- The selected route's own streaming behaviour is preserved (a route without native streaming
  still works).

## Acceptance criteria

- [ ] Each convention tested with fake routes; merged stream output equals the `invoke` output.
- [ ] Async paths never block the event loop (strategy and route are both awaited).
