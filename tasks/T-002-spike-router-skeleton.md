---
id: T-002
title: "Spike: minimal router skeleton"
phase: v0
status: todo
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

- [ ] With two fake routes (`GenericFakeChatModel`), `invoke`, `ainvoke`, `stream`, `astream` and
      `batch` all return the selected route's output unchanged.
- [ ] A strategy that raises or returns an unknown route falls back to the default route with a
      warning.
- [ ] The decision record is readable from the returned message.

## Notes

- Delegation must hand the route the child callback manager so its run nests under the router's
  run — T-004 depends on this.
