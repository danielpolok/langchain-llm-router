---
id: T-112
title: Current-request extraction
phase: v1
status: todo
principles: [R4, C7]
depends_on: [T-111]
prd: ["§3.1", "§3.2", "§11"]
---

# T-112 · Current-request extraction

## Goal

By default, strategies see the user's *current request* — not tool output, system prompts or
conversation length (R4). The PRD's survey found that last-message or length-based routing
misroutes agent conversations.

Requirements: **REQ-R4-1, REQ-R4-2, REQ-R4-4, REQ-C7-1, REQ-C7-2**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- Build the `RoutingRequest` from a transcript. "Current request" is the most recent
  `HumanMessage` by position, ignoring trailing AI and tool messages — so an agent loop keeps
  routing on the request that started it.
- Read messages as LangChain defines them (C7): string content and content blocks. Join text into
  `text`; expose the content blocks and the set of `modalities` so a strategy can route on a
  non-text request.
- Every input form a chat model accepts (string, dicts, `BaseMessage`s, a `ChatPromptValue`)
  reaches extraction through the same path.
- Opt-in wider context: populate `messages` only when the strategy sets `wants_full_context`.
- No user message at all → "can't decide" → default route (R9).

## Acceptance criteria

- [ ] Fixture transcripts: single turn, multi-turn, agent tool loop, multimodal (text + image),
      system prompt only.
- [ ] In a three-iteration agent loop, all three model calls route on the originating user request.
- [ ] The four input forms produce an identical `RoutingRequest`.
