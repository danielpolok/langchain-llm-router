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

## Scope

- Define "current request" for real transcripts: plain chat, and agent loops where the trailing
  messages are AI tool calls and `ToolMessage`s.
- Read messages as LangChain defines them (C7): string content and content blocks. Extract text
  from multimodal content and expose non-text modalities so a strategy can route on them.
- Opt-in wider context for strategies that ask for it.
- No user message at all → the strategy can't decide → default route (R9).

## Acceptance criteria

- [ ] Fixture transcripts: single turn, multi-turn, agent tool loop, multimodal (text + image),
      system prompt only.
- [ ] In an agent loop, every model call routes on the originating user request, not the latest
      tool output.
