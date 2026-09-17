---
id: T-133
title: "Opt-in strategy: embedding similarity"
phase: v1
status: todo
principles: [R6, R7, R8, R3]
depends_on: [T-111, T-112, T-117]
prd: ["§3.2", "§8", "§9"]
---

# T-133 · Opt-in strategy: embedding similarity

## Goal

Route by semantic similarity between the current request and example requests for each route. It
makes extra API calls, so it is an explicit opt-in (R7).

Requirements: **REQ-R7-2**, **REQ-R3-2** ([v1 requirements](../docs/v1-requirements.md)).

## Scope

- The application supplies a LangChain `Embeddings` instance — no new dependencies (R8), no hidden
  credentials.
- Example utterances per route; a similarity threshold below which the strategy can't decide
  (→ default route, R9).
- Route examples are embedded once, not per request; async path included.
- Embedding calls are traced and costed as their own runs (T-117).
- Embedding failure → default route with a warning (R9).

## Acceptance criteria

- [ ] It can't be enabled implicitly; construction requires the embeddings instance.
- [ ] Offline tests with fake embeddings (e.g. `DeterministicFakeEmbedding`); one integration test
      with real embeddings.
