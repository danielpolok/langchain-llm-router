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
- Embedding calls are traced as their own runs, nested in the strategy run (T-117, D9). LangChain's
  `Embeddings` is a plain ABC — no callbacks, no `config`, no usage — so its calls are invisible
  to tracing unless the strategy opens a child run from `request.config` around each one. Do
  that, recording the input length on the run.
- Cost can't be measured through the interface, only estimated from input length. Record the
  estimate method; T-140 uses it for strategy overhead.
- Embedding failure → default route with a warning (R9).

## Acceptance criteria

- [ ] It can't be enabled implicitly; construction requires the embeddings instance.
- [ ] Each per-request embedding call appears as a child run of the strategy run, sync and async.
- [ ] Offline tests with fake embeddings (e.g. `DeterministicFakeEmbedding`); one integration test
      with real embeddings.
