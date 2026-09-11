---
id: T-119
title: Response cache correctness
phase: v1
status: todo
principles: [C10]
depends_on: [T-115, T-116]
prd: ["§3.1", "§11"]
---

# T-119 · Response cache correctness

## Goal

LangChain's response cache (`set_llm_cache`, a model's `cache`) works on the router as on any chat
model, and never serves an answer cached for a different route or configuration (C10).

## Context

`BaseChatModel` looks up the cache *before* `_generate` runs — that is, before routing — keyed on
the prompt plus the model's identifying params and call kwargs. So anything that changes which
route answers must be in the key, or routing must happen before the lookup.

## Scope

- Decide the approach (e.g. identifying params that cover the route set, strategy config and
  forced route; or routing ahead of the cache lookup).
- Cover: changed route set, changed strategy config, forced route, bound tools / structured
  output, non-deterministic strategies.
- Interaction with routes that have their own caches.

## Acceptance criteria

- [ ] With `InMemoryCache`, a repeated identical request is a hit; the same prompt forced to
      another route, or under a changed strategy config, is a miss.
- [ ] Cached responses still carry a correct decision record.
