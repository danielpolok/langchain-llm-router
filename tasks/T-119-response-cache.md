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

Requirements: **REQ-C10-1…4** ([v1 requirements](../docs/v1-requirements.md)).

## Context

`_generate_with_cache` (`chat_models.py:1892`) looks the cache up *before* `_generate` runs — that
is, before routing — keyed by `_get_llm_string` (`:1578`): the model's serialized repr plus the
call kwargs.

**D4 settles the approach: the route owns caching.** Because the router delegates from
`invoke` / `stream` rather than `_generate`, the router's own `_generate_with_cache` never runs;
the selected route's cache applies, keyed on that route's identity. C10 then holds structurally —
an answer cached for one route cannot be found under another's key — rather than by arithmetic the
router has to maintain. This task verifies that and closes the two gaps it leaves.

## Scope

- Verify the delegated-cache behaviour across the cases: changed route set, changed strategy
  config, forced route, bound tools / structured output, non-deterministic strategies (documented
  as the user's concern, not the router's).
- **The router's own `cache=` must not silently no-op** — reject it at construction with a message
  pointing at per-route caching.
- Keep process-unstable reprs out of the key: a bare function bound as a tool renders as
  `<function f at 0x…>`, which changes every process and would miss every time (spike caveat).
- Interaction with routes that have their own caches, and with `set_llm_cache` globally.

## Acceptance criteria

- [ ] With `InMemoryCache`, a repeated identical request is a hit; the same prompt forced to
      another route, or under a changed strategy config, is a miss.
- [ ] `ChatRouter(cache=…)` raises, and the message says where caching belongs.
- [ ] The cache key for the same bound tools is equal across two processes.
- [ ] Cached responses still carry a correct decision record.
