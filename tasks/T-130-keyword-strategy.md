---
id: T-130
title: "Ready-made strategy: keyword"
phase: v1
status: todo
principles: [R6, R7, R8]
depends_on: [T-111, T-112]
prd: ["§3.2", "§4", "§9"]
---

# T-130 · Ready-made strategy: keyword

## Goal

Route on keywords or patterns in the current request, with sensible defaults — the fast start for
domain routing (§4, e.g. code requests to a code-strong model).

Requirements: **REQ-R7-1**, **REQ-R6-1** ([v1 requirements](../docs/v1-requirements.md)).

## Scope

- Rules mapping keywords or patterns to routes; ordering and tie-breaking; case handling.
- No model or API calls (R7); no dependencies beyond `langchain-core` (R8).
- The reason in the decision record names the rule that matched.

## Acceptance criteria

- [ ] Works with a one-line setup using defaults.
- [ ] No match → "can't decide" → default route (R9).
- [ ] A test asserts that no network or model calls are made.
