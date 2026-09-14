---
id: T-131
title: "Ready-made strategy: heuristic"
phase: v1
status: todo
principles: [R6, R7, R8]
depends_on: [T-111, T-112]
prd: ["§3.2", "§4", "§8", "§9"]
---

# T-131 · Ready-made strategy: heuristic

## Goal

Cost tiering without extra calls: score the current request's difficulty from cheap local signals
and map the score to routes (§4, "cost tiering").

## Scope

- Signals computable locally from the request (e.g. length, code, multi-part questions, non-text
  modalities); which ones is a T-101 / T-140 decision.
- Thresholds with sensible, overridable defaults; scores and signals recorded as the reason (R2).
- No model or API calls (R7).

## Acceptance criteria

- [ ] A test asserts that no network or model calls are made.
- [ ] Default thresholds are revisited once T-140 has results.

## Notes

§8 warns that heuristics alone may miss the ≥ 30% target; T-140 decides whether they are enough.
