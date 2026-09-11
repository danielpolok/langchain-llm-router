---
id: T-141
title: Measure provider prompt-caching loss from route switching
phase: v1
status: todo
principles: [R3]
depends_on: [T-140]
prd: ["§8", "§11"]
---

# T-141 · Measure provider prompt-caching loss from route switching

## Goal

Quantify how much of the saving is lost when consecutive turns go to different models and forfeit
the provider's cached prompt prefix (§8).

## Scope

- Multi-turn and agent-loop workloads from T-140; record `cache_read` / `cache_creation` input
  token details.
- Compare cost with routing against an oracle that keeps each conversation on one route.

## Acceptance criteria

- [ ] Report quantifying cache loss per strategy.
- [ ] Decision on whether "keep a conversation on one route" moves from the backlog into a phase.
