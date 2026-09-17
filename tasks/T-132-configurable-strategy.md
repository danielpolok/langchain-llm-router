---
id: T-132
title: Configurable strategy component
phase: v1
status: todo
principles: [R6, R7]
depends_on: [T-111, T-112]
prd: ["§3.2", "§4", "§11"]
---

# T-132 · Configurable strategy component

## Goal

The middle level of R6: define a strategy from scratch through configuration — rules, conditions
and priorities — without writing a strategy class.

Requirements: **REQ-R6-3** ([v1 requirements](../docs/v1-requirements.md)).

## Scope

- A declarative way to combine conditions (keywords, heuristic signals, modality, custom
  predicates) into route decisions.
- Decide whether the ready-made strategies (T-130, T-131) are presets of this component.
- Configuration errors surface at construction, not per request.

## Acceptance criteria

- [ ] The domain-routing and cost-tiering use cases (§4) can each be expressed as configuration
      alone.
- [ ] Invalid configuration (unknown route, conflicting rules) fails at construction.
