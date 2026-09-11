---
id: T-101
title: v1 detailed requirements
phase: v1
status: todo
principles: [all]
depends_on: [T-005]
prd: ["§3", "§9"]
---

# T-101 · v1 detailed requirements

## Goal

Turn the §3 principles into detailed, testable requirements — the step v0 explicitly deferred.

## Scope

- Public API: constructor, route and default-route declaration, strategy attachment, settings
  (e.g. R11 forced-route behaviour).
- The strategy interface (R6) — small and stable — and its versioning promise.
- Decision record schema (R2) and where it lives (message metadata, trace metadata).
- Warning and error types for R9, R10 and R11.
- Rules the principles leave open, e.g. *which* tool-capable route a diverted request goes to
  (R10), and whether a forced route skips the strategy (R11).
- Re-scope T-110 onward to match; add, split or drop tasks as needed.

## Acceptance criteria

- [ ] Every C and R principle maps to at least one requirement with a testable acceptance check.
- [ ] Spike caveats from T-005 are reflected.
- [ ] `tasks/` is updated to match the requirements.
