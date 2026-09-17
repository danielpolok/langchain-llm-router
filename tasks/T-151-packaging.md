---
id: T-151
title: Packaging and publishing
phase: v1
status: todo
principles: [C9, R8]
depends_on: [T-120]
prd: ["§5", "§9"]
---

# T-151 · Packaging and publishing

## Goal

Publish the package so other LangChain developers can install it (§9; packaging was a v0
non-goal, §5).

Requirements: **REQ-R8-1**, **REQ-R8-2**, **REQ-R6-4**, **REQ-C9-2** ([v1 requirements](../docs/v1-requirements.md)).

## Scope

- Package name and import path.
- Supported `langchain-core` range (C9); runtime dependencies limited to `langchain-core` (R8);
  optional extras for anything else.
- Versioning policy, including the stability promise for the strategy interface (R6).
- Release automation to PyPI; changelog.

## Acceptance criteria

- [ ] Installing into a clean environment pulls in only `langchain-core` and its transitive
      dependencies.
- [ ] A tagged release publishes automatically.
