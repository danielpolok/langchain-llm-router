---
id: T-140
title: Cost/quality benchmark
phase: v1
status: todo
principles: [R3, R7]
depends_on: [T-120, T-130, T-131]
prd: ["§7", "§8", "§9"]
---

# T-140 · Cost/quality benchmark

## Goal

Demonstrate the success metric (§7): **≥ 30% lower inference cost at ≥ 95% of always-frontier
quality** on a mixed workload — and settle whether heuristic strategies are enough (§8).

## Scope

- A mixed workload dataset (easy and hard, several domains, single-turn and agent tasks) with
  documented provenance.
- Baseline: always-frontier. Compare strategies: keyword, heuristic and — if built — embedding and
  classifier.
- Cost from recorded usage × price, including strategy overhead (embedding / classifier calls).
- Quality via an evaluation method fixed up front (e.g. LangSmith evals). A judge model evaluates;
  it never routes.
- Reproducible: pinned models, a versioned dataset, one command to rerun.

## Acceptance criteria

- [ ] Report with cost and quality per strategy against the baseline.
- [ ] Verdict on the §7 target.
- [ ] Decision recorded in PRD §11: are embedding / classifier strategies required or optional
      (§8)?
