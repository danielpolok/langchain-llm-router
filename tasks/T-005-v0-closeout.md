---
id: T-005
title: v0 close-out — reread §3 against the spike
phase: v0
status: todo
principles: []
depends_on: [T-003, T-004]
prd: ["§3", "§6", "§10", "§11"]
---

# T-005 · v0 close-out — reread §3 against the spike

## Goal

Meet the v0 exit criteria (§6) and hand v1 a principles section that has survived contact with
code.

## Scope

- Write up the spike findings (T-002–T-004) in one place (e.g. `docs/spike-findings.md`): what
  worked, caveats, surprises.
- Reread §3 line by line. Amend any principle the spike disproved; record each change and its
  reason in §11.
- Any question the spike raised goes into §10 and is answered before v0 closes.
- Update the PRD header (phase, status, date).

## Acceptance criteria

- [ ] §6 criterion 1: C3 and R3 verdicts are recorded (T-003, T-004).
- [ ] §6 criterion 2: §3 stands as written, or has been amended with reasons in §11.
- [ ] §6 criterion 3: §10 has no open questions.
