---
id: T-005
title: v0 close-out — reread §3 against the spike
phase: v0
status: done
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

- [x] §6 criterion 1: C3 and R3 verdicts are recorded (T-003, T-004).
- [x] §6 criterion 2: §3 stands as written, or has been amended with reasons in §11.
- [x] §6 criterion 3: §10 has no open questions.

## Outcome

- Findings written up in [docs/spike-findings.md](../docs/spike-findings.md): the two verdicts,
  the one design decision the spike forced, seven surprises, and six caveats carried into v1
  against the task that will answer each.
- **§3 stands as written.** Rereading it line by line disproved nothing, so no principle is
  amended. The spike exercised C1, C2, C3, C5, C9, R1, R2, R3, R5, R8, R9 and previewed R10;
  C4, C6, C7, C8, C10, R4, R6, R7 and R11 remain v1's to prove.
- §11 gained one row from the spike: the router's own run is a chain run, not a model run.
- PRD header and §6 updated with where v0 actually stands.

**Ran with T-003 and T-004 still `blocked`**, which was a deliberate departure from the
dependency rule: both verdicts were already recorded, and what remained in those tasks was one
real-provider run and one LangSmith trace, each gated on credentials not set in that environment
at the time. Both have since run — against Gemini and Ollama, and a live LangSmith project — and
T-003/T-004 are now `done`. **v0 is closed.**
