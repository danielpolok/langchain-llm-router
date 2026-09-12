# Tasks

Work breakdown for `RouterChatModel`, derived from [PRD.md](../PRD.md). One file per task.

## Conventions

- **Front matter** on every task: `id`, `title`, `phase`, `status`, `principles` (PRD §3 IDs),
  `depends_on`, `prd` (the sections it comes from).
- **Status:** `todo` · `in-progress` · `blocked` · `done` · `dropped`. Change it in the task's
  front matter *and* in the index below.
- **IDs are stable** and never reused. Ranges by phase: `T-0xx` v0, `T-1xx` v1.
- **Picking work:** the lowest-numbered `todo` whose dependencies are all `done`.
- **The PRD wins.** If a task's findings contradict a principle, amend the PRD and record the
  decision in §11 — don't let the code quietly diverge.
- **v1 tasks are provisional.** v0 deliberately says nothing about implementation; T-101 turns the
  principles into detailed requirements and may split, merge or re-scope everything after it.

## v0 — prove the principles (PRD §6)

| ID | Task | Principles | Depends on | Status |
| --- | --- | --- | --- | --- |
| [T-001](T-001-project-scaffold.md) | Project scaffold | C9, R8 | — | done |
| [T-002](T-002-spike-router-skeleton.md) | Spike: minimal router skeleton | C1, C2, R1, R2, R5, R9 | T-001 | done |
| [T-003](T-003-spike-tools-structured-output.md) | Spike: tools and structured output (C3) | C3, C1, R10 | T-002 | blocked |
| [T-004](T-004-spike-cost-and-tracing.md) | Spike: cost counted once, real call traced (R3 vs C5) | R3, C5, R1 | T-002 | todo |
| [T-005](T-005-v0-closeout.md) | v0 close-out — reread §3 against the spike | — | T-003, T-004 | todo |

Exit criterion 3 (§10 open questions answered) is already met — §10 has none open. T-005 checks it
again after the spike.

## v1 — build, benchmark, ship (PRD §9)

### Requirements

| ID | Task | Principles | Depends on | Status |
| --- | --- | --- | --- | --- |
| [T-101](T-101-v1-requirements.md) | v1 detailed requirements | all | T-005 | todo |

### Core router

| ID | Task | Principles | Depends on | Status |
| --- | --- | --- | --- | --- |
| [T-110](T-110-routes-and-default-route.md) | Named routes and the mandatory default route | R5, R9, R2 | T-101 | todo |
| [T-111](T-111-strategy-interface.md) | Strategy interface | R6, R4, R7 | T-101 | todo |
| [T-112](T-112-request-extraction.md) | Current-request extraction | R4, C7 | T-111 | todo |
| [T-113](T-113-calling-conventions.md) | Sync, async, streaming and batch | C2, R1 | T-110, T-111 | todo |
| [T-114](T-114-decision-record.md) | Routing decision record | R1, R2 | T-110 | todo |
| [T-115](T-115-tool-aware-routing.md) | Tool-aware routing | R10, C3, R2 | T-110, T-114 | todo |
| [T-116](T-116-forced-routes.md) | Forced routes via runtime config | C4, R11, R2 | T-110, T-114, T-115 | todo |
| [T-117](T-117-tracing-and-cost.md) | Tracing and cost attribution | C5, R3 | T-113, T-114 | todo |
| [T-118](T-118-error-semantics.md) | Error semantics — leave retries and fallbacks to LangChain | C6, R9 | T-110 | todo |
| [T-119](T-119-response-cache.md) | Response cache correctness | C10 | T-115, T-116 | todo |
| [T-120](T-120-compatibility-suite.md) | LangChain compatibility suite | C1, C8, C9 | T-113, T-115–T-119 | todo |

### Strategies

| ID | Task | Principles | Depends on | Status |
| --- | --- | --- | --- | --- |
| [T-130](T-130-keyword-strategy.md) | Ready-made strategy: keyword | R6, R7, R8 | T-111, T-112 | todo |
| [T-131](T-131-heuristic-strategy.md) | Ready-made strategy: heuristic | R6, R7, R8 | T-111, T-112 | todo |
| [T-132](T-132-configurable-strategy.md) | Configurable strategy component | R6, R7 | T-111, T-112 | todo |
| [T-133](T-133-embedding-strategy.md) | Opt-in strategy: embedding similarity | R6, R7, R8, R3 | T-111, T-112, T-117 | todo |
| [T-134](T-134-classifier-strategy.md) | Opt-in strategy: small-LLM classifier | R6, R7, R3, R9 | T-111, T-112, T-117 | todo |

### Benchmark

| ID | Task | Principles | Depends on | Status |
| --- | --- | --- | --- | --- |
| [T-140](T-140-benchmark.md) | Cost/quality benchmark | R3, R7 | T-120, T-130, T-131 | todo |
| [T-141](T-141-prompt-cache-impact.md) | Measure provider prompt-caching loss from route switching | R3 | T-140 | todo |

### Docs, packaging, adoption

| ID | Task | Principles | Depends on | Status |
| --- | --- | --- | --- | --- |
| [T-150](T-150-documentation.md) | Documentation | C8, R6 | T-120, T-130–T-132 | todo |
| [T-151](T-151-packaging.md) | Packaging and publishing | C9, R8 | T-120 | todo |
| [T-160](T-160-first-adoption.md) | First real-application adoption | — | T-120 | todo |

## Backlog — not broken down yet

| Phase | Item | Source |
| --- | --- | --- |
| v1.x | Agent-middleware form of the router | §9 |
| v1.x | RouteLLM and other trained routers as strategies | §9 |
| v1.x | LangGraph helpers | §9 |
| v2 | Learning from traffic (a non-goal until then, §5) | §9 |
| v2 | Cascade / escalation mode | §9 |
| later | Keep a conversation on one route to preserve provider prompt caching — pending T-141 | §8 |
