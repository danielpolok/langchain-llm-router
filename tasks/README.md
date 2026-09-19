# Tasks

Work breakdown for `ChatRouter`. **Open work lives in [GitHub Issues](https://github.com/danielpolok/langchain-llm-router/issues?q=is%3Aissue+milestone%3Av1);
this directory keeps the rules and the archive of closed tasks.**

The split: what the router *is* and *must do* stays in the repo, where it changes through reviewed
PRs — [PRD.md](../PRD.md) (principles and settled decisions) and
[docs/v1-requirements.md](../docs/v1-requirements.md) (the numbered `REQ-` requirements). *Work*
— who's on what, what's blocked, what's done — lives in the tracker, where it has one status and
PRs close it.

## Conventions

- **One issue per task**, titled `T-NNN · Title`, in milestone `v1`. **IDs are stable and never
  reused** — the requirements' *Task* column and commit messages refer to them. Ranges by phase:
  `T-0xx` v0, `T-1xx` v1.
- **Body:** a header line (principles, PRD sections, dependencies), then *Goal*, a
  *Requirements* line naming the `REQ-` IDs the task owns, *Scope* and *Acceptance criteria*.
- **Labels:** `principle:<ID>` for each PRD §3 principle it serves; one `area:` label —
  `core`, `strategies`, `benchmark` or `ship`.
- **Dependencies** are GitHub's native *blocked by* links, not text. An issue with an open blocker
  shows as *Blocked*.
- **Status is the issue's own state, and nothing else:**

  | Status | On GitHub |
  | --- | --- |
  | todo | open, unassigned |
  | in progress | open, assigned (a PR says `Closes #N`) |
  | blocked | open, with an open *blocked by* link |
  | done | closed as *completed* — usually by its PR merging |
  | dropped | closed as *not planned*, with a comment saying why |

- **Picking work:** the lowest-numbered open, unassigned, unblocked `v1` issue.

  ```bash
  gh issue list --milestone v1 --search "-is:blocked no:assignee" --json number,title
  ```

- **The PRD wins.** If a task's findings contradict a principle, amend the PRD and record the
  decision in §11 — don't let the code quietly diverge.
- **Requirements change in the repo first.** Don't re-decide D1–D9 (PRD §11) in an issue thread.
  When a requirement changes, the PR changes `docs/v1-requirements.md`; the affected issues are
  edited to match once it merges — the one step the old in-repo task files did atomically.
- **Adding a task:** take the next free ID in its phase range, open the issue in the format above,
  and add its row to the lookup below.

## v1 — task ID → issue

A lookup only — status is on the issue.

| ID | Task | Issue |
| --- | --- | --- |
| **Core router** | | |
| T-110 | Named routes, the default route and the public API | [#5](https://github.com/danielpolok/langchain-llm-router/issues/5) |
| T-111 | Strategy interface | [#6](https://github.com/danielpolok/langchain-llm-router/issues/6) |
| T-112 | Current-request extraction | [#7](https://github.com/danielpolok/langchain-llm-router/issues/7) |
| T-113 | Sync, async, streaming and batch | [#8](https://github.com/danielpolok/langchain-llm-router/issues/8) |
| T-114 | Routing decision record | [#9](https://github.com/danielpolok/langchain-llm-router/issues/9) |
| T-115 | Tool-aware routing | [#10](https://github.com/danielpolok/langchain-llm-router/issues/10) |
| T-116 | Forced routes via runtime config | [#11](https://github.com/danielpolok/langchain-llm-router/issues/11) |
| T-117 | Tracing and cost attribution | [#12](https://github.com/danielpolok/langchain-llm-router/issues/12) |
| T-118 | Error semantics — leave retries and fallbacks to LangChain | [#13](https://github.com/danielpolok/langchain-llm-router/issues/13) |
| T-119 | Response cache correctness | [#14](https://github.com/danielpolok/langchain-llm-router/issues/14) |
| T-120 | LangChain compatibility suite | [#15](https://github.com/danielpolok/langchain-llm-router/issues/15) |
| T-121 | Capability reporting — the router's own profile | [#16](https://github.com/danielpolok/langchain-llm-router/issues/16) |
| **Strategies** | | |
| T-130 | Ready-made strategy: keyword | [#17](https://github.com/danielpolok/langchain-llm-router/issues/17) |
| T-131 | Ready-made strategy: heuristic | [#18](https://github.com/danielpolok/langchain-llm-router/issues/18) |
| T-132 | Configurable strategy component | [#19](https://github.com/danielpolok/langchain-llm-router/issues/19) |
| T-133 | Opt-in strategy: embedding similarity | [#20](https://github.com/danielpolok/langchain-llm-router/issues/20) |
| T-134 | Opt-in strategy: small-LLM classifier | [#21](https://github.com/danielpolok/langchain-llm-router/issues/21) |
| **Benchmark** | | |
| T-140 | Cost/quality benchmark | [#22](https://github.com/danielpolok/langchain-llm-router/issues/22) |
| T-141 | Measure provider prompt-caching loss from route switching | [#23](https://github.com/danielpolok/langchain-llm-router/issues/23) |
| **Docs, packaging, adoption** | | |
| T-150 | Documentation | [#24](https://github.com/danielpolok/langchain-llm-router/issues/24) |
| T-151 | Packaging and publishing | [#25](https://github.com/danielpolok/langchain-llm-router/issues/25) |
| T-160 | First real-application adoption | [#26](https://github.com/danielpolok/langchain-llm-router/issues/26) |

## Archive — closed tasks

Closed tasks from before the move stay here as files: T-003 and T-004 hold the spike verdicts that
[docs/spike-findings.md](../docs/spike-findings.md) cites.

### v0 — prove the principles (PRD §6)

| ID | Task | Principles | Depends on | Status |
| --- | --- | --- | --- | --- |
| [T-001](T-001-project-scaffold.md) | Project scaffold | C9, R8 | — | done |
| [T-002](T-002-spike-router-skeleton.md) | Spike: minimal router skeleton | C1, C2, R1, R2, R5, R9 | T-001 | done |
| [T-003](T-003-spike-tools-structured-output.md) | Spike: tools and structured output (C3) | C3, C1, R10 | T-002 | done |
| [T-004](T-004-spike-cost-and-tracing.md) | Spike: cost counted once, real call traced (R3 vs C5) | R3, C5, R1 | T-002 | done |
| [T-005](T-005-v0-closeout.md) | v0 close-out — reread §3 against the spike | — | T-003, T-004 | done |

Exit criterion 3 (§10 open questions answered) is already met — §10 has none open. T-005 checks it
again after the spike.

T-003 and T-004 are `done`; their one credential-gated criterion each has since run — a
real-provider pass (Gemini, cloud; Ollama, local) and a live LangSmith trace. **v0 is closed.**

### v1 — requirements

| ID | Task | Principles | Depends on | Status |
| --- | --- | --- | --- | --- |
| [T-101](T-101-v1-requirements.md) | v1 detailed requirements | all | T-005 | done |

## Backlog — not broken down yet

| Phase | Item | Source |
| --- | --- | --- |
| v1.x | Agent-middleware form of the router | §9 |
| v1.x | RouteLLM and other trained routers as strategies | §9 |
| v1.x | LangGraph helpers | §9 |
| v2 | Learning from traffic (a non-goal until then, §5) | §9 |
| v2 | Cascade / escalation mode | §9 |
| later | Keep a conversation on one route to preserve provider prompt caching — pending T-141 | §8 |
