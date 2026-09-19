---
id: T-101
title: v1 detailed requirements
phase: v1
status: done
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

- [x] Every C and R principle maps to at least one requirement with a testable acceptance check.
      61 requirements in [docs/v1-requirements.md](../docs/v1-requirements.md); the traceability
      table has no empty row.
- [x] Spike caveats from T-005 are reflected — the "Spike caveats answered" table maps all seven
      "Carried into v1" rows and the four C3 caveats to a requirement.
- [x] `tasks/` is updated to match the requirements: T-110–T-120 re-scoped and pointed at the
      requirements they own, T-130–T-151 given requirement pointers, T-121 added.

## Outcome

`docs/v1-requirements.md` — the public API (constructor, strategy interface, decision record,
warning and error types), 61 `REQ-` requirements each with a check phrased as a test and one
owning task, and eight settled rules **D1–D8**, recorded in PRD §11:

| # | What it settles |
| --- | --- |
| D1 | Which tool-capable route a diverted request goes to |
| D2 | A forced route skips the strategy |
| D3 | Where the decision record lives when structured output leaves no room for it |
| D4 | The route owns response caching; `ChatRouter(cache=…)` is rejected |
| D5 | How tool capability is detected |
| D6 | The strategy interface |
| D7 | The `route` configurable key |
| D8 | The decision record schema, on exactly one streamed chunk |

**One new task: [T-121](T-121-capability-reporting.md)** — the router must report a `profile`
(the intersection of its routes'), because `create_agent` picks a structured-output strategy from
it. T-115 reads routes' profiles and T-120 tests placement; nobody owned the router reporting its
own.

D4 supersedes the mechanism in an earlier §11 row ("keyed on the router"); that row was amended
rather than left to contradict.

## Amended 2026-09-19 — D9

Found after close: a strategy that calls a model had no parent run to nest under, because the
spike's design decided *before* opening the router's run — and below Python 3.11 an async call
doesn't propagate callbacks unless handed its config. **D9** settles it: the router opens its run
first, the strategy runs in a child run and receives that run's config as
`RoutingRequest.config`, and the decision moves from the router run's start metadata to the
strategy run's output, the router run's output and the route run's metadata. Two requirements
added (REQ-C5-5, REQ-R6-5; 63 in total), REQ-C5-1, REQ-R2-2 and REQ-R3-2 reworded, and T-111,
T-114, T-117, T-120, T-133, T-134 and T-140 updated. REQ-R3-2 was narrowed on the way:
LangChain's `Embeddings` emits no callbacks and reports no usage, so embedding cost can only be
estimated, not measured.
