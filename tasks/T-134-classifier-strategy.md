---
id: T-134
title: "Opt-in strategy: small-LLM classifier"
phase: v1
status: todo
principles: [R6, R7, R3, R9]
depends_on: [T-111, T-112, T-117]
prd: ["§3.2", "§4", "§9"]
---

# T-134 · Opt-in strategy: small-LLM classifier

## Goal

A small chat model classifies the current request into a route. It makes an extra model call, so
it is an explicit opt-in (R7). It is also the template for plugging in a team's own classifier
(§4, "custom strategy").

Requirements: **REQ-R7-2**, **REQ-R3-2** ([v1 requirements](../docs/v1-requirements.md)).

## Scope

- The application supplies the classifier chat model; route descriptions drive the prompt; output
  is constrained to route names (structured output where available).
- The classifier's cost is real: traced as its own run and attributed to the classifier model —
  never to the router or the answering route (R3, T-117).
- Unparseable output, unknown route, timeout or error → default route with a warning (R9).

## Acceptance criteria

- [ ] Offline tests with a fake classifier cover valid, invalid and failing outputs.
- [ ] The trace shows the classifier run, the decision and the answering run as separate,
      correctly costed runs.
