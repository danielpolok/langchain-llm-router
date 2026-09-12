# v0 spike findings

What the spike (T-002 to T-004) showed about the two riskiest principles, written down so v1
starts from evidence rather than from the spike's code.

**Built against** `langchain-core` 1.6.3, `langchain` 1.4.0, `langgraph` 1.2.11.
**Where it lives:** `spike/` — `router.py` (the naive design), `designs.py` (the candidates),
`fakes.py`, `tracing.py`, and 52 offline tests under `spike/tests/`. None of it is v1 code.

## Verdicts

| Principle | Verdict |
| --- | --- |
| **C3** — tools and structured output through the router | **Holds, with caveats.** Nothing here disproves C1, so the §8 fork is not triggered. |
| **R3 vs C5** — cost counted once while the real call stays traced | **Both hold**, once the router's own run stops being a model run. |

The C3 caveats are in [T-003](../tasks/T-003-spike-tools-structured-output.md); the R3/C5
scoring of three designs is in [T-004](../tasks/T-004-spike-cost-and-tracing.md).

## The one design decision the spike forced

R3 and C5 look like a conflict only while the router emits a **model** run of its own. Two model
runs for one request means the same tokens are billed twice — by `UsageMetadataCallbackHandler`,
which adds up every `on_llm_end` that carries usage and a model name, and by LangSmith, which
prices each LLM run from its own `usage_metadata`.

Neither principle has to give way. The router's own run becomes a **chain** run that delegates,
leaving the selected route's call as the only model run in the trace. Recorded in PRD §11.

## Surprises

Things that were not obvious from the documentation, each of which cost time:

1. **An LLM run cannot have children.** `CallbackManagerForLLMRun` is not a `ParentRunManager`,
   so it has no `get_child()`. A chat model that calls a chat model has to rebuild the child
   callback manager by hand. A *chain* run manager has one — which is half of why the chain-run
   design wins.
2. **Streaming hands `_stream` no run manager at all.** `BaseChatModel.stream` and `astream`
   (`chat_models.py:794`, `:927`), and the streaming branch of `_generate_with_cache`
   (`:1981`, `:2136`) — the path taken whenever anything streams the application around the
   model — call `_stream` without one. Only the v2 protocol path passes it. So a router built on
   `_stream` cannot pass the route any callbacks: the real call disappears from the trace.
3. **The double count is silent.** In the naive design the router's run carries usage but no
   `ls_model_name` of its own, so LangSmith would likely price it at zero while still counting
   the tokens twice. Nothing errors; the number is just wrong.
4. **`with_structured_output`'s default discards provider arguments.** `method=` and `strict=`
   are popped and dropped (`chat_models.py:2530`), so every route would be forced through
   function calling and no route's native JSON-schema mode would ever run.
5. **`create_agent` asks the model what it can do.** It picks `ProviderStrategy` or
   `ToolStrategy` from `model.profile` (`langchain/agents/factory.py:560`), so a router has to
   answer for capabilities it does not own. The honest answer is the intersection of its routes'
   profiles — otherwise a strategy gets chosen that some route cannot serve.
6. **`merge_dicts` concatenates repeated strings.** A decision record attached to every streamed
   chunk aggregates to `"frontierfrontier…"`. It can ride on exactly one chunk.
7. **LangChain already has this problem elsewhere.** `_attach_gateway_metadata`
   (`tracers/core.py:352`) promotes a response-time model identity over the request-time one,
   because a gateway also only learns the real model from the response. It is the supported way
   to relabel a run, if v1 ever needs it.

## Carried into v1

| What | Where |
| --- | --- |
| `generate()` / `agenerate()` still take the base path and would double count | T-117 |
| R2 has no answer for structured output: a parsed object cannot carry the decision | T-114 |
| Cache ownership — delegating means the *route's* cache applies, and the router's own `cache=` is silently ignored | T-119 |
| A route that cannot use tools fails at call time, only when selected | T-115 |
| The router's run being a chain run changes what `stream_mode="messages"` sees; confirm against a real graph | T-113, T-117 |
| Raw tool objects ride in the router's invocation params, which are traced and used as a cache key | T-117, T-119 |

## Still unverified

Both need credentials that were not available when the spike ran, and both are written and
skipping, not missing:

- **One real-provider run across two providers** — a tool call and structured output on each
  (`spike/tests/test_real_providers.py`, needs `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`).
- **A LangSmith trace** showing the decision and the real call, costing one call of the selected
  model (needs `LANGSMITH_API_KEY`). Checked offline against a stand-in that mirrors LangSmith's
  pricing inputs — `usage_metadata` plus `ls_model_name`, per run (`spike/tracing.py`).

## §3 after the spike

Reread line by line. **§3 stands as written** — no principle was disproved, so nothing is
amended. What the spike added is one decision in §11 (the router's run is a chain run) and the
caveats above, which are v1 requirements rather than changes to the principles.

Principles the spike exercised directly: C1, C2, C3, C5, C9, R1, R2, R3, R5, R8, R9, and a
preview of R10. Untouched, and still v1's to prove: C4, C6, C7, C8, C10, R4, R6, R7, R11.
