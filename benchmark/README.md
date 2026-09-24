# Cost/quality benchmark

Does routing actually save money without costing quality — and are the strategies that make their
own model or embedding calls worth their overhead, or do the free ones already get there?

The target: **at least 30% lower inference cost at at least 95% of always-frontier quality** on a
mixed workload. This harness runs five arms over one dataset against real models, prices every call
from real usage, has an LLM judge grade every answer, and reports each arm against an
always-frontier baseline. Rerun it any time — after changing the dataset, the models, or the
embedding threshold — with:

```bash
uv run python -m benchmark.run
```

## Results

One real run, 2026-09-23. Raw per-item data is in
[`results/20260923T085017Z.json`](results/20260923T085017Z.json) and the rendered report in
[`results/20260923T085017Z.md`](results/20260923T085017Z.md).

- **Models:** `gemini-3.8-flash` ("frontier"), `gemini-3.5-flash-lite` ("small"), and
  `gemini-3.1-pro-preview` as the judge — never a route.
- **Cost of the run:** $0.3568 for 5 arms × 32 items × (answer + judge). Zero errors.

| Arm | Cost saved vs baseline | Quality vs baseline | Meets target |
| --- | --- | --- | --- |
| `keyword` | 41.8% | 106.6% | **yes** |
| `heuristic` | 50.1% | 110.4% | **yes** |
| `embedding` | 14.7% | 97.2% | no (cost) |
| `classifier` | 19.0% | 108.5% | no (cost) |

**The free strategies clear the target; the opt-in ones don't, on this run.** `keyword` and
`heuristic` make no extra call at all and beat both opt-in strategies on net cost — even though
`classifier` routes far more *accurately*:

| Arm | Routing accuracy vs the dataset's own easy/hard label | Cost saved |
| --- | --- | --- |
| `classifier` | 31/32 correct | 19.0% |
| `heuristic` | 20/32 correct (all 16 easy right; 12 of 16 hard under-escalated) | 50.1% |

The classifier almost perfectly recovers the intended difficulty split and still saves *less*,
because its own call is a real, billed request (here the same model as the "small" route) and that
overhead eats into the saving faster than the extra accuracy earns it back. The embedding strategy
shows the same shape for a different reason: its threshold is an uncalibrated placeholder (see
below), and it sends two whole domains (`support`, `research`) to the frontier route because their
calibration examples read closer to the "hard" set than the "easy" one. Its 14.7% is mostly a
calibration artefact, not evidence it can't do better.

**Conclusion:** the embedding and classifier strategies are useful but **optional** — reach for
them once the free strategies measurably fall short on *your* traffic, not by default.

**Not an argument for retuning the heuristic's defaults.** It sent 12 of 16 "hard" items to the
small model and still scored *higher* mean quality than the always-frontier baseline (7.31 vs
6.62), so tightening its thresholds would only add cost on this workload. The shipped defaults are
unchanged.

### Surprises

1. **The frontier model reasons even on trivial prompts, and isn't obviously better for it.** A
   direct call with "What is 17 + 26?" cost 122 output tokens on `gemini-3.8-flash` (111 of them
   reasoning) against 10 on `gemini-3.5-flash-lite`, and the heuristic arm's mean quality beat the
   baseline's while routing most traffic to the cheaper model. Either the rubric doesn't reward
   the extra reasoning, or the small model is simply strong for a workload this size.
2. **Routing accuracy and net cost saving are different things.** The classifier is the most
   accurate router and second-worst on cost, purely from paying for its own decision.
3. **One global embedding threshold doesn't travel across domains.** The same calibration
   examples separated `math`/`coder` requests cleanly but not `support`/`research` ones.

### Limitations

- **One run, no repeats.** Routes weren't called at `temperature=0`, so quality figures this close
  (97–110% of baseline) could shift on a rerun. The judge was pinned at `temperature=0`.
- **32 items show a shape, not a tight bound.** Treat the exact percentages as directional.
- **`gemini-3.5-flash-lite` may be an unusually strong "small" model** for this difficulty ceiling,
  which would flatter every strategy at once. A harder dataset, or a bigger capability gap between
  the routes, is what would stress-test this conclusion rather than confirm it.
- **The embedding threshold and example set are placeholders**, so this run measures *that
  configuration*, not what the strategy could do calibrated.
- **Provider prompt-caching loss from switching routes is not measured here.**

## What it does

Five arms, over the same 32-item dataset (`data/workload.json`):

| Arm | What decides |
| --- | --- |
| `baseline-always-frontier` | nothing — every request goes to `gemini-3.8-flash` |
| `keyword` | `KeywordStrategy`: a short list of "this needs the frontier" keywords |
| `heuristic` | `HeuristicStrategy` on its shipped defaults (`DEFAULT_WEIGHTS`/`DEFAULT_THRESHOLD`) |
| `embedding` | `EmbeddingStrategy`, Gemini embeddings, an **uncalibrated** threshold (see below) |
| `classifier` | `ClassifierStrategy`, classifying on the small route's own model |

Two candidate routes throughout, both Gemini so the whole benchmark runs from one API key with no
local-server dependency: `"small"` (`gemini-3.5-flash-lite`) and `"frontier"` (`gemini-3.8-flash`)
— override with `LLM_ROUTER_BENCHMARK_SMALL_MODEL` / `LLM_ROUTER_BENCHMARK_FRONTIER_MODEL`. The
repository's other usual small/frontier pair (`ollama:qwen3:8b` / `gemini-3-flash-preview`) is what
`test_live_ollama_smoke.py` uses for its free, budget-independent sanity check — see "Testing this
harness itself". `arms.py` builds all five arms.

For each (arm, item) pair, `runner.run_item`:

1. Invokes the router — a two-step tool round trip for `kind == "agent"` items (bind the full
   toolkit in `tools.py`, execute at most the first tool call, invoke again with its result).
2. Captures usage with `langchain_core.callbacks.get_usage_metadata_callback` — which the router's
   own tests prove is exactly the routes' own usage, including a classifier strategy's nested call
   (it inherits the router's config, so the same callback sees it without special-casing).
3. Reads the routing decision straight off the response (`routing_decision`) — which route ran,
   why, and whether the strategy decided or the default route fell back.
4. Prices it (`costing.py`, `pricing.py`) — real usage × pinned $/token, plus an *estimate* for
   `EmbeddingStrategy`'s per-request query: `Embeddings` reports no usage at all, so there is
   nothing to measure, and the strategy's own chars-per-token method stands in.
5. Grades the final answer against the item's `rubric` with an LLM judge (`judge.py`) — a model
   distinct from both routes, so grading isn't a route grading its own answer.

`report.py` aggregates every arm's `ItemResult`s into cost-saved-vs-baseline and
quality-vs-baseline percentages and a verdict against the target.

## The dataset

`data/workload.json`, generated by `_build_dataset.py` (edit the `ITEMS` list there, then rerun it
— the JSON is the versioned artefact, not something to hand-edit). 32 items: 4 domains (`support`,
`coder`, `research`, `math`) × 2 difficulties (`easy`, `hard`) × 2 kinds (`single_turn`, `agent`),
2 items per cell. Every item carries `provenance`: all of it is author-written for this benchmark
(no verbatim reuse of any copyrighted problem set), patterned after common question styles where
noted. `dataset.validate()` checks the shape (enough domains, both difficulties, both kinds, every
agent item names a real tool) every time it loads.

## The judge

`DEFAULT_JUDGE_MODEL` (`judge.py`) is `gemini-3.1-pro-preview` — a separate, stronger Gemini model
than either candidate route, so it never grades its own answer under the "frontier" name. It is
never added to any arm's `routes=`; nothing in `runner.py` ever routes to it. Override with
`LLM_ROUTER_BENCHMARK_JUDGE_MODEL` if you'd rather point it at a different provider (e.g. an
Anthropic model, for a judge with no shared vendor with either route at all — this needs
`langchain-anthropic` and its own API key, neither of which this repo installs by default).

## What's uncalibrated, on purpose

- **`EmbeddingStrategy`'s threshold** (`EMBEDDING_THRESHOLD` in `arms.py`, default `0.5`) —
  `embedding.py` ships no default at all, because a number that's silently wrong is worse than one
  the caller has to supply, and a real threshold needs real similarity scores from a live run to
  calibrate against. Override with `LLM_ROUTER_BENCHMARK_EMBEDDING_THRESHOLD`.
- **`HeuristicStrategy`'s defaults** (`DEFAULT_WEIGHTS`, `DEFAULT_THRESHOLD`,
  `DEFAULT_LENGTH_RANGE` in `src/langchain_llm_router/strategies/heuristic.py`) are used as
  shipped (all weights equal, threshold `1.0`). If a future run shows a clear miss — say the
  heuristic arm's routing accuracy is poor on one signal — edit those constants and rerun to check
  the fix, following that module's own notes on what a benchmark result should retune.

## Running it for real

```bash
uv run python -m benchmark.run
```

Needs only `GEMINI_API_KEY` — every route, the classifier's own call, embeddings and the judge are
all Gemini. Every call is real and billed — `pricing.py`'s comments cite the source and date;
recheck it if it's old.

**Measured cost, not guessed:** a live call to each candidate model on one easy and one hard
dataset-style prompt gave $0.0005–0.010/call for `gemini-3.8-flash` and $0.00003–0.0025/call for
`gemini-3.5-flash-lite` (both models reason by default — `gemini-3.8-flash` even on a trivial
prompt — which is *why* the per-call range is so wide); the judge (`gemini-3.1-pro-preview`)
measured ~$0.0024/call. Even the worst case — every arm routing every item to the frontier model —
puts a full run (32 items × 5 arms, answers + judging) at roughly **$1.60–1.70**.

Useful flags:

- `--limit N` — cap the dataset to the first N items, for a cheap smoke test before the full run.
- `--arms baseline-always-frontier heuristic` — run a subset of arms.
- `--pause SECONDS` (default `1.0`) — courtesy delay between items, gentler on a rate-limited key.
- `--report-only path/to/raw.json` — re-render the report beside a saved run's raw results without
  calling anything again.

Each run writes its raw, per-item results to `results/<timestamp>.json` and the rendered report to
`results/<timestamp>.md` beside it — so a run's output is versioned and reproducible, not just its
input.

## Testing this harness itself

`benchmark/tests/` is entirely offline (fake routes, a fake judge, `DeterministicFakeEmbedding`)
except `test_live_ollama_smoke.py`, which makes real calls to a local Ollama server via
`arms.ollama_smoke_model()` — not the benchmark's own `small_model()`/`frontier_model()`
(`@pytest.mark.requires_ollama`, skips without one) — and deliberately never touches Gemini. Run
with the rest of the suite (`uv run pytest`) or on their own:

```bash
uv run pytest benchmark/tests
```
