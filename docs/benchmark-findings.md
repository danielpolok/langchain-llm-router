# T-140 benchmark findings

What a real run of the cost/quality benchmark showed, against the §7 success metric and the §8
open question — written down so the PRD §11 decision starts from evidence, the way
[docs/spike-findings.md](spike-findings.md) did for the spike.

**Run:** `benchmark/results/20260923T085017Z.json` (raw, per-item) — reproduce with
`uv run python -m benchmark.run` (see `benchmark/README.md`). **Models:** `gemini-3.8-flash`
("frontier"), `gemini-3.5-flash-lite` ("small"), `gemini-3.1-pro-preview` (judge, never a
route). **Cost:** $0.3568 total for all 5 arms x 32 items x (answer + judge) — about 1.36 PLN of
a 20 PLN budget. **Dataset:** `benchmark/data/workload.json`, 32 items, unchanged from the one
`benchmark/README.md` describes.

## Verdict on the §7 target

**>= 30% lower cost at >= 95% of always-frontier quality**, on this workload, with this model
pair:

| Arm | Cost saved vs baseline | Quality vs baseline | Meets target |
| --- | --- | --- | --- |
| `keyword` | 41.8% | 106.6% | **yes** |
| `heuristic` | 50.1% | 110.4% | **yes** |
| `embedding` | 14.7% | 97.2% | no (cost) |
| `classifier` | 19.0% | 108.5% | no (cost) |

Full per-arm cost/quality/routing numbers are in `docs/benchmark-findings.md`'s own table below
(regenerated verbatim by `report.render_markdown`, not hand-edited) and in the raw JSON.

**The free-tier strategies clear the target; the opt-in ones don't, on this run.** That answers
§8 for this workload: see "The §8 decision" below.

## The §8 decision: embedding and classifier are optional, not required

`docs/v1-requirements.md` left this open for T-140 to decide. The evidence: `keyword` and
`heuristic` — the two strategies R7 requires to make **no extra call at all** — already clear
the §7 target by a comfortable margin, and beat both opt-in strategies on net cost, despite
`classifier` routing far more *accurately*:

| Arm | Routing accuracy vs the dataset's own easy/hard label | Cost saved |
| --- | --- | --- |
| `classifier` | 31/32 correct | 19.0% |
| `heuristic` | 20/32 correct (all 16 easy items right; 12 of 16 hard items under-escalated) | 50.1% |

`classifier` almost perfectly recovers the intended difficulty split — and still saves *less*
than `heuristic`, which gets a third of the hard items wrong. The reason is R7's own point made
concrete: `classifier`'s own call is a real, billed request (`gemini-3.5-flash-lite` here, the
same model as the "small" route — see `benchmark/arms.py`), and that overhead eats into the
saving faster than the extra routing accuracy earns it back. `embedding` shows the same shape for
a different reason: its threshold is a documented placeholder (`benchmark/README.md`, "What's
uncalibrated"), and it over-routes two whole domains (`support`, `research`) to frontier because
their calibration examples read semantically closer to the "hard" set than the "easy" one —
14.7% cost saved is mostly a calibration artefact, not evidence the strategy can't do better.

**Decision:** embedding and classifier strategies are useful but **optional** — a team should
reach for them only once the free strategies measurably fall short on *their own* traffic, not
by default, since their own overhead (a real call, or real calibration effort) is a cost the free
strategies don't pay. Recorded in PRD §11 as D10.

## What this does *not* argue for retuning

`heuristic.py`'s own docstring says T-140's results settle `DEFAULT_WEIGHTS`,
`DEFAULT_THRESHOLD` and `DEFAULT_LENGTH_RANGE`. This run's evidence is **not** a case for
escalating more traffic to frontier: `heuristic` sent 12 of 16 "hard" items to the small model
anyway, and still scored *higher* mean quality than the always-frontier baseline (7.31 vs 6.62).
Tightening the thresholds to catch those 12 would only add cost, not quality, on this workload —
the small model (`gemini-3.5-flash-lite`) is already answering them well. **No change made to
the shipped defaults**; the honest reading of one run's evidence is "no clear miss to fix," not
"retune anyway."

## Surprises

1. **The frontier model reasons even on trivial prompts, and it isn't obviously smarter for
   it.** A direct call with "What is 17 + 26?" to `gemini-3.8-flash` cost 122 output tokens (111
   of them reasoning) against 10 for `gemini-3.5-flash-lite` (`benchmark/README.md`'s measured
   figures) — and across the whole dataset, `heuristic`'s mean quality (7.31) beat the baseline's
   (6.62) despite routing most traffic to the cheaper model. Either the judge's rubric grading
   doesn't reward verbosity/reasoning the frontier model produces by default, or
   `gemini-3.5-flash-lite` is simply a strong model for a workload this size — see the caveat
   below.
2. **Routing accuracy and net cost saving are not the same thing.** `classifier` is the most
   accurate router and the second-worst on cost, purely from paying for its own decision.
3. **A single global embedding threshold doesn't travel across domains.** The same
   calibration examples separated `math`/`coder` requests cleanly but not `support`/`research`
   ones — `EmbeddingStrategy`'s own module docstring warns a single bar may not mean the same
   thing for every provider; here it doesn't mean the same thing for every *domain* either.

## Limitations

- **One run, no repeats.** Routes weren't called at `temperature=0`, so quality numbers this
  close (97-110% of baseline) could shift on a rerun. The judge was pinned at `temperature=0`.
- **32 items is enough to see a shape, not to bound a percentage tightly.** Each arm's mean
  quality is an average over 32 LLM-judge scores; treat the exact percentages as directional.
- **`gemini-3.5-flash-lite` may be an unusually strong "small" model** for this workload's
  difficulty ceiling, which would flatter every routing strategy at once (there's less to lose
  by under-escalating). A harder dataset, or a bigger small/frontier capability gap, is the next
  thing that would stress-test this conclusion rather than confirm it.
- **Embedding's threshold and example set are placeholders**, not the strategy's ceiling (see
  "The §8 decision" above) — this run measures *this configuration*, not what `EmbeddingStrategy`
  could do calibrated.
- **T-141** (provider prompt-caching loss from route switching) is explicitly out of scope here.

## Full results table

<!-- Regenerated verbatim by `benchmark.report.render_markdown` on every real run; edit the
     prose above, not this table, and rerun `uv run python -m benchmark.run` to refresh it. -->

| Arm | Items | Errors | Cost ($) | Cost saved | Mean quality (/10) | Quality vs baseline | Target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-always-frontier | 32 | 0 | 0.0953 | — | 6.62 | — | — |
| keyword | 32 | 0 | 0.0554 | 41.8% | 7.06 | 106.6% | **met** |
| heuristic | 32 | 0 | 0.0476 | 50.1% | 7.31 | 110.4% | **met** |
| embedding | 32 | 0 | 0.0813 | 14.7% | 6.44 | 97.2% | not met |
| classifier | 32 | 0 | 0.0772 | 19.0% | 7.19 | 108.5% | not met |

| Arm | to small | to frontier | agent tool accuracy |
| --- | --- | --- | --- |
| baseline-always-frontier | 0 | 32 | — |
| keyword | 26 | 6 | 100% |
| heuristic | 28 | 4 | 100% |
| embedding | 12 | 20 | 100% |
| classifier | 17 | 15 | 100% |

**Verdict: target met by keyword, heuristic.**
