"""Aggregate `ItemResult`s into the report the task asks for: cost and quality per strategy
against the always-frontier baseline, and a verdict on the PRD §7 target (>=30% lower cost at
>=95% of baseline quality)."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.runner import ItemResult

BASELINE_ARM = "baseline-always-frontier"
COST_TARGET_PCT = 30.0
QUALITY_TARGET_PCT = 95.0


@dataclass(frozen=True)
class ArmSummary:
    """One arm's numbers, and its two derived figures against the baseline."""

    name: str
    n_items: int
    n_errors: int
    total_cost_usd: float
    mean_quality: float | None
    """Mean judge score out of 10, over items that graded successfully. `None` if none did."""
    frontier_calls: int
    small_calls: int
    agent_items: int
    tool_correct: int
    """How many agent items called the tool `expected_tool` named."""
    cost_saved_pct: float | None = None
    """vs `BASELINE_ARM`'s cost. `None` for the baseline itself, or if the baseline is missing."""
    quality_pct_of_baseline: float | None = None
    """vs `BASELINE_ARM`'s mean quality. Same `None` cases."""

    @property
    def meets_target(self) -> bool | None:
        if self.cost_saved_pct is None or self.quality_pct_of_baseline is None:
            return None
        return self.cost_saved_pct >= COST_TARGET_PCT and self.quality_pct_of_baseline >= (
            QUALITY_TARGET_PCT
        )

    @property
    def tool_accuracy(self) -> float | None:
        return self.tool_correct / self.agent_items if self.agent_items else None


def _summarize_one(name: str, results: list[ItemResult]) -> ArmSummary:
    scores = [r.judge.score for r in results if r.judge is not None]
    agent = [r for r in results if r.tool_correct is not None]
    return ArmSummary(
        name=name,
        n_items=len(results),
        n_errors=sum(1 for r in results if r.error is not None),
        total_cost_usd=sum(r.cost.total_usd for r in results),
        mean_quality=(sum(scores) / len(scores)) if scores else None,
        frontier_calls=sum(1 for r in results if r.decision and r.decision.route == "frontier"),
        small_calls=sum(1 for r in results if r.decision and r.decision.route == "small"),
        agent_items=len(agent),
        tool_correct=len([r for r in agent if r.tool_correct]),
    )


def summarize(by_arm: dict[str, list[ItemResult]]) -> dict[str, ArmSummary]:
    """One `ArmSummary` per arm, with `cost_saved_pct` / `quality_pct_of_baseline` filled in
    against `BASELINE_ARM` wherever it's present in `by_arm`."""
    summaries = {name: _summarize_one(name, results) for name, results in by_arm.items()}
    baseline = summaries.get(BASELINE_ARM)
    if baseline is None or baseline.total_cost_usd == 0 or baseline.mean_quality is None:
        return summaries

    filled = {}
    for name, summary in summaries.items():
        if name == BASELINE_ARM or summary.mean_quality is None:
            filled[name] = summary
            continue
        cost_saved = (1 - summary.total_cost_usd / baseline.total_cost_usd) * 100
        quality_pct = (summary.mean_quality / baseline.mean_quality) * 100
        filled[name] = ArmSummary(
            **{
                **summary.__dict__,
                "cost_saved_pct": cost_saved,
                "quality_pct_of_baseline": quality_pct,
            }
        )
    filled[BASELINE_ARM] = baseline
    return filled


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "—"


def _fmt_target(summary: ArmSummary) -> str:
    if summary.name == BASELINE_ARM:
        return "—"
    met = summary.meets_target
    if met is None:
        return "n/a"
    return "**met**" if met else "not met"


def render_markdown(
    summaries: dict[str, ArmSummary], *, title: str = "T-140 Benchmark Results"
) -> str:
    """The markdown report: one table, a per-arm errors/tool-accuracy note, and a verdict."""
    order = [BASELINE_ARM, "keyword", "heuristic", "embedding", "classifier"]
    rows = [summaries[name] for name in order if name in summaries]

    lines = [
        f"# {title}",
        "",
        f"Target (PRD §7): >= {COST_TARGET_PCT:.0f}% lower cost at "
        f">= {QUALITY_TARGET_PCT:.0f}% of always-frontier quality.",
        "",
        "| Arm | Items | Errors | Cost ($) | Cost saved | Mean quality (/10) "
        "| Quality vs baseline | Target |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in rows:
        quality = f"{summary.mean_quality:.2f}" if summary.mean_quality is not None else "—"
        lines.append(
            f"| {summary.name} | {summary.n_items} | {summary.n_errors} | "
            f"{summary.total_cost_usd:.4f} | {_fmt_pct(summary.cost_saved_pct)} | {quality} | "
            f"{_fmt_pct(summary.quality_pct_of_baseline)} | {_fmt_target(summary)} |"
        )

    lines += ["", "## Routing and tool-use detail", ""]
    lines += ["| Arm | to small | to frontier | agent tool accuracy |", "| --- | --- | --- | --- |"]
    for summary in rows:
        tool_acc = f"{summary.tool_accuracy:.0%}" if summary.tool_accuracy is not None else "—"
        lines.append(
            f"| {summary.name} | {summary.small_calls} | {summary.frontier_calls} | {tool_acc} |"
        )

    met = [s.name for s in rows if s.meets_target]
    lines += ["", "## Verdict", ""]
    if any(s.name != BASELINE_ARM and s.meets_target is None for s in rows):
        lines.append(
            "Incomplete: at least one strategy has no baseline-relative figures to judge "
            "against (see the table above)."
        )
    elif met:
        lines.append(f"Target met by: {', '.join(met)}.")
    else:
        lines.append("Target not met by any strategy on this run.")

    return "\n".join(lines) + "\n"
