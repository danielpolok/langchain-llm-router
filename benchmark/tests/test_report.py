"""Report aggregation: cost-saved and quality-vs-baseline percentages, and the target verdict."""

from __future__ import annotations

import pytest

from benchmark.costing import CostBreakdown
from benchmark.judge import JudgeVerdict
from benchmark.report import BASELINE_ARM, render_markdown, summarize
from benchmark.runner import ItemResult
from langchain_llm_router import RoutingDecision


def _result(
    item_id: str,
    arm: str,
    *,
    route: str,
    cost: float,
    score: int,
    error: str | None = None,
) -> ItemResult:
    return ItemResult(
        item_id=item_id,
        arm=arm,
        decision=RoutingDecision(route=route, reason="test") if error is None else None,
        answer="an answer",
        tool_called=None,
        tool_correct=None,
        usage_by_model={},
        cost=CostBreakdown(chat_usd=cost, embedding_usd=0.0),
        judge=JudgeVerdict(score=score, rationale="test") if error is None else None,
        error=error,
    )


def test_a_strategy_costing_half_as_much_at_full_quality_meets_the_target() -> None:
    by_arm = {
        BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)],
        "heuristic": [_result("1", "heuristic", route="small", cost=0.5, score=10)],
    }
    summaries = summarize(by_arm)
    heuristic = summaries["heuristic"]
    assert heuristic.cost_saved_pct == 50.0
    assert heuristic.quality_pct_of_baseline == 100.0
    assert heuristic.meets_target is True


def test_a_strategy_saving_little_cost_misses_the_target() -> None:
    by_arm = {
        BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)],
        "keyword": [_result("1", "keyword", route="frontier", cost=0.95, score=10)],
    }
    summaries = summarize(by_arm)
    assert summaries["keyword"].cost_saved_pct == pytest.approx(5.0)
    assert summaries["keyword"].meets_target is False


def test_a_strategy_that_tanks_quality_misses_the_target_even_if_cheap() -> None:
    by_arm = {
        BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)],
        "classifier": [_result("1", "classifier", route="small", cost=0.1, score=5)],
    }
    summaries = summarize(by_arm)
    classifier = summaries["classifier"]
    assert classifier.cost_saved_pct == 90.0
    assert classifier.quality_pct_of_baseline == 50.0
    assert classifier.meets_target is False


def test_the_baseline_itself_has_no_relative_figures() -> None:
    by_arm = {BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)]}
    summaries = summarize(by_arm)
    assert summaries[BASELINE_ARM].cost_saved_pct is None
    assert summaries[BASELINE_ARM].meets_target is None


def test_errors_are_excluded_from_quality_but_counted() -> None:
    by_arm = {
        "heuristic": [
            _result("1", "heuristic", route="small", cost=0.1, score=9),
            _result("2", "heuristic", route="small", cost=0.0, score=0, error="boom"),
        ]
    }
    summary = summarize(by_arm)["heuristic"]
    assert summary.n_errors == 1
    assert summary.mean_quality == 9.0  # the errored item is excluded, not scored as 0


def test_render_markdown_includes_every_arm_and_a_verdict() -> None:
    by_arm = {
        BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)],
        "heuristic": [_result("1", "heuristic", route="small", cost=0.5, score=10)],
    }
    report = render_markdown(summarize(by_arm))
    assert BASELINE_ARM in report
    assert "heuristic" in report
    assert "## Verdict" in report
    assert "Target met by: heuristic" in report
