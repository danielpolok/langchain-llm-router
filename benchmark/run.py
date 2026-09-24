"""T-140's one command to reproduce the benchmark:

    uv run python -m benchmark.run

Needs `GEMINI_API_KEY` (the frontier route, the judge) and a local Ollama server serving
`qwen3:8b` (the small route, the classifier's own model) — both real-provider preconditions this
repo already documents in `CLAUDE.md`. Every call is real and billed; see `benchmark/README.md`
before running this against a metered key.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmark.arms import build_arms
from benchmark.costing import CostBreakdown
from benchmark.dataset import load_dataset
from benchmark.judge import JudgeVerdict, judge_model
from benchmark.report import render_markdown, summarize
from benchmark.runner import ItemResult, run_arm
from langchain_llm_router import RoutingDecision

RESULTS_DIR = Path(__file__).parent / "results"


def _to_raw(result: ItemResult) -> dict[str, object]:
    """`ItemResult` -> plain JSON-able dict, for the versioned raw-results file."""
    return {
        "item_id": result.item_id,
        "arm": result.arm,
        "decision": result.decision.as_dict() if result.decision else None,
        "answer": result.answer,
        "tool_called": result.tool_called,
        "tool_correct": result.tool_correct,
        "usage_by_model": result.usage_by_model,
        "cost": asdict(result.cost),
        "judge": result.judge.model_dump() if result.judge else None,
        "error": result.error,
    }


def _from_raw(raw: dict[str, Any]) -> ItemResult:
    """The inverse of `_to_raw`, for `--report-only`'s re-render from a saved run."""
    return ItemResult(
        item_id=raw["item_id"],
        arm=raw["arm"],
        decision=RoutingDecision.from_dict(raw["decision"]) if raw["decision"] else None,
        answer=raw["answer"],
        tool_called=raw["tool_called"],
        tool_correct=raw["tool_correct"],
        usage_by_model=raw["usage_by_model"],
        cost=CostBreakdown(**raw["cost"]),
        judge=JudgeVerdict(**raw["judge"]) if raw["judge"] else None,
        error=raw["error"],
    )


def _run(
    arm_names: list[str] | None, limit: int | None, pause: float
) -> tuple[dict[str, list[ItemResult]], Path]:
    items = list(load_dataset())
    if limit:
        items = items[:limit]
    arms = build_arms()
    if arm_names:
        arms = [arm for arm in arms if arm.name in arm_names]
        if not arms:
            msg = f"no arm matches {arm_names}"
            raise SystemExit(msg)

    judge = judge_model()
    by_arm: dict[str, list[ItemResult]] = {}
    for arm in arms:
        print(f"running {arm.name} ({len(items)} items)...", file=sys.stderr)
        results = run_arm(arm, items, judge, embed=(arm.name == "embedding"), pause_seconds=pause)
        by_arm[arm.name] = results
        errors = sum(1 for r in results if r.error)
        print(f"  {len(results) - errors}/{len(results)} ok", file=sys.stderr)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS_DIR / f"{stamp}.json"
    raw = {arm: [_to_raw(r) for r in results] for arm, results in by_arm.items()}
    raw_path.write_text(json.dumps(raw, indent=2))
    print(f"raw results: {raw_path}", file=sys.stderr)
    return by_arm, raw_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=None, help="Arm names (default: all five).")
    parser.add_argument("--limit", type=int, default=None, help="Cap the dataset to N items.")
    parser.add_argument(
        "--pause", type=float, default=1.0, help="Seconds between items (rate-limit courtesy)."
    )
    parser.add_argument(
        "--report-only",
        type=Path,
        default=None,
        help="Skip running: re-render the report beside a saved raw-results JSON.",
    )
    args = parser.parse_args(argv)

    load_dotenv()

    if args.report_only:
        raw_path = args.report_only
        raw = json.loads(raw_path.read_text())
        by_arm = {arm: [_from_raw(r) for r in results] for arm, results in raw.items()}
    else:
        by_arm, raw_path = _run(args.arms, args.limit, args.pause)

    summaries = summarize(by_arm)
    report = render_markdown(summaries)
    report_path = raw_path.with_suffix(".md")
    report_path.write_text(report)
    print(report)
    print(f"report written to {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
