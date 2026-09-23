"""The mixed workload dataset (T-140): easy/hard, several domains, single-turn and agent tasks.

Loaded from `data/workload.json` rather than written as Python literals, so the dataset can be
inspected, diffed and versioned independently of the code that runs it (the task's "a versioned
dataset" requirement). Every item carries `provenance`: where its content comes from, since the
task asks for that explicitly and a benchmark nobody can audit proves nothing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

Difficulty = Literal["easy", "hard"]
Kind = Literal["single_turn", "agent"]

_DIFFICULTIES: tuple[Difficulty, ...] = get_args(Difficulty)
_KINDS: tuple[Kind, ...] = get_args(Kind)

DATASET_PATH = Path(__file__).parent / "data" / "workload.json"


@dataclass(frozen=True)
class WorkloadItem:
    """One request the benchmark routes and grades.

    `tool` and `expected_tool` are set only for `kind == "agent"`: the full toolkit
    (`benchmark.tools.ALL_TOOLS`) is bound for every agent item regardless, but `expected_tool`
    names the one a competent agent should call, so the runner can score tool-selection
    separately from answer quality.
    """

    id: str
    domain: str
    difficulty: Difficulty
    kind: Kind
    prompt: str
    rubric: str
    """What the judge should check for — passed into its grading prompt verbatim."""
    provenance: str
    """Where this item's content comes from (task requirement: documented provenance)."""
    expected_tool: str | None = None


def _item_from_dict(raw: dict[str, object]) -> WorkloadItem:
    return WorkloadItem(
        id=str(raw["id"]),
        domain=str(raw["domain"]),
        difficulty=str(raw["difficulty"]),  # type: ignore[arg-type]
        kind=str(raw["kind"]),  # type: ignore[arg-type]
        prompt=str(raw["prompt"]),
        rubric=str(raw["rubric"]),
        provenance=str(raw["provenance"]),
        expected_tool=str(raw["expected_tool"]) if raw.get("expected_tool") else None,
    )


def load_dataset(path: Path = DATASET_PATH) -> tuple[WorkloadItem, ...]:
    """Load and validate the workload dataset. Raises `ValueError` for anything malformed."""
    raw = json.loads(path.read_text())
    items = tuple(_item_from_dict(entry) for entry in raw)
    validate(items)
    return items


def validate(items: Sequence[WorkloadItem]) -> None:
    """Checks the "mixed workload" shape the task asks for, not just well-formed JSON.

    Raises `ValueError` naming what's missing, so a dataset edit that narrows coverage fails
    loudly instead of quietly shrinking what the benchmark can conclude.
    """
    if not items:
        msg = "the dataset is empty"
        raise ValueError(msg)

    ids = [item.id for item in items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        msg = f"duplicate item ids: {duplicates}"
        raise ValueError(msg)

    for item in items:
        if item.difficulty not in _DIFFICULTIES:
            msg = f"{item.id}: difficulty {item.difficulty!r} must be one of {_DIFFICULTIES}"
            raise ValueError(msg)
        if item.kind not in _KINDS:
            msg = f"{item.id}: kind {item.kind!r} must be one of {_KINDS}"
            raise ValueError(msg)
        if not item.prompt.strip():
            msg = f"{item.id}: prompt is blank"
            raise ValueError(msg)
        if not item.rubric.strip():
            msg = f"{item.id}: rubric is blank"
            raise ValueError(msg)
        if not item.provenance.strip():
            msg = f"{item.id}: provenance is blank"
            raise ValueError(msg)
        if item.kind == "agent" and not item.expected_tool:
            msg = f"{item.id}: kind is 'agent' but expected_tool is not set"
            raise ValueError(msg)
        if item.kind == "single_turn" and item.expected_tool:
            msg = f"{item.id}: kind is 'single_turn' but expected_tool is set"
            raise ValueError(msg)

    domains = {item.domain for item in items}
    if len(domains) < 3:
        msg = f"only {len(domains)} domain(s) present ({sorted(domains)}): need at least 3"
        raise ValueError(msg)

    for difficulty in _DIFFICULTIES:
        if not any(item.difficulty == difficulty for item in items):
            msg = f"no items with difficulty={difficulty!r}"
            raise ValueError(msg)

    for kind in _KINDS:
        if not any(item.kind == kind for item in items):
            msg = f"no items with kind={kind!r}"
            raise ValueError(msg)
