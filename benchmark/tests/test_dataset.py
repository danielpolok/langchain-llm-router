"""The shipped dataset validates, and `validate` actually catches a narrowed workload."""

from __future__ import annotations

import pytest

from benchmark.dataset import WorkloadItem, load_dataset, validate


def _mk(
    id_: str, domain: str, difficulty: str, kind: str, *, expected_tool: str | None = None
) -> WorkloadItem:
    return WorkloadItem(
        id=id_,
        domain=domain,
        difficulty=difficulty,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        prompt="p",
        rubric="r",
        provenance="author-written",
        expected_tool=expected_tool,
    )


ITEM = _mk("x", "d", "easy", "single_turn")


def test_the_shipped_dataset_loads_and_validates() -> None:
    items = load_dataset()
    assert len(items) >= 1
    domains = {item.domain for item in items}
    assert len(domains) >= 3
    assert {item.difficulty for item in items} == {"easy", "hard"}
    assert {item.kind for item in items} == {"single_turn", "agent"}


def test_agent_items_all_name_an_expected_tool() -> None:
    for item in load_dataset():
        if item.kind == "agent":
            assert item.expected_tool


def test_every_item_has_provenance() -> None:
    for item in load_dataset():
        assert item.provenance.strip()


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate([])


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate([ITEM, ITEM])


def test_too_few_domains_is_rejected() -> None:
    items = [
        _mk("1", "a", "easy", "single_turn"),
        _mk("2", "a", "hard", "agent", expected_tool="calculator"),
        _mk("3", "b", "easy", "single_turn"),
    ]
    with pytest.raises(ValueError, match="domain"):
        validate(items)


def test_missing_a_difficulty_is_rejected() -> None:
    items = [
        _mk("1", "a", "easy", "single_turn"),
        _mk("2", "b", "easy", "agent", expected_tool="calculator"),
        _mk("3", "c", "easy", "single_turn"),
    ]
    with pytest.raises(ValueError, match="difficulty"):
        validate(items)  # 3 domains, but never "hard"


def test_missing_a_kind_is_rejected() -> None:
    items = [
        _mk("1", "a", "easy", "single_turn"),
        _mk("2", "b", "hard", "single_turn"),
        _mk("3", "c", "easy", "single_turn"),
    ]
    with pytest.raises(ValueError, match="kind"):
        validate(items)  # 3 domains, both difficulties, but never "agent"


def test_agent_item_without_expected_tool_is_rejected() -> None:
    bad = _mk("y", "d", "easy", "agent")
    with pytest.raises(ValueError, match="expected_tool"):
        validate([bad])


def test_single_turn_item_with_expected_tool_is_rejected() -> None:
    bad = _mk("y", "d", "easy", "single_turn", expected_tool="calculator")
    with pytest.raises(ValueError, match="expected_tool"):
        validate([bad])
