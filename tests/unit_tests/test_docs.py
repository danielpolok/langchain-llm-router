"""The documentation stays true offline: its code compiles, its links resolve, it covers the API.

The examples call real models, so running them is the live test's job
(`tests/integration_tests/test_docs_live.py`). What can be checked without a provider is checked
here, on every run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import langchain_llm_router
from tests.docs import (
    PAGES,
    README,
    ROOT,
    URL,
    Block,
    anchors,
    check_imports,
    links,
    python_blocks,
    relative,
)

BLOCKS = [block for page in PAGES for block in python_blocks(page)]


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.location)
def test_python_blocks_compile_and_their_imports_resolve(block: Block) -> None:
    block.compiled()
    check_imports(block.code, block.location)


@pytest.mark.parametrize("page", PAGES, ids=relative)
def test_relative_links_resolve(page: Path) -> None:
    broken = []
    for target in links(page):
        if URL.match(target):
            continue  # http(s), mailto: not checked offline
        path, _, fragment = target.partition("#")
        destination = (page.parent / path).resolve() if path else page
        missing_anchor = (
            bool(fragment) and destination.suffix == ".md" and fragment not in anchors(destination)
        )
        if not destination.exists() or missing_anchor:
            broken.append(target)
    assert not broken, f"{relative(page)} has broken links: {broken}"


def test_every_docs_page_is_linked_from_the_readme() -> None:
    linked = {
        (ROOT / target.partition("#")[0]).resolve()
        for target in links(README)
        if not URL.match(target)
    }
    unlinked = [relative(page) for page in PAGES if page != README and page not in linked]
    assert not unlinked, f"not linked from README.md: {unlinked}"


def test_every_public_name_is_documented() -> None:
    docs = "\n".join(page.read_text(encoding="utf-8") for page in PAGES)
    missing = [name for name in langchain_llm_router.__all__ if f"`{name}" not in docs]
    assert not missing, f"public names missing from the documentation: {missing}"


def test_docs_carry_no_internal_nomenclature() -> None:
    pattern = re.compile(r"\b(?:T-\d{3}|REQ-\d+|D\d{1,2}\b|PRD)")
    found = [
        f"{relative(page)}: {match.group()}"
        for page in PAGES
        for match in pattern.finditer(page.read_text(encoding="utf-8"))
    ]
    assert not found, found
