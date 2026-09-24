"""The documentation stays true: its code runs, its links resolve and it covers the public API.

Every ```` ```python ```` block in `README.md` and `docs/` is executed, the blocks of one page
in order and in one shared namespace, so a page reads top to bottom like a script. A block that
can't run offline — one that needs a real provider or credentials — is fenced as
```` ```python skip ````; that is the only opt-out, and it is for those blocks only.
"""

from __future__ import annotations

import re
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import pytest

import langchain_llm_router

ROOT = Path(__file__).resolve().parents[2]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]

FENCE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
LINK = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$", re.MULTILINE)


def python_blocks(page: Path) -> list[tuple[int, str]]:
    """The runnable Python blocks of a page, each with the line it starts on."""
    text = page.read_text(encoding="utf-8")
    blocks = []
    for match in FENCE.finditer(text):
        words = match.group("info").split()
        if words[:1] == ["python"] and "skip" not in words[1:]:
            blocks.append((text.count("\n", 0, match.start("body")) + 1, match.group("body")))
    return blocks


def slug(title: str) -> str:
    """A heading's anchor, the way GitHub makes it."""
    title = re.sub(r"[`*_]|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", title)
    return re.sub(r"[^\w\- ]", "", title.lower()).strip().replace(" ", "-")


def anchors(page: Path) -> set[str]:
    text = FENCE.sub("", page.read_text(encoding="utf-8"))
    return {slug(match.group("title")) for match in HEADING.finditer(text)}


def relative(page: Path) -> str:
    return str(page.relative_to(ROOT))


@pytest.mark.parametrize("page", PAGES, ids=relative)
def test_python_blocks_run(page: Path) -> None:
    module = types.ModuleType("__docs__")  # dataclasses look their module up in `sys.modules`
    sys.modules[module.__name__] = module
    namespace: dict[str, Any] = module.__dict__
    try:
        for line, code in python_blocks(page):
            with warnings.catch_warnings():
                # Pages that demonstrate a warning catch it themselves; any other is a defect.
                warnings.simplefilter("error")
                try:
                    exec(compile(code, f"{relative(page)}:{line}", "exec"), namespace)
                except Exception as error:
                    raise AssertionError(f"{relative(page)}:{line}: {error!r}") from error
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize("page", PAGES, ids=relative)
def test_relative_links_resolve(page: Path) -> None:
    text = FENCE.sub("", page.read_text(encoding="utf-8"))
    broken = []
    for match in LINK.finditer(text):
        target = match.group("target")
        if re.match(r"[a-z][a-z0-9+.-]*:", target):
            continue  # http(s), mailto: not checked offline
        path, _, fragment = target.partition("#")
        destination = (page.parent / path).resolve() if path else page
        missing_anchor = (
            bool(fragment) and destination.suffix == ".md" and fragment not in anchors(destination)
        )
        if not destination.exists() or missing_anchor:
            broken.append(target)
    assert not broken, f"{relative(page)} has broken links: {broken}"


def test_every_public_name_is_documented() -> None:
    docs = "\n".join(page.read_text(encoding="utf-8") for page in PAGES if page.name != "README.md")
    missing = [name for name in langchain_llm_router.__all__ if f"`{name}" not in docs]
    assert not missing, f"public names missing from docs/: {missing}"


def test_every_page_is_linked_from_the_docs_index() -> None:
    index = ROOT / "docs" / "index.md"
    linked = {
        (index.parent / match.group("target").partition("#")[0]).resolve()
        for match in LINK.finditer(index.read_text(encoding="utf-8"))
        if not re.match(r"[a-z][a-z0-9+.-]*:", match.group("target"))
    }
    unlinked = [relative(page) for page in PAGES[1:] if page != index and page not in linked]
    assert not unlinked, f"not linked from docs/index.md: {unlinked}"


def test_docs_carry_no_internal_nomenclature() -> None:
    pattern = re.compile(r"\b(?:T-\d{3}|REQ-\d+|D\d{1,2}\b|PRD)")
    found = [
        f"{relative(page)}: {match.group()}"
        for page in PAGES
        for match in pattern.finditer(page.read_text(encoding="utf-8"))
    ]
    assert not found, found
