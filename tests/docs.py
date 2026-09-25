"""Reading the documentation: its pages, their code blocks, links and headings.

A page's ```` ```python ```` blocks are one script, read top to bottom: setup is written once, and
later blocks use what earlier ones defined. The ```` ```text ```` block straight after a code block
is what the page says that code prints.
"""

from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from types import CodeType

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PAGES = [README, *sorted((ROOT / "docs").rglob("*.md"))]

FENCE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
SOURCE = re.compile(r"\b(?:src|srcset)=\"(?P<target>[^\"]+)\"")  # <img> and <picture> images
HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$", re.MULTILINE)
URL = re.compile(r"[a-z][a-z0-9+.-]*:")


@dataclass(frozen=True)
class Block:
    """One ```` ```python ```` block, and the output the page shows for it."""

    page: Path
    line: int
    code: str
    output: str

    @property
    def location(self) -> str:
        return f"{relative(self.page)}:{self.line}"

    def compiled(self) -> CodeType:
        """The block as code; `await` at the top level is allowed, as in a notebook."""
        code: CodeType = compile(
            self.code, self.location, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        )
        return code


def python_blocks(page: Path) -> list[Block]:
    text = page.read_text(encoding="utf-8")
    fences = list(FENCE.finditer(text))
    blocks = []
    for fence, following in itertools.zip_longest(fences, fences[1:]):
        if fence.group("info").strip() != "python":
            continue
        shows_output = (
            following is not None
            and following.group("info").strip() == "text"
            and not text[fence.end() : following.start()].strip()
        )
        blocks.append(
            Block(
                page=page,
                line=text.count("\n", 0, fence.start("body")) + 1,
                code=fence.group("body"),
                output=following.group("body") if following and shows_output else "",
            )
        )
    return blocks


def links(page: Path) -> list[str]:
    """Every link and image target on the page, outside code blocks."""
    text = FENCE.sub("", page.read_text(encoding="utf-8"))
    return [match.group("target") for pattern in (LINK, SOURCE) for match in pattern.finditer(text)]


def slug(title: str) -> str:
    """A heading's anchor, the way GitHub makes it."""
    title = re.sub(r"[`*_]|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", title)
    return re.sub(r"[^\w\- ]", "", title.lower()).strip().replace(" ", "-")


def anchors(page: Path) -> set[str]:
    text = FENCE.sub("", page.read_text(encoding="utf-8"))
    return {slug(match.group("title")) for match in HEADING.finditer(text)}


def relative(page: Path) -> str:
    return str(page.relative_to(ROOT))
