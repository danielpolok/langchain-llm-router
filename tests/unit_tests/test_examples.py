"""The examples stay true offline: they compile, their imports resolve, each shows what it prints.

Each example calls real models, like the documentation, so running them is the live test's job
(`tests/integration_tests/test_examples_live.py`). What can be checked without a provider is
checked here, on every run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.docs import EXAMPLES, EXAMPLES_README, check_imports, example_output, links, relative


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_compiles_and_its_imports_resolve(script: Path) -> None:
    code = script.read_text(encoding="utf-8")
    compile(code, relative(script), "exec")
    check_imports(code, relative(script))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_ends_with_what_it_prints(script: Path) -> None:
    assert example_output(script), f"{relative(script)} doesn't end with an '# It prints' comment"


def test_every_example_is_listed_in_the_examples_readme() -> None:
    listed = set(links(EXAMPLES_README))
    unlisted = [script.name for script in EXAMPLES if script.name not in listed]
    assert not unlisted, f"not listed in examples/README.md: {unlisted}"


def test_every_example_is_covered() -> None:
    """A new `examples/*.py` file is picked up automatically (the `EXAMPLES` glob); this only
    guards against the glob itself silently finding nothing."""
    assert len(EXAMPLES) >= 6
