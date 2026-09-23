"""REQ-C8-2 and the T-150 acceptance criterion: every documented example runs in CI, offline.

Each script in `examples/` is run the way its own docstring tells a reader to run it —
`python examples/<name>.py`, from the repository root — and must exit zero. None of them needs a
provider credential: every route in every example is a fake (`GenericFakeChatModel` or
`examples/_fakes.py`'s `ScriptedToolChatModel`), so this runs unconditionally, unlike the
`requires_env`/`requires_ollama`-gated tests in `tests/integration_tests`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted(
    path for path in (ROOT / "examples").glob("*.py") if not path.name.startswith("_")
)


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_every_example_is_covered() -> None:
    """A new `examples/*.py` file is picked up automatically (module-level `EXAMPLES` glob);
    this only guards against the glob itself silently finding nothing."""
    assert len(EXAMPLES) >= 6
