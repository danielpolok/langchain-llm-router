"""An offline stand-in for what LangSmith charges for a trace (ported from `spike/tracing.py`).

LangSmith prices a run from its `usage_metadata` plus the `ls_provider` / `ls_model_name` on
its metadata (docs: *Cost tracking*). `LangChainTracer._on_llm_end` is what puts usage on a
run: it reads the run's *own* outputs (`tracers/langchain.py:395`). So every LLM run in a trace
that carries usage and a model name is a separate charge — which is the whole R3 versus C5
question when a chat model calls a chat model.

These helpers walk collected runs the same way, so tests can check the trace's shape and
charges without an API key. The live check against LangSmith still needs one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, NamedTuple

from langchain_core.tracers.langchain import _get_usage_metadata_from_generations

if TYPE_CHECKING:
    from langchain_core.messages.ai import UsageMetadata
    from langchain_core.tracers.schemas import Run

LLM_RUN_TYPES = {"llm", "chat_model"}


class PricedCall(NamedTuple):
    """One line LangSmith would put a price against."""

    model_name: str | None
    usage: UsageMetadata


def walk(runs: Iterable[Run]) -> Iterator[Run]:
    """Every run in the trees, parents before children."""
    for run in runs:
        yield run
        yield from walk(run.child_runs or [])


def model_runs(runs: Iterable[Run]) -> list[Run]:
    """The runs LangSmith treats as model calls."""
    return [run for run in walk(runs) if run.run_type in LLM_RUN_TYPES]


def usage_of(run: Run) -> UsageMetadata | None:
    """The usage LangSmith would read off this run, by the tracer's own rule."""
    outputs = run.outputs or {}
    if "generations" not in outputs:
        return None
    return _get_usage_metadata_from_generations(outputs["generations"])


def model_name_of(run: Run) -> str | None:
    metadata = (run.extra or {}).get("metadata") or {}
    name = metadata.get("ls_model_name")
    return name if isinstance(name, str) else None


def priced_calls(runs: Iterable[Run]) -> list[PricedCall]:
    """Every charge in the trace: one per model run that reports usage.

    More than one entry for a single request is R3 broken — the same tokens billed twice.
    """
    calls = []
    for run in model_runs(runs):
        usage = usage_of(run)
        if usage is not None:
            calls.append(PricedCall(model_name_of(run), usage))
    return calls
