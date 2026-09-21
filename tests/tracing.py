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
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.langchain import _get_usage_metadata_from_generations

from langchain_llm_router import RoutingChoice, RoutingRequest, RoutingStrategy

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
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


def ls_metadata_of(run: Run) -> list[str]:
    """The `ls_*` keys on a run's metadata — the ones LangSmith reads a model's identity from."""
    metadata = (run.extra or {}).get("metadata") or {}
    return sorted(key for key in metadata if key.startswith("ls_"))


class Start(NamedTuple):
    """One run's start, as a callback handler is told of it."""

    run_id: UUID
    parent_run_id: UUID | None
    name: str | None


class StartLog(BaseCallbackHandler):
    """Records every run's start by the callback that announced it.

    The offline tracer files a chat model's run and a text LLM's under the same `run_type`; the
    callbacks don't, and REQ-C5-1 is about `on_chat_model_start` — the one a chat model reports
    — so this keeps the three apart. `on_llm_start` should stay empty for a router.
    """

    run_inline = True

    def __init__(self) -> None:
        self.chat_models: list[Start] = []
        self.llms: list[Start] = []
        self.chains: list[Start] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.chat_models.append(Start(run_id, parent_run_id, kwargs.get("name")))

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.llms.append(Start(run_id, parent_run_id, kwargs.get("name")))

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.chains.append(Start(run_id, parent_run_id, kwargs.get("name")))


class ConsultingStrategy(RoutingStrategy):
    """A strategy that asks a model before it decides, as an opt-in classifier does (R7, D9).

    It always answers with `route`: what these tests are about is the call it makes on the way,
    and where that call lands in the trace and on the bill. `pass_config=False` is the custom
    strategy that forgets `request.config`, which D9 still nests wherever context propagates.
    """

    def __init__(self, model: BaseChatModel, route: str, *, pass_config: bool = True) -> None:
        self.model = model
        self.route = route
        self.pass_config = pass_config

    def _choice(self) -> RoutingChoice:
        return RoutingChoice(route=self.route, reason="the classifier said so")

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        self.model.invoke(request.text, config=request.config if self.pass_config else None)
        return self._choice()

    async def adecide(self, request: RoutingRequest) -> RoutingChoice:
        await self.model.ainvoke(request.text, config=request.config if self.pass_config else None)
        return self._choice()
