"""Run the dataset through one arm and grade every answer.

One item, one arm: invoke the router (a two-step tool round trip for `kind == "agent"`),
capture usage with `get_usage_metadata_callback`, read the routing decision straight off the
response (`routing_decision`), price it (`costing.py`), and grade the final answer with the
judge (`judge.py`). Errors are per item — a rate limit or a flaky local model shouldn't lose the
rest of the run — recorded on the result rather than raised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata, add_usage

from benchmark.arms import Arm
from benchmark.costing import CostBreakdown, cost_of
from benchmark.dataset import WorkloadItem
from benchmark.judge import JudgeVerdict, grade
from benchmark.tools import ALL_TOOLS, BY_NAME
from langchain_llm_router import RoutingDecision, routing_decision


@dataclass
class ItemResult:
    """Everything one (arm, item) pair produced — enough to build the report from, and enough
    to debug a surprising number without re-running anything."""

    item_id: str
    arm: str
    decision: RoutingDecision | None
    answer: str
    tool_called: str | None
    tool_correct: bool | None
    """`None` for a single-turn item, where there is no tool to have called."""
    usage_by_model: dict[str, UsageMetadata]
    cost: CostBreakdown
    judge: JudgeVerdict | None
    error: str | None = None


def _merge_usage(
    into: dict[str, UsageMetadata], added: dict[str, UsageMetadata]
) -> dict[str, UsageMetadata]:
    merged = dict(into)
    for model, usage in added.items():
        merged[model] = add_usage(merged.get(model), usage) if model in merged else usage
    return merged


def _run_single_turn(
    router: BaseChatModel, item: WorkloadItem
) -> tuple[AIMessage, dict[str, UsageMetadata]]:
    with get_usage_metadata_callback() as usage:
        response = router.invoke(item.prompt)
    return response, dict(usage.usage_metadata)


def _run_agent(
    router: BaseChatModel, item: WorkloadItem
) -> tuple[AIMessage, dict[str, UsageMetadata], str | None]:
    """One tool round trip: bind the full toolkit, invoke, execute at most the first tool call
    the model makes, invoke again with its result. The *first* call's routing decision is what
    the report attributes the item to (routing looks at the current request, and the first
    call is the one that saw it as a fresh human turn)."""
    bound = router.bind_tools(list(ALL_TOOLS))
    messages: list[HumanMessage | AIMessage | ToolMessage] = [HumanMessage(item.prompt)]

    with get_usage_metadata_callback() as usage:
        first = bound.invoke(messages)
    total_usage = dict(usage.usage_metadata)
    tool_called: str | None = None

    if first.tool_calls:
        call = first.tool_calls[0]
        tool_called = call["name"]
        tool_fn = BY_NAME.get(call["name"])
        result = tool_fn.invoke(call["args"]) if tool_fn is not None else "error: unknown tool"
        messages += [first, ToolMessage(content=str(result), tool_call_id=call["id"])]
        with get_usage_metadata_callback() as usage2:
            second = bound.invoke(messages)
        total_usage = _merge_usage(total_usage, dict(usage2.usage_metadata))
        return second, total_usage, tool_called

    return first, total_usage, tool_called


def run_item(
    arm: Arm,
    item: WorkloadItem,
    judge: BaseChatModel,
    *,
    embed: bool,
) -> ItemResult:
    """Run and grade one item on one arm. `embed=True` when the arm's strategy embeds the
    request (only `EmbeddingStrategy`'s arm) — the only way to know is to be told, since the
    router's response carries no signal that an embedding call happened (see the embedding
    strategy's docstring: `Embeddings` emits no callbacks at all)."""
    try:
        if item.kind == "agent":
            response, usage_by_model, tool_called = _run_agent(arm.router, item)
        else:
            response, usage_by_model = _run_single_turn(arm.router, item)
            tool_called = None

        answer = response.text
        decision = routing_decision(response)
        cost = cost_of(usage_by_model, embedding_query_text=item.prompt if embed else None)
        verdict = grade(judge, prompt=item.prompt, rubric=item.rubric, answer=answer)
        tool_correct = (tool_called == item.expected_tool) if item.kind == "agent" else None

        return ItemResult(
            item_id=item.id,
            arm=arm.name,
            decision=decision,
            answer=answer,
            tool_called=tool_called,
            tool_correct=tool_correct,
            usage_by_model=usage_by_model,
            cost=cost,
            judge=verdict,
        )
    except Exception as error:
        return ItemResult(
            item_id=item.id,
            arm=arm.name,
            decision=None,
            answer="",
            tool_called=None,
            tool_correct=None,
            usage_by_model={},
            cost=CostBreakdown(chat_usd=0.0, embedding_usd=0.0),
            judge=None,
            error=f"{type(error).__name__}: {error}",
        )


def run_arm(
    arm: Arm,
    items: list[WorkloadItem],
    judge: BaseChatModel,
    *,
    embed: bool,
    pause_seconds: float = 0.0,
) -> list[ItemResult]:
    """Run every item on one arm, in order. `pause_seconds` between items is a courtesy to a
    rate-limited API — Ollama needs none, a free-tier Gemini key may."""
    results = []
    for index, item in enumerate(items):
        if index and pause_seconds:
            time.sleep(pause_seconds)
        results.append(run_item(arm, item, judge, embed=embed))
    return results
