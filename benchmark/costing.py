"""Turn recorded usage into dollars (T-140's "cost from recorded usage x price").

Two sources, matching REQ-R3-2's two cases:

- Chat-model calls (every route, and a `ClassifierStrategy`'s own call) report real
  `usage_metadata`, gathered by `langchain_core.callbacks.get_usage_metadata_callback` — the
  same mechanism `tests/unit_tests/test_cost.py` proves is exactly the routes' own usage,
  keyed by the model that ran, including a strategy's nested call (it inherits the router's
  config per D9, so it is visible to the same callback without any special-casing here).
- `EmbeddingStrategy` calls report none (`Embeddings` is a plain ABC with no callback hook), so
  its cost is the same chars-per-token estimate `embedding.py` itself uses, applied to the
  current request's text — not a measurement, an estimate, and reported as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from langchain_core.messages.ai import UsageMetadata

from benchmark.pricing import EMBEDDING_PRICE_PER_MILLION_TOKENS, chat_prices, price_for
from langchain_llm_router.strategies.embedding import _CHARS_PER_TOKEN


@dataclass(frozen=True)
class CostBreakdown:
    """One item's cost, split by source, so a report can show where the money went."""

    chat_usd: float
    """Every chat-model call in the trace (the answering route, plus a classifier's own call)."""

    embedding_usd: float
    """The embedding strategy's per-request query, estimated from input length."""

    @property
    def total_usd(self) -> float:
        return self.chat_usd + self.embedding_usd


def chat_cost(usage_by_model: dict[str, UsageMetadata]) -> float:
    """Dollar cost of every model call `get_usage_metadata_callback` recorded for one item.

    `usage_by_model` sums usage per model name across every chat-model call inside the
    callback's scope — the answering route and, if the strategy made one, its own call too,
    since both inherit the same config (D9). A model this benchmark hasn't priced fails loudly
    (`price_for`) rather than silently costing $0.00.
    """
    prices = chat_prices()
    return sum(
        price_for(model, prices).cost(usage["input_tokens"], usage["output_tokens"])
        for model, usage in usage_by_model.items()
    )


def estimate_embedding_cost(text: str) -> float:
    """The same chars-per-token estimate `EmbeddingStrategy` itself computes (T-133), priced.

    Only the per-request query is estimated — the one-time route-example embedding is
    amortised construction-time work, not any one item's cost (embedding.py's own reasoning).
    """
    estimated_tokens = max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))
    return estimated_tokens * EMBEDDING_PRICE_PER_MILLION_TOKENS / 1_000_000


def cost_of(
    usage_by_model: dict[str, UsageMetadata], *, embedding_query_text: str | None = None
) -> CostBreakdown:
    """The full cost of one item: its chat usage, plus an embedding estimate if the strategy
    under test made one (`embedding_query_text` is the request text that was embedded, or
    `None` for a strategy that doesn't embed)."""
    embedding_usd = estimate_embedding_cost(embedding_query_text) if embedding_query_text else 0.0
    return CostBreakdown(chat_usd=chat_cost(usage_by_model), embedding_usd=embedding_usd)
