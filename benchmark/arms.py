"""The baseline and the four strategies the benchmark compares.

Two candidate routes throughout, both Gemini so the whole benchmark runs from one API key with
no local server dependency — `"small"` (`gemini-3.5-flash-lite`) and `"frontier"`
(`gemini-3.8-flash`), picked over this repo's usual small/frontier pair
(`ollama:qwen3:8b`/`gemini-3-flash-preview`, still `benchmark.arms.ollama_smoke_model`'s default
for the offline-adjacent live-Ollama test) once a budget-constrained run made an all-cloud pair
worth measuring first. The baseline always answers on `"frontier"`; each strategy decides per
request whether `"small"` is safe.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from langchain_llm_router import ChatRouter
from langchain_llm_router.strategies import (
    ClassifierStrategy,
    EmbeddingStrategy,
    HeuristicStrategy,
    KeywordStrategy,
)

FRONTIER_MODEL = os.environ.get(
    "LLM_ROUTER_BENCHMARK_FRONTIER_MODEL", "google_genai:gemini-3.8-flash"
)
SMALL_MODEL = os.environ.get(
    "LLM_ROUTER_BENCHMARK_SMALL_MODEL", "google_genai:gemini-3.5-flash-lite"
)
EMBED_MODEL = os.environ.get("LLM_ROUTER_GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
OLLAMA_SMOKE_MODEL = os.environ.get("LLM_ROUTER_OLLAMA_MODEL", "ollama:qwen3:8b")
"""Not one of the benchmark's own arms — only `test_live_ollama_smoke.py`'s free, local,
budget-independent sanity check uses this."""

# Uncalibrated pending a real run (embedding.py: EmbeddingStrategy's threshold has no shipped
# default by design). Override with
# LLM_ROUTER_BENCHMARK_EMBEDDING_THRESHOLD once a live run's similarity scores say what a good
# bar actually is; see benchmark/README.md.
EMBEDDING_THRESHOLD = float(os.environ.get("LLM_ROUTER_BENCHMARK_EMBEDDING_THRESHOLD", "0.5"))

# Calibration examples for EmbeddingStrategy and KeywordStrategy — deliberately *not* drawn from
# benchmark/data/workload.json: reusing graded items as the strategy's own few-shot examples
# would let a strategy "recognise" the exact things it is tested on, inflating its apparent
# accuracy over what it would do on traffic it hasn't seen.
_EASY_EXAMPLES = (
    "What is the capital of Japan?",
    "Convert 10 kilometers to miles.",
    "What does 'HTTP' stand for?",
    "What's 12 times 8?",
    "Is a tomato a fruit or a vegetable?",
)
_HARD_EXAMPLES = (
    "Compare two sorting algorithms' time complexity and explain when you'd choose one over "
    "the other.",
    "Here's a stack trace — explain the root cause and how to fix it.",
    "Derive the formula for compound interest and justify each step.",
    "Critique this argument for a logical flaw and suggest a better way to reason about it.",
    "Refactor this function and explain the trade-offs of your approach.",
)

_HARD_KEYWORDS = (
    "compare",
    "contrast",
    "prove",
    "derive",
    "justify",
    "critique",
    "refactor",
    "traceback",
    "root cause",
    "trade-off",
    "trade-offs",
)


@dataclass(frozen=True)
class Arm:
    """One thing under test: a name for the report, and the router that implements it."""

    name: str
    router: ChatRouter


ArmFactory = Callable[[], Arm]


def frontier_model() -> BaseChatModel:
    return init_chat_model(FRONTIER_MODEL)


def small_model() -> BaseChatModel:
    return init_chat_model(SMALL_MODEL)


def ollama_smoke_model() -> BaseChatModel:
    """The one place `OLLAMA_SMOKE_MODEL` is used — a free, local, real-provider sanity check
    that doesn't touch a budget at all. Not part of any arm."""
    return init_chat_model(OLLAMA_SMOKE_MODEL)


def embeddings() -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def build_arms() -> list[Arm]:
    """The baseline plus keyword, heuristic, embedding and classifier — a fresh instance of
    each model per arm, since a route must not be shared between two `ChatRouter`s that might
    run concurrently (nothing here forbids it, but each arm owning its own keeps the arms
    independent for a parallel run)."""
    return [
        Arm(
            "baseline-always-frontier",
            ChatRouter(routes={"frontier": frontier_model()}, default_route="frontier"),
        ),
        Arm(
            "keyword",
            ChatRouter(
                routes={"small": small_model(), "frontier": frontier_model()},
                default_route="small",
                strategy=KeywordStrategy({"frontier": list(_HARD_KEYWORDS)}),
            ),
        ),
        Arm(
            "heuristic",
            ChatRouter(
                routes={"small": small_model(), "frontier": frontier_model()},
                default_route="small",
                strategy=HeuristicStrategy("small", "frontier"),
            ),
        ),
        Arm(
            "embedding",
            ChatRouter(
                routes={"small": small_model(), "frontier": frontier_model()},
                default_route="small",
                strategy=EmbeddingStrategy(
                    embeddings(),
                    {"small": list(_EASY_EXAMPLES), "frontier": list(_HARD_EXAMPLES)},
                    threshold=EMBEDDING_THRESHOLD,
                ),
            ),
        ),
        Arm(
            "classifier",
            ChatRouter(
                routes={"small": small_model(), "frontier": frontier_model()},
                default_route="small",
                strategy=ClassifierStrategy(
                    small_model(),  # "ask a small model" (classifier.py) — the small route's own
                    {
                        "small": "simple, short requests: single facts, basic arithmetic, "
                        "one-step lookups, short definitions",
                        "frontier": "requests that need real reasoning: multi-step analysis, "
                        "comparisons, proofs or derivations, debugging, refactors with "
                        "trade-offs, or several distinct sub-questions at once",
                    },
                ),
            ),
        ),
    ]
