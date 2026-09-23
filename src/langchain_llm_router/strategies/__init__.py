"""The built-in strategies (R6's first two levels: ready-made, and configured from scratch).

Each one implements `RoutingStrategy` and is exported from `langchain_llm_router` itself.
The heuristic ones make no model or API call (R7); the opt-in ones (embedding similarity,
small-LLM classifier) take the model or embeddings they use from the application.
"""

from langchain_llm_router.strategies.classifier import ClassifierStrategy
from langchain_llm_router.strategies.configurable import ConfigurableStrategy
from langchain_llm_router.strategies.embedding import EmbeddingStrategy
from langchain_llm_router.strategies.heuristic import HeuristicStrategy
from langchain_llm_router.strategies.keyword import KeywordStrategy

__all__ = [
    "ClassifierStrategy",
    "ConfigurableStrategy",
    "EmbeddingStrategy",
    "HeuristicStrategy",
    "KeywordStrategy",
]
