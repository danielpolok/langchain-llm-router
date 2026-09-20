"""The built-in strategies (R6's first level: ready-made, with sensible defaults).

Each one implements `RoutingStrategy` and is exported from `langchain_llm_router` itself.
The heuristic ones make no model or API call (R7); the opt-in ones (embedding similarity,
small-LLM classifier) take the model or embeddings they use from the application.
"""

__all__: list[str] = []
