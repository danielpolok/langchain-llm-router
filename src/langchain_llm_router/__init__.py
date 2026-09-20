"""In-process model routing for LangChain.

`ChatRouter` is a chat model that, per request, picks one of several named chat models and
returns that model's response unchanged, plus a record of the routing decision.
"""

from langchain_llm_router.decision import RoutingDecision, last_routing_decision, routing_decision
from langchain_llm_router.errors import (
    FallbackWarning,
    ForcedRouteError,
    ForcedRouteWarning,
    NoToolCapableRouteError,
    RoutingError,
    RoutingWarning,
    ToolSupportWarning,
)
from langchain_llm_router.router import ChatRouter
from langchain_llm_router.strategies import HeuristicStrategy, KeywordStrategy
from langchain_llm_router.strategy import (
    RoutingCallable,
    RoutingChoice,
    RoutingRequest,
    RoutingStrategy,
)

__all__ = [
    "ChatRouter",
    "FallbackWarning",
    "ForcedRouteError",
    "ForcedRouteWarning",
    "HeuristicStrategy",
    "KeywordStrategy",
    "NoToolCapableRouteError",
    "RoutingCallable",
    "RoutingChoice",
    "RoutingDecision",
    "RoutingError",
    "RoutingRequest",
    "RoutingStrategy",
    "RoutingWarning",
    "ToolSupportWarning",
    "last_routing_decision",
    "routing_decision",
]
