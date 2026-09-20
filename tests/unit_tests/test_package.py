"""Package-level guarantees: R8 (langchain-core only) and C9 (langchain-core 1.x)."""

from importlib.metadata import requires, version

from packaging.requirements import Requirement

# The public API pinned in docs/v1-requirements.md. Tasks add their names as they land:
# the remaining built-in strategies (T-132 to T-134) are still to come.
PINNED_EXPORTS = {
    "ChatRouter",
    "KeywordStrategy",
    "RoutingStrategy",
    "RoutingRequest",
    "RoutingChoice",
    "RoutingCallable",
    "RoutingDecision",
    "routing_decision",
    "last_routing_decision",
    "RoutingWarning",
    "FallbackWarning",
    "ToolSupportWarning",
    "ForcedRouteWarning",
    "RoutingError",
    "NoToolCapableRouteError",
    "ForcedRouteError",
    "HeuristicStrategy",
}


def test_package_exports_the_pinned_api() -> None:
    import langchain_llm_router

    assert set(langchain_llm_router.__all__) >= PINNED_EXPORTS
    for name in langchain_llm_router.__all__:
        assert getattr(langchain_llm_router, name) is not None


def test_runtime_dependencies_are_langchain_core_only() -> None:
    declared = [Requirement(spec) for spec in requires("langchain-llm-router") or []]
    runtime = [req for req in declared if req.marker is None]  # extras carry a marker

    assert [req.name for req in runtime] == ["langchain-core"]


def test_targets_langchain_core_1x() -> None:
    (core,) = [Requirement(spec) for spec in requires("langchain-llm-router") or []]

    assert core.specifier.contains(version("langchain-core"))
    assert not core.specifier.contains("2.0.0")
