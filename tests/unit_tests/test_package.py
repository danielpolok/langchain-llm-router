"""Package-level guarantees: it needs langchain-core only, and langchain-core 1.x."""

from dataclasses import fields
from importlib.metadata import requires, version
from inspect import Parameter, signature

from packaging.requirements import Requirement

# The public API this test pins: the names the package exports.
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
    "ConfigurableStrategy",
    "EmbeddingStrategy",
    "ClassifierStrategy",
}


def test_package_exports_the_pinned_api() -> None:
    import langchain_model_router

    assert set(langchain_model_router.__all__) >= PINNED_EXPORTS
    for name in langchain_model_router.__all__:
        assert getattr(langchain_model_router, name) is not None


# Defaults the stability promise settles: changing one is a breaking release.
READS_EARLIER_MESSAGES = ["KeywordStrategy", "ConfigurableStrategy", "HeuristicStrategy"]


def test_reading_earlier_messages_is_off_by_default() -> None:
    import langchain_model_router

    assert langchain_model_router.RoutingStrategy.lookback == 0
    request_fields = {field.name: field for field in fields(langchain_model_router.RoutingRequest)}
    assert request_fields["previous_requests"].default == ()
    for name in READS_EARLIER_MESSAGES:
        lookback = signature(getattr(langchain_model_router, name)).parameters["lookback"]
        assert (lookback.kind, lookback.default) == (Parameter.KEYWORD_ONLY, 0)


def test_runtime_dependencies_are_langchain_core_only() -> None:
    declared = [Requirement(spec) for spec in requires("langchain-model-router") or []]
    runtime = [req for req in declared if req.marker is None]  # extras carry a marker

    assert [req.name for req in runtime] == ["langchain-core"]


def test_targets_langchain_core_1x() -> None:
    (core,) = [Requirement(spec) for spec in requires("langchain-model-router") or []]

    assert core.specifier.contains(version("langchain-core"))
    assert not core.specifier.contains("2.0.0")
