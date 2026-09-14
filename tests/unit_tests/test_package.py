"""Package-level guarantees: R8 (langchain-core only) and C9 (langchain-core 1.x)."""

from importlib.metadata import requires, version

from packaging.requirements import Requirement


def test_package_imports() -> None:
    import langchain_llm_router

    assert langchain_llm_router.__all__ == []


def test_runtime_dependencies_are_langchain_core_only() -> None:
    declared = [Requirement(spec) for spec in requires("langchain-llm-router") or []]
    runtime = [req for req in declared if req.marker is None]  # extras carry a marker

    assert [req.name for req in runtime] == ["langchain-core"]


def test_targets_langchain_core_1x() -> None:
    (core,) = [Requirement(spec) for spec in requires("langchain-llm-router") or []]

    assert core.specifier.contains(version("langchain-core"))
    assert not core.specifier.contains("2.0.0")
