"""Fakes shared across `benchmark/tests/` (mirrors the repo's own `tests/fakes.py` convention:
helpers live outside any one `test_*.py` file so more than one test module can import them
without importing test functions along with them)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from benchmark.judge import JudgeVerdict
from tests.fakes import GenerateOnlyFakeChatModel


class FakeJudge(GenerateOnlyFakeChatModel):
    """A judge that always scores the same way, regardless of what it's asked to grade."""

    fixed_score: int = 8
    fixed_rationale: str = "fake judge, fixed verdict"

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        verdict = JudgeVerdict(score=self.fixed_score, rationale=self.fixed_rationale)
        return RunnableLambda(lambda _input: verdict)


class RaisingModel(BaseChatModel):
    """A route that always fails, to prove one item's error doesn't lose the run."""

    @property
    def _llm_type(self) -> str:
        return "raising"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        msg = "this route always fails"
        raise RuntimeError(msg)
