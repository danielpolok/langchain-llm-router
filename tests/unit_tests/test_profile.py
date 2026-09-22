"""T-121: the router's own `.profile` — the intersection of its routes' (C1, C3, R10).

`_resolve_model_profile` (`router.py`) and the reduction it calls (`_profile.resolve_profile`)
are tested per value kind first (REQ-C3-4's Check, one case per bullet), then end to end through
`create_agent`, the real consumer `_supports_provider_strategy` (`langchain/agents/factory.py:560`)
names in the issue: a router whose routes disagree on `structured_output` must not have
`create_agent` pick a strategy only one of them can serve.

Neighbours, not repeated here: tool capability detection reads `profile["tool_calling"]` too
(D5, REQ-R10-1), but that is the other direction — what the router reads about a *route* — and
lives in `test_tools.py`.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel, ModelProfile
from langchain_core.messages import ToolCall
from pydantic import BaseModel, ValidationError

from langchain_llm_router import ChatRouter, RoutingError
from tests.fakes import FakeChatModel, ToolCallingFakeChatModel


def route(profile: dict[str, object] | None = None, **fields: object) -> FakeChatModel:
    """A route reporting exactly `profile` — nothing else about it matters to these tests."""
    return FakeChatModel(profile=profile, **fields)  # type: ignore[arg-type]


def router_of(**profiles: dict[str, object] | None) -> ChatRouter:
    """A two-or-more-route router, one named route per keyword, default the first."""
    routes: dict[str, BaseChatModel] = {name: route(profile) for name, profile in profiles.items()}
    return ChatRouter(routes=routes, default_route=next(iter(routes)))


# --- REQ-C3-4: the reduction, one value kind at a time ---


def test_booleans_are_anded() -> None:
    """REQ-C3-4: a boolean key is kept only if every route says `True`."""
    both_true = router_of(a={"tool_calling": True}, b={"tool_calling": True})
    one_false = router_of(a={"tool_calling": True}, b={"tool_calling": False})

    assert both_true.profile == {"tool_calling": True}
    assert one_false.profile == {"tool_calling": False}


def test_ints_are_minimised() -> None:
    """REQ-C3-4: a non-bool int key becomes the smallest of the routes' claims."""
    router = router_of(a={"max_input_tokens": 128_000}, b={"max_input_tokens": 32_000})

    assert router.profile == {"max_input_tokens": 32_000}


def test_equal_non_bool_non_int_values_are_kept() -> None:
    """REQ-C3-4: a `str` (or other non-bool, non-int) key survives when every route agrees."""
    router = router_of(a={"status": "active"}, b={"status": "active"})

    assert router.profile == {"status": "active"}


def test_differing_non_bool_non_int_values_are_dropped() -> None:
    """REQ-C3-4: the same key, disagreeing, is left out rather than guessed at."""
    router = router_of(a={"status": "active"}, b={"status": "deprecated"})

    assert router.profile == {}


def test_a_key_only_some_routes_report_is_dropped_not_defaulted() -> None:
    """REQ-C3-4: "the intersection of its routes' profiles" — a route silent on a key is not
    treated as `False`/unlimited/absent-and-ignorable; the key just isn't shared."""
    router = router_of(a={"tool_calling": True, "attachment": True}, b={"tool_calling": True})

    assert router.profile == {"tool_calling": True}


def test_none_when_any_route_reports_no_profile_at_all() -> None:
    """REQ-C3-4: `None` when *any* route has no profile — not only when all of them don't."""
    router = router_of(a={"tool_calling": True}, b=None)

    assert router.profile is None


def test_none_when_every_route_reports_no_profile() -> None:
    router = router_of(a=None, b=None)

    assert router.profile is None


def test_a_single_route_router_reports_that_route_s_profile_unchanged() -> None:
    """REQ-C3-4 acceptance criterion: the one-route case falls out of the algorithm for free."""
    profile = {"tool_calling": True, "max_input_tokens": 128_000, "status": "active"}
    router = ChatRouter(routes={"a": route(profile)}, default_route="a")

    assert router.profile == profile


def test_bool_and_falsy_int_keys_are_reduced_separately_and_correctly() -> None:
    """REQ-C3-4: `bool` is a subclass of `int` in Python, so a reduction that checked `int`
    before `bool` (or dropped the distinction) risks running a genuine boolean pair through
    `min()` instead of `all()`, or a genuine (non-bool) int pair through boolean logic. Both are
    exercised together, with values chosen to be truthy/falsy either way (`0`/`1` for the int
    key, `False`/`True` for the bool key) so a reduction that quietly conflated the two kinds
    would still have to get both of these right independently to pass.
    """
    router = router_of(
        a={"tool_calling": True, "max_output_tokens": 1},
        b={"tool_calling": False, "max_output_tokens": 0},
    )

    assert router.profile == {"tool_calling": False, "max_output_tokens": 0}
    assert router.profile["tool_calling"] is False  # AND, not min() coincidentally agreeing
    assert type(router.profile["max_output_tokens"]) is int  # min(), not bool logic
    assert router.profile["max_output_tokens"] == 0


def test_unknown_profile_keys_do_not_crash_the_reduction() -> None:
    """The issue: "unknown keys must not crash the reduction" (`_warn_unknown_profile_keys`,
    `chat_models.py:443`, runs generically over whatever `self.profile` ends up being — nothing
    here needs to special-case it). A key `ModelProfile` doesn't declare still reduces like any
    other; the base class warns about it once `profile` is assigned, which is not this
    reduction's job to suppress or duplicate."""
    with pytest.warns(UserWarning, match="Unrecognized keys"):
        router = router_of(a={"made_up_key": True}, b={"made_up_key": True})

    assert router.profile == {"made_up_key": True}


def test_explicit_profile_wins_over_the_intersection() -> None:
    """The issue: "an explicitly supplied `profile=` on the router wins" — already handled by
    `BaseChatModel._set_model_profile` (`chat_models.py:427`), which only assigns
    `_resolve_model_profile()`'s result when `profile is None`; confirmed empirically rather
    than assumed, per the brief."""
    routes: dict[str, BaseChatModel] = {
        "a": route({"tool_calling": True}),
        "b": route({"tool_calling": False}),
    }
    given: ModelProfile = {"tool_calling": True, "structured_output": True}

    router = ChatRouter(routes=routes, default_route="a", profile=given)

    assert router.profile == given  # not {"tool_calling": False}, the intersection


def test_empty_routes_is_rejected_before_profile_resolution_would_run() -> None:
    """The issue's "exact handling of an empty `routes` mapping": `resolve_profile` guards
    against it defensively, but it is unreachable through `ChatRouter` — `_check_routes` (a
    *field* validator, REQ-R5-2) raises before any `model_validator(mode="after")`, the base
    class's `_set_model_profile` among them, gets a chance to call `_resolve_model_profile` on
    an empty `routes` (pydantic v2 runs field validation first; confirmed here, not assumed)."""
    with pytest.raises(ValidationError) as caught:
        ChatRouter(routes={}, default_route="a")
    [error] = caught.value.errors()
    assert error["loc"] == ("routes",)
    assert isinstance(error["ctx"]["error"], RoutingError)


# --- Acceptance criterion: the real consumer, `create_agent(response_format=…)` ---


class Answer(BaseModel):
    """A structured answer."""

    answer: str


ANSWER_CALL = ToolCall(name="Answer", args={"answer": "42"}, id="call_1", type="tool_call")


def test_create_agent_settles_on_a_strategy_every_route_can_serve() -> None:
    """REQ-C3-4's acceptance criterion: two routes agree on `tool_calling` but disagree on
    `structured_output`. The router's own profile (AND-ed) reports `structured_output: False`,
    so `create_agent`'s auto-detection (`_supports_provider_strategy`,
    `langchain/agents/factory.py:560`, reads `model.profile.get("structured_output")`) must not
    pick `ProviderStrategy` — a route can't serve it — and instead falls back to `ToolStrategy`,
    which every route here can: it is bound as an ordinary tool (named after the schema,
    `"Answer"`) with `tool_choice="any"`, and the structured response still comes back correctly
    through the router.
    """
    from langchain.agents import create_agent

    capable = ToolCallingFakeChatModel(
        model_name="a",
        tool_calls=[ANSWER_CALL],
        profile={"tool_calling": True, "structured_output": True},
    )
    incapable = ToolCallingFakeChatModel(
        model_name="b", profile={"tool_calling": True, "structured_output": False}
    )
    router = ChatRouter(routes={"a": capable, "b": incapable}, default_route="a")

    assert router.profile == {"tool_calling": True, "structured_output": False}

    agent = create_agent(model=router, response_format=Answer)
    out = agent.invoke({"messages": [{"role": "user", "content": "what is the answer?"}]})

    assert out["structured_response"] == Answer(answer="42")
    # ToolStrategy's signature (REQ-C3-4's Check names it: "a strategy every route can serve") —
    # ProviderStrategy binds no tools at all, so a bound "Answer" tool with tool_choice="any"
    # is exactly the evidence that the fallback, not the provider strategy, ran.
    assert capable.bind_calls[-1]["tool_choice"] == "any"
    assert [tool.name for tool in capable.bind_calls[-1]["tools"]] == ["Answer"]


def test_provider_strategy_is_available_when_every_route_reports_it() -> None:
    """Contrast case for the test above, so it is the *mismatch* being detected and not
    `ToolStrategy` always winning regardless of `profile`: `_supports_provider_strategy` itself,
    called directly, says `True` for a router whose routes agree on `structured_output`, and
    `False` for the mismatched pair — the router's own profile is what changes between them."""
    from langchain.agents.factory import (  # private API: the acceptance criterion names it
        _supports_provider_strategy,
    )

    agreeing = router_of(
        a={"tool_calling": True, "structured_output": True},
        b={"tool_calling": True, "structured_output": True},
    )
    mismatched = router_of(
        a={"tool_calling": True, "structured_output": True},
        b={"tool_calling": True, "structured_output": False},
    )

    assert _supports_provider_strategy(agreeing) is True
    assert _supports_provider_strategy(mismatched) is False
