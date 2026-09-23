"""`ConfigurableStrategy` (R6's middle level): a strategy from rules, without a strategy class.

The two §4 use cases REQ-R6-3 names, cost tiering and domain routing, are configuration here,
and read as what they do:

```python
long_request = signal_at_least(length_signal(20, 200), 0.5, name="length")  # 110 words or more
carries_code = signal_at_least(code_signal, 0.5, name="code")

# Cost tiering: long or code-bearing requests to the frontier model, the rest to the small one.
ConfigurableStrategy(
    [
        Rule("frontier", any_of(long_request, carries_code), name="long or code-bearing"),
        Rule("small", always(), name="short and simple"),
    ]
)

# Domain routing: code requests to the code model; the rest is the router's default (R9).
ConfigurableStrategy(
    [Rule("coder", any_of(keywords("python", "regex", "stack trace"), carries_code), name="code")]
)
```

What is a rule
--------------
A `Rule` is a route, a condition and a rank: `Rule(route, when, priority=0, name=None)`. When
the condition holds for a request, the rule sends it to the route. The condition is built from
what a strategy can see of the current request (R4) — nothing else, so nothing here can misroute
an agent loop on its tool output:

| Condition | Holds when |
| --- | --- |
| `keywords("python", re.compile(...))` | any word is in the text, as `KeywordStrategy` reads it |
| `signal_at_least(signal, 0.5)` | the `Signal` scores the request at least that |
| `modality("image", "audio")` | the request carries any of those modalities (C7) |
| `tools_bound()` | tools or structured output are bound to the call |
| `predicate(func)` | `func(request)` is `True` — the escape hatch for anything else |
| `always()` | always — the rule that answers when nothing above it did |

A keyword is a whole word, ignoring case, and a compiled pattern is searched as compiled. A
`Signal` is any of the functions `strategies/heuristic.py` exports (`code_signal`,
`length_signal(...)`, ...), or one of your own: it takes the request and returns `0.0` to `1.0`.
The reason names a signal by its `name=`, else by its function's own name — which for the closure
`length_signal(...)` returns is just `signal`, so give it one.

What the conditions inherit. Keywords match whole words, and running Chinese, Japanese or Thai
text has no word boundaries to match at, so a keyword in those scripts is a compiled pattern;
`length_signal` counts whitespace-separated words, so the same text reads as a few words however
much it says. A `predicate` on `len(request.text)` is the way round both, and it is a rule like
any other.

**Three combinators, and no more.** `any_of` and `all_of` are what the §4 sentences say ("long
*or* code-bearing"; an image *and* tools). `not_` is the one that could be left out — any use of
it can be rewritten as a higher-priority rule — and it stays because it lets a rule be read
without the rules above it: `not_(any_of(long_request, carries_code))` says "short and simple"
on its own, and the two-rule cost tiering above can be written with both rules that way, in any
order. There are no operators and no nesting syntax to learn, and no other atoms: a condition
that isn't one of these is a `predicate`, and a strategy that needs more is a class (R6).

Nothing is guessed. `when=` takes a condition, never a bare function: a function goes through
`predicate(func)` if it answers yes or no, or `signal_at_least(func, threshold)` if it scores,
because the two can't be told apart by looking at them, and reading one as the other would
route every request the same way.

Which rule wins
---------------
The highest `priority`, and among equals the one declared first. Priority defaults to `0`, so a
list of rules with no priorities is "the first that matches", as `KeywordStrategy` reads its
rules. Priority is there so that the order a rule set is *written* in — assembled from several
places, or grouped by route — need not be the order it is *tried* in. Higher goes first, the
convention of Kubernetes' `PriorityClass` and Traefik's router priority.

Not deciding, and deciding "everything else"
--------------------------------------------
When no rule matches, `decide` returns `None`: it can't decide, and the router falls back to
the default route, with a warning and the reason recorded (R9). That is right for a policy that
covers only what it knows — "code goes to `coder`". It is wrong for one where "everything else"
is a decision, as it is in cost tiering: a simple request answered by the small model is not a
fallback, and shouldn't warn. Say so with a final `Rule(route, always())`, which is then an
ordinary decision with a reason of its own.

The reason
----------
Names the rule and what made it hold (R2): `"rule 'code' matched: keyword 'python'"`,
`"rule 'long or code-bearing' matched: length 0.72 >= 0.50"`. A rule with no `name` goes by its
position in the list, `"rule #2"`. A combination reports the part that decided it — the first
part of an `any_of` that held, every part of an `all_of` — so the reader of a trace sees which
keyword or signal it was, not only which rule.

What is checked, and when
-------------------------
Everything that can be judged from the configuration alone fails at construction, with a
`RoutingError` naming the rule at fault, never once per request (REQ-R6-3): an empty rule set;
a blank route, a bad priority or a duplicate name; an empty `any_of`; an unknown modality; a
blank keyword; a threshold outside `(0, 1]`; an `async` function where a synchronous one is
called; and **a rule that can never fire**, because an earlier one always matches first:

- after an `always()` rule,
- with the same condition as an earlier rule — two rules sending the same requests to different
  routes are in conflict, and one of them is dead whatever the priorities,
- narrower than an earlier rule: `all_of(python, sql)` after `python`, or — the classic mistake
  in tiering — `signal_at_least(s, 0.7)` after `signal_at_least(s, 0.3)`, thresholds out of order.

The check is sound and not complete: it proves a rule dead by structure (a signal or predicate is
the same one only if it is the same object), so an unreachable rule it cannot see is still
accepted, and a rule it reports really is unreachable. It assumes predicates and signals are pure
functions of the request, which they must be anyway — `decide` runs on worker threads (C2).

Route *names* can only be checked against the router's own on a request: a strategy is built
before the router it is given to, so `request.routes` is the first sight of them. The precedent
is `KeywordStrategy`'s, where validating every name up front would have sent every request to
the default route over one mistyped key. Matching therefore comes first here as well: a rule
whose route the router doesn't have returns its choice like any other, and the router reports
it precisely — `chose 'codr', which is not one of the routes` — and falls back (R9). A typo
costs the requests its rule would have taken, and nothing else. Only a rule set in which *no* rule
names a route the router has raises on the request, because that strategy can never decide
anything.

A predicate is user code, so it is checked when it runs: one that raises, or returns anything
but a `bool`, fails the request it ran for and the router falls back (R9). A `bool` is required
rather than "truthy" because the truthy mistakes are silent ones — an un-awaited coroutine, a
match object — that route every request the same way.

The transcript
--------------
`wants_full_context` stays `False`, so a condition sees the current request and `messages` is
`None` (R4). It is a class attribute, which the router reads before it builds the request
(D6), so a configuration can't switch it on — and shouldn't: the opt-in is meant to be
conspicuous. A rule that needs the conversation is a one-line subclass, whose predicates then
find it in `request.messages`:

```python
class WithHistory(ConfigurableStrategy):
    wants_full_context = True
```

The ready-made strategies are not presets
-----------------------------------------
`KeywordStrategy` and `HeuristicStrategy` are left as they are; REQ-R6-3 holds either way.
Tried rather than assumed: a `KeywordStrategy` rebuilt as a subclass of this one matches
identically once its reason is put back in its own words (`"matched keyword 'python'"`), so the
matching is shareable — and is shared, through the whole-word pattern for a keyword and the
signals. What a preset would still have to override is what makes it that strategy: the reason
text, its configuration errors (which are about a mapping of route to keywords) and its
permissiveness (two routes may share a keyword; here the second rule is dead, and refused). That
leaves the saving at one loop. `HeuristicStrategy` is *additive* — signals sum to a score, and a
score falls in a band — which boolean rules do not express, and its reason is the score and its
breakdown. Both stay as they are.

No model, API or network call is made here (R7), and nothing outside the standard library and
`langchain-core` is imported (R8). A predicate or signal *you* write may call one; it is handed
`request` for that reason, and should pass `request.config` on (D9).
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from langchain_llm_router.errors import RoutingError
from langchain_llm_router.strategies.heuristic import Signal
from langchain_llm_router.strategies.keyword import _whole_word, _written
from langchain_llm_router.strategy import RoutingChoice, RoutingRequest, RoutingStrategy

__all__ = [
    "Condition",
    "ConfigurableStrategy",
    "Rule",
    "all_of",
    "always",
    "any_of",
    "keywords",
    "modality",
    "not_",
    "predicate",
    "signal_at_least",
    "tools_bound",
]

_MODALITIES = ("text", "image", "audio", "video", "file", "other")
"""`RoutingRequest.modalities`' closed vocabulary (C7), so a misspelt one fails at construction.

New values arrive with LangChain's content-block types, in a minor release (strategy.py's
stability promise) — and are added here in the same one."""


# --- Conditions ---


class Condition(ABC):
    """When a rule applies: a question about one request, with an answer that says why (R2).

    Build one with the functions in this module — `keywords`, `signal_at_least`, `modality`,
    `tools_bound`, `predicate`, `always` — and combine them with `all_of`, `any_of` and `not_`.
    `predicate` is the extension point; this class is here so a `Rule` can say what it takes,
    not to be subclassed.
    """

    @abstractmethod
    def explain(self, request: RoutingRequest) -> str | None:
        """Why this condition holds for `request` — or `None` when it doesn't."""

    @abstractmethod
    def __str__(self) -> str:
        """The condition as a reader of the configuration would say it."""

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True, repr=False)
class _Keyword(Condition):
    pattern: re.Pattern[str]
    label: str = field(compare=False)
    words: tuple[str, ...] | None = field(default=None, compare=False)
    """The words of a string keyword — `None` for a compiled pattern, which says nothing of them."""

    def explain(self, request: RoutingRequest) -> str | None:
        return self.label if self.pattern.search(request.text) else None

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True, repr=False)
class _SignalAtLeast(Condition):
    signal: Signal
    threshold: float
    name: str = field(compare=False)

    def explain(self, request: RoutingRequest) -> str | None:
        strength = self.signal(request)
        if not isinstance(strength, (int, float)):
            msg = (
                f"signal {self.name!r} returned {type(strength).__name__}; a signal returns a float"
            )
            raise TypeError(msg)
        if strength < self.threshold:
            return None
        return f"{self.name} {strength:.2f} >= {self.threshold:.2f}"

    def __str__(self) -> str:
        return f"{self.name} >= {self.threshold:g}"


@dataclass(frozen=True, repr=False)
class _Modality(Condition):
    names: tuple[str, ...]

    def explain(self, request: RoutingRequest) -> str | None:
        present = next((name for name in self.names if name in request.modalities), None)
        return f"has {present}" if present is not None else None

    def __str__(self) -> str:
        return "has " + " or ".join(self.names)


@dataclass(frozen=True, repr=False)
class _ToolsBound(Condition):
    def explain(self, request: RoutingRequest) -> str | None:
        return "tools bound" if request.tools_bound else None

    def __str__(self) -> str:
        return "tools bound"


@dataclass(frozen=True, repr=False)
class _Predicate(Condition):
    func: Callable[[RoutingRequest], bool]
    name: str = field(compare=False)

    def explain(self, request: RoutingRequest) -> str | None:
        outcome = self.func(request)
        if not isinstance(outcome, bool):
            msg = (
                f"predicate {self.name!r} returned {type(outcome).__name__}; "
                "a predicate returns a bool"
            )
            raise TypeError(msg)
        return self.name if outcome else None

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, repr=False)
class _Always(Condition):
    def explain(self, request: RoutingRequest) -> str | None:
        return "always"

    def __str__(self) -> str:
        return "always"


@dataclass(frozen=True, repr=False)
class _AllOf(Condition):
    conditions: tuple[Condition, ...]

    def explain(self, request: RoutingRequest) -> str | None:
        reasons: list[str] = []
        for condition in self.conditions:
            reason = condition.explain(request)
            if reason is None:
                return None
            reasons.append(reason)
        return " and ".join(reasons)

    def __str__(self) -> str:
        return "(" + " and ".join(str(condition) for condition in self.conditions) + ")"


@dataclass(frozen=True, repr=False)
class _AnyOf(Condition):
    conditions: tuple[Condition, ...]

    def explain(self, request: RoutingRequest) -> str | None:
        for condition in self.conditions:
            reason = condition.explain(request)
            if reason is not None:
                return reason
        return None

    def __str__(self) -> str:
        return "(" + " or ".join(str(condition) for condition in self.conditions) + ")"


@dataclass(frozen=True, repr=False)
class _Not(Condition):
    condition: Condition

    def explain(self, request: RoutingRequest) -> str | None:
        return None if self.condition.explain(request) is not None else str(self)

    def __str__(self) -> str:
        return f"not {self.condition}"


def keywords(*words: str | re.Pattern[str]) -> Condition:
    """Holds when any of the words appears in the request's text — as `KeywordStrategy` reads it.

    Whole words, ignoring case; a compiled pattern is searched exactly as compiled.
    """
    if not words:
        msg = "keywords() needs at least one keyword or compiled pattern"
        raise RoutingError(msg)
    found = tuple(_keyword(word) for word in words)
    return found[0] if len(found) == 1 else _AnyOf(found)


def signal_at_least(signal: Signal, threshold: float, *, name: str | None = None) -> Condition:
    """Holds when `signal` scores the request at `threshold` or above (`0 < threshold <= 1`)."""
    _require_sync("signal", signal)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 < threshold <= 1.0
    ):
        msg = (
            f"signal_at_least needs a threshold above 0 and at most 1, got {threshold!r}: "
            "a signal never scores above 1.0, and one that always reaches 0.0 is `always()`"
        )
        raise RoutingError(msg)
    return _SignalAtLeast(signal, threshold, _named(signal, name))


def modality(*names: str) -> Condition:
    """Holds when the request carries any of these modalities (C7), such as `"image"`."""
    if not names:
        msg = "modality() needs at least one modality"
        raise RoutingError(msg)
    unknown = [name for name in names if name not in _MODALITIES]
    if unknown:
        msg = f"unknown modality {unknown[0]!r}: the modalities are {_quoted(_MODALITIES)}"
        raise RoutingError(msg)
    return _Modality(names)


def tools_bound() -> Condition:
    """Holds when tools or structured output are bound to the call (`request.tools_bound`)."""
    return _ToolsBound()


def predicate(func: Callable[[RoutingRequest], bool], *, name: str | None = None) -> Condition:
    """Holds when `func(request)` is `True` — the escape hatch for anything else."""
    _require_sync("predicate", func)
    return _Predicate(func, _named(func, name))


def always() -> Condition:
    """Holds for every request: the rule that answers when nothing above it did."""
    return _Always()


def all_of(*conditions: Condition) -> Condition:
    """Holds when every condition does; they are tried in the order given, and stop at the first."""
    return _combined("all_of", "for every request", conditions, _AllOf)


def any_of(*conditions: Condition) -> Condition:
    """Holds when any condition does; they are tried in the order given, and stop at the first."""
    return _combined("any_of", "for no request", conditions, _AnyOf)


def not_(condition: Condition) -> Condition:
    """Holds when `condition` does not."""
    _require_conditions("not_", (condition,))
    return _Not(condition)


def _combined(
    kind: str,
    when_empty: str,
    conditions: tuple[Condition, ...],
    combine: Callable[[tuple[Condition, ...]], Condition],
) -> Condition:
    if not conditions:
        msg = f"{kind}() has no conditions, so it would hold {when_empty}"
        raise RoutingError(msg)
    _require_conditions(kind, conditions)
    return conditions[0] if len(conditions) == 1 else combine(conditions)


def _require_conditions(kind: str, conditions: Sequence[object]) -> None:
    for condition in conditions:
        if not isinstance(condition, Condition):
            msg = (
                f"{kind}() takes conditions, not {type(condition).__name__}: give several as "
                "separate arguments, and wrap a function in predicate(...)"
            )
            raise RoutingError(msg)


def _keyword(word: object) -> _Keyword:
    if isinstance(word, re.Pattern):
        if not isinstance(word.pattern, str):
            msg = (
                f"the pattern {word.pattern!r} is compiled from bytes: a request's text "
                "is a string, so compile the pattern from one"
            )
            raise RoutingError(msg)
        return _Keyword(word, f"pattern {_written(word)}")
    if not isinstance(word, str):
        msg = (
            f"the keyword {word!r} is {type(word).__name__}: a keyword is a string, or a "
            "compiled regular expression — give several as separate arguments"
        )
        raise RoutingError(msg)
    if not word.strip():
        msg = f"the keyword {word!r} is blank: a keyword needs a word"
        raise RoutingError(msg)
    wanted = word.strip()
    return _Keyword(
        re.compile(_whole_word(wanted), re.IGNORECASE), f"keyword {wanted!r}", tuple(wanted.split())
    )


def _require_sync(kind: str, func: object) -> None:
    if not callable(func):
        msg = f"the {kind} is {type(func).__name__}: it must be a function"
        raise RoutingError(msg)
    # The type's `__call__` as well, as a call looks it up: a callable object may be `async def`.
    if inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(type(func).__call__):
        msg = (
            f"the {kind} {_named(func, None)!r} is async: it must be a plain function, "
            "because a strategy calls it while it decides"
        )
        raise RoutingError(msg)


def _named(func: object, name: str | None) -> str:
    """What a reason calls `func`: the name it was given, or the function's own."""
    if name is None:
        own = getattr(func, "__name__", None)
        return own if isinstance(own, str) else type(func).__name__
    if not isinstance(name, str) or not name.strip():
        msg = f"name must be a non-blank string, got {name!r}"
        raise RoutingError(msg)
    return name


def _quoted(names: Sequence[str]) -> str:
    return ", ".join(repr(name) for name in names)


# --- Rules ---


@dataclass(frozen=True)
class Rule:
    """One rule: the route a request goes to when `when` holds, and how the rule ranks.

    Args:
        route: The route name that answers.
        when: The condition, built from the functions above.
        priority: Higher is tried first; rules of equal priority are tried in the order they
            were declared. Defaults to `0`.
        name: What the decision's reason calls the rule. Unnamed rules go by their position.
    """

    route: str
    when: Condition
    priority: int = field(default=0, kw_only=True)
    name: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.route, str) or not self.route.strip():
            msg = f"the route {self.route!r} is not a route name: a rule names the route it serves"
            raise RoutingError(msg)
        if not isinstance(self.when, Condition):
            msg = (
                f"when= is {type(self.when).__name__}, not a condition: build one with keywords(), "
                "signal_at_least(), modality(), tools_bound(), predicate(), always(), "
                "all_of(), any_of() or not_()"
            )
            raise RoutingError(msg)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            msg = f"priority must be an int, got {self.priority!r}"
            raise RoutingError(msg)
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            msg = f"name must be a non-blank string, got {self.name!r}"
            raise RoutingError(msg)


# --- The strategy ---


class ConfigurableStrategy(RoutingStrategy):
    """Routes on rules of conditions, priorities and names (R6, R7) — the module docstring has them.

    ```python
    ChatRouter(
        routes={"small": small, "frontier": frontier},
        default_route="small",
        strategy=ConfigurableStrategy(
            [
                Rule("frontier", any_of(long_request, carries_code), name="long or code-bearing"),
                Rule("small", always(), name="short and simple"),
            ]
        ),
    )
    ```

    The highest-priority rule whose condition holds decides, and its reason names it and what
    made it hold. When none does, the strategy returns `None` and the default route answers (R9).

    A strategy is immutable once built: the rules are checked and ordered in `__init__` and never
    touched again, so `decide` is thread-safe as the interface requires — provided the
    predicates and signals it was given are — and one instance can serve several routers.

    Args:
        rules: The rules, in declaration order; at least one.

    Raises:
        RoutingError: for configuration that could never route — an empty rule set, a rule that
            can never fire because an earlier one always matches first. Raised here, at
            construction, not once per request; the module docstring lists them.
    """

    rules: tuple[Rule, ...]
    """The rules, in the order they were declared — not the order they are tried in."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = _checked_rules(rules)
        self._ordered = _evaluation_order(self.rules)
        _reject_dead_rules(self._ordered)

    def __repr__(self) -> str:
        return f"ConfigurableStrategy({list(self.rules)!r})"

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """The route of the first rule that holds, in priority order, or `None` if none does (R9).

        A rule that holds is answered with even when the router has no such route: the router
        says so far better than this can, and one mistyped rule must not take the rest down
        with it (the module docstring has the reasoning).
        """
        self._check_it_can_decide(request.routes)
        for label, rule in self._ordered:
            reason = rule.when.explain(request)
            if reason is not None:
                return RoutingChoice(route=rule.route, reason=f"rule {label} matched: {reason}")
        return None

    def _check_it_can_decide(self, routes: tuple[str, ...]) -> None:
        """At least one rule names a route the router has — the check only a request can make.

        A rule set that names none of them has no answer it could give, whatever the request.
        Raising says so on the first call; the router records it and falls back (R9), where
        abstaining would look like a request nothing happened to match.
        """
        if any(rule.route in routes for rule in self.rules):
            return
        named = dict.fromkeys(rule.route for rule in self.rules)
        msg = (
            f"no rule names a route this router has: the rules name {_quoted(list(named))}; "
            f"the routes are {_quoted(routes)}"
        )
        raise RoutingError(msg)


_Entry: TypeAlias = "tuple[str, Rule]"
"""A rule and what the strategy calls it: its name in quotes, or its position."""


def _checked_rules(rules: Sequence[Rule]) -> tuple[Rule, ...]:
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        msg = (
            f"rules is {type(rules).__name__}: a ConfigurableStrategy takes a list of Rule, "
            "such as [Rule('coder', keywords('python'))]"
        )
        raise RoutingError(msg)
    if not rules:
        msg = "rules is empty: a ConfigurableStrategy needs at least one rule"
        raise RoutingError(msg)
    for position, rule in enumerate(rules, start=1):
        if not isinstance(rule, Rule):
            msg = (
                f"rule #{position} is {type(rule).__name__}: every rule is a Rule(route, when=...)"
            )
            raise RoutingError(msg)
    names = [rule.name for rule in rules if rule.name is not None]
    repeated = next((name for name in names if names.count(name) > 1), None)
    if repeated is not None:
        msg = (
            f"two rules are named {repeated!r}: a name says which rule a decision's reason "
            "means, so each must be unique"
        )
        raise RoutingError(msg)
    return tuple(rules)


def _evaluation_order(rules: tuple[Rule, ...]) -> tuple[_Entry, ...]:
    """The rules as they are tried: highest priority first, declaration order within a priority."""
    entries = [
        (repr(rule.name) if rule.name is not None else f"#{position}", rule)
        for position, rule in enumerate(rules, start=1)
    ]
    return tuple(sorted(entries, key=lambda entry: -entry[1].priority))


def _reject_dead_rules(ordered: tuple[_Entry, ...]) -> None:
    """A rule that an earlier one always answers first can never fire — so it is a mistake."""
    for index, (label, rule) in enumerate(ordered):
        for earlier_label, earlier in ordered[:index]:
            if _implies(rule.when, earlier.when):
                raise RoutingError(_dead_rule(label, rule, earlier_label, earlier))


def _dead_rule(label: str, rule: Rule, earlier_label: str, earlier: Rule) -> str:
    """Why `rule` can never fire, and what to do about it."""
    what = f"rule {label} (for {rule.route!r}) can never fire: rule {earlier_label} is tried first"
    if isinstance(earlier.when, _Always):
        return f"{what} and matches every request"
    if _implies(earlier.when, rule.when):
        # The same condition, said differently or not: no priority can save both.
        if earlier.route == rule.route:
            return f"{what} and has the same condition, so one of the two is redundant"
        return (
            f"{what} and has the same condition, but sends its requests to {earlier.route!r}: "
            "the rules conflict, and one of them has to go"
        )
    if earlier.route == rule.route:
        return f"{what}, matches every request it does, and sends them to the same route"
    return (
        f"{what}, matches every request it does, and sends them to {earlier.route!r} instead — "
        "give this rule a higher priority, or narrow the other"
    )


def _implies(narrow: Condition, wide: Condition) -> bool:
    """Whether every request `narrow` holds for, `wide` holds for as well.

    Sound, not complete: `True` is a proof, `False` is "not shown" — so a rule is only ever
    called dead when it is.
    """
    if isinstance(wide, _Always) or narrow == wide:
        return True
    # These two lose nothing, so they go first: `a or b` implies `w` exactly when both parts do,
    # and `n` implies `x and y` exactly when it implies both.
    if isinstance(narrow, _AnyOf):
        return all(_implies(part, wide) for part in narrow.conditions)
    if isinstance(wide, _AllOf):
        return all(_implies(narrow, part) for part in wide.conditions)
    if isinstance(narrow, _AllOf) and any(_implies(part, wide) for part in narrow.conditions):
        return True
    if isinstance(wide, _AnyOf) and any(_implies(narrow, part) for part in wide.conditions):
        return True
    if isinstance(narrow, _Keyword) and isinstance(wide, _Keyword):
        return _contains_run(narrow.words, wide.words)
    if isinstance(narrow, _SignalAtLeast) and isinstance(wide, _SignalAtLeast):
        return narrow.signal == wide.signal and narrow.threshold >= wide.threshold
    if isinstance(narrow, _Modality) and isinstance(wide, _Modality):
        return set(narrow.names) <= set(wide.names)
    if isinstance(narrow, _Not) and isinstance(wide, _Not):
        return _implies(wide.condition, narrow.condition)
    return False


def _contains_run(words: tuple[str, ...] | None, run: tuple[str, ...] | None) -> bool:
    """Whether the words of one string keyword contain the other's as a run: `unit test` has `test`.

    Every request that has the longer keyword's words has the shorter's, whole words as they are
    matched — so a rule for `test` above a rule for `unit test` leaves the second nothing to do.
    Compiled patterns say nothing about their words, so they never imply anything but themselves.
    """
    if words is None or run is None:
        return False
    return any(words[start : start + len(run)] == run for start in range(len(words) - len(run) + 1))
