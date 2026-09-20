r"""Keyword routing (R6's ready-made level, R7): words in the request name the route.

The fast start for domain routing (PRD §4) — "code requests to the code-strong model" is one
line:

    strategy=KeywordStrategy({"coder": ["python", "regex", "stack trace"]})

**What is matched.** The current request's text, and nothing else (R4): `request.text`, what the
user has just asked. A request with no text — an image on its own — matches no rule, so the
default route answers (R9); a list of keywords has nothing to say about a picture, which is why
`modalities` plays no part here.

**How a keyword matches.** As a whole word, ignoring case: `"python"` matches `"Python?"` and
`"in python,"` but not `"pythonic"`. Substring matching is the tempting default and the wrong
one — `"ai"` would fire on `"said"` and `"go"` on `"ago"`, and a rule set that misroutes is worse
than no rule set. A keyword of several words (`"stack trace"`) matches them literally, in order,
separated by the single space it is written with. A keyword that begins or ends in punctuation
keeps that edge open, so `"c++"` matches `"c++"` and `".net"` matches `"in .net"`.

**Regular expressions are the opt-in.** Pass a compiled pattern where a keyword goes, and it is
searched exactly as compiled: `re.compile(r"\bdef\s+\w+\(")`. Compiling *is* the opt-in — no flag
declares that a string is a pattern, and no string is ever guessed at, so a keyword like `"c++"`
can never be read as a regular expression. Flags are the pattern's own, case sensitivity
included: `re.compile("SQL")` matches `"SQL"` but not `"sql"`, `re.compile("SQL", re.IGNORECASE)`
matches both. Patterns are also the way past whole-word matching — `re.compile(r"regexe?s?")`
catches "regexes" as well.

**Which rule wins.** The first that matches, in the order the rules are written: route by route,
and within a route keyword by keyword. Declaration order is what the router already reads route
order as (D1), it is the order a reader of the rules sees, and unlike "most matches" or "longest
keyword" it does not change with the wording of a request. So put the specific rules first:
`{"coder": ["unit test"], "small": ["test"]}` sends "write a unit test" to `coder`, and the same
two rules the other way round send it to `small`.

**What is checked, and when.** The rules themselves at construction — a blank keyword, a route
with no keywords, an empty rule set — because a strategy that can never decide should not reach
a request. Rules are checked against the *router's* route names on each request instead: a
strategy is built before the router it is given to, so `request.routes` is the first sight of
them (R6). A rule naming a route that doesn't exist then raises, rather than matching into the
void: the router records the reason and falls back with its warning (R9), which is how a typo in
a rule gets noticed.

No model, API, embedding or network call is made here or anywhere below (R7): matching is `re`
over a string. Nothing outside the standard library and `langchain-core` is imported (R8).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from langchain_llm_router.errors import RoutingError
from langchain_llm_router.strategy import RoutingChoice, RoutingRequest, RoutingStrategy

__all__ = ["KeywordStrategy"]

Keyword: TypeAlias = "str | re.Pattern[str]"
"""What a rule looks for: a keyword matched as a whole word, or a pattern used as compiled."""

Keywords: TypeAlias = "Keyword | Iterable[Keyword]"
"""One route's keywords — a single one need not be wrapped in a list."""


@dataclass(frozen=True)
class _Rule:
    """One keyword or pattern, the route it names, and the reason a match writes (R2)."""

    route: str
    pattern: re.Pattern[str]
    reason: str


class KeywordStrategy(RoutingStrategy):
    """Routes on the words of the current request (R6, R7) — the module docstring has the rules.

    Rules map a route name to the keywords and patterns that send a request to it:

        KeywordStrategy(
            {
                "coder": ["python", "regex", "stack trace", re.compile(r"```")],
                "frontier": ["prove", "proof", "derive"],
            }
        )

    The first rule that matches wins, in declaration order, and the reason on the decision
    record names it — `"matched keyword 'python'"` — so a trace says what the router saw. When
    nothing matches, the strategy returns `None` and the default route answers (R9).

    A strategy is immutable once built: the rules are compiled in `__init__` and never touched
    again, so `decide` is thread-safe, as the interface requires, and one instance can serve
    several routers.
    """

    def __init__(self, rules: Mapping[str, Keywords]) -> None:
        """Compile `rules`, raising `RoutingError` for a rule set that could never decide."""
        if not rules:
            msg = "rules is empty: a KeywordStrategy needs at least one rule"
            raise RoutingError(msg)
        compiled: list[_Rule] = []
        for route, keywords in rules.items():
            if not route.strip():
                msg = f"the rule key {route!r} is blank: a rule names the route it routes to"
                raise RoutingError(msg)
            compiled += [_rule(route, keyword) for keyword in _listed(route, keywords)]
        self._rules = tuple(compiled)

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """The route named by the first rule that matches, or `None` if none does (R9)."""
        self._check_routes(request.routes)
        for rule in self._rules:
            if rule.pattern.search(request.text):
                return RoutingChoice(route=rule.route, reason=rule.reason)
        return None

    def _check_routes(self, routes: tuple[str, ...]) -> None:
        """Every rule names a route the router has — the one check only a request can make.

        Raising beats matching nothing: the router turns it into a recorded reason and one
        `FallbackWarning` per request (R9), so a rule naming a route that was renamed or never
        existed is visible on the first call rather than never.
        """
        unknown = dict.fromkeys(rule.route for rule in self._rules if rule.route not in routes)
        if unknown:
            msg = (
                f"rules name routes that don't exist: {_names(unknown)}; "
                f"the routes are {_names(routes)}"
            )
            raise RoutingError(msg)


def _listed(route: str, keywords: Keywords) -> list[object]:
    """One route's keywords as a list, whether it was given one keyword or many."""
    if isinstance(keywords, (str, re.Pattern)):
        # A lone keyword is not a list of one-character keywords: `{"coder": "python"}` would
        # otherwise route on the letters of "python", matching almost every request.
        listed: list[object] = [keywords]
    elif isinstance(keywords, Iterable):
        listed = list(keywords)
    else:
        msg = (
            f"route {route!r} is given {type(keywords).__name__}: a rule takes a keyword, "
            "a compiled regular expression, or a list of them"
        )
        raise RoutingError(msg)
    if not listed:
        msg = f"route {route!r} has no keywords: a rule needs something to match"
        raise RoutingError(msg)
    return listed


def _rule(route: str, keyword: object) -> _Rule:
    """One compiled rule, and the reason a match writes (R2).

    `keyword` is typed `object` because this is where untyped callers land: a list of anything
    passes as `Iterable[Keyword]` when nothing type-checks it, and a clear error here beats an
    `AttributeError` from inside `decide` on the first request.
    """
    if isinstance(keyword, re.Pattern):
        if not isinstance(keyword.pattern, str):
            msg = (
                f"the pattern for route {route!r} is compiled from bytes: a request's text "
                "is a string, so compile the pattern from one"
            )
            raise RoutingError(msg)
        return _Rule(route, keyword, f"matched pattern {keyword.pattern!r}")
    if not isinstance(keyword, str):
        msg = (
            f"the keyword {keyword!r} for route {route!r} is {type(keyword).__name__}: "
            "a keyword is a string, or a compiled regular expression"
        )
        raise RoutingError(msg)
    if not keyword.strip():
        msg = f"the keyword {keyword!r} for route {route!r} is blank: a keyword needs a word"
        raise RoutingError(msg)
    return _Rule(
        route, re.compile(_whole_word(keyword), re.IGNORECASE), f"matched keyword {keyword!r}"
    )


def _whole_word(keyword: str) -> str:
    r"""`keyword` as a pattern that matches it as a whole word.

    The `\b` goes only where the keyword's own edge is a word character. `\bc++\b` would never
    match anything: `\b` after `+` asks for a word character next to a non-word one.
    """
    start = r"\b" if _is_word_char(keyword[0]) else ""
    end = r"\b" if _is_word_char(keyword[-1]) else ""
    return f"{start}{re.escape(keyword)}{end}"


def _is_word_char(char: str) -> bool:
    r"""Whether `\b` would see a word boundary beside `char` — `\w` decides, as it does in `\b`."""
    return re.fullmatch(r"\w", char) is not None


def _names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)
