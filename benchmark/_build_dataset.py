"""One-off generator for `data/workload.json`. Not part of the benchmark's runtime; run by hand
after editing the item list below, then delete or keep for the next edit. Kept out of `__all__`
of nothing, since this module exports nothing — it is a script.
"""

from __future__ import annotations

import json
from pathlib import Path

AUTHORED = "author-written for T-140; original phrasing, no verbatim external source"

ITEMS = [
    # --- support: customer-support policy and account questions ---
    dict(
        id="support-easy-st-1",
        domain="support",
        difficulty="easy",
        kind="single_turn",
        prompt="How long does a refund take to show up after it's approved?",
        rubric=(
            "States refunds post to the original payment method roughly 5-7 business days "
            "after approval; no invented policy details, no more than a sentence or two."
        ),
        provenance=f"{AUTHORED}; general customer-support policy question.",
    ),
    dict(
        id="support-easy-st-2",
        domain="support",
        difficulty="easy",
        kind="single_turn",
        prompt="What's the fastest way to reset my password if I'm locked out of my account?",
        rubric=(
            "Gives sensible, generic guidance (use the 'forgot password' / email reset link, "
            "or contact support); no fabricated URLs or internal tool names presented as fact."
        ),
        provenance=f"{AUTHORED}; general account-support question.",
    ),
    dict(
        id="support-hard-st-1",
        domain="support",
        difficulty="hard",
        kind="single_turn",
        prompt=(
            "My order #A1002 hasn't shipped after 2 weeks, my last refund never posted, and "
            "now I'm being charged twice for the same subscription — can you explain what's "
            "going on and what I should do about each of these, one at a time?"
        ),
        rubric=(
            "Addresses all three issues separately (late shipment, missing refund, double "
            "charge) rather than one generic reply; suggests a concrete next step for each; "
            "doesn't claim to know the account's actual state."
        ),
        provenance=f"{AUTHORED}; multi-part support complaint, patterned after common "
        "customer-support tickets (no verbatim reuse).",
    ),
    dict(
        id="support-hard-st-2",
        domain="support",
        difficulty="hard",
        kind="single_turn",
        prompt=(
            "Write a short policy explaining our refund, retry and caching behaviour for a "
            "technical partner integrating with our API, comparing how each interacts with "
            "idempotency."
        ),
        rubric=(
            "Correctly explains refunds, retries and caching as distinct concepts and connects "
            "each to idempotency (e.g. retries need an idempotent endpoint, caching must key on "
            "enough context not to serve a stale/wrong response); coherent structure, not just "
            "three disconnected sentences."
        ),
        provenance=f"{AUTHORED}; technical-writing/comparison task.",
    ),
    dict(
        id="support-easy-agent-1",
        domain="support",
        difficulty="easy",
        kind="agent",
        prompt="What's the status of order A1001?",
        rubric=(
            "Calls lookup_order_status with order_id 'A1001' and reports back that it shipped "
            "2026-09-18, arriving 2026-09-24, without inventing different details."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.lookup_order_status.",
        expected_tool="lookup_order_status",
    ),
    dict(
        id="support-easy-agent-2",
        domain="support",
        difficulty="easy",
        kind="agent",
        prompt="Can you check on order A1003 for me?",
        rubric=(
            "Calls lookup_order_status with order_id 'A1003' and reports it was delivered "
            "2026-09-15, without inventing different details."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.lookup_order_status.",
        expected_tool="lookup_order_status",
    ),
    dict(
        id="support-hard-agent-1",
        domain="support",
        difficulty="hard",
        kind="agent",
        prompt=(
            "A partner is asking whether failed requests get retried automatically and how "
            "many times — look it up and explain it to them in a paragraph, comparing it to "
            "what would happen if we retried indefinitely."
        ),
        rubric=(
            "Calls search_docs for the retry policy, reports the '3 retries with exponential "
            "backoff on 429/5xx' fact accurately, and adds a comparison to indefinite retries "
            "(e.g. cascading load, no backpressure) rather than just restating the doc."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.search_docs.",
        expected_tool="search_docs",
    ),
    dict(
        id="support-hard-agent-2",
        domain="support",
        difficulty="hard",
        kind="agent",
        prompt=(
            "I need to explain to a customer why two routes never leak into each other's "
            "cached responses — look up how our caching keys work and write a two-sentence "
            "explanation for a non-technical reader."
        ),
        rubric=(
            "Calls search_docs for the cache-key fact, reports that the cache key includes the "
            "route name, and translates that into plain language in roughly two sentences."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.search_docs.",
        expected_tool="search_docs",
    ),
    # --- coder: programming help ---
    dict(
        id="coder-easy-st-1",
        domain="coder",
        difficulty="easy",
        kind="single_turn",
        prompt="What does the Python `%` operator do?",
        rubric="Correctly describes it as the modulo/remainder operator, with a short example.",
        provenance=f"{AUTHORED}; general Python-knowledge question.",
    ),
    dict(
        id="coder-easy-st-2",
        domain="coder",
        difficulty="easy",
        kind="single_turn",
        prompt="In one sentence, what's the difference between a list and a tuple in Python?",
        rubric="Correctly states lists are mutable and tuples are immutable, in about a sentence.",
        provenance=f"{AUTHORED}; general Python-knowledge question.",
    ),
    dict(
        id="coder-hard-st-1",
        domain="coder",
        difficulty="hard",
        kind="single_turn",
        prompt=(
            "I'm getting this error and don't understand why:\n\n"
            "```\n"
            "Traceback (most recent call last):\n"
            '  File "app.py", line 12, in <module>\n'
            "    total = sum(prices) / len(items)\n"
            "ZeroDivisionError: division by zero\n"
            "```\n\n"
            "What's causing it, and how should I fix it?"
        ),
        rubric=(
            "Identifies that `items` (the divisor) is empty, causing the division by zero; "
            "suggests a guard such as checking the length before dividing or handling the "
            "empty case; doesn't invent an unrelated cause."
        ),
        provenance=f"{AUTHORED}; debugging task with a synthetic traceback, patterned after "
        "common Python ZeroDivisionError questions (no verbatim reuse).",
    ),
    dict(
        id="coder-hard-st-2",
        domain="coder",
        difficulty="hard",
        kind="single_turn",
        prompt=(
            "Refactor this function to avoid the repeated database call inside the loop, and "
            "explain the trade-offs of your approach:\n\n"
            "```python\n"
            "def get_order_totals(order_ids):\n"
            "    totals = []\n"
            "    for oid in order_ids:\n"
            "        order = db.fetch_order(oid)  # one query per iteration\n"
            "        totals.append(order.total)\n"
            "    return totals\n"
            "```"
        ),
        rubric=(
            "Proposes batching the fetch into a single call (e.g. db.fetch_orders(order_ids)) "
            "instead of one query per iteration, and names at least one real trade-off (memory, "
            "partial failure handling, query complexity); doesn't just restate the original."
        ),
        provenance=f"{AUTHORED}; refactoring task with a synthetic N+1-query snippet.",
    ),
    dict(
        id="coder-easy-agent-1",
        domain="coder",
        difficulty="easy",
        kind="agent",
        prompt="Does this snippet have a syntax error? `def f(x):\\n    return x +`",
        rubric=(
            "Calls lint_code on the snippet and reports the syntax error it finds, rather than "
            "guessing without checking."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.lint_code.",
        expected_tool="lint_code",
    ),
    dict(
        id="coder-easy-agent-2",
        domain="coder",
        difficulty="easy",
        kind="agent",
        prompt="Check this for syntax errors: `print('hello'`",
        rubric=(
            "Calls lint_code on the snippet and reports the unclosed-parenthesis syntax error "
            "it finds."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.lint_code.",
        expected_tool="lint_code",
    ),
    dict(
        id="coder-hard-agent-1",
        domain="coder",
        difficulty="hard",
        kind="agent",
        prompt=(
            "I'm estimating Big-O cost: if I run a nested loop that's O(n^2) with n=1500, "
            "roughly how many operations is that? Compute it, then explain what that means for "
            "a 50ms-per-operation budget."
        ),
        rubric=(
            "Calls calculator with an expression equivalent to 1500**2 (or 1500*1500), reports "
            "2,250,000 operations, and gives a sane follow-on remark about a 50ms-per-operation "
            "budget being far too slow at that scale."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.calculator.",
        expected_tool="calculator",
    ),
    dict(
        id="coder-hard-agent-2",
        domain="coder",
        difficulty="hard",
        kind="agent",
        prompt=(
            "My cloud bill line item says $0.0000166667 per GB-second, and I used 0.5 GB for "
            "3600 seconds, 30 times this month — what's the total cost? Compute it precisely."
        ),
        rubric=(
            "Calls calculator with an expression equivalent to "
            "0.0000166667 * 0.5 * 3600 * 30, and reports a total close to $0.90."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.calculator.",
        expected_tool="calculator",
    ),
    # --- research: writing, summarization and analysis ---
    dict(
        id="research-easy-st-1",
        domain="research",
        difficulty="easy",
        kind="single_turn",
        prompt="In one sentence, what is photosynthesis?",
        rubric="Correctly describes plants converting light, water and CO2 into energy/glucose.",
        provenance=f"{AUTHORED}; general-knowledge science question.",
    ),
    dict(
        id="research-easy-st-2",
        domain="research",
        difficulty="easy",
        kind="single_turn",
        prompt=(
            "Summarize in one sentence: 'The mitochondria is the powerhouse of the cell, "
            "converting nutrients into ATP through cellular respiration.'"
        ),
        rubric="Produces a faithful one-sentence summary that keeps the mitochondria/ATP fact.",
        provenance=f"{AUTHORED}; one-sentence summarization task.",
    ),
    dict(
        id="research-hard-st-1",
        domain="research",
        difficulty="hard",
        kind="single_turn",
        prompt=(
            "Compare and contrast supervised and unsupervised learning, and explain why the "
            "choice matters for a team with mostly unlabeled data."
        ),
        rubric=(
            "Correctly distinguishes labeled vs. unlabeled training data between the two "
            "approaches, and explains that a team with mostly unlabeled data leans toward "
            "unsupervised methods (or labeling investment); not just a definitions dump."
        ),
        provenance=f"{AUTHORED}; compare/contrast analysis task, general ML knowledge.",
    ),
    dict(
        id="research-hard-st-2",
        domain="research",
        difficulty="hard",
        kind="single_turn",
        prompt=(
            "Critique the following argument for its logical soundness: 'Every product we've "
            "shipped this year has grown revenue, therefore our next product will too.' "
            "Explain the flaw and suggest a better way to reason about it."
        ),
        rubric=(
            "Names the flaw (a small/biased sample, survivorship bias, or "
            "no-guarantee-from-past-success reasoning) and suggests a more sound approach "
            "(e.g. base rates, testing, considering counterexamples)."
        ),
        provenance=f"{AUTHORED}; informal-logic critique task, original argument.",
    ),
    dict(
        id="research-easy-agent-1",
        domain="research",
        difficulty="easy",
        kind="agent",
        prompt="Quick refund policy check — how long do refunds take?",
        rubric=(
            "Calls search_docs and reports the '5-7 business days to the original payment "
            "method' fact it finds."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.search_docs.",
        expected_tool="search_docs",
    ),
    dict(
        id="research-easy-agent-2",
        domain="research",
        difficulty="easy",
        kind="agent",
        prompt="What does our SDK do when a request gets a 429?",
        rubric=(
            "Calls search_docs and reports the 'retries up to 3 times with exponential "
            "backoff' fact it finds."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.search_docs.",
        expected_tool="search_docs",
    ),
    dict(
        id="research-hard-agent-1",
        domain="research",
        difficulty="hard",
        kind="agent",
        prompt=(
            "For a blog post about our reliability story, look up how we handle transient "
            "failures and write two sentences explaining it to a general audience, comparing "
            "it to just failing immediately."
        ),
        rubric=(
            "Calls search_docs, accurately reflects the retry-with-backoff fact, and contrasts "
            "it with failing immediately in plain, non-technical language."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.search_docs.",
        expected_tool="search_docs",
    ),
    dict(
        id="research-hard-agent-2",
        domain="research",
        difficulty="hard",
        kind="agent",
        prompt=(
            "I'm documenting cache correctness for a security review — look up how our cache "
            "keys avoid cross-route leakage and explain why that specifically matters for two "
            "customers on different tiers."
        ),
        rubric=(
            "Calls search_docs, accurately reflects the route-name-in-cache-key fact, and "
            "explains why that prevents one customer's cached answer (from a different tier's "
            "model) leaking to another."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.search_docs.",
        expected_tool="search_docs",
    ),
    # --- math: arithmetic and quantitative reasoning ---
    dict(
        id="math-easy-st-1",
        domain="math",
        difficulty="easy",
        kind="single_turn",
        prompt="What is 17 + 26?",
        rubric="Answers 43.",
        provenance=f"{AUTHORED}; trivial arithmetic.",
    ),
    dict(
        id="math-easy-st-2",
        domain="math",
        difficulty="easy",
        kind="single_turn",
        prompt=(
            "If a train travels 60 miles in 1 hour, how far does it travel in 3 hours at the "
            "same speed?"
        ),
        rubric="Answers 180 miles, with the reasoning shown or implied.",
        provenance=f"{AUTHORED}; simple rate word problem, patterned after common grade-school "
        "arithmetic word problems (no verbatim reuse).",
    ),
    dict(
        id="math-hard-st-1",
        domain="math",
        difficulty="hard",
        kind="single_turn",
        prompt=(
            "A store marks up an item by 40% then offers a 25% discount off the marked-up "
            "price. Is the final price higher or lower than the original, and by what "
            "percentage? Show your reasoning, and prove the general rule for any markup m and "
            "discount d."
        ),
        rubric=(
            "Correctly computes 1.40 * 0.75 = 1.05, so the final price is 5% higher than the "
            "original; states the general rule as final = original * (1+m) * (1-d) and reasons "
            "about when that's above or below 1; shows the derivation, not just the answer."
        ),
        provenance=f"{AUTHORED}; markup/discount word problem with a general-rule proof request, "
        "patterned after common percentage-change problems (no verbatim reuse).",
    ),
    dict(
        id="math-hard-st-2",
        domain="math",
        difficulty="hard",
        kind="single_turn",
        prompt=(
            "Explain why the sum of the first n odd numbers is always a perfect square, and "
            "derive a general formula, justifying each step."
        ),
        rubric=(
            "States the result (sum of first n odd numbers = n^2) and gives a real "
            "justification — e.g. induction, or a geometric/algebraic argument — not just the "
            "formula asserted without support."
        ),
        provenance=f"{AUTHORED}; classic number-theory identity (public mathematical fact), own "
        "phrasing, no single external source.",
    ),
    dict(
        id="math-easy-agent-1",
        domain="math",
        difficulty="easy",
        kind="agent",
        prompt="What's 234 * 18?",
        rubric="Calls calculator with '234 * 18' and reports 4212.",
        provenance=f"{AUTHORED}; agent task against benchmark.tools.calculator.",
        expected_tool="calculator",
    ),
    dict(
        id="math-easy-agent-2",
        domain="math",
        difficulty="easy",
        kind="agent",
        prompt="Compute 144 divided by 12, then add 9.",
        rubric="Calls calculator (one or two calls) and reports 21.",
        provenance=f"{AUTHORED}; agent task against benchmark.tools.calculator.",
        expected_tool="calculator",
    ),
    dict(
        id="math-hard-agent-1",
        domain="math",
        difficulty="hard",
        kind="agent",
        prompt=(
            "I'm compounding $1000 at 5% annual interest for 10 years, compounded once per "
            "year — compute the final amount precisely, then explain in one sentence why "
            "compounding matters."
        ),
        rubric=(
            "Calls calculator with an expression equivalent to 1000 * 1.05**10, reports a "
            "result close to $1628.89, and adds a sensible one-sentence remark about interest "
            "earning interest over time."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.calculator.",
        expected_tool="calculator",
    ),
    dict(
        id="math-hard-agent-2",
        domain="math",
        difficulty="hard",
        kind="agent",
        prompt=(
            "A rectangle's length is 3 times its width. If the perimeter is 96, compute the "
            "width and length precisely, then verify your answer makes sense."
        ),
        rubric=(
            "Uses calculator to solve 2*(w + 3w) = 96, reports width 12 and length 36, and "
            "checks that 2*(12+36) = 96."
        ),
        provenance=f"{AUTHORED}; agent task against benchmark.tools.calculator.",
        expected_tool="calculator",
    ),
]


def main() -> None:
    ids = [item["id"] for item in ITEMS]
    assert len(ids) == len(set(ids)), "duplicate ids in ITEMS"
    out = Path(__file__).parent / "data" / "workload.json"
    out.write_text(json.dumps(ITEMS, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(ITEMS)} items to {out}")


if __name__ == "__main__":
    main()
