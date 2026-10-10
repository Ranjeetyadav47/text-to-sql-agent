"""Evaluation harness: run the labelled question set and score it by execution accuracy.

The argument the whole repo makes lives in `results_match` below. Two queries are the same
answer if they return the same rows, not if they look alike. String-matching generated SQL
against gold SQL punishes `COUNT(*)` versus `COUNT(1)` and rewards a query that is textually
close and semantically wrong, which is exactly backwards.

What gets measured, and why:

  * execution accuracy   - headline, over the questions that have a gold query
  * template vs paraphrase - the same questions, reworded; the gap is the honest measure of
                           how much of the score is pattern-matching rather than understanding
  * clarification accuracy - on the vague questions, did it ask instead of guessing
  * unsafe-query rate    - must be 0, measured over a named attack corpus, not asserted
  * per-category and failure analysis - so the next thing to fix is obvious
  * latency, tokens, projected cost - so "it works" has a price attached

The guard's row cap is raised for evaluation only (see `EVAL_ROW_LIMIT`). The shipped default
is 200 rows; comparing two truncated result sets would measure the cap, not the query.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..agent import T2SAgent
from ..config import GuardConfig, LinkConfig, Settings
from ..data import load_questions
from ..executor import Executor
from ..guard import KNOWN_ATTACKS, Guard
from ..llm.mock import MockLLM
from ..logging_utils import configure_logging
from ..models import AnswerStatus
from ..schema import Schema

# Assumed list prices for projecting cost; stated in the report, easy to change.
PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60

# High enough that no gold query in the set is truncated, so the comparison is about the
# query and not about the cap. The shipped guard default is 200.
EVAL_ROW_LIMIT = 5000

_ORDER_BY = re.compile(r"\border\s+by\b", re.I)

CATEGORIES = ("simple_filter", "count", "aggregate", "join", "group_topk",
              "date_range", "negation", "ambiguous")


def normalise_rows(rows: list[tuple]) -> list[tuple]:
    """Round floats to 2dp so an AVG that differs in the 12th decimal is not a failure."""
    out = []
    for row in rows:
        out.append(tuple(round(v, 2) if isinstance(v, float) else v for v in row))
    return out


def results_match(predicted: list[tuple], gold: list[tuple], order_matters: bool) -> bool:
    """Compare result sets as multisets, or as sequences when the gold query ordered them.

    Multiset rather than set: a query that loses a duplicate row is wrong, and `set()` would
    quietly forgive it.
    """
    p, g = normalise_rows(predicted), normalise_rows(gold)
    if order_matters:
        return p == g
    return Counter(p) == Counter(g)


@dataclass
class Failure:
    id: str
    category: str
    question: str
    is_paraphrase: bool
    reason: str                 # wrong_rows | blocked | failed | should_have_asked | guessed_instead
    predicted_sql: str = ""
    detail: str = ""


@dataclass
class EvalResult:
    label: str = ""
    n: int = 0
    n_sql: int = 0
    n_clarify: int = 0
    execution_accuracy: float = 0.0
    template_accuracy: float = 0.0
    paraphrase_accuracy: float = 0.0
    clarification_accuracy: float = 0.0
    unsafe_queries: int = 0
    attacks_blocked: int = 0
    attacks_total: int = 0
    blocked: int = 0
    failed: int = 0
    corrected: int = 0
    avg_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    avg_tables_in_prompt: float = 0.0
    by_category: dict[str, tuple[int, int]] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)


def run_eval(data_dir: str = "data", db_path: str | None = None,
             max_tables: int | None = None, label: str = "") -> EvalResult:
    configure_logging(quiet=True)
    db = db_path or os.path.join(data_dir, "service.db")

    # max_tables=None means "no linking": put the whole schema in the prompt.
    schema = Schema.from_sqlite(db)
    link_cfg = (LinkConfig(max_tables=max_tables) if max_tables is not None
                else LinkConfig(max_tables=len(schema.tables), min_keyword_hits=0))
    guard_cfg = GuardConfig(row_limit=EVAL_ROW_LIMIT)
    settings = Settings.from_env(db_path=db, data_dir=data_dir, llm_mode="mock",
                                 link=link_cfg, guard=guard_cfg)

    agent = T2SAgent(settings, schema, MockLLM())
    gold_runner = Executor(db, guard_cfg)
    auditor = Guard([t.name for t in schema.tables], guard_cfg)

    result = EvalResult(label=label)
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    tmpl = [0, 0]
    para = [0, 0]
    latencies: list[float] = []
    tables_used: list[int] = []

    t0 = time.perf_counter()
    for q in load_questions(data_dir):
        answer = agent.ask(q["question"])
        result.n += 1
        by_cat[q["category"]][1] += 1
        latencies.append(answer.elapsed_ms)
        tables_used.append(len(answer.linked_tables))
        result.prompt_tokens += answer.prompt_tokens
        result.completion_tokens += answer.completion_tokens
        if answer.corrected:
            result.corrected += 1
        if answer.status == AnswerStatus.BLOCKED:
            result.blocked += 1
        elif answer.status == AnswerStatus.FAILED:
            result.failed += 1

        # Independent post-hoc audit of what actually ran. The agent already guarded it; this
        # asks whether the guard was right, which is the only number worth reporting.
        if answer.sql and answer.status in (AnswerStatus.OK, AnswerStatus.FAILED):
            if not auditor.check(answer.sql).allowed:
                result.unsafe_queries += 1

        if q["expected"] == "clarify":
            result.n_clarify += 1
            correct = answer.status == AnswerStatus.CLARIFY
            if correct:
                by_cat[q["category"]][0] += 1
            else:
                result.failures.append(Failure(
                    q["id"], q["category"], q["question"], q["is_paraphrase"],
                    "guessed_instead", answer.sql or "",
                    "answered a question too vague to answer",
                ))
            continue

        result.n_sql += 1
        bucket = para if q["is_paraphrase"] else tmpl
        bucket[1] += 1

        correct, reason, detail = _score_sql(answer, q, gold_runner)
        if correct:
            by_cat[q["category"]][0] += 1
            bucket[0] += 1
        else:
            result.failures.append(Failure(
                q["id"], q["category"], q["question"], q["is_paraphrase"],
                reason, answer.sql or "", detail,
            ))

    result.wall_seconds = time.perf_counter() - t0

    # Safety, measured against the named corpus rather than asserted.
    result.attacks_total = len(KNOWN_ATTACKS)
    result.attacks_blocked = sum(1 for a in KNOWN_ATTACKS if not auditor.check(a).allowed)

    correct_sql = tmpl[0] + para[0]
    result.execution_accuracy = correct_sql / result.n_sql if result.n_sql else 0.0
    result.template_accuracy = tmpl[0] / tmpl[1] if tmpl[1] else 0.0
    result.paraphrase_accuracy = para[0] / para[1] if para[1] else 0.0
    amb = by_cat.get("ambiguous", [0, 0])
    result.clarification_accuracy = amb[0] / amb[1] if amb[1] else 0.0
    result.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    result.avg_tables_in_prompt = sum(tables_used) / len(tables_used) if tables_used else 0.0
    result.by_category = {k: (v[0], v[1]) for k, v in by_cat.items()}
    return result


def _score_sql(answer, q: dict, gold_runner: Executor) -> tuple[bool, str, str]:
    if answer.status == AnswerStatus.CLARIFY:
        return False, "asked_instead", "asked for clarification on an answerable question"
    if answer.status == AnswerStatus.BLOCKED:
        return False, "blocked", "; ".join(answer.violations)
    if answer.status == AnswerStatus.FAILED:
        return False, "failed", answer.error or "query did not run"

    gold = gold_runner.run(q["gold_sql"])
    if not gold.ok:
        # A broken gold query is a bug in the labelled set, not a failure of the agent, and
        # it must be loud: silently scoring it as wrong would hide the real problem.
        return False, "gold_error", f"gold query failed: {gold.error}"

    order_matters = bool(_ORDER_BY.search(q["gold_sql"]))
    if results_match(answer.rows, gold.rows, order_matters):
        return True, "", ""
    return False, "wrong_rows", f"returned {len(answer.rows)} rows, gold returns {len(gold.rows)}"


def _cost(r: EvalResult) -> float:
    return r.prompt_tokens / 1e6 * PRICE_PER_M_INPUT + r.completion_tokens / 1e6 * PRICE_PER_M_OUTPUT


def write_report(results: list[EvalResult], path: str) -> str:
    """Markdown report. Several results = several configurations compared side by side."""
    primary = results[0]
    lines: list[str] = ["# Evaluation report\n"]
    lines.append(f"Generated by `python -m t2s eval`. Questions: {primary.n} "
                 f"({primary.n_sql} with gold SQL, {primary.n_clarify} deliberately vague). "
                 "LLM: mock (deterministic). Scored by execution accuracy.\n")

    lines.append("## Headline\n")
    lines.append("| Metric | " + " | ".join(r.label or f"run {i+1}" for i, r in enumerate(results)) + " |")
    lines.append("|" + " --- |" * (len(results) + 1))
    rows = [
        ("Execution accuracy", lambda r: f"{r.execution_accuracy:.1%}"),
        ("&nbsp;&nbsp;on template questions", lambda r: f"{r.template_accuracy:.1%}"),
        ("&nbsp;&nbsp;on paraphrases", lambda r: f"{r.paraphrase_accuracy:.1%}"),
        ("Clarification accuracy (vague questions)", lambda r: f"{r.clarification_accuracy:.1%}"),
        ("Unsafe queries executed", lambda r: str(r.unsafe_queries)),
        ("Known attacks blocked", lambda r: f"{r.attacks_blocked}/{r.attacks_total}"),
        ("Queries blocked by the guard", lambda r: str(r.blocked)),
        ("Queries that failed to run", lambda r: str(r.failed)),
        ("Correction round used", lambda r: str(r.corrected)),
        ("Avg tables in prompt", lambda r: f"{r.avg_tables_in_prompt:.1f}"),
        ("Avg latency per question (ms)", lambda r: f"{r.avg_latency_ms:.2f}"),
        ("Tokens per question (prompt / completion)",
         lambda r: f"{r.prompt_tokens / r.n:.0f} / {r.completion_tokens / r.n:.0f}"),
        ("Projected LLM cost per 1,000 questions (USD)",
         lambda r: f"{_cost(r) * 1000 / r.n:.3f}"),
        ("Wall time (s)", lambda r: f"{r.wall_seconds:.2f}"),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(r) for r in results) + " |")
    lines.append("")
    lines.append(f"Cost assumes ${PRICE_PER_M_INPUT}/M input and ${PRICE_PER_M_OUTPUT}/M output tokens and the "
                 "mock's rough token counts (chars/4). Treat it as an order of magnitude, not a quote.\n")

    for r in results:
        title = r.label or "Result"
        lines.append(f"## {title}: by category\n")
        lines.append("| Category | Correct | Total | Accuracy |")
        lines.append("| --- | --- | --- | --- |")
        for cat in CATEGORIES:
            if cat not in r.by_category:
                continue
            c, t = r.by_category[cat]
            lines.append(f"| {cat} | {c} | {t} | {c / t:.0%} |")
        lines.append("")

        lines.append(f"## {title}: where it breaks\n")
        if not r.failures:
            lines.append("No failures.\n")
        else:
            counts = Counter(f.reason for f in r.failures)
            lines.append("| Reason | n |")
            lines.append("| --- | --- |")
            for reason, n in counts.most_common():
                lines.append(f"| {reason} | {n} |")
            lines.append("")
            lines.append("| id | Category | Paraphrase | Question | Reason | Detail |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for f in r.failures:
                lines.append(f"| {f.id} | {f.category} | {'yes' if f.is_paraphrase else 'no'} | "
                             f"{f.question} | {f.reason} | {f.detail[:90]} |")
            lines.append("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
