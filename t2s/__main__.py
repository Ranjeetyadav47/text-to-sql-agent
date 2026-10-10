"""CLI: generate-data, schema, ask, eval.

Printing is deliberate about one thing: an answer is never shown without the SQL that
produced it and the tables that were in the prompt. If you cannot see those, you cannot
tell a right answer from a lucky one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import T2SAgent
from .config import GuardConfig, LinkConfig, Settings
from .data import generate
from .eval import run_eval, write_report
from .llm import build_llm
from .logging_utils import configure_logging
from .models import AnswerStatus
from .schema import Schema


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="t2s", description="Text-to-SQL agent over a synthetic service database.")
    p.add_argument("--data-dir", default=os.getenv("T2S_DATA_DIR", "data"))
    p.add_argument("--db", default=None, help="path to service.db (default: <data-dir>/service.db)")
    p.add_argument("--quiet", action="store_true", help="suppress JSON logs on stderr")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-data", help="rebuild the database and the labelled question set")
    g.add_argument("--out", default=None)
    g.add_argument("--seed", type=int, default=7)

    sub.add_parser("schema", help="print the schema as the model sees it")

    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="emit the full answer as JSON")
    a.add_argument("--max-tables", type=int, default=None, help="cap tables in the prompt (default 4)")

    e = sub.add_parser("eval", help="score the labelled question set")
    e.add_argument("--compare-linking", action="store_true",
                   help="also run with the whole schema in the prompt, to price schema linking")
    e.add_argument("--out", default="reports/eval_report.md")

    args = p.parse_args(argv)
    configure_logging(quiet=args.quiet or args.cmd in ("eval", "schema"))
    db_path = args.db or os.path.join(args.data_dir, "service.db")

    if args.cmd == "generate-data":
        out = args.out or args.data_dir
        counts = generate(out_dir=out, seed=args.seed)
        print(f"wrote {out}/service.db and {out}/questions.jsonl")
        for k in sorted(counts):
            print(f"  {k:<16} {counts[k]}")
        return 0

    if not os.path.exists(db_path):
        print(f"no database at {db_path}; run `python -m t2s generate-data` first", file=sys.stderr)
        return 2

    if args.cmd == "schema":
        print(Schema.from_sqlite(db_path).render())
        return 0

    if args.cmd == "ask":
        return _ask(args, db_path)

    if args.cmd == "eval":
        return _eval(args, db_path)

    return 2


def _ask(args, db_path: str) -> int:
    link = LinkConfig(max_tables=args.max_tables) if args.max_tables else LinkConfig()
    settings = Settings.from_env(db_path=db_path, data_dir=args.data_dir, link=link)
    schema = Schema.from_sqlite(db_path)
    answer = T2SAgent(settings, schema, build_llm(settings)).ask(args.question)

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, default=str))
        return 0 if answer.status == AnswerStatus.OK else 1

    print(f"question : {answer.question}")
    print(f"status   : {answer.status.value}")
    print(f"linked   : {', '.join(answer.linked_tables) or '(none)'}")
    if answer.sql:
        print(f"sql      : {answer.sql}")
    if answer.violations:
        print(f"blocked  : {'; '.join(answer.violations)}")
    if answer.error:
        print(f"error    : {answer.error}")
    if answer.clarifying_question:
        print(f"asking   : {answer.clarifying_question}")
    if answer.corrected:
        print(f"note     : first query failed, corrected once")
    if answer.rows:
        print()
        _print_table(answer.columns, answer.rows)
    print()
    print(answer.summary)
    return 0 if answer.status == AnswerStatus.OK else 1


def _print_table(columns: list[str], rows: list[tuple], limit: int = 20) -> None:
    shown = rows[:limit]
    widths = [len(c) for c in columns]
    for r in shown:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))
    print("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(columns)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in shown:
        print("  " + "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more row(s)")


def _eval(args, db_path: str) -> int:
    results = [run_eval(args.data_dir, db_path, max_tables=LinkConfig().max_tables, label="with linking")]
    if args.compare_linking:
        results.append(run_eval(args.data_dir, db_path, max_tables=None, label="whole schema"))

    for r in results:
        print(f"[{r.label}] execution_accuracy={r.execution_accuracy:.1%}  "
              f"template={r.template_accuracy:.1%}  paraphrase={r.paraphrase_accuracy:.1%}  "
              f"clarify={r.clarification_accuracy:.1%}  unsafe={r.unsafe_queries}  "
              f"attacks_blocked={r.attacks_blocked}/{r.attacks_total}  "
              f"tokens/q={r.prompt_tokens / r.n:.0f}  n={r.n}")
    path = write_report(results, args.out)
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
