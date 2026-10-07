"""The loop: link -> generate -> guard -> execute -> (on error) correct once -> answer.

Five steps, one file, no framework. The whole control flow fits on a screen on purpose: this
is the part a reviewer needs to be able to read end to end before trusting anything it says.

Every exit returns an `Answer` carrying the SQL and the linked tables, including the exits
that failed. An answer you cannot audit is not an answer, and a failure you cannot see the
query for is not debuggable.
"""

from __future__ import annotations

import time
import uuid

from .config import Settings
from .executor import Executor
from .generator import SQLGenerator
from .guard import Guard
from .llm.base import LLMClient
from .logging_utils import EventLogger
from .models import Answer, AnswerStatus, GeneratedSQL, LinkedSchema
from .schema import Schema, SchemaLinker

log = EventLogger()


class T2SAgent:
    def __init__(self, settings: Settings, schema: Schema, llm: LLMClient):
        self.settings = settings
        self.schema = schema
        self.linker = SchemaLinker(schema, settings.link)
        self.generator = SQLGenerator(llm, schema)
        self.guard = Guard([t.name for t in schema.tables], settings.guard)
        self.executor = Executor(settings.db_path, settings.guard)

    def ask(self, question: str) -> Answer:
        ask_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        prompt_tokens = completion_tokens = 0

        # 1. Link. Which few tables is this question about?
        linked = self.linker.link(question)
        log.info("link.done", ask_id=ask_id, tables=linked.names,
                 considered=linked.considered, reasons=linked.reasons)

        def finish(**kw) -> Answer:
            return Answer(
                question=question, linked_tables=linked.names,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                elapsed_ms=(time.perf_counter() - started) * 1000, **kw,
            )

        # 2. Generate.
        gen, resp = self.generator.generate(question, linked)
        prompt_tokens += resp.prompt_tokens
        completion_tokens += resp.completion_tokens

        if gen.needs_clarification or (gen.sql is None and gen.clarifying_question):
            log.info("ask.clarify", ask_id=ask_id)
            return finish(
                status=AnswerStatus.CLARIFY,
                summary="The question is too vague to answer safely, so I asked instead of guessing.",
                clarifying_question=gen.clarifying_question or "Could you be more specific?",
            )

        if gen.sql is None:
            log.warning("ask.no_sql", ask_id=ask_id)
            return finish(status=AnswerStatus.FAILED,
                          summary="The model did not produce a query for this question.",
                          error="no SQL generated")

        # 3. Guard, then 4. execute, with at most one correction round between them.
        attempt = 1
        result, sql, verdict = self._guard_and_run(gen, ask_id, attempt)

        if verdict is not None and not verdict.allowed:
            log.warning("guard.blocked", ask_id=ask_id, violations=verdict.violations)
            return finish(status=AnswerStatus.BLOCKED,
                          summary="The generated query was blocked before it reached the database.",
                          sql=gen.sql, violations=verdict.violations)

        corrected = False
        if result is not None and not result.ok and self.settings.correct_once and not result.timed_out:
            # The error text is the single most useful thing we can give the model, and the
            # only reason a second attempt is worth paying for.
            log.info("correct.start", ask_id=ask_id, error=result.error)
            attempt = 2
            fixed, resp2 = self.generator.correct(question, linked, sql or gen.sql, result.error or "")
            prompt_tokens += resp2.prompt_tokens
            completion_tokens += resp2.completion_tokens
            if fixed.sql:
                corrected = True
                result2, sql2, verdict2 = self._guard_and_run(fixed, ask_id, attempt)
                if verdict2 is not None and not verdict2.allowed:
                    return finish(status=AnswerStatus.BLOCKED, attempts=attempt, corrected=corrected,
                                  summary="The corrected query was blocked before it reached the database.",
                                  sql=fixed.sql, violations=verdict2.violations)
                result, sql = result2, sql2

        # 5. Answer.
        if result is None or not result.ok:
            error = result.error if result else "query could not be run"
            log.warning("ask.failed", ask_id=ask_id, error=error, attempts=attempt)
            return finish(status=AnswerStatus.FAILED, sql=sql, attempts=attempt, corrected=corrected,
                          summary=f"The query failed after {attempt} attempt(s) and was not run again.",
                          error=error)

        log.info("ask.ok", ask_id=ask_id, rows=len(result.rows), attempts=attempt)
        return finish(
            status=AnswerStatus.OK, sql=sql, attempts=attempt, corrected=corrected,
            columns=result.columns, rows=result.rows,
            summary=_summarise(result.columns, result.rows, linked.names),
        )

    def _guard_and_run(self, gen: GeneratedSQL, ask_id: str, attempt: int):
        """Guard then execute. Returns (ExecutionResult|None, sql|None, GuardVerdict)."""
        verdict = self.guard.check(gen.sql)
        if not verdict.allowed:
            return None, None, verdict
        if verdict.rewrites:
            log.info("guard.rewrote", ask_id=ask_id, rewrites=verdict.rewrites)
        result = self.executor.run(verdict.sql or "")
        log.info("execute.done", ask_id=ask_id, attempt=attempt, **result.to_dict())
        return result, verdict.sql, verdict


def _summarise(columns: list[str], rows: list[tuple], tables: list[str]) -> str:
    """One line a person can read without looking at the grid."""
    source = ", ".join(tables) if tables else "the database"
    if not rows:
        return f"No rows matched, from {source}."
    if len(rows) == 1 and len(columns) == 1:
        return f"{columns[0]}: {rows[0][0]}, from {source}."
    noun = "row" if len(rows) == 1 else "rows"
    return f"{len(rows)} {noun} across {len(columns)} column(s), from {source}."
