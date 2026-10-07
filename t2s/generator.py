"""NL -> SQL via the LLM, prompted with the linked schema only.

The prompt is assembled here and nowhere else, so what the model sees is one readable
function rather than string fragments scattered across the agent. The model is asked for
JSON and given an explicit way to decline: a model that answers every question is a model
that guesses, and a guess that runs is worse than a question that does not.
"""

from __future__ import annotations

import json
from typing import Optional

from .llm.base import LINKED_MARKER, Q_MARKER, LLMClient, LLMResponse
from .models import GeneratedSQL, LinkedSchema, PermanentError
from .schema import Schema

SYSTEM = (
    "You translate questions into exactly one SQLite SELECT statement.\n"
    "Use only the tables and columns in SCHEMA. Never invent a column or a table.\n"
    "Return JSON only, with these keys:\n"
    '  sql                  - one SELECT or WITH statement, or null if you cannot answer\n'
    "  confidence           - 0.0 to 1.0\n"
    "  needs_clarification  - true if the question is too vague to answer\n"
    "  clarifying_question  - the question to ask back, or null\n"
    "Rules:\n"
    "- One statement. No semicolons, no comments, no PRAGMA, no ATTACH, no writes.\n"
    "- If the question does not name anything in the schema, set needs_clarification and ask. "
    "Do not guess a query that merely looks plausible.\n"
    "- Prefer explicit JOINs using the relationships given in SCHEMA.\n"
    "- Return only the JSON object, no prose and no code fences."
)

CORRECTION_SYSTEM = (
    "Your previous SQLite query failed. Return corrected JSON in the same shape.\n"
    "Change only what the error requires; do not rewrite the query's meaning.\n"
    "If the error cannot be fixed from the schema given, return sql: null rather than guessing."
)


class SQLGenerator:
    def __init__(self, llm: LLMClient, schema: Schema):
        self.llm = llm
        self.schema = schema

    def build_prompt(self, question: str, linked: LinkedSchema) -> str:
        schema_text = self.schema.render(linked.names) if linked.names else "(no tables matched)"
        return (
            f"SCHEMA:\n{schema_text}\n\n"
            f"{LINKED_MARKER} {', '.join(linked.names) or '(none)'}\n\n"
            f"{Q_MARKER} {question}\n"
        )

    def generate(self, question: str, linked: LinkedSchema) -> tuple[GeneratedSQL, LLMResponse]:
        resp = self.llm.complete(task="generate", system=SYSTEM,
                                 user=self.build_prompt(question, linked), json_mode=True)
        return _parse(resp.content), resp

    def correct(self, question: str, linked: LinkedSchema, previous_sql: str,
                error: str) -> tuple[GeneratedSQL, LLMResponse]:
        """One round only. A model that could not fix it with the error text in hand is not
        going to fix it on the third try; it is going to wander."""
        user = (
            f"{self.build_prompt(question, linked)}\n"
            f"PREVIOUS SQL: {previous_sql}\n\n"
            f"SQLITE ERROR: {error}\n"
        )
        resp = self.llm.complete(task="correct", system=CORRECTION_SYSTEM, user=user, json_mode=True)
        return _parse(resp.content), resp


def _parse(content: str) -> GeneratedSQL:
    """Parse the model's JSON defensively: this is untrusted output, not a return value."""
    text = content.strip()
    if text.startswith("```"):
        # Models add fences despite being asked not to. Strip rather than fail: it is a
        # formatting slip, not a semantic one.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PermanentError(f"model did not return JSON: {content[:200]!r}") from e
    if not isinstance(data, dict):
        raise PermanentError(f"model returned {type(data).__name__}, expected an object")

    sql = data.get("sql")
    if sql is not None and not isinstance(sql, str):
        raise PermanentError(f"model returned a non-string sql field: {sql!r}")

    return GeneratedSQL(
        sql=(sql.strip() if isinstance(sql, str) and sql.strip() else None),
        confidence=_float(data.get("confidence")),
        needs_clarification=bool(data.get("needs_clarification")),
        clarifying_question=_str_or_none(data.get("clarifying_question")),
    )


def _float(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _str_or_none(v) -> Optional[str]:
    return v.strip() if isinstance(v, str) and v.strip() else None
