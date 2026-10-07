"""Typed records that flow through the agent. Plain dataclasses: no framework, easy to serialise."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class TransientError(Exception):
    """Retryable: timeouts, rate limits, flaky downstream."""


class PermanentError(Exception):
    """Not retryable: bad input, schema violation, malformed model output."""


class AnswerStatus(str, Enum):
    OK = "ok"                       # SQL ran, rows returned
    CLARIFY = "clarify"             # question too vague to answer; we asked instead of guessing
    BLOCKED = "blocked"             # guard refused the SQL
    FAILED = "failed"               # SQL errored twice, including after one correction round


@dataclass
class Column:
    name: str
    type: str
    description: str = ""
    is_pk: bool = False
    # Set when this column is a foreign key: ("table", "column") it points at.
    references: Optional[tuple[str, str]] = None


@dataclass
class Table:
    name: str
    description: str
    columns: list[Column] = field(default_factory=list)

    def column(self, name: str) -> Optional[Column]:
        return next((c for c in self.columns if c.name == name), None)


@dataclass
class LinkedSchema:
    """The subset of the schema sent to the model, and why each table was chosen.

    Logged on every ask: schema linking is the main cost and accuracy lever here, so the
    choice it made has to be visible rather than buried in a prompt.
    """

    tables: list[Table]
    reasons: dict[str, str] = field(default_factory=dict)   # table name -> why it was included
    considered: int = 0                                     # tables in the full schema

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tables]

    def to_dict(self) -> dict[str, Any]:
        return {"tables": self.names, "reasons": self.reasons, "considered": self.considered}


@dataclass
class GeneratedSQL:
    """What the model returns. `needs_clarification` is the model declining to guess."""

    sql: Optional[str]
    confidence: float = 0.0
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardVerdict:
    """Deterministic safety check. `sql` is the rewritten statement actually sent to SQLite."""

    allowed: bool
    sql: Optional[str] = None
    violations: list[str] = field(default_factory=list)
    rewrites: list[str] = field(default_factory=list)       # e.g. "appended LIMIT 200"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "row_count": len(self.rows),
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "timed_out": self.timed_out,
        }


@dataclass
class Answer:
    """Everything an ask produced. The SQL and the linked tables are always present,
    answered or not, because an answer you cannot audit is not an answer."""

    question: str
    status: AnswerStatus
    summary: str
    sql: Optional[str] = None
    linked_tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    clarifying_question: Optional[str] = None
    violations: list[str] = field(default_factory=list)
    error: Optional[str] = None
    attempts: int = 1                                       # 2 means the correction round ran
    corrected: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["rows"] = [list(r) for r in self.rows]
        return d
