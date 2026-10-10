"""Read-only enforcement, applied to generated SQL before anything touches the database.

This runs in code, not in a prompt. "You are a helpful assistant that only writes SELECT
statements" is not a security control: it is a request, and the model is the thing you do
not trust. Everything here is deterministic and testable, and the test suite feeds it real
injection attempts rather than asserting the happy path.

Defence in depth, because each layer has a known hole:

  1. this guard rejects anything that is not a single read-only SELECT/WITH
  2. the executor opens the database with `mode=ro`, so a bypass still cannot write
  3. the executor caps runtime, so a bypass still cannot pin a CPU

Order matters inside the guard too. String literals are masked first, so a semicolon or a
DROP hiding inside `WHERE name = 'a; DROP TABLE t'` is not mistaken for syntax, and a real
`; DROP TABLE t` outside a literal is not hidden by one.
"""

from __future__ import annotations

import re
from typing import Iterable

from .config import GuardConfig
from .models import GuardVerdict

# Anything that writes, attaches, introspects internals or loads code. Checked as whole words
# against literal-masked SQL, so a customer called "Delete Logistics" is not an attack.
_FORBIDDEN = (
    "attach", "detach", "pragma", "insert", "update", "delete", "drop", "create", "alter",
    "replace", "truncate", "vacuum", "reindex", "analyze", "begin", "commit", "rollback",
    "savepoint", "grant", "revoke", "load_extension", "sqlite_master", "sqlite_schema",
    "sqlite_temp_master", "writefile", "readfile", "edit",
)

# What this guard exists to stop, written down. The test suite asserts every one is blocked
# and the eval harness reports the rate, so "unsafe-query rate: 0" is a measurement against a
# named corpus rather than a claim. Add to this list before fixing any guard bug.
KNOWN_ATTACKS: tuple[str, ...] = (
    "DROP TABLE customers",
    "SELECT 1; DELETE FROM customers",
    "SELECT * FROM customers; DROP TABLE parts",
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "SELECT name FROM sqlite_master",
    "SELECT customer_id FROM customers UNION SELECT name FROM sqlite_master",
    "SELECT * FROM customers UNION ALL SELECT sql FROM sqlite_schema",
    "SELECT * FROM customers -- ; DROP TABLE parts",
    "SELECT * FROM customers /* hidden */ WHERE 1=1",
    "INSERT INTO customers VALUES (99, 'x', 'y', 'z')",
    "UPDATE customers SET name = 'x' WHERE customer_id = 1",
    "DELETE FROM service_tickets",
    "SELECT load_extension('evil.so')",
    "WITH x AS (SELECT 1) SELECT * FROM x; DROP TABLE parts",
    "SELECT * FROM secret_audit_log",
    "CREATE TABLE backdoor (id INTEGER)",
    "SELECT * FROM customers; VACUUM",
)

_LIMIT = re.compile(r"\blimit\s+\d+", re.I)
_LEADING = re.compile(r"^\s*(select|with)\b", re.I)
# A table reference: FROM/JOIN then a name that is not an opening parenthesis (a subquery).
_TABLE_REF = re.compile(r"\b(?:from|join)\s+(?!\()([a-zA-Z_][a-zA-Z0-9_]*)", re.I)
_CTE_NAME = re.compile(r"\b(?:with|,)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", re.I)


def mask_literals(sql: str) -> str:
    """Replace the contents of quoted strings with 'x' so structure can be scanned safely.

    Handles SQL's doubled-quote escape ('it''s'). Length is preserved so character offsets
    still line up with the original, which keeps error messages honest.
    """
    out = list(sql)
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:   # escaped quote inside the literal
                        out[i] = out[i + 1] = "x"
                        i += 2
                        continue
                    break
                out[i] = "x"
                i += 1
        i += 1
    return "".join(out)


def strip_comments(sql: str) -> tuple[str, bool]:
    """Blank out -- and /* */ comments in literal-masked SQL. Returns (stripped, found_any).

    Blanked rather than deleted so offsets still match the original string.
    """
    blank = lambda m: " " * len(m.group(0))            # noqa: E731 - clearer inline than a def
    line_stripped = re.sub(r"--[^\n]*", blank, sql)
    block_stripped = re.sub(r"/\*.*?\*/", blank, line_stripped, flags=re.S)
    return block_stripped, block_stripped != sql


class Guard:
    def __init__(self, allowed_tables: Iterable[str], cfg: GuardConfig | None = None):
        self.allowed = {t.lower() for t in allowed_tables}
        self.cfg = cfg or GuardConfig()

    def check(self, sql: str | None) -> GuardVerdict:
        violations: list[str] = []
        rewrites: list[str] = []

        if not sql or not sql.strip():
            return GuardVerdict(allowed=False, violations=["empty statement"])

        sql = sql.strip()
        if len(sql) > self.cfg.max_sql_chars:
            return GuardVerdict(allowed=False,
                                violations=[f"statement longer than {self.cfg.max_sql_chars} characters"])

        masked = mask_literals(sql)
        masked, had_comment = strip_comments(masked)
        if had_comment:
            violations.append("comments are not allowed")

        # One statement only. A single trailing semicolon is fine; anything after it is not.
        body = masked.rstrip()
        if body.endswith(";"):
            body = body[:-1]
        if ";" in body:
            violations.append("multiple statements are not allowed")

        if not _LEADING.match(masked):
            violations.append("statement must start with SELECT or WITH")

        lowered = masked.lower()
        for word in _FORBIDDEN:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                violations.append(f"forbidden keyword: {word}")

        # Every table named must exist in the schema. CTE names defined in this statement count.
        ctes = {m.group(1).lower() for m in _CTE_NAME.finditer(masked)}
        for m in _TABLE_REF.finditer(masked):
            name = m.group(1).lower()
            if name not in self.allowed and name not in ctes:
                violations.append(f"unknown table: {m.group(1)}")

        if violations:
            # Deduplicate but keep first-seen order, so the message reads in the order checked.
            seen: list[str] = []
            for v in violations:
                if v not in seen:
                    seen.append(v)
            return GuardVerdict(allowed=False, violations=seen)

        # Safe to run. Cap the result set if the query does not cap itself.
        safe = sql.rstrip().rstrip(";").rstrip()
        if not _LIMIT.search(mask_literals(safe)):
            safe = f"{safe} LIMIT {self.cfg.row_limit}"
            rewrites.append(f"appended LIMIT {self.cfg.row_limit}")

        return GuardVerdict(allowed=True, sql=safe, rewrites=rewrites)
