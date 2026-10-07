"""Run guarded SQL against a read-only connection, with a hard time limit.

Two independent protections, because the guard upstream is code and code has bugs:

  * `file:...?mode=ro` - SQLite itself refuses writes, so a guard bypass still cannot mutate
  * a progress handler that aborts past a deadline - so a guard bypass still cannot hang the
    process on a cartesian join

The progress handler is used rather than a watchdog thread: it runs on the same thread as
the query, so there is no race between "the timer fired" and "the query finished", and no
risk of interrupting the *next* query because a timer outlived the one it was meant to stop.
"""

from __future__ import annotations

import sqlite3
import time

from .config import GuardConfig
from .models import ExecutionResult

# How often SQLite checks in with us, in virtual-machine instructions. Small enough that a
# runaway query is caught promptly, large enough that the callback is not the bottleneck.
_PROGRESS_INTERVAL = 1000


class Executor:
    def __init__(self, db_path: str, cfg: GuardConfig | None = None):
        self.db_path = db_path
        self.cfg = cfg or GuardConfig()

    def _connect(self) -> sqlite3.Connection:
        # uri=True is what makes mode=ro mean anything; without it this is just a filename.
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=self.cfg.timeout_s)

    def run(self, sql: str, timeout_s: float | None = None) -> ExecutionResult:
        limit = self.cfg.timeout_s if timeout_s is None else timeout_s
        started = time.perf_counter()
        conn = self._connect()
        deadline = time.monotonic() + limit
        timed_out = False

        def _abort_if_over_deadline() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1        # non-zero aborts the running statement
            return 0

        try:
            conn.set_progress_handler(_abort_if_over_deadline, _PROGRESS_INTERVAL)
            cur = conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = [tuple(r) for r in cur.fetchmany(self.cfg.row_limit)]
            return ExecutionResult(
                columns=columns, rows=rows,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except sqlite3.Error as e:
            # An aborted statement surfaces as a generic OperationalError, so the flag set by
            # the handler is what distinguishes "too slow" from "bad SQL". The difference
            # matters: one is worth retrying, the other is worth correcting.
            message = "query exceeded the time limit" if timed_out else str(e)
            return ExecutionResult(
                error=message, timed_out=timed_out,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()
