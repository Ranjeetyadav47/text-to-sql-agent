"""Shared fixtures (unittest-style so the suite runs with or without pytest)."""

from __future__ import annotations

import os
import shutil
import tempfile

from t2s.config import GuardConfig, LinkConfig, Settings
from t2s.agent import T2SAgent
from t2s.data import generate
from t2s.llm.mock import MockLLM
from t2s.logging_utils import configure_logging
from t2s.schema import Schema


class Workspace:
    """A temp directory with a generated database and question set.

    Built once per test class: generating 2,000 tickets for every test method is the
    difference between a suite that gets run and one that gets skipped.
    """

    def __init__(self, seed: int = 7):
        configure_logging(quiet=True)
        self.dir = tempfile.mkdtemp(prefix="t2s-test-")
        self.data_dir = os.path.join(self.dir, "data")
        generate(out_dir=self.data_dir, seed=seed)
        self.db_path = os.path.join(self.data_dir, "service.db")
        self._schema: Schema | None = None

    @property
    def schema(self) -> Schema:
        if self._schema is None:
            self._schema = Schema.from_sqlite(self.db_path)
        return self._schema

    def settings(self, **overrides) -> Settings:
        base = dict(db_path=self.db_path, data_dir=self.data_dir, llm_mode="mock")
        base.update(overrides)
        return Settings.from_env(**base)

    def agent(self, llm=None, **overrides) -> T2SAgent:
        s = self.settings(**overrides)
        return T2SAgent(s, self.schema, llm or MockLLM())

    def table_names(self) -> list[str]:
        return [t.name for t in self.schema.tables]

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def gold_rows(workspace: Workspace, sql: str) -> list[tuple]:
    from t2s.executor import Executor
    return Executor(workspace.db_path, GuardConfig(row_limit=5000)).run(sql).rows


__all__ = ["Workspace", "gold_rows", "GuardConfig", "LinkConfig"]
