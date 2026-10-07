"""Runtime configuration. Everything a reviewer might ask "why this number?" about lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LinkConfig:
    """Schema linking: how much of the schema the model is allowed to see."""

    max_tables: int = 4             # hard cap on tables in the prompt; the accuracy/cost lever
    min_keyword_hits: int = 1       # a table needs this many keyword matches before it is a seed
    follow_foreign_keys: bool = True


@dataclass(frozen=True)
class GuardConfig:
    """Deterministic safety layer. These are not suggestions to a model; they are checks in code."""

    row_limit: int = 200            # appended as LIMIT when the query has none
    timeout_s: float = 5.0          # per query, enforced by interrupting the connection
    max_sql_chars: int = 4000


@dataclass(frozen=True)
class Settings:
    db_path: str = "data/service.db"
    data_dir: str = "data"
    llm_mode: str = "mock"          # "mock" | "live"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    correct_once: bool = True       # one self-correction round on a SQLite error
    link: LinkConfig = field(default_factory=LinkConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Build settings from environment variables, with explicit overrides winning."""
        base = dict(
            db_path=os.getenv("T2S_DB", "data/service.db"),
            data_dir=os.getenv("T2S_DATA_DIR", "data"),
            llm_mode=os.getenv("T2S_LLM_MODE", "mock"),
            llm_base_url=os.getenv("T2S_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=os.getenv("T2S_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            llm_model=os.getenv("T2S_LLM_MODEL", "gpt-4o-mini"),
        )
        base.update(overrides)
        return cls(**base)
