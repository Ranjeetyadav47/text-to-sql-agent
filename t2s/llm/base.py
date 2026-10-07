"""The only contract agents depend on. Swap providers without touching agent code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    content: str                                   # raw text (JSON for extraction tasks)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "mock"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    def complete(self, task: str, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        """Run one completion.

        `task` is a short label ("generate", "correct") so a mock can route deterministically
        and a live client can tag usage by task.
        """
        ...


# Section markers in the user prompt. The generator writes them and the mock reads them, so
# they are a contract between the two and belong next to the protocol rather than in either.
Q_MARKER = "QUESTION:"
LINKED_MARKER = "LINKED TABLES:"
PREV_SQL_MARKER = "PREVIOUS SQL:"
ERROR_MARKER = "SQLITE ERROR:"


def after_marker(text: str, marker: str) -> str:
    """Text following `marker` up to the next blank line, or '' if the marker is absent."""
    idx = text.find(marker)
    if idx < 0:
        return ""
    rest = text[idx + len(marker):]
    end = rest.find("\n\n")
    return (rest if end < 0 else rest[:end]).strip()
