"""LLM client selection. Agents depend on the protocol in `base`, never on a concrete client."""

from __future__ import annotations

from ..config import Settings
from .base import LLMClient, LLMResponse
from .mock import MockLLM
from .openai_compat import OpenAICompatibleClient

__all__ = ["LLMClient", "LLMResponse", "MockLLM", "OpenAICompatibleClient", "build_llm"]


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_mode == "live":
        if not settings.llm_api_key:
            raise ValueError("T2S_LLM_MODE=live but no API key: set T2S_LLM_API_KEY or OPENAI_API_KEY")
        return OpenAICompatibleClient(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model
        )
    return MockLLM()
