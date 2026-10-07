"""Live client for any OpenAI-compatible chat endpoint (OpenAI, Gemini's compatibility API,
Azure OpenAI with a base_url, local servers such as Ollama/vLLM).

Standard library only: no SDK pinned, nothing to install. Errors are mapped onto the
workflow's own exception types so the orchestrator decides what to retry.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..models import PermanentError, TransientError
from .base import LLMResponse

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, task: str, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in _RETRYABLE_STATUS:
                raise TransientError(f"{task}: HTTP {e.code} from LLM endpoint") from e
            raise PermanentError(f"{task}: HTTP {e.code} from LLM endpoint: {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise TransientError(f"{task}: network error: {e}") from e

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise PermanentError(f"{task}: unexpected response shape: {str(body)[:200]}") from e

        usage = body.get("usage", {}) or {}
        return LLMResponse(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            model=body.get("model", self.model),
        )
