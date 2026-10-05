"""Local inference through Ollama's HTTP API. Independent of AWS."""
from __future__ import annotations

import requests

from .config import Settings


class LocalInferenceError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
        self.max_tokens = settings.max_output_tokens

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": self.max_tokens},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LocalInferenceError(f"Ollama call failed: {exc}") from exc
