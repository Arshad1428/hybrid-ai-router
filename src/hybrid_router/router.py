"""Centralized hybrid routing: the ONLY place that decides local vs cloud."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional, Protocol

from .config import Settings


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class InferenceResult:
    response: str
    provider: str                 # "local" | "bedrock"  (who ACTUALLY answered)
    requested_route: str          # "local" | "cloud"
    fallback_used: bool
    fallback_reason: Optional[str]
    model: str
    latency_ms: int
    cloud_calls_used: int

    def to_dict(self) -> dict:
        return asdict(self)


class HybridRouter:
    def __init__(self, settings: Settings, local: Generator, cloud: Generator):
        self.settings = settings
        self._local = local
        self._cloud = cloud
        self.cloud_calls_used = 0  # per-session counter (per router instance)

    def generate(self, prompt: str, route: Optional[str] = None) -> InferenceResult:
        route = (route or self.settings.default_provider).lower()
        if route not in {"local", "cloud"}:
            raise ValueError("route must be 'local' or 'cloud'")

        if route == "local":
            return self._run_local(prompt, route, fallback_reason=None)

        # ---- cloud requested: guards first ----
        if not self.settings.aws_cloud_enabled:
            return self._run_local(prompt, route, "cloud_disabled")
        if self.cloud_calls_used >= self.settings.max_cloud_calls_per_session:
            return self._run_local(prompt, route, "cloud_call_limit_reached")

        start = time.perf_counter()
        self.cloud_calls_used += 1  # count the attempt, even if it fails
        try:
            text = self._cloud.generate(prompt)
        except Exception as exc:  # Bedrock-specific failures must not crash the app
            return self._run_local(prompt, route, f"bedrock_error: {type(exc).__name__}")
        return InferenceResult(
            response=text,
            provider="bedrock",
            requested_route=route,
            fallback_used=False,
            fallback_reason=None,
            model=self.settings.bedrock_model_id,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cloud_calls_used=self.cloud_calls_used,
        )

    def _run_local(self, prompt: str, route: str, fallback_reason: Optional[str]) -> InferenceResult:
        start = time.perf_counter()
        text = self._local.generate(prompt)  # if this fails there is nothing left to fall back to
        return InferenceResult(
            response=text,
            provider="local",
            requested_route=route,
            fallback_used=fallback_reason is not None,
            fallback_reason=fallback_reason,
            model=self.settings.ollama_model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cloud_calls_used=self.cloud_calls_used,
        )
