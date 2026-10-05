"""Environment-based configuration. No credentials live here."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean(value: str | None, default: str) -> str:
    # strips inline "# comments" so .env.example works as-is
    if value is None:
        return default
    return value.split("#", 1)[0].strip() or default


@dataclass(frozen=True)
class Settings:
    default_provider: str = "local"
    aws_cloud_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: int = 120
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "amazon.nova-micro-v1:0"
    max_output_tokens: int = 256
    max_cloud_calls_per_session: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        g = os.environ.get
        provider = _clean(g("DEFAULT_PROVIDER"), "local").lower()
        if provider not in {"local", "cloud"}:
            raise ValueError("DEFAULT_PROVIDER must be 'local' or 'cloud'")
        return cls(
            default_provider=provider,
            aws_cloud_enabled=_bool(_clean(g("AWS_CLOUD_ENABLED"), "false")),
            ollama_base_url=_clean(g("OLLAMA_BASE_URL"), cls.ollama_base_url),
            ollama_model=_clean(g("OLLAMA_MODEL"), cls.ollama_model),
            ollama_timeout=int(_clean(g("OLLAMA_TIMEOUT_SECONDS"), "120")),
            aws_region=_clean(g("AWS_REGION"), cls.aws_region),
            bedrock_model_id=_clean(g("BEDROCK_MODEL_ID"), cls.bedrock_model_id),
            max_output_tokens=int(_clean(g("MAX_OUTPUT_TOKENS"), "256")),
            max_cloud_calls_per_session=int(_clean(g("MAX_CLOUD_CALLS_PER_SESSION"), "5")),
        )
