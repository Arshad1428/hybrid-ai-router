"""Cloud inference through Amazon Bedrock (on-demand, Converse API)."""
from __future__ import annotations

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .config import Settings


class BedrockInferenceError(RuntimeError):
    pass


class BedrockClient:
    def __init__(self, settings: Settings):
        self.model_id = settings.bedrock_model_id
        self.max_tokens = settings.max_output_tokens
        self._region = settings.aws_region
        self._client = None  # created lazily: no AWS activity unless cloud is actually used

    def _runtime(self):
        if self._client is None:
            # Credentials come from the standard AWS chain (profile / env / SSO), never from code.
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=Config(retries={"max_attempts": 2}, read_timeout=60, connect_timeout=10),
            )
        return self._client

    def generate(self, prompt: str) -> str:
        try:
            resp = self._runtime().converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": self.max_tokens},  # caps generation cost
            )
            parts = resp["output"]["message"]["content"]
            return "".join(p.get("text", "") for p in parts)
        except (BotoCoreError, ClientError, KeyError) as exc:
            raise BedrockInferenceError(f"Bedrock call failed: {exc}") from exc
