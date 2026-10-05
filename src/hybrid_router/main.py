"""CLI entry point.

One-shot:     python -m hybrid_router.main "your prompt" --route cloud
Interactive:  python -m hybrid_router.main --interactive   (session cloud-call guard applies across prompts)
"""
from __future__ import annotations

import argparse
import json
import sys

from .bedrock_inference import BedrockClient
from .config import Settings
from .local_inference import LocalInferenceError, OllamaClient
from .router import HybridRouter


def build_router() -> HybridRouter:
    settings = Settings.from_env()
    return HybridRouter(settings, OllamaClient(settings), BedrockClient(settings))


def _show(result) -> None:
    print(f"\n{result.response}\n")
    meta = result.to_dict()
    meta.pop("response")
    print("[meta]", json.dumps(meta))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Hybrid Local (Ollama) + Amazon Bedrock router")
    p.add_argument("prompt", nargs="?", help="prompt text")
    p.add_argument("--route", choices=["local", "cloud"], help="override DEFAULT_PROVIDER")
    p.add_argument("--interactive", action="store_true", help="REPL; prefix a line with 'cloud:' to use cloud")
    args = p.parse_args(argv)

    router = build_router()
    try:
        if args.interactive:
            print("Interactive mode. 'cloud: <prompt>' = cloud route, 'exit' to quit.")
            while True:
                line = input("> ").strip()
                if line.lower() in {"exit", "quit"}:
                    return 0
                if not line:
                    continue
                route = None
                if line.lower().startswith("cloud:"):
                    route, line = "cloud", line[6:].strip()
                elif line.lower().startswith("local:"):
                    route, line = "local", line[6:].strip()
                _show(router.generate(line, route))
        if not args.prompt:
            p.error("provide a prompt or use --interactive")
        _show(router.generate(args.prompt, args.route))
        return 0
    except LocalInferenceError as exc:
        print(f"Local inference failed and no fallback remains: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    sys.exit(main())
