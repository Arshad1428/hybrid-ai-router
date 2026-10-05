# Hybrid AI Router: Ollama + Amazon Bedrock

A Python app that runs LLM inference locally through Ollama by default and can route requests to
Amazon Bedrock. Routing is centralized, falls back to the local model when the cloud path is
unavailable, and reports which provider answered each request.

## Project structure

```text
src/hybrid_router/
  config.py             environment-based settings
  local_inference.py    Ollama client
  bedrock_inference.py  Amazon Bedrock client (Converse API)
  router.py             centralized routing, guards and fallback
  main.py               CLI entry point
tests/test_router.py    unit tests (no Ollama or AWS needed)
docs/bedrock-iam-policy.json   least-privilege IAM policy example
.env.example            configuration template
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2               # then set OLLAMA_MODEL in .env to a model you have
```

## Run

```bash
export PYTHONPATH=src              # Windows PowerShell: $env:PYTHONPATH="src"

python -m pytest -q                                                      # tests
python -m hybrid_router.main "Your prompt"                               # local (default)
python -m hybrid_router.main "Your prompt" --route cloud                 # cloud route
python -m hybrid_router.main --interactive                               # prefix a line with 'cloud:' for cloud
```

Each response prints a `[meta]` line showing the provider that answered, whether a fallback
occurred, and the number of cloud calls used.

## Configuration

All settings live in `.env` (see `.env.example`): `DEFAULT_PROVIDER`, `AWS_CLOUD_ENABLED`,
`OLLAMA_MODEL`, `AWS_REGION`, `BEDROCK_MODEL_ID`, `MAX_OUTPUT_TOKENS`, `MAX_CLOUD_CALLS_PER_SESSION`.
The cloud route requires `AWS_CLOUD_ENABLED=true` and AWS credentials configured through a
standard AWS profile (`AWS_PROFILE`). Credentials are never stored in this repository.
