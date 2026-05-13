# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run tests
pytest                                    # full suite
pytest tests/test_routing.py -v           # single file
pytest -x                                 # stop on first failure
pytest tests/ --ignore=tests/test_server.py  # CI config (server tests excluded)

# Run the server
nadirclaw serve                           # default port 8856
python -m nadirclaw.server                # alternative

# Build
python -m build
```

No linter/formatter is configured. Keep code consistent with surrounding style.

## Architecture

NadirClaw is an LLM routing proxy that sits between AI coding tools and LLM providers. It classifies prompt complexity in ~10ms and routes simple prompts to cheap models, complex ones to premium models.

### Request Flow

```
Client → /v1/chat/completions or /v1/messages
  → auth.py (Bearer/X-API-Key)
  → rate_limit.py (per-model sliding window)
  → routing.py (profile → alias → session cache → smart routing)
  → classifier.py (embedding-based binary complexity scoring)
  → routing.py apply_routing_modifiers() (agentic/reasoning/vision/role detection cascade)
  → server.py _dispatch_model() → provider call
  → anthropic_api.py or server.py _call_litellm/_call_gemini
  → fallback chain on failure (provider_health.py skips cooling-down models)
```

### Three Dispatch Paths

1. **Direct Anthropic** (`anthropic_api.py`): Anthropic, ZAI, Kimi, MiniMax — raw httpx to Anthropic Messages API endpoints. No format conversion.
2. **LiteLLM** (`server.py _call_litellm`): OpenAI, Ollama, DeepSeek, vLLM — converts to OpenAI format, LiteLLM dispatches.
3. **Native Gemini** (`server.py _call_gemini`): Google models via `google-genai` SDK directly.

The `/v1/messages` endpoint (`anthropic_api.py`) adds bidirectional Anthropic↔OpenAI format conversion for non-Anthropic providers.

### Key Modules

- `routing.py` — Model registry/aliases, routing profiles (auto/eco/premium/free/reasoning), modifier detection cascade, model pools, context window checks. `resolve_alias()` is a single dict lookup (no chain).
- `classifier.py` — Binary complexity classifier using `all-MiniLM-L6-v2` sentence embeddings with pre-computed centroids (`.npy` files). `classify()` returns simple/mid/complex.
- `server.py` — FastAPI app, main dispatch logic, streaming, fallback orchestration. Largest file (~1400 lines).
- `anthropic_api.py` — `/v1/messages` endpoint, PPChat multi-domain failover (429 rotation with cooldown).
- `settings.py` — All config via env vars with `@property` accessors. Loads from `~/.nadirclaw/.env` or `./.env`.
- `credentials.py` — Resolution chain: OpenClaw stored → NadirClaw `~/.nadirclaw/credentials.json` → env var.
- `quota.py` — Daily quota tracking for ppchat-paid models, persists to disk, resets at Beijing midnight.
- `provider_health.py` — In-memory rolling health tracker with cooldown for failing models.

### Configuration

All config is env vars loaded by `python-dotenv`. Tier model slots: `NADIRCLAW_SIMPLE_MODEL`, `COMPLEX_MODEL`, `EXPLORE_MODEL`, `REASONING_MODEL`, `SUBAGENT_MODEL`, `EXECUTION_MODEL`, `FREE_MODEL`, `REVIEW_MODEL`, `LONG_CONTEXT_MODEL`. Per-tier fallback chains via `NADIRCLAW_*_FALLBACK`.

## Key Patterns

- **Lazy imports everywhere** — Most cross-module imports happen inside functions to avoid circular deps and reduce cold-start. Don't fight this pattern.
- **Thread-safe singletons** — `encoder.py`, `classifier.py`, `cache.py`, `quota.py`, `provider_health.py` all use `threading.Lock` with double-checked locking for lazy init.
- **Provider auto-detection** — Model name prefix determines dispatch path: `claude-` → anthropic, `gpt-` → openai, `ollama/` → ollama, `deepseek/` → deepseek, `gemini-` → gemini, etc.
- **Fallback chains** — `_call_with_fallback()` tries models in order, skipping ones in cooldown via `provider_health_tracker.ordered_candidates()`.
- **Model pools** — Weighted random load balancing across multiple models in a tier (e.g., `pool-reasoning=gpt-5.5/gpt-5.4:glm-5.1 = 5:5`).

## Testing

Tests use `autouse` fixtures to redirect credentials to `tmp_path` and clear env vars via `monkeypatch`. E2E tests use `fastapi.testclient.TestClient` with mocked LLM calls. `test_server.py` is excluded from CI.

## Upstream

Fork of `doramirdor/NadirClaw` (now `NadirRouter/NadirClaw`). Branch `integration/all-features` accumulates merged feature branches before syncing with upstream main.
