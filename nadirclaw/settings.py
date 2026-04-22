"""Minimal env-based configuration for NadirClaw."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_settings_logger = logging.getLogger(__name__)

# Load .env from ~/.nadirclaw/.env if it exists
_nadirclaw_dir = Path.home() / ".nadirclaw"
_env_file = _nadirclaw_dir / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    # Fallback to current directory .env
    load_dotenv()


class Settings:
    """All configuration from environment variables."""

    @property
    def AUTH_TOKEN(self) -> str:
        return os.getenv("NADIRCLAW_AUTH_TOKEN", "")

    @property
    def SIMPLE_MODEL(self) -> str:
        """Model for simple prompts. Falls back to last model in MODELS list."""
        explicit = os.getenv("NADIRCLAW_SIMPLE_MODEL", "")
        if explicit:
            return explicit
        models = self.MODELS
        return models[-1] if models else "gemini-3-flash-preview"

    @property
    def COMPLEX_MODEL(self) -> str:
        """Model for complex prompts. Falls back to first model in MODELS list."""
        explicit = os.getenv("NADIRCLAW_COMPLEX_MODEL", "")
        if explicit:
            return explicit
        models = self.MODELS
        return models[0] if models else "openai-codex/gpt-5.3-codex"

    @property
    def MODELS(self) -> list[str]:
        raw = os.getenv(
            "NADIRCLAW_MODELS",
            "openai-codex/gpt-5.3-codex,gemini-3-flash-preview",
        )
        return [m.strip() for m in raw.split(",") if m.strip()]

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def OPENAI_API_KEY(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def GEMINI_API_KEY(self) -> str:
        return os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    @property
    def OLLAMA_API_BASE(self) -> str:
        return os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    @property
    def API_BASE(self) -> str:
        """Custom base URL for OpenAI-compatible endpoints (vLLM, LocalAI, etc.).

        When set, passed as api_base to all non-Ollama, non-Gemini LiteLLM calls.
        """
        return os.getenv("NADIRCLAW_API_BASE", "")

    @property
    def ZAI_API_BASE(self) -> str:
        return os.getenv("ZAI_API_BASE", "")

    @property
    def ZAI_API_KEY(self) -> str:
        return os.getenv("ZAI_API_KEY", "")

    @property
    def KIMI_API_BASE(self) -> str:
        return os.getenv("KIMI_API_BASE", "")

    @property
    def KIMI_API_KEY(self) -> str:
        return os.getenv("KIMI_API_KEY", "")

    @property
    def MINIMAX_API_BASE(self) -> str:
        return os.getenv("MINIMAX_API_BASE", "")

    @property
    def MINIMAX_API_KEY(self) -> str:
        return os.getenv("MINIMAX_API_KEY", "")

    @property
    def HOSTED_VLLM_API_BASE(self) -> str:
        return os.getenv("HOSTED_VLLM_API_BASE", "http://localhost:8000/v1")

    @property
    def HOSTED_VLLM_API_KEY(self) -> str:
        return os.getenv("HOSTED_VLLM_API_KEY", "none")

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        return float(os.getenv("NADIRCLAW_CONFIDENCE_THRESHOLD", "0.06"))

    @property
    def MID_MODEL(self) -> str:
        """Model for mid-complexity prompts. Falls back to SIMPLE_MODEL."""
        return os.getenv("NADIRCLAW_MID_MODEL", "") or self.SIMPLE_MODEL

    @property
    def has_mid_tier(self) -> bool:
        """True if MID_MODEL is explicitly set via env."""
        return bool(os.getenv("NADIRCLAW_MID_MODEL"))

    @property
    def PORT(self) -> int:
        return int(os.getenv("NADIRCLAW_PORT", "8856"))

    @property
    def LOG_RAW(self) -> bool:
        """When True, log full raw request messages and response content."""
        return os.getenv("NADIRCLAW_LOG_RAW", "").lower() in ("1", "true", "yes")

    @property
    def LOG_DIR(self) -> Path:
        return Path(os.getenv("NADIRCLAW_LOG_DIR", "~/.nadirclaw/logs")).expanduser()

    @property
    def CREDENTIALS_FILE(self) -> Path:
        return Path.home() / ".nadirclaw" / "credentials.json"

    @property
    def LOG_MAX_SIZE_MB(self) -> int:
        return int(os.getenv("NADIRCLAW_LOG_MAX_SIZE_MB", "50"))

    @property
    def LOG_RETENTION_DAYS(self) -> int:
        return int(os.getenv("NADIRCLAW_LOG_RETENTION_DAYS", "30"))

    @property
    def LOG_COMPRESS(self) -> bool:
        return os.getenv("NADIRCLAW_LOG_COMPRESS", "true").lower() in ("1", "true", "yes")

    @property
    def OPTIMIZE(self) -> str:
        return os.getenv("NADIRCLAW_OPTIMIZE", "off")

    @property
    def OPTIMIZE_MAX_TURNS(self) -> int:
        return int(os.getenv("NADIRCLAW_OPTIMIZE_MAX_TURNS", "20"))

    @property
    def REASONING_MODEL(self) -> str:
        """Model for reasoning tasks. Falls back to COMPLEX_MODEL."""
        return os.getenv("NADIRCLAW_REASONING_MODEL", "") or self.COMPLEX_MODEL

    @property
    def SONNET_MODEL(self) -> str:
        """Model for medium-complexity tasks (2000-20000 tokens). Falls back to COMPLEX_MODEL."""
        return os.getenv("NADIRCLAW_SONNET_MODEL", "") or self.COMPLEX_MODEL

    @property
    def EXPLORE_MODEL(self) -> str:
        """Model for Claude Code explore agent. Falls back to COMPLEX_MODEL."""
        return os.getenv("NADIRCLAW_EXPLORE_MODEL", "") or self.COMPLEX_MODEL

    @property
    def FREE_MODEL(self) -> str:
        """Free fallback model. Falls back to SIMPLE_MODEL."""
        return os.getenv("NADIRCLAW_FREE_MODEL", "") or self.SIMPLE_MODEL

    @property
    def SUBAGENT_MODEL(self) -> str:
        """Model for Claude Code subagent/execution tasks.
        These are background coding-plan tasks that use API quota.
        Falls back to COMPLEX_MODEL if not set.
        """
        return os.getenv("NADIRCLAW_SUBAGENT_MODEL", "") or self.COMPLEX_MODEL

    @property
    def EXECUTION_MODEL(self) -> str:
        """Model for Claude Code execution tasks. Falls back to SUBAGENT_MODEL."""
        return os.getenv("NADIRCLAW_EXECUTION_MODEL", "") or self.SUBAGENT_MODEL

    @property
    def LONG_CONTEXT_MODEL(self) -> str:
        """Model for long context requests. Falls back to REASONING_MODEL."""
        return os.getenv("NADIRCLAW_LONG_CONTEXT_MODEL", "") or self.REASONING_MODEL

    # ============================================================
    # Complex Coding Detection Configuration
    # ============================================================

    @property
    def COMPLEX_THRESHOLD(self) -> float:
        """Threshold for complex coding detection.

        Higher values = more conservative (fewer requests routed to Complex).
        Lower values = more aggressive (more requests routed to Complex).

        Recommended values:
        - 0.50: Aggressive (start here for testing)
        - 0.60: Conservative (production default)
        - 0.70: Very conservative (reduce cost)
        """
        return float(os.getenv("NADIRCLAW_COMPLEX_THRESHOLD", "0.60"))

    @property
    def COMPLEX_WEIGHT_EDITING(self) -> float:
        """Weight for heavy editing signal (3+ Edit/Write calls)."""
        return float(os.getenv("NADIRCLAW_COMPLEX_WEIGHT_EDITING", "0.50"))

    @property
    def COMPLEX_WEIGHT_COMBO(self) -> float:
        """Weight for tool combination signal (Read + Edit + Bash)."""
        return float(os.getenv("NADIRCLAW_COMPLEX_WEIGHT_COMBO", "0.30"))

    @property
    def COMPLEX_WEIGHT_CONVERSATION(self) -> float:
        """Weight for deep conversation signal (10+ messages)."""
        return float(os.getenv("NADIRCLAW_COMPLEX_WEIGHT_CONVERSATION", "0.20"))

    @property
    def COMPLEX_WEIGHT_KEYWORDS(self) -> float:
        """Weight for coding keywords signal (implement, refactor, etc.)."""
        return float(os.getenv("NADIRCLAW_COMPLEX_WEIGHT_KEYWORDS", "0.30"))

    @property
    def REVIEW_MODEL(self) -> str:
        """Model for code review/verification tasks. Falls back to COMPLEX_MODEL."""
        return os.getenv("NADIRCLAW_REVIEW_MODEL", "") or self.COMPLEX_MODEL

    @property
    def FALLBACK_CHAIN(self) -> list[str]:
        """Ordered fallback chain. When a model fails, try the next one.

        Defaults to [COMPLEX_MODEL, SIMPLE_MODEL] (existing behavior).
        Set NADIRCLAW_FALLBACK_CHAIN to customize, e.g.:
          NADIRCLAW_FALLBACK_CHAIN=gpt-5.4,claude-sonnet-4-6,gemini-2.5-flash
        """
        raw = os.getenv("NADIRCLAW_FALLBACK_CHAIN", "")
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        # Default: deduplicated list of all configured tier models
        chain = []
        for m in [self.COMPLEX_MODEL, self.MID_MODEL, self.SIMPLE_MODEL, self.REASONING_MODEL, self.FREE_MODEL]:
            if m and m not in chain:
                chain.append(m)
        return chain

    def get_tier_fallback_chain(self, tier: str) -> list[str]:
        """Get the fallback chain for a specific tier.

        Per-tier chains are configured via env vars:
          NADIRCLAW_SIMPLE_FALLBACK=gemini-2.5-flash,gemini-3-flash-preview
          NADIRCLAW_MID_FALLBACK=gpt-5.4-mini,gemini-2.5-flash
          NADIRCLAW_COMPLEX_FALLBACK=claude-sonnet-4-6,gpt-5.4

        When a per-tier chain is set, it is used instead of the global chain.
        If no per-tier chain is configured, falls back to the global FALLBACK_CHAIN.
        """
        env_key = f"NADIRCLAW_{tier.upper()}_FALLBACK"
        raw = os.getenv(env_key, "")
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        return self.FALLBACK_CHAIN

    @property
    def MODEL_RATE_LIMITS(self) -> str:
        """Per-model rate limits. Format: model=rpm,model2=rpm2."""
        return os.getenv("NADIRCLAW_MODEL_RATE_LIMITS", "")

    @property
    def DEFAULT_MODEL_RPM(self) -> int:
        """Default max requests/minute per model. 0 = unlimited."""
        try:
            return max(0, int(os.getenv("NADIRCLAW_DEFAULT_MODEL_RPM", "0")))
        except ValueError:
            return 0

    @property
    def has_explicit_tiers(self) -> bool:
        """True if SIMPLE_MODEL and COMPLEX_MODEL are explicitly set via env."""
        return bool(
            os.getenv("NADIRCLAW_SIMPLE_MODEL") and os.getenv("NADIRCLAW_COMPLEX_MODEL")
        )

    @property
    def tier_models(self) -> list[str]:
        """Deduplicated list of all tier models + Claude Code model aliases."""
        seen = set()
        models = []
        # Include all tier models + full fallback chain
        all_models = [
            self.SIMPLE_MODEL,
            self.COMPLEX_MODEL,
            self.MID_MODEL,
            self.SONNET_MODEL,
            self.REASONING_MODEL,
            self.REVIEW_MODEL,
            self.FREE_MODEL,
        ] + self.FALLBACK_CHAIN
        for m in all_models:
            if m and m not in seen:
                seen.add(m)
                models.append(m)
        # Claude Code queries /v1/models to verify its selected model exists.
        # Expose common Claude model IDs so CC doesn't reject them.
        for alias in [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ]:
            if alias not in seen:
                seen.add(alias)
                models.append(alias)
        return models

    @property
    def CONTEXT_COMPRESSION(self) -> bool:
        """Enable context compression for long conversations."""
        return os.getenv("NADIRCLAW_CONTEXT_COMPRESSION", "false").lower() in ("true", "1", "yes")

    @property
    def COMPRESS_MIN_MESSAGES(self) -> int:
        """Minimum message count before compression kicks in."""
        return int(os.getenv("NADIRCLAW_COMPRESS_MIN_MESSAGES", "30"))

    @property
    def COMPRESS_RECENT_WINDOW(self) -> int:
        """Number of recent messages to preserve intact."""
        return int(os.getenv("NADIRCLAW_COMPRESS_RECENT_WINDOW", "20"))

    @property
    def COMPRESS_TOOL_OUTPUT_MAX(self) -> int:
        """Max characters for truncated tool output."""
        return int(os.getenv("NADIRCLAW_COMPRESS_TOOL_MAX", "500"))

    @property
    def COMPRESS_VLLM_SUMMARY(self) -> bool:
        """Use vLLM to summarize old messages instead of truncating."""
        return os.getenv("NADIRCLAW_COMPRESS_VLLM_SUMMARY", "false").lower() in ("true", "1", "yes")

    @property
    def COMPRESS_VLLM_URL(self) -> str:
        """vLLM endpoint for context summarization."""
        return os.getenv("NADIRCLAW_COMPRESS_VLLM_URL", "http://localhost:8000/v1/chat/completions")

    @property
    def COMPRESS_VLLM_MODEL(self) -> str:
        """Model name for vLLM summarization."""
        return os.getenv("NADIRCLAW_COMPRESS_VLLM_MODEL", "Qwopus3.5-27B-v3")

    @property
    def AGENT_ROLE_DETECTION(self) -> bool:
        """Enable agent role detection for coding agents (opt-in)."""
        return os.getenv("NADIRCLAW_AGENT_ROLE_DETECTION", "false").lower() in ("true", "1", "yes")


settings = Settings()
