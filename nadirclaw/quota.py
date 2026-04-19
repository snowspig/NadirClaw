"""Quota tracking system for NadirClaw.

Tracks daily quota consumption for paid models (Claude Opus/Sonnet).
Quota is shared across all paid models and resets daily.

When quota is exhausted, requests are downgraded to cheaper/free models.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Set

from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw.quota")


# Quota costs per model (ppchat.vip pricing)
# Normal: standard request cost
# High token: cost when total tokens > threshold (Anthropic doubles pricing at 125k)
QUOTA_COSTS = {
    "claude-opus-4-6": {"normal": 35, "high_token": 70, "threshold": 125000},
    "claude-sonnet-4-6": {"normal": 20, "high_token": 40, "threshold": 125000},
    # Future support (not currently used)
    "openai-codex/gpt-5.4": {"normal": 12, "high_token": 12, "threshold": 999999},
}

# Opus→Sonnet routing switch threshold (lower than quota doubling threshold)
OPUS_TO_SONNET_TOKEN_THRESHOLD = 96000

# Models that share ppchat quota (suspend together when ppchat quota exhausted)
PPCHAT_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6", "openai-codex/gpt-5.4"}


class QuotaTracker:
    """Track daily quota consumption for paid models.

    Quota is shared across all paid models (Opus + Sonnet).
    State is persisted to disk and reset daily.

    When quota is exhausted, paid models are disabled until midnight (UTC).
    """

    def __init__(
        self,
        daily_quota: Optional[int] = None,
        state_file: Optional[Path] = None,
    ):
        self.daily_quota = daily_quota or 160000
        self._state_file = state_file or (settings.LOG_DIR / "quota_state.json")
        self._lock = Lock()

        # Quota accumulators
        self._quota_used: int = 0
        self._current_day: str = ""
        self._request_count: int = 0

        # Per-model tracking
        self._model_quota: Dict[str, int] = {}
        self._model_requests: Dict[str, int] = {}

        # Alert state
        self._warn_sent = False
        self._limit_sent = False

        # Paid models disabled flag (set when quota exhausted, reset at midnight)
        self._paid_models_disabled: bool = False

        # Per-provider suspension (ppchat quota exhausted from API error)
        self._suspended_providers: Set[str] = set()  # e.g., {"ppchat"}

        self._load_state()

    def _load_state(self) -> None:
        """Load persisted quota state from disk."""
        if not self._state_file.exists():
            self._reset_day()
            return

        try:
            data = json.loads(self._state_file.read_text())
            today = datetime.now(self.BEIJING_TZ).strftime("%Y-%m-%d")

            if data.get("day") == today:
                self._quota_used = data.get("quota_used", 0)
                self._request_count = data.get("request_count", 0)
                self._current_day = today
                self._model_quota = data.get("model_quota", {})
                self._model_requests = data.get("model_requests", {})
                self._paid_models_disabled = data.get("paid_models_disabled", False)
                # 恢复 alert 状态
                self._warn_sent = data.get("warn_sent", False)
                self._limit_sent = data.get("limit_sent", False)
                # 恢复 suspended providers
                self._suspended_providers = set(data.get("suspended_providers", []))
            else:
                self._reset_day()

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Quota state file corrupt, resetting: %s", e)
            self._reset_day()

    # Beijing timezone for midnight reset (UTC+8)
    BEIJING_TZ = timezone(timedelta(hours=8))

    def _reset_day(self) -> None:
        """Reset daily quota counters."""
        self._quota_used = 0
        self._request_count = 0
        self._current_day = datetime.now(self.BEIJING_TZ).strftime("%Y-%m-%d")
        self._model_quota = {}
        self._model_requests = {}
        self._warn_sent = False
        self._limit_sent = False
        self._paid_models_disabled = False
        self._suspended_providers = set()
        logger.info("Quota reset for new day: %s - paid models re-enabled", self._current_day)

    def _save_state(self) -> None:
        """Persist current quota state to disk."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "day": self._current_day,
            "quota_used": self._quota_used,
            "quota_limit": self.daily_quota,
            "request_count": self._request_count,
            "model_quota": dict(self._model_quota),
            "model_requests": dict(self._model_requests),
            "paid_models_disabled": self._paid_models_disabled,
            "warn_sent": self._warn_sent,
            "limit_sent": self._limit_sent,
            "suspended_providers": list(self._suspended_providers),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._state_file.write_text(json.dumps(data, indent=2))

    def calculate_cost(self, model: str, total_tokens: int) -> int:
        """Calculate quota cost for a request.

        Args:
            model: Model identifier
            total_tokens: Estimated total tokens (prompt + completion)

        Returns:
            Quota cost (35/70 for Opus, 20/40 for Sonnet, 0 for unknown models)
        """
        cost_info = QUOTA_COSTS.get(model)
        if not cost_info:
            # Unknown model = free/local, no quota cost
            return 0

        threshold = cost_info["threshold"]
        if total_tokens > threshold:
            return cost_info["high_token"]
        return cost_info["normal"]

    def can_afford(self, model: str, estimated_tokens: int) -> bool:
        """Check if there's enough quota for this request.

        Returns:
            True if quota is available, False if exhausted or paid models disabled
        """
        cost = self.calculate_cost(model, estimated_tokens)
        with self._lock:
            # 如果是付费模型且已禁用（配额耗尽），拒绝请求
            if cost > 0 and self._paid_models_disabled:
                return False
            return (self._quota_used + cost) <= self.daily_quota

    def is_paid_models_disabled(self) -> bool:
        """Check if paid models are currently disabled due to quota exhaustion."""
        with self._lock:
            return self._paid_models_disabled

    def is_model_suspended(self, model: str) -> bool:
        """Check if a specific model is suspended due to API quota error.

        Returns True if:
        - Global paid_models_disabled is set, OR
        - The model's provider is in suspended_providers
        """
        with self._lock:
            if self._paid_models_disabled:
                return True
            # Check if model belongs to a suspended provider
            if model in PPCHAT_MODELS and "ppchat" in self._suspended_providers:
                return True
            return False

    def suspend_provider(self, provider: str) -> None:
        """Suspend a provider until midnight UTC due to API quota error.

        Args:
            provider: Provider name (e.g., "ppchat")
        """
        with self._lock:
            if provider not in self._suspended_providers:
                self._suspended_providers.add(provider)
                logger.warning(
                    "🚫 Provider %s suspended until midnight UTC due to quota exhaustion",
                    provider,
                )
                self._save_state()

    def get_suspended_providers(self) -> Set[str]:
        """Get currently suspended providers.

        Also checks for day rollover and resets if needed.
        """
        with self._lock:
            # Check for day rollover before returning
            today = datetime.now(self.BEIJING_TZ).strftime("%Y-%m-%d")
            if today != self._current_day:
                self._reset_day()
            return set(self._suspended_providers)

    def increment_request_count(self) -> int:
        """Increment and return the daily request count."""
        with self._lock:
            # Check for day rollover
            today = datetime.now(self.BEIJING_TZ).strftime("%Y-%m-%d")
            if today != self._current_day:
                self._reset_day()
            self._request_count += 1
            return self._request_count

    def get_request_count(self) -> int:
        """Get current daily request count."""
        with self._lock:
            return self._request_count

    def record(self, model: str, total_tokens: int) -> Dict[str, Any]:
        """Record quota consumption for a completed request.

        Returns dict with: cost, quota_used, quota_remaining, alerts
        """
        cost = self.calculate_cost(model, total_tokens)

        with self._lock:
            # Check for day rollover
            today = datetime.now(self.BEIJING_TZ).strftime("%Y-%m-%d")
            if today != self._current_day:
                self._reset_day()

            self._quota_used += cost
            self._request_count += 1

            self._model_quota[model] = self._model_quota.get(model, 0) + cost
            self._model_requests[model] = self._model_requests.get(model, 0) + 1

            alerts = self._check_alerts()

            # Save every 5 requests to reduce IO
            if self._request_count % 5 == 0:
                self._save_state()

            return {
                "cost": cost,
                "quota_used": self._quota_used,
                "quota_limit": self.daily_quota,
                "quota_remaining": self.daily_quota - self._quota_used,
                "alerts": alerts,
            }

    def _check_alerts(self) -> list[str]:
        """Check quota status and return any new alerts."""
        alerts = []

        if self.daily_quota:
            ratio = self._quota_used / self.daily_quota
            if ratio >= 1.0 and not self._limit_sent:
                self._limit_sent = True
                # 配额耗尽，禁用付费模型到午夜
                self._paid_models_disabled = True
                msg = f"Daily quota exhausted: {self._quota_used} / {self.daily_quota} - paid models disabled until midnight"
                alerts.append(msg)
                logger.warning("🚨 %s", msg)
                # 立即保存状态
                self._save_state()
            elif ratio >= 0.8 and not self._warn_sent:
                self._warn_sent = True
                msg = f"Daily quota warning: {self._quota_used} / {self.daily_quota} ({ratio:.0%})"
                alerts.append(msg)
                logger.warning("⚠️ %s", msg)

        return alerts

    def get_status(self) -> Dict[str, Any]:
        """Get current quota status."""
        with self._lock:
            return {
                "day": self._current_day,
                "quota_used": self._quota_used,
                "quota_limit": self.daily_quota,
                "quota_remaining": self.daily_quota - self._quota_used,
                "request_count": self._request_count,
                "paid_models_disabled": self._paid_models_disabled,
                "suspended_providers": list(self._suspended_providers),
                "ppchat_suspended": "ppchat" in self._suspended_providers,
                "top_models": sorted(
                    [
                        {
                            "model": m,
                            "quota": q,
                            "requests": self._model_requests.get(m, 0),
                        }
                        for m, q in self._model_quota.items()
                    ],
                    key=lambda x: x["quota"],
                    reverse=True,
                )[:10],
            }

    def flush(self) -> None:
        """Force-save state to disk."""
        with self._lock:
            self._save_state()


# ---------------------------------------------------------------------------
# Global quota tracker (lazy init from env vars)
# ---------------------------------------------------------------------------

_quota_tracker: Optional[QuotaTracker] = None
_quota_init_lock = Lock()


def get_quota_tracker() -> QuotaTracker:
    """Get the global quota tracker, initializing from env vars if needed."""
    global _quota_tracker
    if _quota_tracker is None:
        with _quota_init_lock:
            if _quota_tracker is None:
                import os

                daily = os.getenv("NADIRCLAW_DAILY_QUOTA", "160000")
                _quota_tracker = QuotaTracker(
                    daily_quota=int(daily) if daily else 160000,
                )
    return _quota_tracker
