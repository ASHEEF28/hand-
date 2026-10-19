"""Anomaly detection for gesture actions.

Two failure modes matter for a gesture-driven desktop:

1. **Spurious repeats** - a tracking glitch fires ``left_click`` 40 times a
   second.  Caught by a per-action rate limiter plus a z-score test against the
   recent action-rate distribution.
2. **Low-confidence noise** - actions dispatched from barely-recognised poses.
   Caught by a confidence floor.

``main.py`` consults :meth:`AnomalyDetector.is_anomaly` before executing an
action; anything filtered is counted in :meth:`stats` so the UI can show it.
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from utils import deep_merge

logger = logging.getLogger(__name__)

DEFAULT_ANOMALY_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "z_score_threshold": 3.0,
    "window_size": 100,
    "min_confidence": 0.35,
    "max_rate_per_action": 12.0,    # actions/second hard ceiling
    "rate_window": 1.0,
    "max_global_rate": 30.0,        # all actions combined
    "burst_size": 5,                # identical actions within burst_window = anomaly
    "burst_window": 0.25,
    "warmup_samples": 10,           # don't z-score before we have some history
}


class AnomalyDetector:
    """Flags actions that look like tracking noise rather than intent."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_ANOMALY_CONFIG), config or {})
        window = int(self.config.get("window_size", 100))
        self._intervals: Deque[float] = deque(maxlen=window)
        self._timestamps: Deque[float] = deque(maxlen=window * 4)
        self._per_action: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._recent_by_action: Counter = Counter()
        self.filtered: Counter = Counter()
        self.seen: Counter = Counter()
        self.reasons: List[Dict[str, Any]] = []
        self._last_action: Optional[str] = None
        self._last_action_time = 0.0
        self.enabled = bool(self.config.get("enabled", True))

    # ------------------------------------------------------------------ #
    def is_anomaly(self, action: str, confidence: float = 1.0, duration: float = 0.0) -> bool:
        """Return ``True`` when this action should be dropped."""
        action = str(action or "")
        now = time.time()
        self.seen[action] += 1

        if not self.enabled or not action:
            self._observe(action, now)
            return False

        reason = self._classify(action, float(confidence), float(duration), now)
        self._observe(action, now)

        if reason:
            self.filtered[action] += 1
            self._remember_reason(action, reason, confidence, now)
            return True
        return False

    def _classify(self, action: str, confidence: float, duration: float, now: float) -> Optional[str]:
        # 1. Confidence floor
        min_confidence = float(self.config.get("min_confidence", 0.35))
        if confidence < min_confidence:
            return f"low_confidence:{confidence:.2f}<{min_confidence:.2f}"

        # 2. Impossible duration (negative, or absurdly long for a discrete gesture)
        if duration < -0.01:
            return "negative_duration"

        history = self._per_action[action]

        # 3. Hard per-action rate ceiling
        max_rate = float(self.config.get("max_rate_per_action", 12.0))
        if max_rate > 0 and history:
            window = float(self.config.get("rate_window", 1.0))
            recent = sum(1 for t in history if now - t <= window)
            if recent / max(window, 1e-6) > max_rate:
                return f"rate_limit:{recent}/s>{max_rate}/s"

        # 4. Burst: the same action several times in a very short window
        burst_size = int(self.config.get("burst_size", 5))
        burst_window = float(self.config.get("burst_window", 0.25))
        if burst_size > 1:
            bursts = sum(1 for t in history if now - t <= burst_window)
            if bursts >= burst_size:
                return f"burst:{bursts}<{burst_window}s"

        # 5. Global rate ceiling (all actions combined)
        max_global = float(self.config.get("max_global_rate", 30.0))
        if max_global > 0 and self._timestamps:
            recent_global = sum(1 for t in self._timestamps if now - t <= 1.0)
            if recent_global > max_global:
                return f"global_rate:{recent_global}/s>{max_global}/s"

        # 6. Z-score against recent inter-arrival times
        z_threshold = float(self.config.get("z_score_threshold", 3.0))
        warmup = int(self.config.get("warmup_samples", 10))
        if z_threshold > 0 and len(self._intervals) >= warmup and self._last_action == action:
            interval = now - self._last_action_time
            if interval > 0:
                z = self._z_score(interval)
                if z is not None and z < -z_threshold:
                    # Interval far *shorter* than usual => machine-gun gesture.
                    return f"z_score:{z:.2f}"

        return None

    def _z_score(self, value: float) -> Optional[float]:
        if len(self._intervals) < 2:
            return None
        mean = sum(self._intervals) / len(self._intervals)
        variance = sum((v - mean) ** 2 for v in self._intervals) / max(len(self._intervals) - 1, 1)
        std = math.sqrt(variance)
        if std < 1e-6:
            return None
        return (value - mean) / std

    def _observe(self, action: str, now: float) -> None:
        if self._last_action is not None:
            interval = now - self._last_action_time
            if 0 < interval < 30:
                self._intervals.append(interval)
        self._last_action = action
        self._last_action_time = now
        self._timestamps.append(now)
        self._per_action[action].append(now)

    def _remember_reason(self, action: str, reason: str, confidence: float, now: float) -> None:
        self.reasons.append({"action": action, "reason": reason, "confidence": confidence, "t": now})
        if len(self.reasons) > 200:
            del self.reasons[: len(self.reasons) - 200]
        logger.debug("Anomaly filtered '%s' (%s)", action, reason)

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        total_seen = sum(self.seen.values())
        total_filtered = sum(self.filtered.values())
        return {
            "enabled": self.enabled,
            "seen": total_seen,
            "filtered": total_filtered,
            "filter_rate": (total_filtered / total_seen) if total_seen else 0.0,
            "top_filtered": self.filtered.most_common(5),
            "recent_reasons": self.reasons[-5:],
            "mean_interval": (sum(self._intervals) / len(self._intervals)) if self._intervals else 0.0,
        }

    def last_reason(self) -> Optional[str]:
        return self.reasons[-1]["reason"] if self.reasons else None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def reset(self) -> None:
        self._intervals.clear()
        self._timestamps.clear()
        self._per_action.clear()
        self.filtered.clear()
        self.seen.clear()
        self.reasons.clear()
        self._last_action = None
        self._last_action_time = 0.0


__all__ = ["AnomalyDetector", "DEFAULT_ANOMALY_CONFIG"]
