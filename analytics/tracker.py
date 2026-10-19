"""Gesture analytics: accuracy, latency and usage tracking.

``main.py`` calls :meth:`record_gesture` for every dispatched action and
:meth:`save` on shutdown.  Data is kept in memory and periodically flushed to
``analytics/analytics.json`` so the Settings/Analytics tab can show real numbers
(FPS, latency, per-gesture success rate, most used actions).
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from utils import deep_merge, project_root

logger = logging.getLogger(__name__)

DEFAULT_ANALYTICS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "save_interval": 10,            # records between auto-saves
    "track_accuracy": True,
    "store_path": "analytics/analytics.json",
    "latency_window": 300,
    "fps_window": 120,
    "session_history": 50,
}


class GestureAnalytics:
    """Collects per-session and cumulative gesture statistics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        section = config.get("analytics", config) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_ANALYTICS_CONFIG), section or {})

        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self.session_start = time.time()
        self.records_since_save = 0
        self.total_gestures = 0

        self._latencies: Deque[float] = deque(maxlen=int(self.config.get("latency_window", 300)))
        self._fps: Deque[float] = deque(maxlen=int(self.config.get("fps_window", 120)))
        self._gesture_counts: Counter = Counter()
        self._action_counts: Counter = Counter()
        self._success: Dict[str, List[bool]] = defaultdict(list)
        self._confidence: Dict[str, List[float]] = defaultdict(list)
        self._timeline: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.sessions: List[Dict[str, Any]] = []
        self.enabled = bool(self.config.get("enabled", True))
        self._load()

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    @property
    def store_path(self) -> str:
        path = str(self.config.get("store_path", "analytics/analytics.json"))
        return path if os.path.isabs(path) else os.path.join(project_root(), path)

    def _load(self) -> None:
        path = self.store_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.sessions = list(data.get("sessions", []))[-int(self.config.get("session_history", 50)):]
            self.total_gestures = int(data.get("total_gestures", 0))
            self._gesture_counts.update(data.get("gesture_counts", {}) or {})
            self._action_counts.update(data.get("action_counts", {}) or {})
            logger.info("Loaded analytics (%d sessions, %d gestures)", len(self.sessions), self.total_gestures)
        except Exception as exc:
            logger.error("Could not load analytics from %s: %s", path, exc)

    def save(self) -> bool:
        """Persist cumulative counters plus a summary of this session."""
        if not self.enabled:
            return False
        path = self.store_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            sessions = list(self.sessions)
            sessions.append(self.session_summary())
            sessions = sessions[-int(self.config.get("session_history", 50)):]
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "total_gestures": self.total_gestures,
                "gesture_counts": dict(self._gesture_counts),
                "action_counts": dict(self._action_counts),
                "accuracy": self.accuracy_by_gesture(),
                "sessions": sessions,
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self.sessions = sessions
            self.records_since_save = 0
            return True
        except Exception as exc:
            logger.error("Could not save analytics: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record_gesture(
        self,
        gesture: str,
        action: str,
        confidence: float = 1.0,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record one dispatched gesture."""
        if not self.enabled:
            return
        now = time.time()
        self.total_gestures += 1
        self.records_since_save += 1
        self._gesture_counts[str(gesture)] += 1
        self._action_counts[str(action)] += 1

        if self.config.get("track_accuracy", True):
            self._success[str(gesture)].append(bool(success))
            self._confidence[str(gesture)].append(float(confidence))

        if latency_ms and latency_ms > 0:
            self._latencies.append(float(latency_ms))

        self._timeline.append({
            "gesture": str(gesture), "action": str(action),
            "confidence": round(float(confidence), 3),
            "latency_ms": round(float(latency_ms), 2),
            "success": bool(success), "t": now,
        })

        interval = int(self.config.get("save_interval", 10))
        if interval > 0 and self.records_since_save >= interval:
            self.save()

    def record_fps(self, fps: float) -> None:
        if self.enabled and fps and fps > 0:
            self._fps.append(float(fps))

    def record_latency(self, latency_ms: float) -> None:
        if self.enabled and latency_ms > 0:
            self._latencies.append(float(latency_ms))

    def record_error(self, where: str, message: str) -> None:
        if not self.enabled:
            return
        self._timeline.append({"error": where, "message": str(message)[:300], "t": time.time()})

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def accuracy_by_gesture(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for gesture, outcomes in self._success.items():
            if outcomes:
                result[gesture] = round(sum(1 for ok in outcomes if ok) / len(outcomes), 4)
        return result

    def average_confidence(self, gesture: Optional[str] = None) -> float:
        if gesture:
            values = self._confidence.get(gesture, [])
        else:
            values = [v for values in self._confidence.values() for v in values]
        return round(statistics.fmean(values), 4) if values else 0.0

    def latency_stats(self) -> Dict[str, float]:
        values = list(self._latencies)
        if not values:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "samples": 0}
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
        return {
            "avg": round(statistics.fmean(values), 2),
            "min": round(ordered[0], 2),
            "max": round(ordered[-1], 2),
            "p95": round(ordered[index], 2),
            "samples": len(values),
        }

    def fps_stats(self) -> Dict[str, float]:
        values = list(self._fps)
        if not values:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "samples": 0}
        return {
            "avg": round(statistics.fmean(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "samples": len(values),
        }

    def top_actions(self, count: int = 10) -> List[tuple]:
        return self._action_counts.most_common(int(count))

    def top_gestures(self, count: int = 10) -> List[tuple]:
        return self._gesture_counts.most_common(int(count))

    def recent(self, count: int = 20) -> List[Dict[str, Any]]:
        items = list(self._timeline)
        return items[-max(0, int(count)):]

    def session_summary(self) -> Dict[str, Any]:
        duration = max(time.time() - self.session_start, 1e-6)
        session_gestures = sum(self._gesture_counts.values())
        return {
            "session_id": self.session_id,
            "started_at": self.session_start,
            "duration_s": round(duration, 1),
            "gestures": session_gestures,
            "gestures_per_minute": round(session_gestures / (duration / 60.0), 2),
            "accuracy": self.accuracy_by_gesture(),
            "latency": self.latency_stats(),
            "fps": self.fps_stats(),
            "top_actions": self.top_actions(5),
        }

    def summary(self) -> Dict[str, Any]:
        """Everything the UI needs in one call."""
        return {
            "enabled": self.enabled,
            "session": self.session_summary(),
            "total_gestures": self.total_gestures + sum(self._gesture_counts.values()) - sum(self._gesture_counts.values()),
            "lifetime_gestures": self.total_gestures,
            "gesture_counts": dict(self._gesture_counts),
            "action_counts": dict(self._action_counts),
            "accuracy": self.accuracy_by_gesture(),
            "average_confidence": self.average_confidence(),
            "latency": self.latency_stats(),
            "fps": self.fps_stats(),
            "top_actions": self.top_actions(10),
            "sessions_recorded": len(self.sessions),
        }

    def reset(self) -> None:
        self._gesture_counts.clear()
        self._action_counts.clear()
        self._success.clear()
        self._confidence.clear()
        self._latencies.clear()
        self._fps.clear()
        self._timeline.clear()
        self.records_since_save = 0
        self.session_start = time.time()
        self.session_id = time.strftime("%Y%m%d-%H%M%S")

    def export_csv(self, path: Optional[str] = None) -> Optional[str]:
        """Write the in-memory timeline to CSV; returns the path."""
        target = path or os.path.join(project_root(), "analytics", "timeline.csv")
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            import csv

            with open(target, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "gesture", "action", "confidence", "latency_ms", "success", "error"])
                for entry in self._timeline:
                    writer.writerow([
                        entry.get("t", ""), entry.get("gesture", ""), entry.get("action", ""),
                        entry.get("confidence", ""), entry.get("latency_ms", ""),
                        entry.get("success", ""), entry.get("message", ""),
                    ])
            return target
        except Exception as exc:
            logger.error("Could not export analytics CSV: %s", exc)
            return None


__all__ = ["GestureAnalytics", "DEFAULT_ANALYTICS_CONFIG"]
