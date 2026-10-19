"""Recent-gesture history used by the UI and analytics."""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """One dispatched action."""

    action: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    duration: float = 0.0
    gesture: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("callback", None)
        return payload


class GestureHistory:
    """Ring buffer of recent actions with cheap aggregate queries."""

    def __init__(self, max_size: int = 50, on_push: Optional[Callable[[HistoryEntry], None]] = None) -> None:
        self.max_size = max(1, int(max_size))
        self._entries: Deque[HistoryEntry] = deque(maxlen=self.max_size)
        self._callbacks: Deque[Callable[[], Any]] = deque(maxlen=self.max_size)
        self._counts: Counter = Counter()
        self.on_push = on_push
        self.total_pushed = 0

    # ------------------------------------------------------------------ #
    def push(self, action: str, callback: Optional[Callable[[], Any]] = None,
             data: Optional[Dict[str, Any]] = None) -> HistoryEntry:
        """Record an action.

        ``main.py`` calls ``push(action, lambda: None, {})``; the callback slot
        exists so an entry can be replayed/undone later.
        """
        data = data or {}
        entry = HistoryEntry(
            action=str(action),
            timestamp=float(data.get("timestamp", time.time())),
            confidence=float(data.get("confidence", 1.0)),
            duration=float(data.get("duration", 0.0)),
            gesture=str(data.get("gesture", "")),
            data={k: v for k, v in data.items() if k not in ("timestamp", "confidence", "duration", "gesture")},
        )
        self._entries.append(entry)
        self._callbacks.append(callback if callable(callback) else (lambda: None))
        self._counts[entry.action] += 1
        self.total_pushed += 1
        if self.on_push:
            try:
                self.on_push(entry)
            except Exception as exc:  # pragma: no cover
                logger.debug("GestureHistory callback failed: %s", exc)
        return entry

    # ------------------------------------------------------------------ #
    def recent(self, count: int = 10) -> List[HistoryEntry]:
        items = list(self._entries)
        return items[-max(0, int(count)):]

    def actions(self, count: int = 10) -> List[str]:
        return [entry.action for entry in self.recent(count)]

    def last(self) -> Optional[HistoryEntry]:
        return self._entries[-1] if self._entries else None

    def last_action(self) -> str:
        entry = self.last()
        return entry.action if entry else ""

    def count_for(self, action: str) -> int:
        return int(self._counts.get(action, 0))

    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def most_common(self, count: int = 5) -> List[tuple]:
        return self._counts.most_common(int(count))

    def rate(self, window_seconds: float = 5.0) -> float:
        """Actions per second over the recent window."""
        if not self._entries:
            return 0.0
        now = time.time()
        recent = [e for e in self._entries if now - e.timestamp <= window_seconds]
        return len(recent) / max(window_seconds, 1e-6)

    def find(self, action: str) -> List[HistoryEntry]:
        return [e for e in self._entries if e.action == action]

    def replay(self, index: int = -1) -> bool:
        """Re-run the callback stored with an entry."""
        try:
            callback = self._callbacks[index]
        except IndexError:
            return False
        if not callable(callback):
            return False
        try:
            callback()
            return True
        except Exception as exc:
            logger.error("History replay failed: %s", exc)
            return False

    def clear(self) -> None:
        self._entries.clear()
        self._callbacks.clear()
        self._counts.clear()
        self.total_pushed = 0

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ #
    def to_list(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries]

    def export_json(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "entries": self.to_list()}, handle, indent=2)
            return True
        except Exception as exc:
            logger.error("Could not export gesture history: %s", exc)
            return False


__all__ = ["GestureHistory", "HistoryEntry"]
