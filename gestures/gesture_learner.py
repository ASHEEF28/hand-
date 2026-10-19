"""Adaptive gesture learning.

Lets a user *teach* GestureOS a new gesture by holding it in front of the
camera for a moment.  Templates are stored as normalised landmark sequences and
matched with cosine similarity, so this needs no ML dependencies and works
offline.  Persisted to ``gestures/learned.json``.

Used by ``main.py``:

    match = self.gesture_learner.match(hand)
    if match and match[1] > 0.8:
        self.macros.execute(match[0])
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import clamp, deep_merge, project_root
from vision.hand_tracker import Hand

logger = logging.getLogger(__name__)

DEFAULT_LEARNER_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "match_threshold": 0.8,
    "history_frames": 15,
    "store_path": "gestures/learned.json",
    "max_templates_per_gesture": 8,
    "min_record_frames": 5,
    "auto_learn": False,
    "auto_learn_confidence": 0.92,
}


class GestureLearner:
    """Records and matches user-defined gestures."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_LEARNER_CONFIG), config or {})
        self.templates: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._recording: Optional[str] = None
        self._recorded: Deque[np.ndarray] = deque(maxlen=int(self.config.get("history_frames", 15)))
        self._recent_matches: Dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _store_path(self) -> str:
        path = str(self.config.get("store_path", "gestures/learned.json"))
        return path if os.path.isabs(path) else os.path.join(project_root(), path)

    def _load(self) -> None:
        path = self._store_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            for name, vectors in (raw.get("templates", {}) or {}).items():
                for vector in vectors:
                    self.templates[name].append(np.asarray(vector, dtype=np.float32))
            logger.info("Loaded %d learned gesture(s): %s", len(self.templates), ", ".join(self.templates))
        except Exception as exc:
            logger.error("Could not load learned gestures from %s: %s", path, exc)

    def save(self) -> bool:
        path = self._store_path()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "templates": {
                    name: [vec.tolist() for vec in vectors] for name, vectors in self.templates.items()
                },
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            return True
        except Exception as exc:
            logger.error("Could not save learned gestures: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #
    @staticmethod
    def _vectorize(hand: Hand) -> np.ndarray:
        """Scale/translation invariant 42-D descriptor for a hand pose."""
        if len(hand.landmarks) < 21:
            return np.zeros(42, dtype=np.float32)
        wrist = hand.landmarks[0]
        scale = hand.hand_size or 1e-6
        values: List[float] = []
        for lm in hand.landmarks[1:21]:
            values.append((lm.x - wrist.x) / scale)
            values.append((lm.y - wrist.y) / scale)
        values.extend(1.0 if up else 0.0 for up in hand.gesture_data.get("fingers_up", [False] * 5)[:2])
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return clamp(float(np.dot(a, b) / (na * nb)), -1.0, 1.0)

    # ------------------------------------------------------------------ #
    # Teaching
    # ------------------------------------------------------------------ #
    def start_recording(self, name: str) -> None:
        """Begin capturing frames for a new gesture called ``name``."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Gesture name must not be empty")
        self._recording = name
        self._recorded.clear()
        logger.info("Recording gesture '%s' - hold it steady", name)

    def is_recording(self) -> bool:
        return self._recording is not None

    def recording_name(self) -> Optional[str]:
        return self._recording

    def record(self, hand: Hand) -> int:
        """Add one frame to the active recording; returns frames captured."""
        if self._recording is None or hand is None:
            return 0
        if not hand.gesture_data:
            hand.refresh_gesture_data()
        self._recorded.append(self._vectorize(hand))
        return len(self._recorded)

    def record_many(self, hands: Sequence[Hand]) -> int:
        if not hands:
            return 0
        return self.record(hands[0])

    def stop_recording(self) -> Optional[str]:
        """Finish the recording and store a template. Returns the gesture name."""
        name = self._recording
        self._recording = None
        if name is None:
            return None

        frames = list(self._recorded)
        self._recorded.clear()
        minimum = int(self.config.get("min_record_frames", 5))
        if len(frames) < minimum:
            logger.warning(
                "Gesture '%s' discarded: only %d frames captured (need %d)",
                name, len(frames), minimum,
            )
            return None

        mean_vector = np.mean(np.vstack(frames), axis=0).astype(np.float32)
        templates = self.templates[name]
        templates.append(mean_vector)
        max_templates = int(self.config.get("max_templates_per_gesture", 8))
        while len(templates) > max_templates:
            templates.pop(0)
        self.save()
        logger.info("Learned gesture '%s' from %d frames", name, len(frames))
        return name

    def forget(self, name: str) -> bool:
        """Delete a learned gesture."""
        if name in self.templates:
            del self.templates[name]
            self.save()
            return True
        return False

    def known_gestures(self) -> List[str]:
        return sorted(self.templates.keys())

    # ------------------------------------------------------------------ #
    # Matching
    # ------------------------------------------------------------------ #
    def match(self, hand: Hand) -> Optional[Tuple[str, float]]:
        """Best matching learned gesture for ``hand`` as ``(name, score)``."""
        if not self.templates or hand is None:
            return None
        if not hand.gesture_data:
            hand.refresh_gesture_data()

        vector = self._vectorize(hand)
        best_name: Optional[str] = None
        best_score = -1.0
        for name, templates in self.templates.items():
            score = max(self._cosine(vector, template) for template in templates)
            if score > best_score:
                best_name, best_score = name, score

        if best_name is None:
            return None
        self._recent_matches[best_name] = best_score
        return best_name, clamp(best_score, 0.0, 1.0)

    def match_if_confident(self, hand: Hand) -> Optional[str]:
        """Return a gesture name only when the score passes ``match_threshold``."""
        result = self.match(hand)
        if not result:
            return None
        name, score = result
        return name if score >= float(self.config.get("match_threshold", 0.8)) else None

    def last_scores(self) -> Dict[str, float]:
        return dict(self._recent_matches)

    def reset(self) -> None:
        self._recording = None
        self._recorded.clear()
        self._recent_matches.clear()


__all__ = ["GestureLearner", "DEFAULT_LEARNER_CONFIG"]
