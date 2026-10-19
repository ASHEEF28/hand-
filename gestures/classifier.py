"""ML gesture classifier (optional).

A thin scikit-learn wrapper that learns to label a static hand pose from its
landmark geometry.  It is *off* by default (``gesture_classifier.enabled``);
when enabled and a trained model exists at ``model_path`` it is consulted
alongside the rule-based recogniser.

scikit-learn / joblib are imported lazily so GestureOS still runs when they are
missing - :meth:`GestureClassifier.is_available` reports that.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import clamp, deep_merge, project_root, try_import
from vision.hand_tracker import Hand, HandLandmark

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "model_path": "models/gesture_classifier.pkl",
    "confidence_threshold": 0.6,
    "feature_version": 1,
    "test_size": 0.2,
    "random_state": 42,
    "max_features": 200,
}

#: Landmark indices used to build the feature vector.
_FEATURE_JOINTS = (
    HandLandmark.WRIST.value,
    HandLandmark.THUMB_TIP.value,
    HandLandmark.INDEX_MCP.value,
    HandLandmark.INDEX_TIP.value,
    HandLandmark.MIDDLE_MCP.value,
    HandLandmark.MIDDLE_TIP.value,
    HandLandmark.RING_MCP.value,
    HandLandmark.RING_TIP.value,
    HandLandmark.PINKY_MCP.value,
    HandLandmark.PINKY_TIP.value,
)


class GestureClassifier:
    """Random-forest gesture classifier over scale-normalised landmarks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_CLASSIFIER_CONFIG), config or {})
        self.model: Optional[Any] = None
        self.classes: List[str] = []
        self.feature_version = int(self.config.get("feature_version", 1))
        self.trained_samples = 0
        self._load()

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_features(hand: Hand) -> np.ndarray:
        """Scale/translation invariant feature vector for one hand.

        All joints are expressed relative to the wrist and divided by hand
        size, so the classifier does not simply learn "where the hand was".
        """
        if len(hand.landmarks) < 21:
            return np.zeros(len(_FEATURE_JOINTS) * 2 + 6, dtype=np.float32)

        wrist = hand.landmarks[HandLandmark.WRIST.value]
        scale = hand.hand_size or 1e-6
        features: List[float] = []
        for idx in _FEATURE_JOINTS:
            lm = hand.landmarks[idx]
            features.append((lm.x - wrist.x) / scale)
            features.append((lm.y - wrist.y) / scale)

        data = hand.gesture_data or hand.refresh_gesture_data()
        features.extend(float(data.get(f"{name}_pinch_ratio", 0.0)) for name in
                        ("index", "middle", "ring", "pinky"))
        features.extend(1.0 if up else 0.0 for up in data.get("fingers_up", [False] * 5)[:5])
        features.append(float(data.get("extended_count", 0)))
        return np.asarray(features, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _resolved_model_path(self) -> str:
        path = str(self.config.get("model_path", "models/gesture_classifier.pkl"))
        return path if os.path.isabs(path) else os.path.join(project_root(), path)

    def is_available(self) -> bool:
        return try_import("sklearn") is not None and try_import("joblib") is not None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        path = self._resolved_model_path()
        if not os.path.exists(path):
            logger.debug("No gesture classifier model at %s", path)
            return
        joblib = try_import("joblib")
        if joblib is None:
            logger.warning("joblib not installed - cannot load gesture classifier")
            return
        try:
            bundle = joblib.load(path)
            self.model = bundle.get("model")
            self.classes = list(bundle.get("classes", []))
            self.feature_version = int(bundle.get("feature_version", self.feature_version))
            self.trained_samples = int(bundle.get("samples", 0))
            logger.info(
                "Loaded gesture classifier (%d classes, %d samples)",
                len(self.classes), self.trained_samples,
            )
        except Exception as exc:
            logger.error("Failed to load gesture classifier from %s: %s", path, exc)
            self.model = None

    def save(self, path: Optional[str] = None) -> bool:
        if self.model is None:
            logger.warning("Nothing to save - classifier is not trained")
            return False
        joblib = try_import("joblib")
        if joblib is None:
            logger.error("joblib not installed - cannot save gesture classifier")
            return False
        target = path or self._resolved_model_path()
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        try:
            joblib.dump(
                {
                    "model": self.model,
                    "classes": self.classes,
                    "feature_version": self.feature_version,
                    "samples": self.trained_samples,
                },
                target,
            )
            logger.info("Saved gesture classifier to %s", target)
            return True
        except Exception as exc:
            logger.error("Failed to save gesture classifier: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Training / inference
    # ------------------------------------------------------------------ #
    def train(self, samples: Sequence[Tuple[Hand, str]]) -> Dict[str, Any]:
        """Train on ``(hand, label)`` pairs and persist the model."""
        sklearn = try_import("sklearn")
        if sklearn is None:
            raise RuntimeError("scikit-learn is required to train the gesture classifier")
        if not samples:
            raise ValueError("No training samples supplied")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X = np.vstack([self.extract_features(hand) for hand, _ in samples])
        y = np.asarray([label for _, label in samples])

        labels, counts = np.unique(y, return_counts=True)
        stratify = y if len(labels) > 1 and counts.min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(self.config.get("test_size", 0.2)),
            random_state=int(self.config.get("random_state", 42)),
            stratify=stratify,
        )

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=2,
            random_state=int(self.config.get("random_state", 42)),
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        accuracy = float(model.score(X_test, y_test)) if len(X_test) else float(model.score(X, y))
        self.model = model
        self.classes = [str(c) for c in model.classes_]
        self.trained_samples = len(samples)
        self.save()
        logger.info("Trained gesture classifier on %d samples (accuracy %.3f)", len(samples), accuracy)
        return {"samples": len(samples), "classes": self.classes, "accuracy": accuracy}

    def predict(self, hand: Hand) -> Optional[Tuple[str, float]]:
        """Return ``(label, confidence)`` or ``None`` when no model is loaded."""
        if self.model is None or hand is None:
            return None
        try:
            features = self.extract_features(hand).reshape(1, -1)
            probabilities = self.model.predict_proba(features)[0]
            best = int(np.argmax(probabilities))
            label = str(self.model.classes_[best])
            confidence = clamp(float(probabilities[best]), 0.0, 1.0)
            return label, confidence
        except Exception as exc:
            logger.debug("Gesture classifier prediction failed: %s", exc)
            return None

    def predict_if_confident(self, hand: Hand) -> Optional[str]:
        """Label only when confidence exceeds the configured threshold."""
        prediction = self.predict(hand)
        if not prediction:
            return None
        label, confidence = prediction
        return label if confidence >= float(self.config.get("confidence_threshold", 0.6)) else None


__all__ = ["GestureClassifier", "DEFAULT_CLASSIFIER_CONFIG"]
