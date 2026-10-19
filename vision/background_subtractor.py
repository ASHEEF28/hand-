"""Background subtraction (optional pre-processing step).

Isolates the moving foreground (the user's hands) before hand tracking, which
can help in cluttered rooms.  Implemented with OpenCV's MOG2 / KNN background
models - no extra dependencies.

``main.py`` calls :meth:`apply_background_removal` on every frame when
``background_subtraction.enabled`` is true.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from utils import clamp, deep_merge, try_import

logger = logging.getLogger(__name__)

DEFAULT_BACKGROUND_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "method": "mog2",              # "mog2" | "knn"
    "history": 500,
    "var_threshold": 16,
    "detect_shadows": True,
    "learning_rate": 0.01,
    "dist2_threshold": 400.0,      # KNN only
    "open_kernel": 3,              # morphological opening to drop speckle
    "close_kernel": 9,             # morphological closing to fill the subject
    "min_foreground_ratio": 0.002, # below this the frame is treated as static
    "mode": "mask",                # "mask" (black background) | "blend" | "mask_only"
    "blend_alpha": 0.85,
}


class BackgroundSubtractor:
    """Removes the static background from camera frames."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_BACKGROUND_CONFIG), config or {})
        self._subtractor = None
        self._kernel_open = None
        self._kernel_close = None
        self._frames = 0
        self._last_ratio = 0.0
        self._initialize()

    # ------------------------------------------------------------------ #
    def _initialize(self) -> None:
        cv2 = try_import("cv2")
        if cv2 is None:
            logger.warning("OpenCV unavailable - background subtraction disabled")
            return
        try:
            method = str(self.config.get("method", "mog2")).lower()
            history = int(self.config.get("history", 500))
            detect_shadows = bool(self.config.get("detect_shadows", True))
            if method == "knn":
                self._subtractor = cv2.createBackgroundSubtractorKNN(
                    history=history,
                    dist2Threshold=float(self.config.get("dist2_threshold", 400.0)),
                    detectShadows=detect_shadows,
                )
            else:
                self._subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=history,
                    varThreshold=float(self.config.get("var_threshold", 16)),
                    detectShadows=detect_shadows,
                )
            open_size = int(self.config.get("open_kernel", 3))
            close_size = int(self.config.get("close_kernel", 9))
            if open_size > 1:
                self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            if close_size > 1:
                self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            logger.info("Background subtraction ready (method=%s)", method)
        except Exception as exc:
            logger.error("Could not initialise background subtractor: %s", exc)
            self._subtractor = None

    @property
    def is_available(self) -> bool:
        return self._subtractor is not None

    def reset(self) -> None:
        """Forget the learned background (e.g. after moving the camera)."""
        self._initialize()
        self._frames = 0

    # ------------------------------------------------------------------ #
    def foreground_mask(self, frame):
        """Binary foreground mask for ``frame``."""
        if frame is None or self._subtractor is None:
            return None
        import cv2
        import numpy as np

        learning_rate = float(self.config.get("learning_rate", -1))
        mask = self._subtractor.apply(frame, learningRate=learning_rate if learning_rate > 0 else -1)
        # MOG2 marks shadows as 127 - drop them.
        if self.config.get("detect_shadows", True):
            mask = np.where(mask == 255, 255, 0).astype(np.uint8)
        if self._kernel_open is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        if self._kernel_close is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        return mask

    def apply_background_removal(self, frame):
        """Return ``frame`` with the background suppressed (same shape/dtype)."""
        self._frames += 1
        if frame is None or self._subtractor is None:
            return frame

        import cv2
        import numpy as np

        mask = self.foreground_mask(frame)
        if mask is None:
            return frame

        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        self._last_ratio = ratio

        # Nothing moving: pass the frame through untouched so tracking never
        # starves (also covers the model warm-up period).
        if ratio < float(self.config.get("min_foreground_ratio", 0.002)) or self._frames < 15:
            return frame

        mode = str(self.config.get("mode", "mask")).lower()
        if mode == "mask_only":
            return mask
        if mode == "blend":
            alpha = clamp(float(self.config.get("blend_alpha", 0.85)), 0.0, 1.0)
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            return cv2.addWeighted(frame, 1.0 - alpha, masked, alpha, 0)
        return cv2.bitwise_and(frame, frame, mask=mask)

    @property
    def last_foreground_ratio(self) -> float:
        return self._last_ratio

    @property
    def frames_processed(self) -> int:
        return self._frames


__all__ = ["BackgroundSubtractor", "DEFAULT_BACKGROUND_CONFIG"]
