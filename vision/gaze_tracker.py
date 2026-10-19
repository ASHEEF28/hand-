"""Gaze direction estimation (optional feature, off by default).

Uses OpenCV Haar cascades - which ship with every OpenCV wheel, so no extra
model download is needed - to find the face and eyes, then estimates a coarse
gaze direction from where the eyes sit inside the face box.

This is deliberately coarse (left / right / up / down / center): it is meant for
"look left to switch tab" style shortcuts, not for precise eye tracking.  When
MediaPipe FaceMesh is available it is used instead, which is noticeably more
stable.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from utils import clamp, deep_merge, try_import

logger = logging.getLogger(__name__)

DEFAULT_GAZE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "smoothing": 0.5,
    "min_detection_confidence": 0.5,
    "backend": "auto",            # "auto" | "haar" | "facemesh"
    "dead_zone": 0.12,            # fraction of face box considered "center"
    "vertical_dead_zone": 0.18,
    "history": 8,
    "min_face_size": 60,
    "scale_factor": 1.2,
    "min_neighbors": 5,
    "gaze_actions": {
        "left": "swipe_left",
        "right": "swipe_right",
        "up": "volume_up",
        "down": "volume_down",
    },
}


class GazeTracker:
    """Estimates coarse gaze direction from a BGR frame."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_GAZE_CONFIG), config or {})
        self._face_cascade = None
        self._eye_cascade = None
        self._face_mesh = None
        self._backend = str(self.config.get("backend", "auto")).lower()
        self._history: Deque[Tuple[float, float]] = deque(maxlen=int(self.config.get("history", 8)))
        self._last_result: Dict[str, Any] = {"face_detected": False, "gaze_direction": "none"}
        self._initialized = False
        self.frames_processed = 0

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        if self._backend in ("auto", "facemesh"):
            mp = try_import("mediapipe")
            if mp is not None and hasattr(getattr(mp, "solutions", None), "face_mesh"):
                try:
                    self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=float(self.config.get("min_detection_confidence", 0.5)),
                        min_tracking_confidence=0.5,
                    )
                    self._backend = "facemesh"
                    logger.info("Gaze tracker ready (backend=facemesh)")
                    return
                except Exception as exc:
                    logger.debug("FaceMesh init failed: %s", exc)

        cv2 = try_import("cv2")
        if cv2 is None:
            logger.warning("OpenCV unavailable - gaze tracking disabled")
            self._backend = "none"
            return
        try:
            self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self._eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            self._backend = "haar"
            logger.info("Gaze tracker ready (backend=haar)")
        except Exception as exc:
            logger.error("Could not load Haar cascades: %s", exc)
            self._backend = "none"

    @property
    def backend(self) -> str:
        return self._backend

    def is_available(self) -> bool:
        self._ensure_initialized()
        return self._backend in ("haar", "facemesh")

    def close(self) -> None:
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
        self._face_mesh = None
        self._history.clear()

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #
    def process(self, frame) -> Dict[str, Any]:
        """Return ``{'face_detected', 'gaze_direction', 'face_box', 'eyes', 'yaw', 'pitch'}``."""
        self._ensure_initialized()
        if frame is None or self._backend == "none":
            return dict(self._last_result, face_detected=False)

        self.frames_processed += 1
        if self._backend == "facemesh":
            result = self._process_facemesh(frame)
        else:
            result = self._process_haar(frame)

        self._last_result = result
        return result

    def _process_haar(self, frame) -> Dict[str, Any]:
        import cv2

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=float(self.config.get("scale_factor", 1.2)),
            minNeighbors=int(self.config.get("min_neighbors", 5)),
            minSize=(int(self.config.get("min_face_size", 60)),) * 2,
        )
        if len(faces) == 0:
            return {"face_detected": False, "gaze_direction": "none", "face_box": None, "eyes": []}

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[fy : fy + fh, fx : fx + fw]
        eyes = self._eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=4)

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            centers = [(fx + ex + ew / 2.0, fy + ey + eh / 2.0) for ex, ey, ew, eh in eyes]
            eye_x = sum(c[0] for c in centers) / len(centers)
            eye_y = sum(c[1] for c in centers) / len(centers)
            yaw = (eye_x - (fx + fw / 2.0)) / max(fw, 1)
            pitch = (eye_y - (fy + fh / 2.0)) / max(fh, 1)
        else:
            centers = []
            yaw, pitch = 0.0, 0.0

        direction = self._classify(yaw, pitch)
        return {
            "face_detected": True,
            "gaze_direction": direction,
            "face_box": (int(fx), int(fy), int(fw), int(fh)),
            "eyes": centers,
            "yaw": yaw,
            "pitch": pitch,
        }

    def _process_facemesh(self, frame) -> Dict[str, Any]:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {"face_detected": False, "gaze_direction": "none", "face_box": None, "eyes": []}

        landmarks = results.multi_face_landmarks[0].landmark
        # Iris centres (refined landmarks) with a nose-bridge fallback.
        left_iris = (landmarks[468].x, landmarks[468].y) if len(landmarks) > 468 else (landmarks[33].x, landmarks[33].y)
        right_iris = (landmarks[473].x, landmarks[473].y) if len(landmarks) > 473 else (landmarks[263].x, landmarks[263].y)
        nose = (landmarks[1].x, landmarks[1].y)
        face_left = min(lm.x for lm in landmarks)
        face_right = max(lm.x for lm in landmarks)
        face_top = min(lm.y for lm in landmarks)
        face_bottom = max(lm.y for lm in landmarks)

        eye_x = (left_iris[0] + right_iris[0]) / 2.0
        eye_y = (left_iris[1] + right_iris[1]) / 2.0
        face_w = max(face_right - face_left, 1e-6)
        face_h = max(face_bottom - face_top, 1e-6)
        yaw = (eye_x - nose[0]) / face_w
        pitch = (eye_y - nose[1]) / face_h

        height, width = frame.shape[:2]
        return {
            "face_detected": True,
            "gaze_direction": self._classify(yaw, pitch),
            "face_box": (int(face_left * width), int(face_top * height),
                         int(face_w * width), int(face_h * height)),
            "eyes": [(int(left_iris[0] * width), int(left_iris[1] * height)),
                     (int(right_iris[0] * width), int(right_iris[1] * height))],
            "yaw": yaw,
            "pitch": pitch,
        }

    def _classify(self, yaw: float, pitch: float) -> str:
        """Smooth the raw offsets and bucket them into a direction."""
        smoothing = clamp(float(self.config.get("smoothing", 0.5)), 0.0, 0.95)
        if self._history:
            prev_x = sum(h[0] for h in self._history) / len(self._history)
            prev_y = sum(h[1] for h in self._history) / len(self._history)
            yaw = prev_x * smoothing + yaw * (1 - smoothing)
            pitch = prev_y * smoothing + pitch * (1 - smoothing)
        self._history.append((yaw, pitch))

        dead_zone = float(self.config.get("dead_zone", 0.12))
        vertical_dead_zone = float(self.config.get("vertical_dead_zone", 0.18))

        if abs(yaw) < dead_zone and abs(pitch) < vertical_dead_zone:
            return "center"
        if abs(yaw) >= abs(pitch):
            return "right" if yaw > 0 else "left"
        return "down" if pitch > 0 else "up"

    def action_for(self, direction: str) -> str:
        """Map a gaze direction to an action name (configurable)."""
        return str(self.config.get("gaze_actions", {}).get(direction, ""))

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def draw_gaze(self, frame, data: Dict[str, Any]):
        """Draw the face box, eye centres and gaze direction on ``frame``."""
        if frame is None or not data or not data.get("face_detected"):
            return frame
        cv2 = try_import("cv2")
        if cv2 is None:
            return frame

        annotated = frame
        box = data.get("face_box")
        if box:
            x, y, w, h = box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 120, 200), 2)
            for ex, ey in data.get("eyes", []) or []:
                cv2.circle(annotated, (int(ex), int(ey)), 4, (0, 215, 255), -1)
            if data.get("eyes"):
                cx = int(sum(e[0] for e in data["eyes"]) / len(data["eyes"]))
                cy = int(sum(e[1] for e in data["eyes"]) / len(data["eyes"]))
                direction = str(data.get("gaze_direction", "center"))
                vector = {
                    "left": (-30, 0), "right": (30, 0),
                    "up": (0, -30), "down": (0, 30), "center": (0, 0),
                }.get(direction, (0, 0))
                if vector != (0, 0):
                    cv2.arrowedLine(annotated, (cx, cy), (cx + vector[0], cy + vector[1]), (0, 255, 180), 2, tipLength=0.3)
            label = f"gaze: {data.get('gaze_direction', 'none')}"
            cv2.putText(annotated, label, (box[0], max(15, box[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 200), 1, cv2.LINE_AA)
        return annotated


__all__ = ["GazeTracker", "DEFAULT_GAZE_CONFIG"]
