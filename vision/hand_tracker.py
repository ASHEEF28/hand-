"""Hand tracking for GestureOS.

Wraps MediaPipe Hands and exposes a small, typed data model that the rest of
the application uses:

* :class:`Landmark`      - a single normalised 3D point
* :class:`HandLandmark`  - the 21 MediaPipe landmark indices
* :class:`Hand`          - one detected hand + derived geometry
* :class:`HandTracker`   - camera frame -> ``list[Hand]``

MediaPipe is imported lazily so the gesture maths and the test-suite work on
machines (and CI containers) where MediaPipe is not installed.  A detector can
also be injected for deterministic testing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:  # numpy is a hard dependency of the whole project
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from utils import clamp, deep_merge, distance, ema


# --------------------------------------------------------------------------- #
# Landmark model
# --------------------------------------------------------------------------- #
class HandLandmark(Enum):
    """MediaPipe hand landmark indices (0-20).

    The short names are canonical (``main.py`` uses ``HandLandmark.INDEX_TIP``);
    the long MediaPipe spellings are kept as aliases for convenience.
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # Long MediaPipe aliases (Enum treats equal values as aliases).
    THUMB_FINGER_MCP = 2
    THUMB_FINGER_IP = 3
    THUMB_FINGER_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_FINGER_MCP = 17
    PINKY_FINGER_PIP = 18
    PINKY_FINGER_DIP = 19
    PINKY_FINGER_TIP = 20


#: Bones used when drawing the hand skeleton.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGER_TIPS = (
    HandLandmark.THUMB_TIP.value,
    HandLandmark.INDEX_TIP.value,
    HandLandmark.MIDDLE_TIP.value,
    HandLandmark.RING_TIP.value,
    HandLandmark.PINKY_TIP.value,
)


@dataclass
class Landmark:
    """A normalised landmark: ``x``/``y`` in 0..1 of the frame, ``z`` relative depth."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_tuple3(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_pixel(self, width: int, height: int) -> Tuple[int, int]:
        return int(round(self.x * width)), int(round(self.y * height))

    def distance_to(self, other: "Landmark") -> float:
        return distance(self.as_tuple(), other.as_tuple())

    def copy(self) -> "Landmark":
        return Landmark(self.x, self.y, self.z, self.visibility)


@dataclass
class Hand:
    """One detected hand and the geometry derived from its landmarks."""

    landmarks: List[Landmark] = field(default_factory=list)
    handedness: str = "Right"
    handedness_score: float = 1.0
    gesture_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.landmarks)

    def landmark(self, index: HandLandmark | int) -> Optional[Landmark]:
        idx = index.value if isinstance(index, HandLandmark) else int(index)
        if 0 <= idx < len(self.landmarks):
            return self.landmarks[idx]
        return None

    @property
    def index_tip(self) -> Optional[Landmark]:
        return self.landmark(HandLandmark.INDEX_TIP)

    @property
    def thumb_tip(self) -> Optional[Landmark]:
        return self.landmark(HandLandmark.THUMB_TIP)

    @property
    def wrist(self) -> Optional[Landmark]:
        return self.landmark(HandLandmark.WRIST)

    @property
    def palm_center(self) -> Tuple[float, float]:
        """Mean of the wrist and the four finger MCP joints - a stable palm point."""
        indices = (
            HandLandmark.WRIST.value,
            HandLandmark.INDEX_MCP.value,
            HandLandmark.MIDDLE_MCP.value,
            HandLandmark.RING_MCP.value,
            HandLandmark.PINKY_MCP.value,
        )
        pts = [self.landmarks[i] for i in indices if i < len(self.landmarks)]
        if not pts:
            return (0.5, 0.5)
        return (sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))

    @property
    def hand_size(self) -> float:
        """Wrist -> middle-MCP distance; used to make thresholds scale-invariant."""
        wrist = self.landmark(HandLandmark.WRIST)
        middle = self.landmark(HandLandmark.MIDDLE_MCP)
        if wrist is None or middle is None:
            return 0.0
        return wrist.distance_to(middle)

    @property
    def fingers_up(self) -> List[bool]:
        """``[thumb, index, middle, ring, pinky]`` extended flags."""
        cached = self.gesture_data.get("fingers_up")
        if cached:
            return list(cached)
        return list(self.compute_fingers_up())

    @property
    def extended_count(self) -> int:
        return sum(1 for up in self.fingers_up if up)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def compute_fingers_up(self) -> Tuple[bool, bool, bool, bool, bool]:
        """Heuristic finger-extension test on normalised landmarks.

        Non-thumb fingers are extended when the tip is farther from the wrist
        than the PIP joint.  The thumb is extended when its tip is farther from
        the index MCP than the thumb IP joint (works for both hands because it
        is distance based rather than axis based).
        """
        if len(self.landmarks) < 21:
            return (False, False, False, False, False)

        wrist = self.landmarks[HandLandmark.WRIST.value]
        result: List[bool] = []

        # Thumb
        thumb_ip = self.landmarks[HandLandmark.THUMB_IP.value]
        thumb_tip = self.landmarks[HandLandmark.THUMB_TIP.value]
        index_mcp = self.landmarks[HandLandmark.INDEX_MCP.value]
        result.append(thumb_tip.distance_to(index_mcp) > thumb_ip.distance_to(index_mcp) * 1.05)

        # Index, middle, ring, pinky
        for mcp, pip_, tip in (
            (HandLandmark.INDEX_MCP.value, HandLandmark.INDEX_PIP.value, HandLandmark.INDEX_TIP.value),
            (HandLandmark.MIDDLE_MCP.value, HandLandmark.MIDDLE_PIP.value, HandLandmark.MIDDLE_TIP.value),
            (HandLandmark.RING_MCP.value, HandLandmark.RING_PIP.value, HandLandmark.RING_TIP.value),
            (HandLandmark.PINKY_MCP.value, HandLandmark.PINKY_PIP.value, HandLandmark.PINKY_TIP.value),
        ):
            result.append(
                self.landmarks[tip].distance_to(wrist)
                > self.landmarks[pip_].distance_to(wrist) * 1.05
            )

        return tuple(result)  # type: ignore[return-value]

    def pinch_distance(self, tip: int = HandLandmark.INDEX_TIP.value) -> float:
        """Distance between the thumb tip and another fingertip (default index)."""
        thumb = self.landmark(HandLandmark.THUMB_TIP)
        other = self.landmark(tip)
        if thumb is None or other is None:
            return 1.0
        return thumb.distance_to(other)

    def refresh_gesture_data(self) -> Dict[str, Any]:
        """Populate ``gesture_data`` with values other modules rely on.

        ``main.py`` reads ``hand.gesture_data['pinch_distance']`` for the
        brightness controller, and the recogniser reuses everything else.
        Distances are stored both raw (normalised) and relative to hand size so
        thresholds behave the same near and far from the camera.
        """
        data: Dict[str, Any] = {}
        if len(self.landmarks) < 21:
            self.gesture_data.update(data)
            return self.gesture_data

        size = self.hand_size or 1e-6
        fingers = self.compute_fingers_up()

        data["fingers_up"] = list(fingers)
        data["extended_count"] = sum(1 for up in fingers if up)
        data["hand_size"] = self.hand_size
        data["palm_center"] = self.palm_center

        for name, tip in (
            ("index", HandLandmark.INDEX_TIP.value),
            ("middle", HandLandmark.MIDDLE_TIP.value),
            ("ring", HandLandmark.RING_TIP.value),
            ("pinky", HandLandmark.PINKY_TIP.value),
        ):
            raw = self.pinch_distance(tip)
            data[f"{name}_pinch_distance"] = raw
            data[f"{name}_pinch_ratio"] = raw / size

        # The canonical "pinch" is thumb + index.
        data["pinch_distance"] = data["index_pinch_distance"]
        data["pinch_ratio"] = data["index_pinch_ratio"]

        # Two-hand zoom consumers use the palm centre; expose it flat too.
        cx, cy = data["palm_center"]
        data["palm_x"], data["palm_y"] = cx, cy

        self.gesture_data.update(data)
        return self.gesture_data


# --------------------------------------------------------------------------- #
# Default configuration
# --------------------------------------------------------------------------- #
DEFAULT_TRACKING_CONFIG: Dict[str, Any] = {
    "max_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "smoothing_factor": 0.5,
    "min_presence_confidence": 0.5,
    "flip_horizontal": True,
    # "auto" | "solutions" (MediaPipe <=0.10) | "tasks" (MediaPipe >=1.0)
    "backend": "auto",
    # Only used by the MediaPipe Tasks API (>=1.0), which needs an external model.
    "model_path": "models/hand_landmarker.task",
}

#: Where to fetch the model for the MediaPipe Tasks API backend.
TASKS_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


class HandTrackingUnavailable(RuntimeError):
    """Raised when MediaPipe is required but not installed / not loadable."""


class HandTracker:
    """Detects hands in a BGR frame and returns :class:`Hand` objects.

    Two MediaPipe backends are supported because the API changed in 1.0:

    * ``solutions`` - ``mp.solutions.hands`` (MediaPipe 0.10.x). The model is
      bundled in the wheel, so this is the zero-setup path and the one pinned
      in ``requirements.txt``.
    * ``tasks`` - ``mp.tasks.vision.HandLandmarker`` (MediaPipe 1.0+). Needs a
      ``hand_landmarker.task`` model file on disk (see :data:`TASKS_MODEL_URL`).

    Pass ``detector=`` any object with a ``process(frame) -> list[Hand]`` method
    to bypass MediaPipe entirely (used by the tests and the simulator).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, detector: Optional[Any] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_TRACKING_CONFIG), config or {})
        self._detector = detector          # injectable: any object with .process(frame)
        self._mp = None
        self._hands = None                 # legacy solutions pipeline
        self._landmarker = None            # tasks API landmarker
        self._initialized = detector is not None
        self._backend = "injected" if detector is not None else "auto"
        self._previous: Dict[str, List[Landmark]] = {}
        self._frames_processed = 0
        self._last_process_ms = 0.0
        self._video_timestamp_ms = 0
        self._closed = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._hands is not None or self._landmarker is not None or self._detector is not None
        self._initialized = True

        try:
            import mediapipe as mp  # lazy: heavy import, only needed at runtime
        except Exception as exc:
            raise HandTrackingUnavailable(
                "MediaPipe is required for hand tracking but could not be imported "
                f"({exc}). Install it with: pip install 'mediapipe>=0.10.14,<1.0'"
            ) from exc

        self._mp = mp
        preferred = str(self.config.get("backend", "auto")).lower()

        if preferred in ("auto", "solutions") and hasattr(getattr(mp, "solutions", None), "hands"):
            self._init_solutions(mp)
            return True

        if preferred in ("auto", "tasks") and hasattr(mp, "tasks"):
            try:
                self._init_tasks(mp)
                return True
            except HandTrackingUnavailable:
                if preferred == "tasks":
                    raise
                logger.warning("MediaPipe Tasks backend unavailable and legacy API missing")

        raise HandTrackingUnavailable(
            "No usable MediaPipe hand-landmark backend found. Either install the "
            "legacy API (pip install 'mediapipe>=0.10.14,<1.0') or download the "
            f"Tasks model to '{self.config.get('model_path')}' from {TASKS_MODEL_URL}"
        )

    def _init_solutions(self, mp: Any) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=int(self.config.get("max_hands", 2)),
            model_complexity=int(self.config.get("model_complexity", 1)),
            min_detection_confidence=float(self.config.get("min_detection_confidence", 0.7)),
            min_tracking_confidence=float(self.config.get("min_tracking_confidence", 0.5)),
        )
        self._backend = "solutions"
        logger.info(
            "MediaPipe hand tracker ready (backend=solutions, max_hands=%s, complexity=%s)",
            self.config.get("max_hands"),
            self.config.get("model_complexity"),
        )

    def _init_tasks(self, mp: Any) -> None:
        import os

        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        model_path = str(self.config.get("model_path", "models/hand_landmarker.task"))
        if not os.path.isabs(model_path):
            from utils import project_root

            model_path = os.path.join(project_root(), model_path)
        if not os.path.exists(model_path):
            raise HandTrackingUnavailable(
                f"MediaPipe Tasks model not found at '{model_path}'. Download it from "
                f"{TASKS_MODEL_URL} or install 'mediapipe>=0.10.14,<1.0' for the bundled model."
            )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=int(self.config.get("max_hands", 2)),
            min_hand_detection_confidence=float(self.config.get("min_detection_confidence", 0.7)),
            min_hand_presence_confidence=float(self.config.get("min_presence_confidence", 0.5)),
            min_tracking_confidence=float(self.config.get("min_tracking_confidence", 0.5)),
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._backend = "tasks"
        logger.info("MediaPipe hand tracker ready (backend=tasks, model=%s)", model_path)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def last_process_ms(self) -> float:
        return self._last_process_ms

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("Error closing MediaPipe hands: %s", exc)
        self._hands = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("Error closing MediaPipe landmarker: %s", exc)
        self._landmarker = None
        self._previous.clear()

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #
    def process(self, frame) -> List[Hand]:
        """Process one BGR frame and return the detected hands (may be empty)."""
        if frame is None:
            return []

        start = time.perf_counter()
        try:
            if self._detector is not None:
                hands = self._detector.process(frame) or []
            else:
                hands = self._process_mediapipe(frame)
        except HandTrackingUnavailable:
            raise
        except Exception as exc:
            logger.error("Hand tracking failed: %s", exc)
            hands = []

        hands = self._post_process(hands)
        self._frames_processed += 1
        self._last_process_ms = (time.perf_counter() - start) * 1000.0
        return hands

    def _process_mediapipe(self, frame) -> List[Hand]:
        try:
            if not self._ensure_initialized():
                return []
        except Exception as exc:
            raise HandTrackingUnavailable(str(exc)) from exc

        if self._backend == "tasks":
            return self._process_tasks(frame)
        return self._process_solutions(frame)

    def _process_solutions(self, frame) -> List[Hand]:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        hands: List[Hand] = []
        if not results.multi_hand_landmarks:
            return hands

        handedness_list = results.multi_handedness or []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label, score = "Right", 1.0
            if i < len(handedness_list) and handedness_list[i].classification:
                cls = handedness_list[i].classification[0]
                label, score = cls.label, float(cls.score)

            landmarks = [Landmark(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            hands.append(
                Hand(
                    landmarks=landmarks,
                    handedness=label,
                    handedness_score=score,
                    timestamp=time.time(),
                )
            )
        return hands

    def _process_tasks(self, frame) -> List[Hand]:
        import cv2

        mp = self._mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO running mode requires strictly increasing timestamps.
        self._video_timestamp_ms = max(self._video_timestamp_ms + 1, int(time.monotonic() * 1000))
        result = self._landmarker.detect_for_video(image, self._video_timestamp_ms)

        hands: List[Hand] = []
        for i, hand_landmarks in enumerate(result.hand_landmarks or []):
            label, score = "Right", 1.0
            handedness = getattr(result, "handedness", None)
            if handedness and i < len(handedness) and handedness[i]:
                category = handedness[i][0]
                label = getattr(category, "category_name", None) or getattr(category, "display_name", "Right")
                score = float(getattr(category, "score", 1.0))
            landmarks = [
                Landmark(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))) for lm in hand_landmarks
            ]
            hands.append(
                Hand(
                    landmarks=landmarks,
                    handedness=label,
                    handedness_score=score,
                    timestamp=time.time(),
                )
            )
        return hands

    def _post_process(self, hands: List[Hand]) -> List[Hand]:
        """Filter by confidence, smooth landmarks and derive gesture data."""
        min_presence = float(self.config.get("min_presence_confidence", 0.5))
        out: List[Hand] = []

        for hand in hands:
            if hand.handedness_score < min_presence:
                continue
            if len(hand.landmarks) >= 21:
                if self.config.get("smooth_landmarks", True):
                    hand.landmarks = self._smooth(hand)
                hand.refresh_gesture_data()
            out.append(hand)

        # Drop stale smoothing state for hands that disappeared.
        seen = {h.handedness for h in out}
        for key in list(self._previous):
            if key not in seen:
                del self._previous[key]
        return out

    def _smooth(self, hand: Hand) -> List[Landmark]:
        factor = clamp(float(self.config.get("smoothing_factor", 0.5)), 0.0, 0.95)
        key = hand.handedness
        previous = self._previous.get(key)
        if not previous or len(previous) != len(hand.landmarks):
            self._previous[key] = [lm.copy() for lm in hand.landmarks]
            return hand.landmarks

        smoothed: List[Landmark] = []
        for prev, cur in zip(previous, hand.landmarks):
            smoothed.append(
                Landmark(
                    x=ema(prev.x, cur.x, factor),
                    y=ema(prev.y, cur.y, factor),
                    z=ema(prev.z, cur.z, factor),
                    visibility=cur.visibility,
                )
            )
        self._previous[key] = smoothed
        return smoothed

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def draw_landmarks(self, frame, hands: Sequence[Hand], draw_connections: bool = True):
        """Draw the hand skeleton on ``frame`` and return the annotated copy."""
        if frame is None:
            return frame
        if np is None:
            return frame
        if not hands:
            return frame

        import cv2

        annotated = frame.copy()
        height, width = annotated.shape[:2]

        for hand in hands:
            color = (80, 220, 100) if hand.handedness == "Right" else (255, 170, 80)
            points = [lm.as_pixel(width, height) for lm in hand.landmarks]

            if draw_connections:
                for a, b in HAND_CONNECTIONS:
                    if a < len(points) and b < len(points):
                        cv2.line(annotated, points[a], points[b], color, 2)

            for i, pt in enumerate(points):
                radius = 5 if i in FINGER_TIPS else 3
                cv2.circle(annotated, pt, radius, color, -1)

            # Pinch indicator: line + circle whose radius shows pinch distance.
            thumb = hand.landmark(HandLandmark.THUMB_TIP)
            index = hand.landmark(HandLandmark.INDEX_TIP)
            if thumb is not None and index is not None:
                p1 = thumb.as_pixel(width, height)
                p2 = index.as_pixel(width, height)
                pinching = hand.gesture_data.get("pinch_ratio", 1.0) < 0.45
                cv2.line(annotated, p1, p2, (0, 215, 255) if pinching else (160, 160, 160), 1)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.circle(annotated, mid, 8 if pinching else 5, (0, 215, 255) if pinching else color, 2)

            palm = hand.palm_center
            cv2.putText(
                annotated,
                f"{hand.handedness} {hand.handedness_score:.2f}",
                (int(palm[0] * width) - 30, max(15, int(palm[1] * height) - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return annotated


def make_hand(
    fingertips: Optional[Dict[int, Tuple[float, float]]] = None,
    handedness: str = "Right",
    score: float = 1.0,
    base: Optional[Sequence[Tuple[float, float]]] = None,
) -> Hand:
    """Build a synthetic :class:`Hand` for tests and the simulation mode.

    ``fingertips`` maps landmark index -> (x, y); every other landmark falls
    back to ``base`` (default: an open hand centred at 0.5, 0.5).
    """
    if base is None:
        base = _OPEN_HAND_BASE
    coords = list(base)
    for index, xy in (fingertips or {}).items():
        if 0 <= index < len(coords):
            coords[index] = xy
    landmarks = [Landmark(x, y, 0.0) for x, y in coords]
    hand = Hand(landmarks=landmarks, handedness=handedness, handedness_score=score)
    hand.refresh_gesture_data()
    return hand


#: A rough open right hand, normalised coordinates (index 0..20).
_OPEN_HAND_BASE: Tuple[Tuple[float, float], ...] = (
    (0.50, 0.80),  # 0  wrist
    (0.42, 0.70),  # 1  thumb cmc
    (0.36, 0.62),  # 2  thumb mcp
    (0.31, 0.55),  # 3  thumb ip
    (0.27, 0.48),  # 4  thumb tip
    (0.44, 0.55),  # 5  index mcp
    (0.43, 0.42),  # 6  index pip
    (0.43, 0.33),  # 7  index dip
    (0.43, 0.24),  # 8  index tip
    (0.51, 0.53),  # 9  middle mcp
    (0.51, 0.39),  # 10 middle pip
    (0.51, 0.30),  # 11 middle dip
    (0.51, 0.21),  # 12 middle tip
    (0.58, 0.55),  # 13 ring mcp
    (0.59, 0.42),  # 14 ring pip
    (0.59, 0.34),  # 15 ring dip
    (0.59, 0.26),  # 16 ring tip
    (0.65, 0.58),  # 17 pinky mcp
    (0.66, 0.48),  # 18 pinky pip
    (0.67, 0.41),  # 19 pinky dip
    (0.67, 0.35),  # 20 pinky tip
)

__all__ = [
    "HandLandmark",
    "HAND_CONNECTIONS",
    "FINGER_TIPS",
    "Landmark",
    "Hand",
    "HandTracker",
    "HandTrackingUnavailable",
    "DEFAULT_TRACKING_CONFIG",
    "TASKS_MODEL_URL",
    "make_hand",
]
