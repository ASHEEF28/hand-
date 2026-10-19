"""Synthetic hand source: drives the whole pipeline without a webcam.

:class:`SyntheticHandDetector` implements the same ``process(frame)`` contract
as :class:`~vision.hand_tracker.HandTracker`, but returns a *scripted* sequence
of hand poses instead of running MediaPipe.  It exists for three reasons:

1. **Development without hardware** - ``python main.py --demo`` exercises
   cursor control, clicks, swipes, drags and two-hand zoom end to end.
2. **CI / smoke tests** - combined with ``--dry-run`` it verifies the engine
   wiring on a headless machine with no camera and no display.
3. **Tuning** - reproducible gestures make threshold changes easy to compare.

Inject it with ``engine.tracker = SyntheticHandDetector()`` or let ``main.py``
do it for you via ``--synthetic-hands`` / ``--demo``.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vision.hand_tracker import Hand, make_hand

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pose recipes: landmark index overrides on top of the open-hand base pose.
# --------------------------------------------------------------------------- #
def _open_palm() -> Dict[int, Tuple[float, float]]:
    return {}


def _pinch() -> Dict[int, Tuple[float, float]]:
    """Thumb tip meets the *index* tip; the other fingers stay clear."""
    return {3: (0.34, 0.48), 4: (0.40, 0.44), 7: (0.42, 0.34), 8: (0.405, 0.42)}


def _middle_pinch() -> Dict[int, Tuple[float, float]]:
    """Thumb tip meets an extended *middle* fingertip; index is pulled away."""
    return {3: (0.42, 0.34), 4: (0.48, 0.26), 12: (0.49, 0.25), 8: (0.40, 0.16)}


def _point() -> Dict[int, Tuple[float, float]]:
    """Index extended, the other fingers curled into the palm."""
    return {4: (0.44, 0.60), 12: (0.50, 0.60), 16: (0.57, 0.62), 20: (0.63, 0.64)}


def _scroll_pose() -> Dict[int, Tuple[float, float]]:
    """Index + middle extended, ring/pinky/thumb curled."""
    return {4: (0.44, 0.60), 16: (0.57, 0.62), 20: (0.63, 0.64)}


def _fist() -> Dict[int, Tuple[float, float]]:
    """Every fingertip folded down towards the wrist."""
    return {4: (0.46, 0.62), 8: (0.47, 0.60), 12: (0.51, 0.60), 16: (0.57, 0.61), 20: (0.62, 0.63)}


def _thumbs_up() -> Dict[int, Tuple[float, float]]:
    return {4: (0.48, 0.26), 8: (0.47, 0.60), 12: (0.51, 0.60), 16: (0.57, 0.61), 20: (0.62, 0.63)}


def _thumbs_down() -> Dict[int, Tuple[float, float]]:
    return {
        1: (0.48, 0.72), 2: (0.47, 0.84), 3: (0.46, 0.90), 4: (0.45, 0.96),
        8: (0.47, 0.60), 12: (0.51, 0.60), 16: (0.57, 0.61), 20: (0.62, 0.63),
    }


def _three_fingers() -> Dict[int, Tuple[float, float]]:
    """Index + middle + ring extended, pinky + thumb curled."""
    return {4: (0.44, 0.60), 20: (0.63, 0.64)}


def _rock_sign() -> Dict[int, Tuple[float, float]]:
    """Index + pinky extended ("call me" / AR menu toggle)."""
    return {4: (0.43, 0.62), 12: (0.50, 0.60), 16: (0.57, 0.61), 20: (0.67, 0.33)}


def _five_finger_pinch() -> Dict[int, Tuple[float, float]]:
    """All five fingertips gathered on the thumb tip (Start Menu)."""
    return {4: (0.47, 0.55), 8: (0.475, 0.53), 12: (0.50, 0.53), 16: (0.53, 0.54), 20: (0.55, 0.56)}


#: The default demo script: ``(pose name, seconds, motion)``.
DEFAULT_SCRIPT: Sequence[Tuple[str, float, str]] = (
    ("open_palm", 1.5, "idle"),
    ("point", 2.0, "circle"),          # cursor movement
    ("pinch", 0.6, "idle"),           # left click
    ("open_palm", 0.5, "idle"),
    ("pinch", 0.15, "idle"),          # double click (two fast pinches)
    ("pinch", 0.15, "idle"),
    ("open_palm", 0.8, "idle"),
    ("middle_pinch", 0.6, "idle"),    # right click
    ("open_palm", 0.6, "idle"),
    ("scroll_pose", 2.0, "scroll"),   # scroll mode + palm motion
    ("point", 1.0, "idle"),
    ("open_palm", 1.2, "swipe_right"),
    ("open_palm", 1.2, "swipe_left"),
    ("three_fingers", 1.0, "swipe_right"),   # media next
    ("three_fingers", 1.0, "swipe_up"),      # volume up
    ("thumbs_up", 1.0, "idle"),       # play / pause
    ("thumbs_down", 1.0, "idle"),     # mute
    ("fist", 1.2, "drag"),           # grab window + move
    ("open_palm", 0.8, "idle"),       # release
    ("rock_sign", 1.0, "idle"),      # AR menu
    ("five_finger_pinch", 1.0, "idle"),  # start menu
    ("open_palm", 1.0, "idle"),
)

POSES: Dict[str, Dict[int, Tuple[float, float]]] = {
    "open_palm": _open_palm(),
    "point": _point(),
    "pinch": _pinch(),
    "middle_pinch": _middle_pinch(),
    "scroll_pose": _scroll_pose(),
    "fist": _fist(),
    "thumbs_up": _thumbs_up(),
    "thumbs_down": _thumbs_down(),
    "three_fingers": _three_fingers(),
    "rock_sign": _rock_sign(),
    "five_finger_pinch": _five_finger_pinch(),
}


class SyntheticHandDetector:
    """Drop-in replacement for :class:`HandTracker` producing scripted hands."""

    def __init__(
        self,
        script: Optional[Sequence[Tuple[str, float, str]]] = None,
        loop: bool = True,
        handedness: str = "Right",
        two_hand_segments: bool = True,
        fps: float = 30.0,
        time_scale: float = 1.0,
    ) -> None:
        self.script: List[Tuple[str, float, str]] = list(script or DEFAULT_SCRIPT)
        self.loop = loop
        self.handedness = handedness
        self.two_hand_segments = two_hand_segments
        self.fps = max(1.0, float(fps))
        #: Runs the script faster (>1) or slower (<1) than wall clock.  Handy for
        #: a snappy demo and for tests that must cover the whole script quickly.
        self.time_scale = max(0.01, float(time_scale))
        self._start = time.time()
        self._frame = 0
        self._renderer = None
        self.backend = "synthetic"
        self.last_process_ms = 0.0
        self.current_segment: Optional[Tuple[str, float, str]] = None
        self.finished = False
        self.hands_emitted = 0
        self.config: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def _segment_at(self, elapsed: float) -> Tuple[Optional[Tuple[str, float, str]], float]:
        """Return the active script segment and the time spent inside it."""
        total = sum(duration for _, duration, _ in self.script)
        if total <= 0:
            if not self.loop:
                self.finished = True
            return None, 0.0
        if elapsed >= total:
            if not self.loop:
                self.finished = True
                return None, 0.0
            elapsed = elapsed % total

        for segment in self.script:
            if elapsed < segment[1]:
                return segment, elapsed
            elapsed -= segment[1]
        return self.script[-1], 0.0

    def _motion_offset(self, motion: str, local: float) -> Tuple[float, float]:
        """Normalised (dx, dy) applied to every landmark for the given motion."""
        if motion == "circle":
            return (0.10 * math.sin(local * 2.2), 0.07 * math.cos(local * 2.2))
        if motion == "scroll":
            return (0.0, 0.10 * math.sin(local * 4.0))
        if motion == "swipe_right":
            return (min(local * 1.6, 0.45), 0.0)
        if motion == "swipe_left":
            return (-min(local * 1.6, 0.45), 0.0)
        if motion == "swipe_up":
            return (0.0, -min(local * 1.6, 0.35))
        if motion == "swipe_down":
            return (0.0, min(local * 1.6, 0.35))
        if motion == "drag":
            return (min(local * 0.35, 0.18), min(local * 0.25, 0.12))
        if motion == "jitter":
            return (0.004 * math.sin(local * 40.0), 0.004 * math.cos(local * 40.0))
        return (0.0, 0.0)

    def _offset_hand(self, pose: Dict[int, Tuple[float, float]], dx: float, dy: float,
                     handedness: str = "Right", scale: float = 1.0) -> Hand:
        overrides: Dict[int, Tuple[float, float]] = {}
        for index, (x, y) in pose.items():
            overrides[index] = (
                min(max(x * scale + dx, 0.02), 0.98),
                min(max(y * scale + dy, 0.02), 0.98),
            )
        hand = make_hand(fingertips=overrides, handedness=handedness, score=0.99)
        # Shift every landmark, not just the overridden ones.
        for landmark in hand.landmarks:
            landmark.x = min(max(landmark.x * scale + dx, 0.02), 0.98)
            landmark.y = min(max(landmark.y * scale + dy, 0.02), 0.98)
        hand.refresh_gesture_data()
        return hand

    # ------------------------------------------------------------------ #
    def process(self, frame: Any = None) -> List[Hand]:
        """Return the scripted hands for "now"."""
        started = time.perf_counter()
        self._frame += 1
        elapsed = (time.time() - self._start) * self.time_scale
        segment, local = self._segment_at(elapsed)
        if segment is None:
            self.last_process_ms = (time.perf_counter() - started) * 1000.0
            return []

        pose_name, _duration, motion = segment
        self.current_segment = segment
        pose = POSES.get(pose_name, {})
        dx, dy = self._motion_offset(motion, local)

        hands = [self._offset_hand(pose, dx, dy, self.handedness)]

        # Two-hand zoom for a moment in the middle of every open-palm segment.
        if self.two_hand_segments and pose_name == "open_palm" and 0.6 < local < 1.3:
            spread = 0.12 + 0.10 * math.sin((local - 0.6) * math.pi / 0.7)
            hands.append(self._offset_hand(_open_palm(), dx + spread, dy + 0.10, "Left", scale=0.85))

        self.hands_emitted += len(hands)
        self.last_process_ms = (time.perf_counter() - started) * 1000.0
        return hands

    # ------------------------------------------------------------------ #
    # HandTracker-compatible surface
    # ------------------------------------------------------------------ #
    def draw_landmarks(self, frame: Any, hands: Sequence[Hand], draw_connections: bool = True):
        """Annotate ``frame`` using the real tracker renderer.

        ``HandTracker`` only touches MediaPipe inside ``process()``, so holding
        one purely for drawing is free - the synthetic detector keeps working on
        machines where MediaPipe is not installed.
        """
        try:
            if self._renderer is None:
                from vision.hand_tracker import HandTracker

                self._renderer = HandTracker(self.config)
            return self._renderer.draw_landmarks(frame, hands, draw_connections)
        except Exception as exc:  # pragma: no cover - only when OpenCV is missing
            logger.debug("Synthetic draw_landmarks fell back to raw frame: %s", exc)
            return frame

    def close(self) -> None:
        return None

    def reset(self) -> None:
        self._start = time.time()
        self._frame = 0
        self.finished = False

    def describe(self) -> str:
        segment = self.current_segment
        if not segment:
            return "idle"
        return f"{segment[0]} ({segment[2]})"


__all__ = ["SyntheticHandDetector", "POSES", "DEFAULT_SCRIPT"]
