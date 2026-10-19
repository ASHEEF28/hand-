"""Gesture recognition engine for GestureOS.

Turns a list of :class:`~vision.hand_tracker.Hand` objects into discrete
:class:`GestureEvent` objects, and maps those events onto the action names that
``main.py`` dispatches (``left_click``, ``swipe_left``, ``volume_up`` ...).

Design notes
------------
* **Scale invariance** - pinch thresholds are expressed relative to hand size
  (wrist -> middle MCP) so gestures work close to or far from the camera.  The
  absolute ``pinch_threshold`` from ``settings.json`` is still honoured via
  ``use_absolute_pinch``.
* **Stateful** - double click, hold-to-drag, swipes and two-hand zoom all need
  history, kept per handedness in :class:`_HandState`.
* **Cooldowns** - every gesture family has its own cooldown taken from config,
  which prevents one physical gesture from firing dozens of actions.
* **No hard dependencies** - only numpy, so the recogniser is fully unit
  testable on any platform.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from utils import clamp, deep_merge, distance, signed_rotation
from vision.hand_tracker import Hand, HandLandmark

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Gesture vocabulary
# --------------------------------------------------------------------------- #
class GestureType(Enum):
    """Every gesture the recogniser can emit."""

    NONE = "none"
    POINT = "point"
    OPEN_PALM = "open_palm"
    FIST = "fist"

    # Pinch family
    PINCH = "pinch"
    MIDDLE_PINCH = "middle_pinch"
    DOUBLE_PINCH = "double_pinch"
    HOLD_PINCH = "hold_pinch"
    PINCH_RELEASE = "pinch_release"

    # Static poses
    SCROLL_MODE = "scroll_mode"          # index + middle extended
    ROCK_SIGN = "rock_sign"              # index + pinky extended
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    FOUR_FINGER_PINCH = "four_finger_pinch"
    FIVE_FINGER_PINCH = "five_finger_pinch"
    GRAB_WINDOW = "grab_window"

    # One-hand swipes (open palm / four fingers)
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"

    # Three-finger swipes (media + volume)
    THREE_FINGER_SWIPE_LEFT = "three_finger_swipe_left"
    THREE_FINGER_SWIPE_RIGHT = "three_finger_swipe_right"
    THREE_FINGER_SWIPE_UP = "three_finger_swipe_up"
    THREE_FINGER_SWIPE_DOWN = "three_finger_swipe_down"

    # Two-hand gestures
    TWO_HAND_ZOOM_IN = "two_hand_zoom_in"
    TWO_HAND_ZOOM_OUT = "two_hand_zoom_out"
    BRIGHTNESS_UP = "brightness_up"
    BRIGHTNESS_DOWN = "brightness_down"

    # Wrist rotation (computed by controllers.media_controller.RotationController)
    WRIST_ROTATE_CW = "wrist_rotate_cw"
    WRIST_ROTATE_CCW = "wrist_rotate_ccw"

    # AR overlay menu
    AR_MENU_TOGGLE = "ar_menu_toggle"
    AR_MENU_SELECT = "ar_menu_select"


@dataclass
class GestureEvent:
    """A recognised gesture.

    Positional order matters: ``main.py`` builds events as
    ``GestureEvent(GestureType.NONE, None, 1.0, time.time())``.
    """

    gesture: GestureType
    hand: Optional[Hand] = None
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> str:
        return self.gesture.value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GestureEvent({self.gesture.value}, conf={self.confidence:.2f})"


#: Default gesture -> action mapping consumed by ``GestureOSEngine._do_action``.
DEFAULT_ACTION_MAP: Dict[str, str] = {
    GestureType.PINCH.value: "left_click",
    GestureType.DOUBLE_PINCH.value: "double_click",
    GestureType.MIDDLE_PINCH.value: "right_click",
    GestureType.HOLD_PINCH.value: "hold_pinch",
    GestureType.SCROLL_MODE.value: "scroll_mode",
    GestureType.FIST.value: "grab_window",
    GestureType.GRAB_WINDOW.value: "grab_window",
    GestureType.FOUR_FINGER_PINCH.value: "show_desktop",
    GestureType.FIVE_FINGER_PINCH.value: "start_menu",
    GestureType.SWIPE_LEFT.value: "swipe_left",
    GestureType.SWIPE_RIGHT.value: "swipe_right",
    GestureType.SWIPE_UP.value: "swipe_up",
    GestureType.SWIPE_DOWN.value: "swipe_down",
    GestureType.THREE_FINGER_SWIPE_LEFT.value: "media_prev",
    GestureType.THREE_FINGER_SWIPE_RIGHT.value: "media_next",
    GestureType.THREE_FINGER_SWIPE_UP.value: "volume_up",
    GestureType.THREE_FINGER_SWIPE_DOWN.value: "volume_down",
    GestureType.THUMBS_UP.value: "media_play_pause",
    GestureType.THUMBS_DOWN.value: "mute_toggle",
    GestureType.TWO_HAND_ZOOM_IN.value: "zoom_in",
    GestureType.TWO_HAND_ZOOM_OUT.value: "zoom_out",
    GestureType.BRIGHTNESS_UP.value: "brightness_up",
    GestureType.BRIGHTNESS_DOWN.value: "brightness_down",
    GestureType.WRIST_ROTATE_CW.value: "rotate_cw",
    GestureType.WRIST_ROTATE_CCW.value: "rotate_ccw",
    GestureType.ROCK_SIGN.value: "ar_menu_toggle",
    GestureType.AR_MENU_TOGGLE.value: "ar_menu_toggle",
    GestureType.AR_MENU_SELECT.value: "ar_menu_select",
}


DEFAULT_GESTURE_CONFIG: Dict[str, Any] = {
    # Pinch
    "pinch_threshold": 0.06,
    "pinch_ratio_threshold": 0.34,
    "use_absolute_pinch": False,
    "pinch_release_ratio": 0.45,        # hysteresis: must open beyond this to re-arm
    "middle_pinch_ratio_threshold": 0.34,
    "double_click_interval": 0.35,
    "pinch_hold_threshold": 0.45,
    "pinch_cooldown": 0.25,
    "hold_cooldown": 0.2,
    # Poses
    "hold_duration": 0.35,
    "pose_cooldown": 0.6,
    "finger_curl_margin": 1.05,
    # Swipes
    "swipe_threshold": 0.15,
    "swipe_min_distance": 0.15,
    "swipe_max_time": 0.5,
    "swipe_velocity_threshold": 500,    # px/s at 1280 px width (normalised internally)
    "swipe_cooldown": 0.6,
    "swipe_direction_ratio": 1.6,       # dominant axis must beat the other by this
    "reference_frame_width": 1280,
    # Two-hand
    "two_hand_distance_threshold": 0.1,
    "two_hand_delta_threshold": 0.035,
    "zoom_sensitivity": 1.0,
    "two_hand_cooldown": 0.25,
    # Rotation
    "rotation_threshold": 15.0,
    "rotation_sensitivity": 1.0,
    "rotation_cooldown": 0.4,
    # Misc
    "gesture_cooldown": 0.3,
    "min_hand_confidence": 0.5,
    "trail_length": 24,
}


@dataclass
class _HandState:
    """Per-hand temporal state."""

    pinching: bool = False
    middle_pinching: bool = False
    pinch_started_at: float = 0.0
    pinch_starts: Deque[float] = field(default_factory=lambda: deque(maxlen=6))
    hold_emitted: bool = False
    pose: GestureType = GestureType.NONE
    pose_since: float = 0.0
    pose_emitted_at: Dict[str, float] = field(default_factory=dict)
    trail: Deque[Tuple[float, float, float, int]] = field(default_factory=lambda: deque(maxlen=64))
    last_seen: float = 0.0


class GestureRecognizer:
    """Stateful gesture recogniser.

    ``main.py`` constructs this with the *whole* config dict, so gesture
    settings are read from ``config['gestures']`` and optionally refined by
    ``config['calibration']`` (the Calibration tab writes those keys).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        gesture_cfg = dict(DEFAULT_GESTURE_CONFIG)
        deep_merge(gesture_cfg, config.get("gestures", {}) or {})

        # The calibration panel overrides a handful of live-tunable values.
        calibration = config.get("calibration", {}) or {}
        if calibration.get("calibrated"):
            for key in ("pinch_threshold", "swipe_threshold"):
                if key in calibration:
                    gesture_cfg[key] = calibration[key]
            if "cursor_sensitivity" in calibration:
                gesture_cfg.setdefault("cursor_sensitivity", calibration["cursor_sensitivity"])

        self.config = gesture_cfg
        self.action_map: Dict[str, str] = dict(DEFAULT_ACTION_MAP)
        self.action_map.update((config.get("gestures", {}) or {}).get("action_map", {}) or {})

        self._states: Dict[str, _HandState] = {}
        self._cooldowns: Dict[str, float] = {}
        self._prev_palm_distance: Optional[float] = None
        self._prev_two_hand_pinching: bool = False
        self._last_events: List[GestureEvent] = []
        self._current_pose: str = GestureType.NONE.value
        self._swipe_velocity_norm: float = self._normalised_swipe_velocity()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_hands(
        self,
        hands: Sequence[Hand],
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> List[GestureEvent]:
        """Analyse the hands in one frame and return the events to dispatch."""
        now = time.time()
        events: List[GestureEvent] = []

        if frame_size:
            width = max(int(frame_size[0]), 1)
            self._swipe_velocity_norm = float(self.config["swipe_velocity_threshold"]) / width

        if not hands:
            self._current_pose = GestureType.NONE.value
            self._prev_palm_distance = None
            self._prev_two_hand_pinching = False
            self._prune_states(now)
            self._last_events = []
            return events

        poses: List[GestureType] = []
        pinching_hands = 0

        for hand in hands:
            if not hand.gesture_data:
                hand.refresh_gesture_data()
            if hand.handedness_score < float(self.config.get("min_hand_confidence", 0.5)):
                continue

            state = self._states.setdefault(self._key(hand), _HandState())
            state.last_seen = now
            self._push_trail(state, hand, now)

            pose = self.classify_pose(hand)
            poses.append(pose)
            state.pose = pose

            if self._is_pinching(hand):
                pinching_hands += 1

            events.extend(self._handle_pinch(hand, state, now))
            events.extend(self._handle_pose(hand, state, pose, now))
            events.extend(self._handle_swipe(hand, state, now))

        events.extend(self._handle_two_hands(hands, pinching_hands, now))

        self._prune_states(now)
        self._current_pose = poses[0].value if poses else GestureType.NONE.value
        # Keep at most one event per gesture family per frame (deterministic UI).
        self._last_events = events
        return events

    def classify_pose(self, hand: Hand) -> GestureType:
        """Static pose of a single hand (no temporal information).

        Order matters: the specific pinches are tested before the generic
        fist / open-palm poses, otherwise a five-finger gather or a middle-finger
        pinch gets swallowed by the "extended == 0/5" checks.
        """
        if hand is None or len(hand.landmarks) < 21:
            return GestureType.NONE
        if not hand.gesture_data:
            hand.refresh_gesture_data()
        data = hand.gesture_data
        fingers: List[bool] = list(data.get("fingers_up", hand.compute_fingers_up()))
        if len(fingers) < 5:
            return GestureType.NONE
        thumb, index, middle, ring, pinky = fingers[:5]
        extended = sum(1 for f in fingers if f)

        ratio_thr = float(self.config["pinch_ratio_threshold"])
        index_near = float(data.get("index_pinch_ratio", 1.0)) < ratio_thr
        middle_near = float(data.get("middle_pinch_ratio", 1.0)) < ratio_thr
        ring_near = float(data.get("ring_pinch_ratio", 1.0)) < ratio_thr
        pinky_near = float(data.get("pinky_pinch_ratio", 1.0)) < ratio_thr
        tips_to_thumb = sum(1 for near in (index_near, middle_near, ring_near, pinky_near) if near)

        # 1. Gathered fingertips ("claw" pinches) - the most specific poses.
        if tips_to_thumb >= 3 and extended <= 1:
            return GestureType.FIVE_FINGER_PINCH       # -> Start Menu
        if tips_to_thumb >= 3 and pinky and not index and not middle:
            return GestureType.FOUR_FINGER_PINCH       # -> Show Desktop

        # 2. Index-tip pinch.  The middle and ring fingers must stay clear of
        #    the thumb, otherwise this is a claw rather than a click.
        pinched = index_near and not middle_near and not ring_near
        if pinched:
            return GestureType.PINCH

        # 3. Middle-finger pinch.  The middle finger has to be extended: a thumb
        #    resting against curled fingers is not a pinch.
        if middle_near and middle and not index_near:
            return GestureType.MIDDLE_PINCH

        # 4. Generic hand shapes.
        if extended == 0:
            return GestureType.FIST
        if extended == 5:
            return GestureType.OPEN_PALM

        if thumb and not index and not middle and not ring and not pinky:
            return GestureType.THUMBS_UP if self._thumb_points_up(hand) else GestureType.THUMBS_DOWN

        if index and pinky and not middle and not ring:
            return GestureType.ROCK_SIGN

        if index and middle and not ring and not pinky:
            return GestureType.SCROLL_MODE

        if index and not middle and not ring and not pinky:
            return GestureType.POINT

        return GestureType.NONE

    def action_for(self, event: GestureEvent) -> str:
        """Map a gesture event to the action name used by ``_do_action``."""
        if event is None or event.gesture is None:
            return ""
        return self.action_map.get(event.gesture.value, "")

    def current_pose(self) -> str:
        """Pose of the primary hand in the last processed frame."""
        return self._current_pose

    def last_events(self) -> List[GestureEvent]:
        return list(self._last_events)

    def set_action_map(self, mapping: Dict[str, str]) -> None:
        """Replace the gesture -> action mapping (used by profile switching)."""
        self.action_map = dict(DEFAULT_ACTION_MAP)
        self.action_map.update(mapping or {})

    def reset(self) -> None:
        """Forget all temporal state (profile switch, pause, camera restart)."""
        self._states.clear()
        self._cooldowns.clear()
        self._prev_palm_distance = None
        self._prev_two_hand_pinching = False
        self._last_events = []
        self._current_pose = GestureType.NONE.value

    # ------------------------------------------------------------------ #
    # Pinch handling
    # ------------------------------------------------------------------ #
    def _is_pinching(self, hand: Hand) -> bool:
        data = hand.gesture_data
        if self.config.get("use_absolute_pinch"):
            pinched = float(data.get("pinch_distance", 1.0)) < float(self.config["pinch_threshold"])
        else:
            pinched = float(data.get("index_pinch_ratio", 1.0)) < float(self.config["pinch_ratio_threshold"])
        if not pinched:
            return False
        # A claw (several fingertips on the thumb) must not register as a click.
        threshold = float(self.config["pinch_ratio_threshold"])
        return (
            float(data.get("middle_pinch_ratio", 1.0)) >= threshold
            and float(data.get("ring_pinch_ratio", 1.0)) >= threshold
        )

    def _is_middle_pinching(self, hand: Hand) -> bool:
        """Thumb tip touching an *extended* middle fingertip (right click)."""
        data = hand.gesture_data or {}
        middle_near = float(data.get("middle_pinch_ratio", 1.0)) < float(
            self.config["middle_pinch_ratio_threshold"]
        )
        index_near = float(data.get("index_pinch_ratio", 1.0)) < float(self.config["pinch_ratio_threshold"])
        if index_near or not middle_near:
            return False
        fingers = list(data.get("fingers_up") or hand.compute_fingers_up())
        return len(fingers) >= 3 and bool(fingers[2])

    def _handle_pinch(self, hand: Hand, state: _HandState, now: float) -> List[GestureEvent]:
        events: List[GestureEvent] = []
        release_ratio = float(self.config["pinch_release_ratio"])
        ratio = float(hand.gesture_data.get("pinch_ratio", 1.0))
        confidence = clamp(1.0 - ratio, 0.0, 1.0)

        # --- middle-finger pinch => right click -------------------------
        middle_now = self._is_middle_pinching(hand)
        if middle_now and not state.middle_pinching:
            if self._cooldown_ok("middle_pinch", self.config["pinch_cooldown"], now):
                events.append(
                    GestureEvent(GestureType.MIDDLE_PINCH, hand, confidence, now, {"ratio": ratio})
                )
        state.middle_pinching = middle_now

        # --- thumb + index pinch ----------------------------------------
        pinching_now = self._is_pinching(hand)
        if pinching_now and not state.pinching:
            # A fresh pinch: decide single vs double click.
            state.pinch_started_at = now
            state.hold_emitted = False
            state.pinch_starts.append(now)

            previous = state.pinch_starts[-2] if len(state.pinch_starts) >= 2 else None
            interval = float(self.config["double_click_interval"])
            if previous is not None and (now - previous) <= interval:
                state.pinch_starts.clear()
                if self._cooldown_ok("double_pinch", 0.15, now):
                    events.append(
                        GestureEvent(
                            GestureType.DOUBLE_PINCH, hand, confidence, now,
                            {"interval": now - previous} if previous else {},
                        )
                    )
            elif self._cooldown_ok("pinch", self.config["pinch_cooldown"], now):
                events.append(GestureEvent(GestureType.PINCH, hand, confidence, now, {"ratio": ratio}))

        elif pinching_now and state.pinching:
            hold_for = float(self.config["pinch_hold_threshold"])
            if not state.hold_emitted and now - state.pinch_started_at >= hold_for:
                state.hold_emitted = True
                if self._cooldown_ok("hold_pinch", self.config["hold_cooldown"], now):
                    events.append(
                        GestureEvent(
                            GestureType.HOLD_PINCH, hand, confidence, now,
                            {"held": now - state.pinch_started_at},
                        )
                    )

        elif not pinching_now and state.pinching:
            state.pinching = False
            if ratio > release_ratio:  # hysteresis - fully open before re-arming
                events.append(GestureEvent(GestureType.PINCH_RELEASE, hand, confidence, now))
                state.hold_emitted = False

        state.pinching = pinching_now
        return [e for e in events if e.gesture is not GestureType.PINCH_RELEASE or self.config.get("emit_release", False)]

    # ------------------------------------------------------------------ #
    # Static pose handling (held gestures)
    # ------------------------------------------------------------------ #
    def _handle_pose(self, hand: Hand, state: _HandState, pose: GestureType, now: float) -> List[GestureEvent]:
        if pose in (GestureType.NONE, GestureType.POINT, GestureType.OPEN_PALM, GestureType.PINCH):
            state.pose_since = 0.0
            return []

        # Pose must be held briefly to avoid transient misfires.  With
        # hold_duration == 0 (tests / very twitchy setups) the first frame fires.
        hold_required = float(self.config["hold_duration"])
        if state.pose_since == 0.0:
            state.pose_since = now
            if hold_required > 0:
                return []
        held = now - state.pose_since
        if held < hold_required:
            return []

        cooldown_key = f"pose:{pose.value}:{hand.handedness}"
        if not self._cooldown_ok(cooldown_key, self.config["pose_cooldown"], now):
            return []

        confidence = clamp(0.6 + min(held, 1.0) * 0.4, 0.0, 1.0)
        return [GestureEvent(pose, hand, confidence, now, {"held": held})]

    # ------------------------------------------------------------------ #
    # Swipe handling
    # ------------------------------------------------------------------ #
    def _push_trail(self, state: _HandState, hand: Hand, now: float) -> None:
        cx, cy = hand.palm_center
        state.trail.append((now, cx, cy, int(hand.gesture_data.get("extended_count", 0))))

    def _handle_swipe(self, hand: Hand, state: _HandState, now: float) -> List[GestureEvent]:
        max_time = float(self.config["swipe_max_time"])
        min_distance = float(self.config["swipe_min_distance"])
        min_velocity = self._swipe_velocity_norm

        trail = [p for p in state.trail if now - p[0] <= max_time]
        if len(trail) < 3:
            return []

        t0, x0, y0, ext0 = trail[0]
        t1, x1, y1, ext1 = trail[-1]
        dt = t1 - t0
        dx = x1 - x0
        dy = y1 - y0
        if dt <= 1e-4:
            return []

        dist = math.hypot(dx, dy)
        velocity = dist / dt
        if dist < min_distance or velocity < min_velocity:
            return []

        ratio = float(self.config["swipe_direction_ratio"])
        if abs(dx) >= abs(dy) * ratio:
            direction = "right" if dx > 0 else "left"
        elif abs(dy) >= abs(dx) * ratio:
            direction = "down" if dy > 0 else "up"
        else:
            return []

        # Extended-finger count picks the gesture family.
        extended = max(ext0, ext1)
        if extended == 3:
            gesture = {
                "left": GestureType.THREE_FINGER_SWIPE_LEFT,
                "right": GestureType.THREE_FINGER_SWIPE_RIGHT,
                "up": GestureType.THREE_FINGER_SWIPE_UP,
                "down": GestureType.THREE_FINGER_SWIPE_DOWN,
            }[direction]
        elif extended >= 4:
            gesture = {
                "left": GestureType.SWIPE_LEFT,
                "right": GestureType.SWIPE_RIGHT,
                "up": GestureType.SWIPE_UP,
                "down": GestureType.SWIPE_DOWN,
            }[direction]
        else:
            return []  # pointing / pinched hand: no swipe (it is a drag or cursor move)

        if not self._cooldown_ok(f"swipe:{hand.handedness}", self.config["swipe_cooldown"], now):
            return []

        confidence = clamp(dist / max(min_distance, 1e-6) * 0.5, 0.5, 1.0)
        state.trail.clear()
        return [
            GestureEvent(
                gesture, hand, confidence, now,
                {"dx": dx, "dy": dy, "velocity": velocity, "distance": dist},
            )
        ]

    def _normalised_swipe_velocity(self) -> float:
        return float(self.config["swipe_velocity_threshold"]) / float(
            self.config.get("reference_frame_width", 1280)
        )

    # ------------------------------------------------------------------ #
    # Two-hand gestures
    # ------------------------------------------------------------------ #
    def _handle_two_hands(self, hands: Sequence[Hand], pinching_hands: int, now: float) -> List[GestureEvent]:
        if len(hands) < 2:
            self._prev_palm_distance = None
            self._prev_two_hand_pinching = False
            return []

        h1, h2 = hands[0], hands[1]
        palm_gap = distance(h1.palm_center, h2.palm_center)
        both_pinching = pinching_hands >= 2

        events: List[GestureEvent] = []
        previous = self._prev_palm_distance
        self._prev_palm_distance = palm_gap

        if previous is None:
            self._prev_two_hand_pinching = both_pinching
            return []

        delta = (palm_gap - previous) * float(self.config.get("zoom_sensitivity", 1.0))
        threshold = float(self.config["two_hand_delta_threshold"])
        if abs(delta) < threshold:
            self._prev_two_hand_pinching = both_pinching
            return []

        if both_pinching and self._prev_two_hand_pinching:
            gesture = GestureType.BRIGHTNESS_UP if delta > 0 else GestureType.BRIGHTNESS_DOWN
            key = "two_hand:brightness"
        else:
            gesture = GestureType.TWO_HAND_ZOOM_IN if delta > 0 else GestureType.TWO_HAND_ZOOM_OUT
            key = "two_hand:zoom"

        self._prev_two_hand_pinching = both_pinching

        if not self._cooldown_ok(key, self.config["two_hand_cooldown"], now):
            return []

        confidence = clamp(abs(delta) / max(threshold, 1e-6) * 0.5, 0.5, 1.0)
        events.append(
            GestureEvent(
                gesture, h1, confidence, now,
                {"delta": delta, "palm_gap": palm_gap, "both_pinching": both_pinching},
            )
        )
        return events

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _fingertips_near_thumb(self, hand: Hand) -> int:
        """How many of index/middle/ring/pinky tips are gathered at the thumb tip."""
        threshold = float(self.config["pinch_ratio_threshold"]) * 1.25
        count = 0
        for key in ("index_pinch_ratio", "middle_pinch_ratio", "ring_pinch_ratio", "pinky_pinch_ratio"):
            if float(hand.gesture_data.get(key, 1.0)) < threshold:
                count += 1
        return count

    def _thumb_points_up(self, hand: Hand) -> bool:
        tip = hand.landmark(HandLandmark.THUMB_TIP)
        mcp = hand.landmark(HandLandmark.THUMB_MCP)
        if tip is None or mcp is None:
            return False
        # Image coordinates: y grows downwards, so "up" means a smaller y.
        return tip.y < mcp.y - 0.02

    def _key(self, hand: Hand) -> str:
        return hand.handedness or "unknown"

    def _cooldown_ok(self, key: str, duration: float, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        last = self._cooldowns.get(key)
        if last is not None and now - last < float(duration):
            return False
        self._cooldowns[key] = now
        return True

    def _prune_states(self, now: float, max_age: float = 2.0) -> None:
        for key in [k for k, s in self._states.items() if now - s.last_seen > max_age]:
            del self._states[key]
        for key in [k for k, t in self._cooldowns.items() if now - t > 10.0]:
            del self._cooldowns[key]


def hand_angle(hand: Hand) -> float:
    """Palm axis angle in degrees (index MCP -> pinky MCP); used for rotation."""
    idx = hand.landmark(HandLandmark.INDEX_MCP)
    pnk = hand.landmark(HandLandmark.PINKY_MCP)
    if idx is None or pnk is None:
        return 0.0
    return math.degrees(math.atan2(pnk.y - idx.y, pnk.x - idx.x))


def rotation_delta(previous_angle: float, current_angle: float) -> float:
    """Signed shortest-path rotation between two palm angles."""
    return signed_rotation(previous_angle, current_angle)


__all__ = [
    "GestureType",
    "GestureEvent",
    "GestureRecognizer",
    "DEFAULT_ACTION_MAP",
    "DEFAULT_GESTURE_CONFIG",
    "hand_angle",
    "rotation_delta",
]
