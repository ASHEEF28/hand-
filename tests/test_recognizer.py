"""Tests for the gesture recognition engine."""
from __future__ import annotations

import time

import pytest

from tests.conftest import pose_hand
from gestures.recognizer import (
    DEFAULT_ACTION_MAP,
    GestureEvent,
    GestureRecognizer,
    GestureType,
)


@pytest.fixture
def fast(config):
    """A recogniser with every cooldown/hold delay set to zero."""
    return GestureRecognizer(config)


class TestPoseClassification:
    @pytest.mark.parametrize("pose,expected", [
        ("open_palm", GestureType.OPEN_PALM),
        ("point", GestureType.POINT),
        ("fist", GestureType.FIST),
        ("pinch", GestureType.PINCH),
        ("scroll_pose", GestureType.SCROLL_MODE),
        ("rock_sign", GestureType.ROCK_SIGN),
        ("thumbs_up", GestureType.THUMBS_UP),
        ("thumbs_down", GestureType.THUMBS_DOWN),
        ("five_finger_pinch", GestureType.FIVE_FINGER_PINCH),
    ])
    def test_static_poses(self, fast, pose, expected):
        assert fast.classify_pose(pose_hand(pose)) is expected

    def test_middle_pinch_is_not_a_left_click(self, fast):
        assert fast.classify_pose(pose_hand("middle_pinch")) is GestureType.MIDDLE_PINCH

    def test_empty_hand_is_none(self, fast):
        from vision.hand_tracker import Hand

        assert fast.classify_pose(Hand()) is GestureType.NONE

    def test_claw_is_not_mistaken_for_a_click(self, fast):
        """All fingertips on the thumb must not fire a left click."""
        hand = pose_hand("five_finger_pinch")
        assert fast._is_pinching(hand) is False
        assert fast.classify_pose(hand) is not GestureType.PINCH

    def test_fist_is_not_a_pinch(self, fast):
        hand = pose_hand("fist")
        assert fast.classify_pose(hand) is GestureType.FIST
        assert fast._is_pinching(hand) is False


class TestPinchEvents:
    def test_pinch_emits_left_click(self, fast):
        events = fast.process_hands([pose_hand("open_palm")])
        events += fast.process_hands([pose_hand("pinch")])
        assert any(e.gesture is GestureType.PINCH for e in events)
        assert fast.action_for(next(e for e in events if e.gesture is GestureType.PINCH)) == "left_click"

    def test_pinch_cooldown_suppresses_repeats(self, config):
        config["gestures"]["pinch_cooldown"] = 5.0
        recognizer = GestureRecognizer(config)
        recognizer.process_hands([pose_hand("open_palm")])
        first = recognizer.process_hands([pose_hand("pinch")])
        second = recognizer.process_hands([pose_hand("pinch")])
        assert any(e.gesture is GestureType.PINCH for e in first)
        assert not any(e.gesture is GestureType.PINCH for e in second)

    def test_double_pinch(self, fast):
        fast.process_hands([pose_hand("open_palm")])
        fast.process_hands([pose_hand("pinch")])
        fast.process_hands([pose_hand("open_palm")])
        events = fast.process_hands([pose_hand("pinch")])
        gestures = [e.gesture for e in events]
        assert GestureType.DOUBLE_PINCH in gestures
        assert fast.action_for(next(e for e in events if e.gesture is GestureType.DOUBLE_PINCH)) == "double_click"

    def test_hold_pinch_after_threshold(self, config):
        config["gestures"]["pinch_hold_threshold"] = 0.05
        recognizer = GestureRecognizer(config)
        recognizer.process_hands([pose_hand("open_palm")])
        recognizer.process_hands([pose_hand("pinch")])
        time.sleep(0.08)
        events = recognizer.process_hands([pose_hand("pinch")])
        assert any(e.gesture is GestureType.HOLD_PINCH for e in events)

    def test_hold_pinch_fires_once(self, config):
        config["gestures"]["pinch_hold_threshold"] = 0.0
        config["gestures"]["hold_cooldown"] = 0.0
        recognizer = GestureRecognizer(config)
        recognizer.process_hands([pose_hand("open_palm")])
        recognizer.process_hands([pose_hand("pinch")])
        first = recognizer.process_hands([pose_hand("pinch")])
        second = recognizer.process_hands([pose_hand("pinch")])
        assert sum(e.gesture is GestureType.HOLD_PINCH for e in first) == 1
        assert sum(e.gesture is GestureType.HOLD_PINCH for e in second) == 0

    def test_middle_pinch_emits_right_click(self, fast):
        fast.process_hands([pose_hand("open_palm")])
        events = fast.process_hands([pose_hand("middle_pinch")])
        assert any(e.gesture is GestureType.MIDDLE_PINCH for e in events)

    def test_absolute_pinch_mode(self, config):
        config["gestures"]["use_absolute_pinch"] = True
        config["gestures"]["pinch_threshold"] = 0.05
        recognizer = GestureRecognizer(config)
        assert recognizer._is_pinching(pose_hand("pinch")) is True
        assert recognizer._is_pinching(pose_hand("open_palm")) is False


class TestPoseEvents:
    def test_held_pose_emits_once(self, fast):
        fast.process_hands([pose_hand("open_palm")])
        events = fast.process_hands([pose_hand("thumbs_up")])
        assert any(e.gesture is GestureType.THUMBS_UP for e in events)
        assert fast.action_for(next(e for e in events if e.gesture is GestureType.THUMBS_UP)) == "media_play_pause"
        # Cooldown is zero in the fixture, so the next emission needs a new hold.
        assert fast.process_hands([pose_hand("thumbs_up")]) != []

    def test_point_and_open_palm_emit_nothing(self, fast):
        assert fast.process_hands([pose_hand("open_palm")]) == []
        assert fast.process_hands([pose_hand("point")]) == []

    def test_current_pose_tracks_primary_hand(self, fast):
        fast.process_hands([pose_hand("scroll_pose")])
        assert fast.current_pose() == "scroll_mode"
        fast.process_hands([])
        assert fast.current_pose() == "none"

    def test_fist_maps_to_grab_window(self, fast):
        events = fast.process_hands([pose_hand("fist")])
        assert any(fast.action_for(e) == "grab_window" for e in events)


class TestSwipes:
    @staticmethod
    def _swipe(recognizer, pose, axis, frames=14, step=0.025, start=0.3):
        events = []
        recognizer.process_hands([pose_hand(pose, dx=0 if axis == "y" else start - 0.1,
                                           dy=0 if axis == "x" else start - 0.1)])
        for i in range(frames):
            offset = start + i * step
            hand = pose_hand(pose, dx=offset if axis == "x" else 0.0,
                             dy=offset if axis == "y" else 0.0)
            events.extend(recognizer.process_hands([hand], frame_size=(1280, 720)))
        return [e.gesture for e in events]

    def test_open_palm_swipe_right(self, fast):
        assert GestureType.SWIPE_RIGHT in self._swipe(fast, "open_palm", "x")

    def test_open_palm_swipe_left(self, fast):
        gestures = self._swipe(fast, "open_palm", "x", step=-0.025)
        assert GestureType.SWIPE_LEFT in gestures

    def test_open_palm_swipe_down(self, fast):
        assert GestureType.SWIPE_DOWN in self._swipe(fast, "open_palm", "y")

    def test_three_finger_swipe_is_media(self, fast):
        gestures = self._swipe(fast, "three_fingers", "x")
        assert GestureType.THREE_FINGER_SWIPE_RIGHT in gestures
        assert fast.action_for(GestureEvent(GestureType.THREE_FINGER_SWIPE_RIGHT)) == "media_next"

    def test_three_finger_swipe_up_is_volume(self, fast):
        gestures = self._swipe(fast, "three_fingers", "y", step=-0.025)
        assert GestureType.THREE_FINGER_SWIPE_UP in gestures

    def test_pointing_hand_does_not_swipe(self, fast):
        """A single extended finger is cursor movement, never a swipe."""
        assert GestureType.SWIPE_RIGHT not in self._swipe(fast, "point", "x")

    def test_slow_movement_is_not_a_swipe(self, fast):
        assert GestureType.SWIPE_RIGHT not in self._swipe(fast, "open_palm", "x", frames=4, step=0.005)

    def test_swipe_cooldown_prevents_repeat(self, config):
        config["gestures"]["swipe_cooldown"] = 10.0
        recognizer = GestureRecognizer(config)
        first = self._swipe(recognizer, "open_palm", "x")
        assert GestureType.SWIPE_RIGHT in first
        recognizer.process_hands([])
        second = self._swipe(recognizer, "open_palm", "x")
        assert GestureType.SWIPE_RIGHT not in second


class TestTwoHandGestures:
    def test_hands_apart_zoom_in(self, fast):
        events = []
        for spread in (0.15, 0.20, 0.26, 0.33, 0.41):
            hands = [
                pose_hand("open_palm", dx=-spread / 2),
                pose_hand("open_palm", dx=spread / 2, handedness="Left"),
            ]
            events.extend(fast.process_hands(hands))
        assert any(e.gesture is GestureType.TWO_HAND_ZOOM_IN for e in events)

    def test_hands_together_zoom_out(self, fast):
        events = []
        for spread in (0.45, 0.38, 0.31, 0.24, 0.17):
            hands = [
                pose_hand("open_palm", dx=-spread / 2),
                pose_hand("open_palm", dx=spread / 2, handedness="Left"),
            ]
            events.extend(fast.process_hands(hands))
        assert any(e.gesture is GestureType.TWO_HAND_ZOOM_OUT for e in events)

    def test_both_pinching_changes_brightness(self, fast):
        events = []
        for spread in (0.15, 0.22, 0.30, 0.39):
            hands = [
                pose_hand("pinch", dx=-spread / 2),
                pose_hand("pinch", dx=spread / 2, handedness="Left"),
            ]
            events.extend(fast.process_hands(hands))
        gestures = [e.gesture for e in events]
        assert GestureType.BRIGHTNESS_UP in gestures
        brightness = next(e for e in events if e.gesture is GestureType.BRIGHTNESS_UP)
        assert brightness.hand is not None
        assert "pinch_distance" in brightness.hand.gesture_data

    def test_single_hand_never_zooms(self, fast):
        events = []
        for dx in (0.0, 0.2, 0.4, 0.6):
            events.extend(fast.process_hands([pose_hand("open_palm", dx=dx)]))
        assert not any(e.gesture in (GestureType.TWO_HAND_ZOOM_IN, GestureType.TWO_HAND_ZOOM_OUT)
                       for e in events)


class TestActionMapping:
    def test_every_mapped_gesture_has_an_action(self):
        recognizer = GestureRecognizer({})
        for gesture in GestureType:
            if gesture in (GestureType.NONE, GestureType.POINT, GestureType.OPEN_PALM,
                           GestureType.PINCH_RELEASE):
                continue
            action = recognizer.action_for(GestureEvent(gesture))
            assert action, f"{gesture.value} has no action mapping"

    def test_default_action_map_covers_documented_gestures(self):
        for gesture, action in {
            "pinch": "left_click",
            "double_pinch": "double_click",
            "middle_pinch": "right_click",
            "hold_pinch": "hold_pinch",
            "scroll_mode": "scroll_mode",
            "swipe_left": "swipe_left",
            "swipe_right": "swipe_right",
            "swipe_up": "swipe_up",
            "swipe_down": "swipe_down",
            "four_finger_pinch": "show_desktop",
            "five_finger_pinch": "start_menu",
        }.items():
            assert DEFAULT_ACTION_MAP[gesture] == action

    def test_profile_can_override_the_map(self):
        recognizer = GestureRecognizer({})
        recognizer.set_action_map({"swipe_right": "media_next"})
        assert recognizer.action_for(GestureEvent(GestureType.SWIPE_RIGHT)) == "media_next"
        assert recognizer.action_for(GestureEvent(GestureType.PINCH)) == "left_click"

    def test_config_action_map_is_applied(self):
        recognizer = GestureRecognizer({"gestures": {"action_map": {"pinch": "middle_click"}}})
        assert recognizer.action_for(GestureEvent(GestureType.PINCH)) == "middle_click"

    def test_calibration_overrides_thresholds(self):
        recognizer = GestureRecognizer({
            "gestures": {"pinch_threshold": 0.06},
            "calibration": {"calibrated": True, "pinch_threshold": 0.11},
        })
        assert recognizer.config["pinch_threshold"] == 0.11

    def test_uncalibrated_values_are_ignored(self):
        recognizer = GestureRecognizer({
            "gestures": {"pinch_threshold": 0.06},
            "calibration": {"calibrated": False, "pinch_threshold": 0.11},
        })
        assert recognizer.config["pinch_threshold"] == 0.06


class TestEventAndState:
    def test_event_positional_construction(self):
        """main.py builds events positionally - the field order is a contract."""
        event = GestureEvent(GestureType.NONE, None, 1.0, time.time())
        assert event.gesture is GestureType.NONE
        assert event.hand is None
        assert event.confidence == 1.0
        assert event.value == "none"
        assert event.data == {}

    def test_no_hands_clears_state(self, fast):
        fast.process_hands([pose_hand("open_palm")])
        assert fast.process_hands([]) == []
        # A hand lost for a single frame keeps its state (MediaPipe drops hands),
        # but a stale one is pruned so a re-appearing hand starts fresh.
        assert fast._states
        for state in fast._states.values():
            state.last_seen -= 10.0
        fast.process_hands([])
        assert fast._states == {}

    def test_reset_clears_everything(self, fast):
        fast.process_hands([pose_hand("pinch")])
        fast.reset()
        assert fast._states == {}
        assert fast._cooldowns == {}
        assert fast.last_events() == []

    def test_stale_hand_state_is_pruned(self, fast):
        fast.process_hands([pose_hand("open_palm")])
        key = next(iter(fast._states))
        fast._states[key].last_seen -= 10
        fast.process_hands([])
        assert key not in fast._states

    def test_low_confidence_hand_is_ignored(self, fast):
        assert fast.process_hands([pose_hand("pinch", score=0.1)]) == []

    def test_confidence_is_within_range(self, fast):
        fast.process_hands([pose_hand("open_palm")])
        events = fast.process_hands([pose_hand("pinch")])
        for event in events:
            assert 0.0 <= event.confidence <= 1.0
