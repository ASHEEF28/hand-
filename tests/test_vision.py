"""Tests for the vision layer (landmark maths, tracker, synthetic source)."""
from __future__ import annotations


import numpy as np
import pytest

from tests.conftest import pose_hand
from vision.hand_tracker import (
    DEFAULT_TRACKING_CONFIG,
    HAND_CONNECTIONS,
    Hand,
    HandLandmark,
    HandTracker,
    Landmark,
    make_hand,
)


class TestLandmarkModel:
    def test_landmark_helpers(self):
        point = Landmark(0.25, 0.5, 0.1)
        assert point.as_tuple() == (0.25, 0.5)
        assert point.as_tuple3() == (0.25, 0.5, 0.1)
        assert point.as_pixel(640, 480) == (160, 240)
        assert point.distance_to(Landmark(0.25, 0.5)) == 0.0
        assert Landmark(0, 0).distance_to(Landmark(3 / 10, 4 / 10)) == pytest.approx(0.5)

    def test_enum_indices_match_mediapipe(self):
        assert HandLandmark.WRIST.value == 0
        assert HandLandmark.THUMB_TIP.value == 4
        assert HandLandmark.INDEX_TIP.value == 8
        assert HandLandmark.MIDDLE_TIP.value == 12
        assert HandLandmark.PINKY_TIP.value == 20
        # Long MediaPipe spellings are aliases of the same index.
        assert HandLandmark.INDEX_FINGER_TIP is HandLandmark.INDEX_TIP
        assert HandLandmark.PINKY_FINGER_MCP is HandLandmark.PINKY_MCP
        assert len({m.value for m in HandLandmark}) == 21

    def test_connections_are_valid_indices(self):
        for a, b in HAND_CONNECTIONS:
            assert 0 <= a <= 20 and 0 <= b <= 20

    def test_make_hand_has_21_landmarks(self):
        hand = make_hand()
        assert len(hand.landmarks) == 21
        assert hand.handedness == "Right"
        assert hand.handedness_score == 1.0


class TestHandGeometry:
    def test_palm_center_is_inside_frame(self):
        hand = make_hand()
        x, y = hand.palm_center
        assert 0.0 < x < 1.0 and 0.0 < y < 1.0

    def test_open_hand_has_five_fingers_up(self):
        hand = pose_hand("open_palm")
        assert hand.fingers_up == [True, True, True, True, True]
        assert hand.extended_count == 5

    def test_fist_has_no_fingers_up(self):
        hand = pose_hand("fist")
        assert hand.extended_count == 0

    def test_point_has_only_index(self):
        fingers = pose_hand("point").fingers_up
        assert fingers[1] is True
        assert fingers[2] is False and fingers[3] is False and fingers[4] is False

    def test_pinch_distance_shrinks_when_pinching(self):
        open_hand = pose_hand("open_palm")
        pinched = pose_hand("pinch")
        assert pinched.gesture_data["pinch_distance"] < open_hand.gesture_data["pinch_distance"]
        assert pinched.gesture_data["pinch_ratio"] < 0.2
        assert open_hand.gesture_data["pinch_ratio"] > 0.5

    def test_gesture_data_is_populated(self):
        data = pose_hand("open_palm").gesture_data
        for key in ("fingers_up", "extended_count", "hand_size", "palm_center",
                    "pinch_distance", "pinch_ratio", "index_pinch_ratio",
                    "middle_pinch_ratio", "ring_pinch_ratio", "pinky_pinch_ratio"):
            assert key in data, f"missing gesture_data key: {key}"

    def test_hand_size_scales_with_distance_from_camera(self):
        near = pose_hand("open_palm", scale=1.0)
        far = pose_hand("open_palm", scale=0.5)
        assert far.hand_size < near.hand_size
        # ... but the pinch *ratio* stays roughly stable, which is the point.
        near_pinch = pose_hand("pinch", scale=1.0).gesture_data["pinch_ratio"]
        far_pinch = pose_hand("pinch", scale=0.5).gesture_data["pinch_ratio"]
        assert abs(near_pinch - far_pinch) < 0.15

    def test_landmark_lookup_by_enum_and_index(self):
        hand = make_hand()
        assert hand.landmark(HandLandmark.INDEX_TIP) is hand.landmarks[8]
        assert hand.landmark(8) is hand.landmarks[8]
        assert hand.landmark(99) is None
        assert hand.index_tip is hand.landmarks[8]
        assert hand.wrist is hand.landmarks[0]
        assert len(hand) == 21

    def test_short_landmark_list_is_safe(self):
        hand = Hand(landmarks=[Landmark(0.5, 0.5)], handedness="Left")
        assert hand.palm_center == (0.5, 0.5)
        assert hand.fingers_up == [False, False, False, False, False]
        assert hand.hand_size == 0.0
        assert hand.pinch_distance() == 1.0
        hand.refresh_gesture_data()
        assert hand.gesture_data == {}


class TestHandTracker:
    def test_defaults_are_merged(self):
        tracker = HandTracker({"max_hands": 1})
        assert tracker.config["max_hands"] == 1
        assert tracker.config["min_detection_confidence"] == DEFAULT_TRACKING_CONFIG["min_detection_confidence"]

    def test_injected_detector_is_used(self):
        class FakeDetector:
            def process(self, frame):
                return [pose_hand("open_palm")]

        tracker = HandTracker(detector=FakeDetector())
        assert tracker.backend == "injected"
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        hands = tracker.process(frame)
        assert len(hands) == 1
        assert hands[0].gesture_data["extended_count"] == 5

    def test_none_frame_returns_empty(self):
        tracker = HandTracker(detector=object())
        assert tracker.process(None) == []

    def test_low_confidence_hands_are_filtered(self):
        class FakeDetector:
            def process(self, frame):
                return [pose_hand("open_palm", score=0.1)]

        tracker = HandTracker({"min_presence_confidence": 0.5}, detector=FakeDetector())
        assert tracker.process(np.zeros((240, 320, 3), dtype=np.uint8)) == []

    def test_smoothing_reduces_jitter(self):
        positions = iter([0.50, 0.50, 0.90, 0.50, 0.50])

        class JitteryDetector:
            def process(self, frame):
                return [pose_hand("point", dx=next(positions) - 0.5)]

        tracker = HandTracker({"smoothing_factor": 0.7}, detector=JitteryDetector())
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        xs = [tracker.process(frame)[0].landmarks[8].x for _ in range(5)]
        # The spike at frame 3 must be damped, not passed through at full size.
        assert max(xs) < 0.85

    def test_smoothing_state_is_pruned(self):
        class OneShot:
            def __init__(self):
                self.calls = 0

            def process(self, frame):
                self.calls += 1
                return [pose_hand("open_palm")] if self.calls == 1 else []

        tracker = HandTracker(detector=OneShot())
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker.process(frame)
        assert tracker._previous
        tracker.process(frame)
        assert tracker._previous == {}

    def test_draw_landmarks_returns_same_shape(self):
        tracker = HandTracker(detector=object())
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        annotated = tracker.draw_landmarks(frame, [pose_hand("open_palm")])
        assert annotated.shape == frame.shape
        assert annotated is not frame  # a copy, the caller's frame is untouched
        assert tracker.draw_landmarks(frame, []) is frame
        assert tracker.draw_landmarks(None, []) is None

    def test_close_is_idempotent(self):
        tracker = HandTracker(detector=object())
        tracker.close()
        tracker.close()
        assert tracker._closed is True

    def test_missing_mediapipe_raises_helpful_error(self, monkeypatch):
        tracker = HandTracker()
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mediapipe":
                raise ImportError("no mediapipe here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from vision.hand_tracker import HandTrackingUnavailable

        with pytest.raises(HandTrackingUnavailable) as info:
            tracker.process(np.zeros((240, 320, 3), dtype=np.uint8))
        assert "pip install" in str(info.value)


class TestSyntheticHands:
    def test_script_produces_hands(self):
        from vision.synthetic_hands import SyntheticHandDetector

        detector = SyntheticHandDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        hands = detector.process(frame)
        assert len(hands) >= 1
        assert detector.backend == "synthetic"
        assert detector.hands_emitted >= 1

    def test_poses_are_distinguishable(self):
        from vision.synthetic_hands import POSES

        assert pose_hand("pinch").gesture_data["pinch_ratio"] < 0.25
        assert pose_hand("open_palm").gesture_data["pinch_ratio"] > 0.4
        assert pose_hand("fist").extended_count == 0
        assert len(POSES) >= 8

    def test_motion_moves_the_hand(self):
        from vision.synthetic_hands import SyntheticHandDetector

        detector = SyntheticHandDetector(script=[("point", 2.0, "swipe_right")])
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        first = detector.process(frame)[0].palm_center[0]
        import time as _time

        _time.sleep(0.35)
        second = detector.process(frame)[0].palm_center[0]
        assert second > first

    def test_non_looping_script_finishes(self):
        from vision.synthetic_hands import SyntheticHandDetector

        detector = SyntheticHandDetector(script=[("open_palm", 0.0, "idle")], loop=False)
        detector.process(None)
        detector._start -= 10
        assert detector.process(None) == []
        assert detector.finished is True

    def test_draw_landmarks_and_reset(self):
        from vision.synthetic_hands import SyntheticHandDetector

        detector = SyntheticHandDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        hands = detector.process(frame)
        annotated = detector.draw_landmarks(frame, hands)
        assert annotated.shape == frame.shape
        detector.close()
        detector.reset()
        assert detector.finished is False
        assert isinstance(detector.describe(), str)


class TestRealMediaPipe:
    """Exercises the actual MediaPipe backend when it is installed.

    Skipped on mediapipe 1.x (Tasks-only wheels ship without a model) and in
    environments where the native libraries cannot load.
    """

    @staticmethod
    def _legacy_available():
        mediapipe = pytest.importorskip("mediapipe")
        if not hasattr(mediapipe, "solutions"):
            pytest.skip("mediapipe 1.x installed - legacy solutions API unavailable")
        return mediapipe

    def test_tracker_initialises_and_processes_a_frame(self, config):
        self._legacy_available()
        from vision.hand_tracker import Hand, HandTracker

        tracker = HandTracker(config["tracking"])
        try:
            tracker._ensure_initialized()
            assert tracker.backend in ("solutions", "tasks")

            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            hands = tracker.process(frame)
            assert isinstance(hands, list)
            assert all(isinstance(hand, Hand) for hand in hands)

            annotated = tracker.draw_landmarks(frame, hands)
            assert annotated.shape == frame.shape
        finally:
            tracker.close()
            tracker.close()            # closing twice must be safe

    def test_repeated_processing_is_stable(self, config):
        self._legacy_available()
        from vision.hand_tracker import HandTracker

        tracker = HandTracker(config["tracking"])
        try:
            frame = np.full((240, 320, 3), 128, dtype=np.uint8)
            for _ in range(5):
                assert isinstance(tracker.process(frame), list)
        finally:
            tracker.close()
