"""Shared fixtures for the GestureOS test-suite.

Nothing here needs a webcam, a display or MediaPipe:

* input goes to :class:`RecordingBackend` (dry run - the OS is never touched),
* hands come from :mod:`vision.synthetic_hands` (``pose_hand`` below),
* Qt runs on the ``offscreen`` platform plugin,
* every on-disk store (macros, apps, profiles, learned gestures, analytics) is
  redirected into pytest's ``tmp_path`` so the repository stays clean.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Qt without a display server (CI, ssh, Windows services).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false")


@pytest.fixture(autouse=True)
def _offscreen_qt():
    """Keep Qt headless for every test, even if the env var was unset."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    yield


@pytest.fixture
def backend():
    """Install a dry-run input backend as the process-wide default."""
    from utils.input_backend import RecordingBackend, set_backend

    created = RecordingBackend((1920, 1080), max_events=5000)
    set_backend(created)
    try:
        yield created
    finally:
        set_backend(None)


@pytest.fixture
def config(tmp_path) -> Dict[str, Any]:
    """Full config dict: defaults with fast timings and temporary stores."""
    from config import DEFAULT_CONFIG

    data = copy.deepcopy(DEFAULT_CONFIG)

    data["camera"].update({
        "device_id": 0, "width": 320, "height": 240, "fps": 30,
        "buffer_size": 4, "simulate": True, "simulate_on_failure": True,
        "simulate_fps": 60, "read_timeout": 1.0,
    })
    data["tracking"].update({"flip_horizontal": False, "smoothing": 0.0})

    # Deterministic cursor maths: no smoothing, full-frame movement region.
    data["cursor"].update({
        "smoothing": 0.0, "mirror_x": True, "click_cooldown": 0.0,
        "edge_margin": 0.0, "dead_zone": 0.0, "max_speed": 100000,
        "movement_region": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
    })

    # Zero every cooldown/hold so tests do not have to sleep.
    data["gestures"].update({
        "hold_duration": 0.0, "pose_cooldown": 0.0, "pinch_cooldown": 0.0,
        "swipe_cooldown": 0.0, "hold_cooldown": 0.0, "two_hand_cooldown": 0.0,
        "rotation_cooldown": 0.0, "pinch_hold_threshold": 0.2,
        "double_click_interval": 0.35, "double_pinch_threshold": 0.35,
    })
    data["scroll"]["cooldown"] = 0.0
    data["media"]["cooldown"] = 0.0
    data["brightness"]["cooldown"] = 0.0
    data["zoom"]["cooldown"] = 0.0
    data["rotation"]["cooldown"] = 0.0
    data["window"]["cooldown"] = 0.0
    data["navigation"]["cooldown"] = 0.0
    data["window_manager"]["cooldown"] = 0.0
    data["window_manager"]["action_cooldown"] = 0.0
    data["window_manager"]["move_smoothing"] = 0.0
    data["window_manager"]["min_drag_distance"] = 1
    data["window_manager"]["grab_timeout"] = 0.0

    data["macros"].update({
        "store_path": str(tmp_path / "macros.json"), "cooldown": 0.0,
        "async": False, "allow_shell": False,
    })
    data["apps"].update({"store_path": str(tmp_path / "apps.json"), "cooldown": 0.0})
    data["profiles"].update({
        "store_path": str(tmp_path / "profiles.json"), "current": "default",
        "auto_switch": False,
    })
    data["gesture_learner"].update({"store_path": str(tmp_path / "learned.json"), "enabled": True})
    data["gesture_classifier"].update({"model_path": str(tmp_path / "classifier.pkl"), "enabled": False})
    data["analytics"].update({
        "store_path": str(tmp_path / "analytics.json"), "enabled": True,
        "save_interval": 100000,
    })
    data["gesture_history"]["max_size"] = 20
    data["anomaly_detection"]["enabled"] = True
    data["gaze"]["enabled"] = False
    data["background_subtraction"]["enabled"] = False
    data["voice"]["enabled"] = False
    data["network"]["enabled"] = False
    data["obs"]["enabled"] = False
    data["onboarding"]["completed"] = True
    data["ui"]["refresh_fps"] = 30
    return data


@pytest.fixture
def mouse(config, backend):
    """A MouseController wired to the recording backend."""
    from controllers.mouse_controller import MouseController

    controller = MouseController(config, backend=backend)
    yield controller
    controller.shutdown()


def pose_hand(name: str, dx: float = 0.0, dy: float = 0.0, handedness: str = "Right",
              scale: float = 0.7, score: float = 0.99):
    """Build a synthetic :class:`~vision.hand_tracker.Hand` in a scripted pose.

    The hand is scaled about its wrist, so translations (``dx``/``dy``) keep every
    landmark inside the normalised frame - a hand clipped at the image border
    would make swipe tests impossible.
    """
    from vision.hand_tracker import make_hand
    from vision.synthetic_hands import POSES

    overrides = POSES.get(name, {})
    hand = make_hand(fingertips=dict(overrides), handedness=handedness, score=score)

    wrist = hand.landmarks[0]
    for landmark in hand.landmarks:                      # scale about the wrist
        landmark.x = wrist.x + (landmark.x - wrist.x) * scale
        landmark.y = wrist.y + (landmark.y - wrist.y) * scale

    xs = [landmark.x for landmark in hand.landmarks]     # recentre on the frame
    ys = [landmark.y for landmark in hand.landmarks]
    offset_x = 0.5 - (min(xs) + max(xs)) / 2.0
    offset_y = 0.5 - (min(ys) + max(ys)) / 2.0
    for landmark in hand.landmarks:
        landmark.x = min(max(landmark.x + offset_x + dx, 0.01), 0.99)
        landmark.y = min(max(landmark.y + offset_y + dy, 0.01), 0.99)

    hand.refresh_gesture_data()
    return hand
