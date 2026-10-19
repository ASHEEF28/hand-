"""End-to-end tests for the GestureOS engine.

These run the *real* engine (camera -> tracker -> recogniser -> controllers)
with a simulated camera, a scripted hand source and a dry-run input backend, so
the whole pipeline is exercised without a webcam, a display or MediaPipe.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

import config as config_module
from main import GestureOSEngine, build_parser, check_installation


#: A compact version of the demo script: every gesture family, ~2 s of "hand time".
FAST_SCRIPT = [
    ("open_palm", 0.20, "idle"),
    ("point", 0.30, "circle"),            # cursor movement
    ("pinch", 0.25, "idle"),              # left click
    ("open_palm", 0.15, "idle"),
    ("pinch", 0.10, "idle"),              # double click
    ("pinch", 0.10, "idle"),
    ("open_palm", 0.15, "idle"),
    ("middle_pinch", 0.25, "idle"),       # right click
    ("open_palm", 0.15, "idle"),
    ("scroll_pose", 0.35, "scroll"),      # scroll mode + wheel
    ("open_palm", 0.30, "swipe_left"),    # swipe
    ("fist", 0.30, "drag"),               # grab window
    ("open_palm", 0.20, "idle"),          # release
    ("five_finger_pinch", 0.25, "idle"),  # start menu
]

#: Wall-clock seconds the fast script needs (it runs 4x speed).
SCRIPT_SECONDS = sum(segment[1] for segment in FAST_SCRIPT) / 4.0


@pytest.fixture
def engine(config, tmp_path, backend):
    """A fully wired engine in dry-run + simulation mode."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(config))

    config_module._config_instance = None
    from camera.capture import CameraCapture
    from vision.synthetic_hands import SyntheticHandDetector

    camera = CameraCapture(config["camera"])
    detector = SyntheticHandDetector(script=FAST_SCRIPT, loop=False, time_scale=4.0)
    created = GestureOSEngine(
        config_path=str(settings),
        simulate=True,
        dry_run=True,
        tracker=detector,
        camera=camera,
    )
    yield created
    created.stop()
    config_module._config_instance = None


class TestEngineConstruction:
    def test_all_modules_are_wired(self, engine):
        for attribute in ("camera", "tracker", "gesture_recognizer", "mouse", "scroll",
                          "window_mgr", "window_ctl", "desktop_nav", "media", "brightness",
                          "zoom", "rotation", "macros", "app_launcher", "profiles",
                          "analytics", "gesture_history", "anomaly_detector"):
            assert getattr(engine, attribute) is not None, f"{attribute} was not initialised"

    def test_optional_modules_are_off_by_default(self, engine):
        assert engine.voice is None
        assert engine.network is None
        assert engine.obs is None
        assert engine.gaze_tracker is None
        assert engine.background_subtractor is None
        assert engine.gesture_classifier is None
        assert engine.gesture_learner is not None      # enabled by default

    def test_optional_modules_enable_from_config(self, config, tmp_path, backend):
        config["gaze"]["enabled"] = True
        config["background_subtraction"]["enabled"] = True
        config["gesture_learner"]["enabled"] = False
        settings = tmp_path / "opt.json"
        settings.write_text(json.dumps(config))
        config_module._config_instance = None
        from camera.capture import CameraCapture
        from vision.synthetic_hands import SyntheticHandDetector

        created = GestureOSEngine(config_path=str(settings), simulate=True, dry_run=True,
                                  tracker=SyntheticHandDetector(), camera=CameraCapture(config["camera"]))
        try:
            assert created.gaze_tracker is not None
            assert created.background_subtractor is not None
            assert created.gesture_learner is None
        finally:
            created.stop()
            config_module._config_instance = None

    def test_start_and_stop(self, engine):
        assert engine.running is False
        assert engine.start() is True
        assert engine.running is True
        assert engine.camera.is_running() is True
        engine.stop()
        assert engine.running is False
        assert engine.camera.is_running() is False

    def test_pause_resume_toggle(self, engine):
        assert engine.paused is False
        engine.pause()
        assert engine.paused is True
        engine.resume()
        assert engine.paused is False
        engine.toggle_pause()
        assert engine.paused is True
        engine.toggle_pause()
        assert engine.paused is False

    def test_snapshot_reports_state(self, engine):
        engine.start()
        snapshot = engine.snapshot()
        for key in ("running", "paused", "mode", "gesture", "action", "profile",
                    "fps", "latency_ms", "hands", "camera", "tracker_backend"):
            assert key in snapshot
        assert snapshot["tracker_backend"] == "synthetic"
        assert snapshot["camera"]["simulated"] is True


class TestPenModeFallback:
    """``--mode pen`` is a roadmap feature: it must warn, not hang."""

    def test_engine_falls_back_to_hand_mode(self, config, tmp_path, backend):
        settings = tmp_path / "pen.json"
        settings.write_text(json.dumps(config))
        config_module._config_instance = None
        from camera.capture import CameraCapture
        from vision.synthetic_hands import SyntheticHandDetector

        created = GestureOSEngine(config_path=str(settings), operation_mode="pen",
                                  simulate=True, dry_run=True,
                                  tracker=SyntheticHandDetector(),
                                  camera=CameraCapture(config["camera"]))
        try:
            assert created.operation_mode == "hand"
        finally:
            created.stop()
            config_module._config_instance = None

    def test_cli_pen_mode_completes(self, tmp_path, backend, capsys):
        from main import main as entry

        config_module._config_instance = None
        try:
            code = entry(["--demo", "--frames", "5", "--mode", "pen",
                          "--config", str(tmp_path / "pen-cli.json")])
        finally:
            config_module._config_instance = None
        assert code == 0
        assert "frames processed : 5" in capsys.readouterr().out

    def test_unknown_mode_is_normalised(self, config, tmp_path, backend):
        settings = tmp_path / "odd.json"
        settings.write_text(json.dumps(config))
        config_module._config_instance = None
        from camera.capture import CameraCapture
        from vision.synthetic_hands import SyntheticHandDetector

        created = GestureOSEngine(config_path=str(settings), operation_mode="  HAND  ",
                                  simulate=True, dry_run=True,
                                  tracker=SyntheticHandDetector(),
                                  camera=CameraCapture(config["camera"]))
        try:
            assert created.operation_mode == "hand"
        finally:
            created.stop()
            config_module._config_instance = None


class TestFramePipeline:
    def test_process_frame_returns_annotated_image(self, engine):
        engine.start()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        annotated = engine.process_frame(frame)
        assert annotated is not None
        assert annotated.shape == frame.shape
        assert engine._total_frames == 1

    def test_none_frame_is_passed_through(self, engine):
        assert engine.process_frame(None) is None

    def test_overlay_reports_fps_and_mode(self, engine):
        engine.start()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        engine.process_frame(frame)
        annotated = engine._add_overlay(frame.copy(), engine._last_processed_hands)
        assert annotated.shape == frame.shape

    @staticmethod
    def _run_script(engine, frame=None):
        """Feed frames until the scripted hand has performed its whole routine."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8) if frame is None else frame
        deadline = time.time() + SCRIPT_SECONDS + 1.0
        frames = 0
        while time.time() < deadline:
            engine.process_frame(frame)
            frames += 1
            time.sleep(0.005)
        return frames

    def test_pipeline_runs_a_full_demo_script(self, engine, backend):
        assert engine.start() is True
        frames = self._run_script(engine)

        snapshot = engine.snapshot()
        assert engine._total_frames == frames
        assert engine._hands_detected_frames > 0
        assert backend.count("move_to") > 0, "cursor was never moved"
        assert snapshot["fps"] >= 0
        assert snapshot["latency_ms"] >= 0
        assert engine.tracker.finished is True      # the whole script ran

    def test_actions_were_dispatched(self, engine, backend):
        engine.start()
        self._run_script(engine)

        kinds = {call["kind"] for call in backend.calls()}
        assert "move_to" in kinds
        # The script pinches, scrolls and grabs, so real input must have fired.
        assert kinds & {"click", "mouse_down", "scroll", "press", "hotkey"}, (
            f"no gesture actions were dispatched, only: {kinds}"
        )
        assert engine.analytics.summary()["session"]["gestures"] > 0

    def test_gesture_history_is_populated(self, engine):
        engine.start()
        self._run_script(engine)
        assert len(engine.gesture_history) > 0
        assert engine.gesture_history.total_pushed > 0

    def test_paused_engine_does_not_act(self, engine, backend):
        engine.start()
        engine.pause()
        self._run_script(engine)
        assert backend.count("click") == 0
        assert backend.count("mouse_down") == 0
        assert backend.count("hotkey") == 0

    def test_headless_run_stops_after_frames(self, engine):
        from main import run_frames

        assert run_frames(engine, 25) == 0
        assert engine._total_frames >= 25
        assert engine.running is False


class TestActionDispatch:
    """Every action the recogniser can emit must execute without raising."""

    ACTIONS = [
        "left_click", "right_click", "double_click", "hold_pinch", "scroll_mode",
        "swipe_left", "swipe_right", "swipe_up", "swipe_down",
        "show_desktop", "start_menu", "task_view",
        "virtual_desktop_left", "virtual_desktop_right",
        "grab_window", "release_window", "close_window", "minimize_window",
        "maximize_window", "snap_left", "snap_right",
        "volume_up", "volume_down", "mute_toggle",
        "media_next", "media_prev", "media_play_pause",
        "rotate_cw", "rotate_ccw", "zoom_in", "zoom_out",
        "brightness_up", "brightness_down",
        "open_explorer", "open_search", "open_settings", "screenshot",
        "ar_menu_toggle", "ar_menu_select",
        "copy", "paste", "unknown_action_name",
    ]

    @pytest.mark.parametrize("action", ACTIONS)
    def test_action_executes(self, engine, backend, action):
        from gestures.recognizer import GestureEvent, GestureType
        from tests.conftest import pose_hand

        engine.start()
        hand = pose_hand("open_palm")
        hand.gesture_data["pinch_distance"] = 0.08
        event = GestureEvent(GestureType.PINCH, hand, 0.9, time.time(), {"delta": 0.05})
        engine._do_action(action, event)          # must not raise
        assert engine.current_action == action

    def test_action_without_a_hand_is_safe(self, engine):
        from gestures.recognizer import GestureEvent, GestureType

        engine.start()
        event = GestureEvent(GestureType.NONE, None, 1.0, time.time())
        for action in ("brightness_up", "brightness_down", "grab_window", "left_click"):
            engine._do_action(action, event)      # hand=None must not raise

    def test_empty_action_is_ignored(self, engine):
        from gestures.recognizer import GestureEvent, GestureType

        engine._do_action("", GestureEvent(GestureType.NONE, None, 1.0, time.time()))
        assert engine.current_action == "none"

    def test_left_click_reaches_the_backend(self, engine, backend):
        from gestures.recognizer import GestureEvent, GestureType

        engine.start()
        engine._do_action("left_click", GestureEvent(GestureType.PINCH, None, 0.9, time.time()))
        time.sleep(0.6)                            # click is coalesced for double-click detection
        assert backend.count("click") >= 1

    def test_drag_toggles(self, engine, backend):
        from gestures.recognizer import GestureEvent, GestureType

        engine.start()
        event = GestureEvent(GestureType.HOLD_PINCH, None, 0.9, time.time())
        engine._do_action("hold_pinch", event)
        assert engine.mouse.is_dragging() is True
        engine._do_action("hold_pinch", event)
        assert engine.mouse.is_dragging() is False

    def test_scroll_mode_toggles(self, engine):
        from gestures.recognizer import GestureEvent, GestureType

        event = GestureEvent(GestureType.SCROLL_MODE, None, 0.9, time.time())
        assert engine.current_mode == "cursor"
        engine._do_action("scroll_mode", event)
        assert engine.current_mode == "scroll"
        engine._do_action("scroll_mode", event)
        assert engine.current_mode == "cursor"

    def test_ar_menu_cycle(self, engine):
        from gestures.recognizer import GestureEvent, GestureType

        event = GestureEvent(GestureType.NONE, None, 1.0, time.time())
        assert engine._ar_menu_active is False
        engine._do_action("ar_menu_toggle", event)
        assert engine._ar_menu_active is True
        assert engine._ar_menu_items
        engine._do_action("ar_menu_select", event)
        assert engine._ar_menu_active is False

    def test_ar_action_mapping(self, engine):
        engine._execute_ar_action("Left Click")
        assert engine.current_action == "left_click"
        engine._execute_ar_action("Unknown Item")
        assert engine.current_action == "left_click"   # unchanged

    def test_anomaly_detector_filters_spam(self, engine, backend):
        from gestures.recognizer import GestureEvent, GestureType

        engine.start()
        event = GestureEvent(GestureType.PINCH, None, 0.9, time.time())
        for _ in range(25):
            engine._do_action("left_click", event)
        stats = engine.anomaly_detector.stats()
        assert stats["filtered"] > 0
        assert stats["seen"] == 25

    def test_hand_to_screen_uses_mouse_bounds(self, engine):
        x, y = engine._hand_to_screen(0.5, 0.5)
        width = engine.mouse.screen_bounds[2]
        height = engine.mouse.screen_bounds[3]
        assert 0 <= x <= width and 0 <= y <= height

    def test_cooldown_helper(self, engine):
        assert engine._check_cooldown("unit_test_key", 1.0) is True
        assert engine._check_cooldown("unit_test_key", 1.0) is False
        assert engine._check_cooldown("other_key", 1.0) is True

    def test_action_cooldowns_follow_the_config(self, engine):
        # The fast fixture zeroes the gesture cooldowns, so click/swipe are 0...
        assert engine._action_cooldown("left_click") == 0.0
        assert engine._action_cooldown("swipe_left") == 0.0
        # ...but the hard-coded floors still stop a gesture machine-gunning the OS.
        assert engine._action_cooldown("double_click") > 0
        assert engine._action_cooldown("grab_window") > 0
        assert engine._action_cooldown("ar_menu_toggle") > 0
        assert engine._action_cooldown("zoom_in") > 0
        assert engine._action_cooldown("something_else") > 0

    def test_action_cooldowns_with_shipped_defaults(self):
        """With settings.json as shipped, every action is rate limited."""
        import copy

        from config import DEFAULT_CONFIG
        from main import GestureOSEngine

        class _ConfigStub:
            def get_section(self, name):
                return copy.deepcopy(DEFAULT_CONFIG).get(name, {})

        class _Stub:
            config = _ConfigStub()

        cooldown = GestureOSEngine._action_cooldown
        for action in ("left_click", "right_click", "double_click", "hold_pinch",
                       "scroll_mode", "swipe_left", "media_next", "volume_up",
                       "brightness_up", "zoom_in", "rotate_cw", "grab_window",
                       "ar_menu_toggle", "unknown_action"):
            assert cooldown(_Stub(), action) > 0, action


class TestProfilesAndContext:
    def test_profile_switch_applies_to_controllers(self, engine):
        engine.start()
        assert engine.profiles.set_active("design") is True
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        engine.process_frame(frame)
        assert engine.active_profile_name == "design"
        assert engine.mouse.sensitivity == pytest.approx(0.9)

    def test_context_update_is_throttled_and_safe(self, engine):
        engine.start()
        engine._update_context()
        first = engine._last_context_update
        engine._update_context()
        assert engine._last_context_update == first      # throttled

    def test_context_profiles_reference_real_profiles(self, engine):
        for profile in engine.CONTEXT_PROFILES.values():
            assert engine.profiles.has_profile(profile), f"unknown profile '{profile}'"


class TestVoiceAndNetwork:
    @pytest.mark.parametrize("command,expected", [
        ({"action": "click"}, "left_click"),
        ({"action": "right_click"}, "right_click"),
        ({"action": "double_click"}, "double_click"),
        ({"action": "volume_up"}, "volume_up"),
        ({"action": "mute"}, "mute_toggle"),
        ({"action": "alt_tab"}, "swipe_left"),
        ({"action": "close"}, "close_window"),
        ({"action": "task_view"}, "task_view"),
        ({"action": "show_desktop"}, "show_desktop"),
        ({"action": "start_menu"}, "start_menu"),
        ({"action": "next_track"}, "media_next"),
        ({"action": "zoom_in"}, "zoom_in"),
    ])
    def test_voice_commands_route_to_actions(self, engine, command, expected):
        engine.start()
        engine._on_voice_command(command)
        assert engine.current_action == expected
        assert engine._last_voice_command == command

    def test_voice_pause_and_resume(self, engine):
        engine._on_voice_command({"action": "pause"})
        assert engine.paused is True
        engine._on_voice_command({"action": "resume"})
        assert engine.paused is False

    def test_voice_scroll_and_type(self, engine, backend):
        engine.start()
        engine._on_voice_command({"action": "scroll", "params": {"direction": "down", "clicks": 3}})
        assert backend.last("scroll")["clicks"] == -3
        engine._on_voice_command({"action": "type", "params": {"text": "hello"}})
        assert backend.last("write")["text"] == "hello"

    def test_voice_open_launches_app(self, engine):
        engine.start()
        engine._on_voice_command({"action": "open", "params": {"app": "notepad"}})
        assert engine.app_launcher.launch_log

    def test_unknown_voice_command_is_ignored(self, engine):
        engine.start()
        engine._on_voice_command({"action": "make_coffee"})
        assert engine.current_action == "none"

    def test_network_commands(self, engine, backend):
        engine.start()
        assert engine._on_network_command({"command": "click"}) is True
        assert engine._on_network_command({"command": "volume_up"}) is True
        assert engine._on_network_command({"command": "scroll", "params": {"clicks": 2}}) is True
        assert engine._on_network_command({"command": "key_press", "params": {"keys": ["a"]}}) is True
        assert engine._on_network_command({"command": "key_press", "params": {"keys": ["ctrl", "c"]}}) is True
        assert engine._on_network_command({"command": "type_text", "params": {"text": "hi"}}) is True
        assert engine._on_network_command({"command": "set_mode", "params": {"mode": "scroll"}}) is True
        assert engine.current_mode == "scroll"
        assert engine._on_network_command({"command": "pause"}) is True
        assert engine.paused is True
        assert engine._on_network_command({"command": "resume"}) is True
        assert engine._on_network_command({"command": "set_profile", "params": {"profile": "gaming"}}) is True
        assert engine.profiles.get_active() == "gaming"

    def test_network_command_validation(self, engine):
        assert engine._on_network_command({}) is False
        assert engine._on_network_command({"command": "set_mode", "params": {"mode": "turbo"}}) is False
        assert engine._on_network_command({"command": "set_profile", "params": {"profile": "nope"}}) is False
        assert engine._on_network_command({"command": "run_macro", "params": {"macro": "nope"}}) is False

    def test_network_status_command_without_server(self, engine):
        assert engine._on_network_command({"command": "status"}) is False


class TestCLI:
    def test_parser_flags(self):
        parser = build_parser()
        args = parser.parse_args(["--demo", "--frames", "10", "--profile", "coding"])
        assert args.demo is True
        assert args.frames == 10
        assert args.profile == "coding"

    def test_demo_implies_headless(self):
        parser = build_parser()
        args = parser.parse_args(["--demo"])
        assert args.demo is True and args.headless is False
        # main() derives headless from demo/frames - check that logic here.
        headless = args.headless or args.demo or args.frames > 0
        assert headless is True

    def test_mode_choices(self):
        parser = build_parser()
        assert parser.parse_args(["--mode", "pen"]).mode == "pen"
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "telepathy"])

    def test_check_installation_runs(self, capsys):
        code = check_installation()
        captured = capsys.readouterr().out
        assert "GestureOS installation check" in captured
        assert code in (0, 1)

    def test_print_gesture_table(self, capsys):
        pytest.importorskip("PySide6")
        from main import print_gesture_table

        print_gesture_table()
        captured = capsys.readouterr().out
        assert "pinch" in captured and "left_click" in captured

    def test_version_flag(self, capsys):
        from main import main as entry

        assert entry(["--version"]) == 0
        assert "GestureOS" in capsys.readouterr().out

    def test_list_gestures_flag(self, capsys, tmp_path):
        from main import main as entry

        assert entry(["--list-gestures"]) == 0
        assert "swipe_left" in capsys.readouterr().out

    def test_frames_flag_runs_headless(self, capsys, tmp_path, backend):
        from main import main as entry

        code = entry(["--demo", "--frames", "12", "--config", str(tmp_path / "cli.json")])
        captured = capsys.readouterr().out
        assert code == 0
        assert "Run summary" in captured
        assert "frames processed : 12" in captured
