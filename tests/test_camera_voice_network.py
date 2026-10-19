"""Tests for camera capture, the simulated source, and voice parsing."""
from __future__ import annotations

import time

import numpy as np
import pytest

from camera.capture import CameraCapture, SimulatedCamera


class TestSimulatedCamera:
    def test_frames_have_requested_shape(self):
        camera = SimulatedCamera(width=160, height=120, fps=30)
        assert camera.isOpened() is True
        ok, frame = camera.read()
        assert ok is True
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8

    def test_frames_change_over_time(self):
        camera = SimulatedCamera(width=80, height=60, fps=60)
        _, first = camera.read()
        time.sleep(0.05)
        _, second = camera.read()
        assert not np.array_equal(first, second)

    def test_properties_are_reported(self):
        camera = SimulatedCamera(width=320, height=240, fps=25)
        assert camera.get(3) == 320.0        # CAP_PROP_FRAME_WIDTH
        assert camera.get(4) == 240.0        # CAP_PROP_FRAME_HEIGHT
        assert camera.get(5) == 25.0         # CAP_PROP_FPS
        camera.read()
        assert camera.get(7) == 1.0          # CAP_PROP_FRAME_COUNT

    def test_release_is_safe(self):
        camera = SimulatedCamera()
        assert camera.release() is None


class TestCameraCapture:
    def test_simulate_config_uses_synthetic_source(self, config):
        config["camera"]["simulate"] = True
        camera = CameraCapture(config["camera"])
        assert camera.start() is True
        assert camera.simulated is True
        assert camera.is_running() is True
        frame, timestamp = camera.read(timeout=2.0)
        assert frame is not None
        assert timestamp > 0
        camera.stop()
        assert camera.is_running() is False

    def test_starting_twice_is_a_noop(self, config):
        config["camera"]["simulate"] = True
        camera = CameraCapture(config["camera"])
        assert camera.start() is True
        assert camera.start() is True
        camera.stop()
        camera.stop()          # idempotent

    def test_get_latest_frame_never_blocks(self, config):
        config["camera"]["simulate"] = True
        camera = CameraCapture(config["camera"])
        assert camera.get_latest_frame() == (None, 0.0)
        camera.start()
        deadline = time.time() + 3
        frame = None
        while time.time() < deadline:
            frame, _ = camera.get_latest_frame()
            if frame is not None:
                break
            time.sleep(0.02)
        assert frame is not None
        camera.stop()

    def test_measured_fps_is_positive(self, config):
        config["camera"].update({"simulate": True, "fps": 60, "simulate_fps": 60})
        camera = CameraCapture(config["camera"])
        camera.start()
        time.sleep(0.5)
        fps = camera.measured_fps()
        assert fps > 5
        assert camera.frames_read() > 5
        camera.stop()

    def test_explicit_source_is_used(self):
        class FakeSource(SimulatedCamera):
            pass

        source = FakeSource(width=64, height=48, fps=30)
        camera = CameraCapture({"simulate_on_failure": False}, source=source)
        assert camera.start() is True
        assert camera.simulated is True
        frame, _ = camera.read(timeout=1.0)
        assert frame.shape == (48, 64, 3)
        camera.stop()

    def test_missing_device_falls_back_to_simulation(self, config):
        config["camera"].update({"device_id": 99, "simulate": False, "simulate_on_failure": True})
        camera = CameraCapture(config["camera"])
        assert camera.start() is True
        assert camera.simulated is True
        assert camera.last_error
        camera.stop()

    def test_missing_device_can_be_fatal(self, config):
        config["camera"].update({"device_id": 99, "simulate": False, "simulate_on_failure": False})
        camera = CameraCapture(config["camera"])
        assert camera.start() is False
        assert camera.is_running() is False

    def test_context_manager(self, config):
        config["camera"]["simulate"] = True
        with CameraCapture(config["camera"]) as camera:
            assert camera.is_running() is True
        assert camera.is_running() is False

    def test_read_buffered_drains(self, config):
        config["camera"].update({"simulate": True, "buffer_size": 4})
        camera = CameraCapture(config["camera"])
        camera.start()
        time.sleep(0.2)
        frames = camera.read_buffered()
        assert len(frames) >= 1
        assert camera.read_buffered() == []
        camera.stop()

    def test_resolution_is_reported(self, config):
        config["camera"].update({"simulate": True, "width": 200, "height": 150})
        camera = CameraCapture(config["camera"])
        camera.start()
        assert camera.resolution == (200, 150)
        camera.stop()

    def test_failing_source_stops_cleanly(self, config):
        class BrokenSource:
            def isOpened(self):
                return True

            def read(self):
                return False, None

            def set(self, *args):
                return True

            def get(self, prop):
                return 0.0

            def release(self):
                return None

        camera = CameraCapture({"simulate_on_failure": False}, source=BrokenSource())
        assert camera.start() is True
        time.sleep(0.2)
        assert camera.is_running() is False      # the loop gave up
        assert camera.last_error
        camera.stop()


class TestVoiceParsing:
    @pytest.mark.parametrize("phrase,action", [
        ("click", "click"),
        ("left click", "click"),
        ("right click", "right_click"),
        ("double click", "double_click"),
        ("scroll down", "scroll_down"),
        ("volume up", "volume_up"),
        ("turn the volume down please", "volume_down"),
        ("mute", "mute"),
        ("next track", "next_track"),
        ("previous song", "prev_track"),
        ("alt tab", "alt_tab"),
        ("task view", "task_view"),
        ("show desktop", "show_desktop"),
        ("minimize", "minimize"),
        ("maximize", "maximize"),
        ("close window", "close"),
        ("start menu", "start_menu"),
        ("brightness up", "brightness_up"),
        ("zoom in", "zoom_up"),
        ("zoom out", "zoom_down"),
        ("take a screenshot", "screenshot"),
        ("stop tracking", "pause"),
        ("resume gestureos", "resume"),
    ])
    def test_known_phrases(self, phrase, action):
        from voice.assistant import parse_command

        command = parse_command(phrase)
        assert command is not None, phrase
        assert command["action"] == action

    def test_parameters_are_extracted(self):
        from voice.assistant import parse_command

        assert parse_command("open notepad")["params"]["app"] == "notepad"
        assert parse_command("type hello world")["params"]["text"] == "hello world"
        assert parse_command("scroll down")["params"]["direction"] == "down"
        assert parse_command("launch chrome")["params"]["app"] == "chrome"

    def test_unknown_phrases_are_ignored(self):
        from voice.assistant import parse_command

        for phrase in ("", "the weather is nice", "hmm", "   ", "what is the meaning of life"):
            assert parse_command(phrase) is None

    def test_wake_word_is_required_when_set(self):
        from voice.assistant import parse_command

        assert parse_command("volume up", wake_word="jarvis") is None
        assert parse_command("jarvis volume up", wake_word="jarvis")["action"] == "volume_up"

    def test_punctuation_is_ignored(self):
        from voice.assistant import parse_command

        assert parse_command("Right-click, please!")["action"] == "right_click"

    def test_assistant_reports_availability(self):
        from voice.assistant import VoiceAssistant

        assistant = VoiceAssistant({"enabled": True})
        assert isinstance(assistant.is_available(), bool)
        assert assistant.is_running is False
        assert assistant.get_event(timeout=0.01) is None
        assert assistant.pending() == 0
        assert assistant.last_transcript() is None
        assistant.stop()          # safe when never started


class TestNetworkServer:
    def test_defaults_and_state(self):
        from network.server import GestureNetworkServer

        server = GestureNetworkServer({"host": "127.0.0.1", "port": 8799})
        assert server.host == "127.0.0.1"
        assert server.port == 8799
        assert server.is_running is False
        assert server.client_count == 0
        assert server.broadcast({"gesture": "pinch"}) is False    # loop not running
        assert server.status()["running"] is False

    def test_main_can_set_host_and_port_directly(self):
        """main.py assigns _host/_port before starting the loop."""
        from network.server import GestureNetworkServer

        server = GestureNetworkServer()
        server._host = "0.0.0.0"
        server._port = 9001
        assert server.host == "0.0.0.0" and server.port == 9001

    def test_callback_registration(self):
        from network.server import GestureNetworkServer

        server = GestureNetworkServer()
        seen = []
        server.set_callback(lambda message: seen.append(message))
        assert server._callback is not None

    def test_start_without_websockets_is_graceful(self):
        from network.server import GestureNetworkServer

        server = GestureNetworkServer()
        if server.is_available():
            pytest.skip("websockets is installed - covered by the integration test")
        assert server.start("127.0.0.1", 8799) is False


class TestOBSIntegration:
    def test_connect_fails_gracefully(self):
        from obs.integration import OBSIntegration

        obs = OBSIntegration({"host": "127.0.0.1", "port": 4455})
        assert obs.connect() is False
        assert obs.is_connected is False
        assert obs.last_error
        assert obs.status()["connected"] is False
        obs.disconnect()

    def test_commands_are_noops_when_disconnected(self):
        from obs.integration import OBSIntegration

        obs = OBSIntegration({})
        assert obs.set_scene("cam") is False
        assert obs.toggle_recording() is False
        assert obs.toggle_mute() is False
        assert obs.scenes() == []
        assert obs.current_scene() is None

    def test_gesture_mapping(self):
        from obs.integration import OBSIntegration

        obs = OBSIntegration({"scene_gestures": {"swipe_left": "previous_scene"}})
        assert obs.handle_gesture("swipe_left") is False     # not connected
        assert obs.handle_gesture("unknown_gesture") is False
