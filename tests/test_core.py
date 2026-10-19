"""Tests for the pure-Python helpers (utils, config, history, anomaly)."""
from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------------- #
# utils
# --------------------------------------------------------------------------- #
class TestUtils:
    def test_clamp(self):
        from utils import clamp

        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(11, 0, 10) == 10
        assert clamp(5, 10, 0) == 5  # inverted bounds are normalised

    def test_lerp_and_remap(self):
        from utils import lerp, remap

        assert lerp(0, 10, 0.5) == 5
        assert lerp(0, 10, 2.0) == 10        # clamped
        assert remap(0.5, 0.0, 1.0, 0.0, 100.0) == 50
        assert remap(5, 5, 5, 0, 1) == 0     # degenerate input range

    def test_ema(self):
        from utils import ema

        assert ema(0.0, 10.0, 0.0) == 10.0
        assert ema(0.0, 10.0, 1.0) == pytest.approx(0.0, abs=0.1)
        assert ema(0.0, 10.0, 0.5) == 5.0

    def test_distance_and_angle(self):
        from utils import angle_between, distance, signed_rotation

        assert distance((0, 0), (3, 4)) == 5
        assert angle_between((0, 0), (1, 0), (0, 1)) == pytest.approx(90.0, abs=1e-6)
        assert signed_rotation(170, -170) == pytest.approx(20.0, abs=1e-6)
        assert signed_rotation(0, 90) == pytest.approx(90.0)

    def test_deep_merge_is_recursive(self):
        from utils import deep_merge

        base = {"a": {"b": 1, "c": 2}, "d": 3}
        deep_merge(base, {"a": {"c": 99, "e": 5}})
        assert base == {"a": {"b": 1, "c": 99, "e": 5}, "d": 3}

    def test_cfg_get(self):
        from utils import cfg_get

        data = {"a": {"b": {"c": 7}}}
        assert cfg_get(data, "a.b.c") == 7
        assert cfg_get(data, "a.x.y", "fallback") == "fallback"
        assert cfg_get(None, "a", 1) == 1

    def test_try_import(self):
        from utils import module_available, try_import

        assert try_import("json") is not None
        assert try_import("definitely_not_a_module_xyz") is None
        assert module_available("json") is True
        assert module_available("definitely_not_a_module_xyz") is False

    def test_screen_size_returns_positive(self):
        from utils import screen_size

        width, height = screen_size((1024, 768))
        assert width > 0 and height > 0


# --------------------------------------------------------------------------- #
# config manager
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_defaults_are_complete(self):
        from config import DEFAULT_CONFIG

        for section in ("camera", "tracking", "gestures", "cursor", "profiles",
                        "ui", "scroll", "media", "brightness", "zoom", "rotation",
                        "window", "navigation", "window_manager", "macros", "apps",
                        "voice", "network", "obs", "gaze", "analytics"):
            assert section in DEFAULT_CONFIG, f"missing config section: {section}"

    def test_dotted_get_and_set(self, tmp_path):
        from config import ConfigManager

        manager = ConfigManager(str(tmp_path / "settings.json"))
        assert manager.get("camera.width") == 1280
        manager.set("camera.width", 800)
        assert manager.get("camera.width") == 800
        assert manager.get("nope.nope", "default") == "default"

    def test_creates_file_when_missing(self, tmp_path):
        from config import ConfigManager

        path = tmp_path / "nested" / "settings.json"
        manager = ConfigManager(str(path))
        assert path.exists()
        assert json.loads(path.read_text())["camera"]["width"] == 1280
        assert manager.config_path == path

    def test_merge_preserves_defaults(self, tmp_path):
        from config import ConfigManager

        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"camera": {"width": 320}, "gestures": {"pinch_threshold": 0.09}}))
        manager = ConfigManager(str(path))
        assert manager.get("camera.width") == 320          # overridden
        assert manager.get("camera.height") == 720         # default kept
        assert manager.get("gestures.pinch_threshold") == 0.09

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path):
        from config import ConfigManager

        path = tmp_path / "settings.json"
        path.write_text("{not json at all")
        manager = ConfigManager(str(path))
        assert manager.get("camera.width") == 1280

    def test_update_section_and_reset(self, tmp_path):
        from config import ConfigManager

        manager = ConfigManager(str(tmp_path / "settings.json"))
        manager.update_section("cursor", {"sensitivity": 3.3})
        assert manager.get("cursor.sensitivity") == 3.3
        manager.reset_to_defaults()
        assert manager.get("cursor.sensitivity") == 1.5

    def test_settings_json_matches_defaults(self):
        """The shipped settings.json must not drift from DEFAULT_CONFIG."""
        from config import DEFAULT_CONFIG

        path = json.load(open("settings.json", encoding="utf-8"))
        for section, values in path.items():
            assert section in DEFAULT_CONFIG, f"settings.json has unknown section '{section}'"
            if isinstance(values, dict):
                for key in values:
                    assert key in DEFAULT_CONFIG[section], (
                        f"settings.json['{section}'] has unknown key '{key}'"
                    )

    def test_get_config_is_singleton(self, tmp_path):
        import config as config_module

        config_module._config_instance = None
        first = config_module.get_config(str(tmp_path / "s.json"))
        second = config_module.get_config()
        assert first is second
        config_module._config_instance = None


# --------------------------------------------------------------------------- #
# gesture history
# --------------------------------------------------------------------------- #
class TestGestureHistory:
    def test_push_and_queries(self):
        from utils.gesture_history import GestureHistory

        history = GestureHistory(max_size=3)
        history.push("left_click")
        history.push("swipe_left", lambda: None, {"confidence": 0.8, "gesture": "swipe_left"})
        history.push("left_click")

        assert len(history) == 3
        assert history.actions(2) == ["swipe_left", "left_click"]
        assert history.count_for("left_click") == 2
        assert history.last_action() == "left_click"
        assert history.most_common(1)[0][0] == "left_click"
        assert history.rate(5.0) > 0

    def test_ring_buffer_drops_oldest(self):
        from utils.gesture_history import GestureHistory

        history = GestureHistory(max_size=2)
        for name in ("a", "b", "c"):
            history.push(name)
        assert history.actions(10) == ["b", "c"]
        assert history.total_pushed == 3

    def test_replay_runs_callback(self):
        from utils.gesture_history import GestureHistory

        calls = []
        history = GestureHistory()
        history.push("undo", lambda: calls.append(1))
        assert history.replay(-1) is True
        assert calls == [1]

    def test_export_json(self, tmp_path):
        from utils.gesture_history import GestureHistory

        history = GestureHistory()
        history.push("left_click", None, {"confidence": 0.9})
        target = tmp_path / "history.json"
        assert history.export_json(str(target)) is True
        payload = json.loads(target.read_text())
        assert payload["entries"][0]["action"] == "left_click"


# --------------------------------------------------------------------------- #
# anomaly detector
# --------------------------------------------------------------------------- #
class TestAnomalyDetector:
    def test_normal_action_passes(self):
        from utils.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector({})
        assert detector.is_anomaly("left_click", 0.9, 0.1) is False

    def test_low_confidence_filtered(self):
        from utils.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector({"min_confidence": 0.5})
        assert detector.is_anomaly("left_click", 0.1, 0.1) is True
        assert "low_confidence" in detector.last_reason()

    def test_burst_filtered(self):
        from utils.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector({"burst_size": 3, "burst_window": 1.0, "max_rate_per_action": 999})
        results = [detector.is_anomaly("left_click", 0.9, 0.05) for _ in range(6)]
        assert results[0] is False
        assert any(results), "a burst of identical actions must be filtered"
        assert detector.stats()["filtered"] > 0

    def test_rate_limit(self):
        from utils.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector({"max_rate_per_action": 1.0, "burst_size": 99,
                                    "rate_window": 0.5, "z_score_threshold": 0,
                                    "min_confidence": 0.0})
        assert detector.is_anomaly("swipe_left", 0.9, 0.0) is False
        assert detector.is_anomaly("swipe_left", 0.9, 0.0) is True

    def test_disabled_never_filters(self):
        from utils.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector({"enabled": False})
        assert all(detector.is_anomaly("left_click", 0.0, -5) is False for _ in range(20))

    def test_reset_clears_state(self):
        from utils.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector({"min_confidence": 0.5})
        detector.is_anomaly("left_click", 0.1, 0.0)
        assert detector.stats()["filtered"] == 1
        detector.reset()
        assert detector.stats()["filtered"] == 0
