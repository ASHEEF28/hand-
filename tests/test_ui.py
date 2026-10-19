"""UI tests.

Skipped automatically when PySide6 or a usable Qt platform plugin is missing
(the suite still runs on a plain Windows/Linux dev box without a display by
using QT_QPA_PLATFORM=offscreen).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def engine_stub(config, backend):
    """A minimal engine-like object with the attributes the UI touches."""
    from analytics.tracker import GestureAnalytics
    from controllers.mouse_controller import MouseController
    from controllers.scroll_controller import ScrollController
    from gestures.recognizer import GestureRecognizer
    from macros.engine import MacroEngine
    from profiles.manager import ProfileManager

    class ConfigManagerStub:
        config_path = "settings.json"

        def __init__(self, data):
            self.config = data

        def set(self, key, value):
            node = self.config
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value

        def get(self, key, default=None):
            node = self.config
            for part in str(key).split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return node

        def get_section(self, name):
            return self.config.get(name, {})

        def update_section(self, section, values):
            self.config.setdefault(section, {}).update(values)

        def save(self):
            return True

    class EngineStub:
        processing_time = 8.5
        running = True
        paused = False
        current_mode = "cursor"
        _last_processed_hands = []

        def __init__(self):
            self.config = ConfigManagerStub(json.loads(json.dumps(config)))
            self.mouse = MouseController(config, backend=backend)
            self.scroll = ScrollController(config, backend=backend)
            self.gesture_recognizer = GestureRecognizer(config)
            self.profiles = ProfileManager(config)
            self.macros = MacroEngine(config, backend=backend)
            self.app_launcher = self.macros.launcher
            self.analytics = GestureAnalytics(config)

        def toggle_pause(self):
            self.paused = not self.paused

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    return EngineStub()


@pytest.fixture
def window(qapp, engine_stub):
    from ui.main_window import MainWindow

    created = MainWindow(engine_stub.config.config, engine_stub)
    created.show()
    yield created
    created.close()


class TestMainWindow:
    def test_window_builds(self, window):
        assert window.windowTitle().startswith("GestureOS")
        assert window.tabs.count() == 6
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
            "Gestures", "Calibration", "Profiles", "Macros", "Analytics", "Settings",
        ]

    def test_update_camera_feed(self, window):
        frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        window.update_camera_feed(frame, 58.0, "cursor", "center")
        assert window.camera_feed.frames == 1
        assert window.control_panel.labels["fps"].text() == "58"
        assert "8.5" in window.control_panel.labels["latency"].text()

    def test_update_camera_feed_with_none(self, window):
        window.update_camera_feed(None, 0.0, "cursor", "")
        assert window.camera_feed.frames == 0

    def test_control_panel_status(self, window):
        window.control_panel.update_status(True, True, "pinch", "left_click", "coding", 0.87)
        labels = window.control_panel.labels
        assert labels["gesture"].text() == "pinch"
        assert labels["action"].text() == "left_click"
        assert labels["profile"].text() == "coding"
        assert labels["confidence"].text() == "87%"
        assert window.control_panel.confidence_bar.value() == 87

    def test_update_hand_visualizer(self, window):
        from tests.conftest import pose_hand

        hand = pose_hand("open_palm")
        window.engine._last_processed_hands = [hand]
        window.update_hand_visualizer(hand.landmarks, hand.handedness)
        assert window.hand_visualizer._landmarks

    def test_add_log_and_throttling(self, window):
        window.add_log("first")
        window.add_log("second")
        assert window.log_panel.line_count >= 2

        # The same key only logs once per interval - this is what keeps the log
        # readable while a gesture repeats 30x per second.
        window.add_log_throttled("key", "alpha")
        window.add_log_throttled("key", "beta")
        text = window.log_panel.toPlainText()
        assert "alpha" in text
        assert "beta" not in text

        window.add_log_throttled("other_key", "gamma")
        assert "gamma" in window.log_panel.toPlainText()

    def test_notify_gesture_deduplicates(self, window):
        window.notify_gesture("pinch", "left_click", 0.9)
        before = window.log_panel.line_count
        window.notify_gesture("pinch", "left_click", 0.9)
        assert window.log_panel.line_count == before
        window.notify_gesture("swipe_left", "swipe_left", 0.9)
        assert window.log_panel.line_count == before + 1

    def test_pause_button_toggles_engine(self, window):
        window._on_pause()
        assert window.engine.paused is True
        assert window.control_panel.pause_button.text() == "Resume"
        window._on_pause()
        assert window.engine.paused is False

    def test_profile_switch_updates_controllers(self, window):
        window._on_profile("design")
        assert window.engine.profiles.get_active() == "design"
        assert window.engine.mouse.sensitivity == pytest.approx(0.9)

    def test_mode_switch(self, window):
        window._on_mode("scroll")
        assert window.engine.current_mode == "scroll"

    def test_calibration_apply(self, window):
        window._apply_calibration({
            "cursor.sensitivity": 2.5,
            "cursor.smoothing": 0.1,
            "gestures.pinch_ratio_threshold": 0.42,
            "scroll.sensitivity": 1.7,
        })
        assert window.engine.mouse.sensitivity == pytest.approx(2.5)
        assert window.engine.scroll.config["sensitivity"] == pytest.approx(1.7)
        assert window.engine.gesture_recognizer.config["pinch_ratio_threshold"] == pytest.approx(0.42)
        assert window.engine.config.get("cursor.sensitivity") == pytest.approx(2.5)
        assert window.engine.config.get("calibration.calibrated") is True

    def test_calibration_apply_empty_is_noop(self, window):
        window._apply_calibration({})

    def test_macro_tab_lists_and_runs(self, window):
        assert window.macros_tab.macro_list.count() > 0
        assert window.macros_tab.app_combo.count() > 0
        window.macros_tab.macro_list.setCurrentRow(0)
        name = window.macros_tab.name_edit.text()
        assert name
        window._run_macro(name)
        assert window.engine.macros.execution_log

    def test_macro_editor_saves_and_deletes(self, window):
        tab = window.macros_tab
        tab.name_edit.setText("ui_macro")
        tab.steps_edit.setPlainText(json.dumps([{"type": "key", "key": "f9"}]))
        tab._save_macro()
        assert "ui_macro" in window.engine.macros.list_macros()
        tab.name_edit.setText("ui_macro")
        tab._delete_macro()
        assert "ui_macro" not in window.engine.macros.list_macros()

    def test_macro_editor_rejects_bad_json(self, window):
        tab = window.macros_tab
        tab.name_edit.setText("bad")
        tab.steps_edit.setPlainText("{not json")
        tab._save_macro()
        assert "bad" not in window.engine.macros.list_macros()

    def test_profiles_tab_lists_profiles(self, window):
        window.profiles_tab.reload()
        names = [window.profiles_tab.list.item(i).text()
                 for i in range(window.profiles_tab.list.count())]
        assert "default" in names

    def test_gesture_tab_is_populated(self, window):
        assert window.gesture_tab.table.rowCount() >= 20
        assert window.gesture_tab.table.item(0, 1).text()

    def test_analytics_tab_refresh(self, window):
        window.engine.analytics.record_gesture("pinch", "left_click", 0.9, 12, True)
        window.analytics_tab.refresh()
        assert "Session" in window.analytics_tab.summary.toPlainText()
        assert window.analytics_tab.table.rowCount() >= 1

    def test_settings_tab_roundtrip(self, window):
        window.settings_tab.reload()
        payload = json.loads(window.settings_tab.editor.toPlainText())
        assert payload["camera"]["width"] == 320
        payload["camera"]["width"] = 256
        window.settings_tab.editor.setPlainText(json.dumps(payload))
        window.settings_tab.apply()
        assert window.engine.config.config["camera"]["width"] == 256

    def test_settings_tab_rejects_invalid_json(self, window):
        window.settings_tab.editor.setPlainText("{oops")
        window.settings_tab.apply()
        assert "Invalid JSON" in window.settings_tab.status_label.text()

    def test_reload_settings_does_not_duplicate_tabs(self, window):
        before = window.tabs.count()
        window.reload_settings()
        window.reload_settings()
        assert window.tabs.count() == before

    def test_update_timer_exists(self, window, qapp):
        assert window.update_timer.interval() > 0
        window.update_timer.start()
        qapp.processEvents()
        window.update_timer.stop()

    def test_fullscreen_toggle(self, window):
        window._toggle_fullscreen()
        assert window.isFullScreen()
        window._toggle_fullscreen()
        assert not window.isFullScreen()

    def test_close_stops_engine(self, window):
        window.close()
        assert window.engine.running is False


class TestWidgets:
    def test_frame_to_pixmap(self):
        from ui.widgets import frame_to_pixmap

        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        pixmap = frame_to_pixmap(frame)
        assert pixmap is not None
        assert pixmap.width() == 80 and pixmap.height() == 60
        assert frame_to_pixmap(None) is None

    def test_camera_feed_message(self, qapp):
        from ui.widgets import CameraFeedWidget

        widget = CameraFeedWidget()
        widget.set_message("waiting")
        assert widget.text() == "waiting"
        assert widget.frames == 0

    def test_hand_visualizer_handles_empty(self, qapp):
        from ui.widgets import HandVisualizer

        widget = HandVisualizer(size=120)
        widget.clear()
        widget.grab()                 # forces a paint event
        assert widget._landmarks == []

    def test_control_panel_set_profiles(self, qapp):
        from ui.widgets import ControlPanel

        panel = ControlPanel(profiles=["default", "gaming"])
        panel.set_profiles(["default", "gaming", "design"], "design")
        assert panel.profile_combo.currentText() == "design"
        panel.set_paused(True)
        assert panel.pause_button.text() == "Resume"

    def test_log_panel_ring_buffer(self, qapp):
        from ui.widgets import LogPanel

        panel = LogPanel(max_lines=3)
        for index in range(6):
            panel.add_log(f"line {index}")
        assert panel.line_count == 6
        assert len(panel.toPlainText().splitlines()) <= 3


class TestAROverlayAndOnboarding:
    def test_render_menu(self):
        from ui.ar_overlay import AROverlay

        overlay = AROverlay({"enabled": True, "menu_radius": 60})
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        items = ["Left Click", "Right Click", "Scroll", "Volume", "Close"]
        annotated = overlay.render_menu(frame, (0.5, 0.5), items, 2)
        assert annotated.shape == frame.shape
        assert overlay.last_items == items
        assert overlay.last_selected == 2
        assert overlay.render_menu(frame, (0.5, 0.5), [], 0) is frame

    def test_selection_from_finger(self):
        from ui.ar_overlay import AROverlay

        overlay = AROverlay({})
        assert overlay.selection_from_finger((0.5, 0.5), (0.5, 0.2), 4) == 0   # straight up
        assert overlay.selection_from_finger((0.5, 0.5), (0.8, 0.5), 4) == 1   # right
        assert overlay.selection_from_finger((0.5, 0.5), (0.5, 0.5), 4) == 0   # no direction

    def test_render_hud_and_status(self):
        from ui.ar_overlay import AROverlay

        overlay = AROverlay({"show_hud": True})
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        hud = overlay.render_hud(frame, {"fps": 60, "mode": "cursor"})
        assert hud.shape == frame.shape
        banner = overlay.render_status(frame, "Camera lost")
        assert banner.shape == frame.shape

    def test_onboarding_skips_without_display(self, monkeypatch):
        import ui.onboarding as onboarding

        monkeypatch.setattr(onboarding, "_qt_available", lambda: False)
        assert onboarding.exec_onboarding(None, None) is False
        assert onboarding.exec_mode_selection() == "hand"

    def test_mode_selection_default(self, monkeypatch):
        import ui.onboarding as onboarding

        monkeypatch.setattr(onboarding, "_qt_available", lambda: False)
        assert onboarding.exec_mode_selection(default="pen") == "pen"
