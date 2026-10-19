"""GestureOS main window (PySide6).

Composition root for the UI: live camera feed, hand skeleton, control panel,
tabs (gestures / calibration / profiles / macros / analytics / settings) and a
rolling log.

``main.py`` drives it with four calls per tick:

    window.update_camera_feed(frame, fps, mode, gaze)
    window.update_hand_visualizer(landmarks, handedness)
    window.control_panel.update_status(...)
    window.add_log(message)

and connects ``window.update_timer`` to its own refresh callback.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.widgets import (
    AnalyticsTab,
    CalibrationTab,
    CameraFeedWidget,
    ControlPanel,
    GestureReferenceTab,
    HandVisualizer,
    LogPanel,
    MacrosTab,
    ProfilesTab,
    SettingsTab,
)

logger = logging.getLogger(__name__)

DARK_STYLE = """
QMainWindow, QWidget { background: #17171d; color: #e8e8ef; font-size: 13px; }
QGroupBox { border: 1px solid #26262f; border-radius: 8px; margin-top: 12px; padding: 10px;
            font-weight: 600; color: #c9c9d4; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background: #23232c; border: 1px solid #33333f; border-radius: 6px;
              padding: 6px 14px; color: #e8e8ef; }
QPushButton:hover { background: #2c2c38; border-color: #00d4ff; }
QPushButton:pressed { background: #1b1b22; }
QPushButton:disabled { color: #6b6b78; border-color: #26262f; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: #101014; border: 1px solid #2b2b35;
              border-radius: 6px; padding: 4px 8px; color: #e8e8ef; }
QComboBox QAbstractItemView { background: #17171d; color: #e8e8ef; selection-background-color: #2c2c38; }
QListWidget, QTableWidget { background: #101014; border: 1px solid #26262f; border-radius: 8px;
              alternate-background-color: #14141a; }
QHeaderView::section { background: #1b1b22; color: #8a8a99; border: 0; padding: 6px; }
QTabWidget::pane { border: 1px solid #26262f; border-radius: 8px; top: -1px; }
QTabBar::tab { background: #1b1b22; color: #8a8a99; padding: 8px 16px;
               border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #23232c; color: #00d4ff; }
QProgressBar { background: #101014; border: 0; border-radius: 3px; }
QProgressBar::chunk { background: #00d4ff; border-radius: 3px; }
QStatusBar { background: #101014; color: #8a8a99; }
QSplitter::handle { background: #23232c; }
QMenuBar { background: #17171d; color: #c9c9d4; }
QMenuBar::item:selected { background: #23232c; }
QMenu { background: #1b1b22; color: #e8e8ef; border: 1px solid #2b2b35; }
QMenu::item:selected { background: #2c2c38; }
QLabel#hint { color: #8a8a99; }
"""


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, engine: Any = None) -> None:
        super().__init__()
        self.config: Dict[str, Any] = dict(config or {})
        self.engine = engine
        self.config_manager = getattr(engine, "config", None)

        ui_cfg = self.config.get("ui", {}) or {}
        self.setWindowTitle("GestureOS - AI Gesture Control")
        self.resize(int(ui_cfg.get("window_width", 1280)), int(ui_cfg.get("window_height", 720)))
        self.setStyleSheet(DARK_STYLE)

        self._started_at = time.time()
        self._log_throttle: Dict[str, float] = {}
        self._last_gesture = ""
        self._last_action = ""

        self._build_ui(ui_cfg)
        self._build_menu()
        self._wire_engine()

        # main.py connects its refresh callback to this timer.
        interval = int(1000 / max(float(ui_cfg.get("refresh_fps", 30)), 1.0))
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(interval)
        self.update_timer.setTimerType(Qt.PreciseTimer)

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(2000)
        self._stats_timer.timeout.connect(self._refresh_analytics)
        self._stats_timer.start()

        profiles = self._profile_names()
        self.control_panel.set_profiles(profiles, self._active_profile())
        self.add_log(f"GestureOS UI ready ({len(profiles)} profiles loaded)")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_ui(self, ui_cfg: Dict[str, Any]) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)

        # --- left column: feed + skeleton + log -------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_feed = CameraFeedWidget()
        feed_width = int(ui_cfg.get("camera_feed_width", 640))
        feed_height = int(ui_cfg.get("camera_feed_height", 360))
        self.camera_feed.setMinimumSize(feed_width, feed_height)
        left_layout.addWidget(self.camera_feed, 3)

        side_row = QHBoxLayout()
        self.hand_visualizer = HandVisualizer(size=180)
        side_row.addWidget(self.hand_visualizer)

        hints = QWidget()
        hints_layout = QVBoxLayout(hints)
        hints_layout.setContentsMargins(8, 0, 0, 0)
        hints_title = QLabel("Quick reference")
        hints_title.setStyleSheet("font-weight: 600;")
        hints_layout.addWidget(hints_title)
        for line in (
            "Index finger - move cursor",
            "Thumb + index pinch - left click",
            "Thumb + middle pinch - right click",
            "Hold pinch - drag & drop",
            "Index + middle up - scroll mode",
            "Open-hand swipe - Alt+Tab / Task View",
            "Closed fist - grab window",
        ):
            label = QLabel("- " + line)
            label.setObjectName("hint")
            hints_layout.addWidget(label)
        hints_layout.addStretch(1)
        side_row.addWidget(hints, 1)
        left_layout.addLayout(side_row)

        self.log_panel = LogPanel(max_lines=int(ui_cfg.get("log_lines", 500)))
        self.log_panel.setMaximumHeight(150)
        left_layout.addWidget(self.log_panel)
        splitter.addWidget(left)

        # --- right column: controls + tabs ------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.control_panel = ControlPanel(profiles=self._profile_names())
        right_layout.addWidget(self.control_panel)

        self.tabs = QTabWidget()
        self.gesture_tab = GestureReferenceTab(engine=self.engine)
        self.calibration_tab = CalibrationTab(config=self.config)
        self.profiles_tab = ProfilesTab(engine=self.engine)
        self.macros_tab = MacrosTab(engine=self.engine)
        self.analytics_tab = AnalyticsTab(engine=self.engine)
        self.settings_tab = SettingsTab(config_manager=self.config_manager)

        self.tabs.addTab(self.gesture_tab, "Gestures")
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.tabs.addTab(self.profiles_tab, "Profiles")
        self.tabs.addTab(self.macros_tab, "Macros")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.settings_tab, "Settings")
        right_layout.addWidget(self.tabs, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 480])
        root.addWidget(splitter)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.status_fps = QLabel("FPS: -")
        self.status_latency = QLabel("Latency: -")
        self.status_profile = QLabel("Profile: -")
        self.status_backend = QLabel("")
        for widget in (self.status_fps, self.status_latency, self.status_profile, self.status_backend):
            self.statusBar().addPermanentWidget(widget)

    def _build_menu(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self.act_reload = QAction("&Reload settings", self)
        self.act_reload.setShortcut(QKeySequence("Ctrl+R"))
        self.act_reload.triggered.connect(self.reload_settings)
        self.act_quit = QAction("&Quit", self)
        self.act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        self.act_quit.triggered.connect(self.close)
        file_menu.addAction(self.act_reload)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        control_menu = bar.addMenu("&Control")
        self.act_start = QAction("&Start tracking", self)
        self.act_start.setShortcut(QKeySequence("F5"))
        self.act_start.triggered.connect(self._on_start)
        self.act_pause = QAction("&Pause / Resume", self)
        self.act_pause.setShortcut(QKeySequence("Space"))
        self.act_pause.triggered.connect(self._on_pause)
        self.act_stop = QAction("S&top tracking", self)
        self.act_stop.setShortcut(QKeySequence("Shift+F5"))
        self.act_stop.triggered.connect(self._on_stop)
        for action in (self.act_start, self.act_pause, self.act_stop):
            control_menu.addAction(action)

        view_menu = bar.addMenu("&View")
        for index in range(self.tabs.count()):
            action = QAction(self.tabs.tabText(index), self)
            action.triggered.connect(lambda _checked=False, i=index: self.tabs.setCurrentIndex(i))
            view_menu.addAction(action)
        view_menu.addSeparator()
        self.act_fullscreen = QAction("Toggle &fullscreen", self)
        self.act_fullscreen.setShortcut(QKeySequence("F11"))
        self.act_fullscreen.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(self.act_fullscreen)

        help_menu = bar.addMenu("&Help")
        self.act_gestures = QAction("&Gesture reference", self)
        self.act_gestures.setShortcut(QKeySequence("F1"))
        self.act_gestures.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        self.act_about = QAction("&About GestureOS", self)
        self.act_about.triggered.connect(self._show_about)
        help_menu.addAction(self.act_gestures)
        help_menu.addAction(self.act_about)

    # ------------------------------------------------------------------ #
    # Engine wiring
    # ------------------------------------------------------------------ #
    def _profile_names(self) -> List[str]:
        manager = getattr(self.engine, "profiles", None)
        if manager is None:
            profiles = (self.config.get("profiles", {}) or {}).get("profiles", {}) or {}
            return sorted(profiles.keys()) or ["default"]
        try:
            return manager.list_profiles()
        except Exception:
            return ["default"]

    def _active_profile(self) -> str:
        manager = getattr(self.engine, "profiles", None)
        if manager is not None:
            try:
                return manager.get_active()
            except Exception:
                pass
        return str((self.config.get("profiles", {}) or {}).get("current", "default"))

    def _wire_engine(self) -> None:
        self.control_panel.start_requested.connect(self._on_start)
        self.control_panel.stop_requested.connect(self._on_stop)
        self.control_panel.pause_requested.connect(self._on_pause)
        self.control_panel.profile_requested.connect(self._on_profile)
        self.control_panel.mode_requested.connect(self._on_mode)
        self.calibration_tab.values_changed.connect(self._apply_calibration)
        self.profiles_tab.profile_selected.connect(self._on_profile)
        self.macros_tab.macro_triggered.connect(self._run_macro)
        self.settings_tab.settings_applied.connect(self._on_settings_applied)

    def _on_start(self) -> None:
        engine = self.engine
        if engine is None:
            return
        try:
            if not getattr(engine, "running", False):
                engine.start()
            if not self.update_timer.isActive():
                self.update_timer.start()
            self.add_log("Tracking started")
        except Exception as exc:
            logger.error("Start failed: %s", exc)
            self.add_log(f"Start failed: {exc}")
            QMessageBox.warning(self, "GestureOS", f"Could not start tracking:\n{exc}")

    def _on_stop(self) -> None:
        engine = self.engine
        if engine is None:
            return
        try:
            self.update_timer.stop()
            engine.stop()
            self.camera_feed.set_message("Tracking stopped")
            self.hand_visualizer.clear()
            self.add_log("Tracking stopped")
        except Exception as exc:
            logger.error("Stop failed: %s", exc)
            self.add_log(f"Stop failed: {exc}")

    def _on_pause(self) -> None:
        engine = self.engine
        if engine is None:
            return
        try:
            engine.toggle_pause()
            paused = bool(getattr(engine, "paused", False))
            self.control_panel.set_paused(paused)
            self.add_log("Paused - gestures ignored" if paused else "Resumed")
        except Exception as exc:
            logger.error("Pause toggle failed: %s", exc)

    def _on_profile(self, name: str) -> None:
        manager = getattr(self.engine, "profiles", None)
        if manager is None or not name:
            return
        try:
            if manager.set_active(name):
                manager.apply(
                    mouse=getattr(self.engine, "mouse", None),
                    recognizer=getattr(self.engine, "gesture_recognizer", None),
                    scroll=getattr(self.engine, "scroll", None),
                    config_manager=self.config_manager,
                )
                self.control_panel.set_profiles(manager.list_profiles(), name)
                self.profiles_tab.reload(name)
                self.gesture_tab.refresh()
                self.status_profile.setText(f"Profile: {name}")
                self.add_log(f"Profile switched to '{name}'")
        except Exception as exc:
            logger.error("Profile switch failed: %s", exc)
            self.add_log(f"Profile switch failed: {exc}")

    def _on_mode(self, mode: str) -> None:
        engine = self.engine
        if engine is None or not mode:
            return
        try:
            engine.current_mode = mode
            scroll = getattr(engine, "scroll", None)
            if hasattr(scroll, "reset"):
                scroll.reset()
            self.add_log(f"Mode set to '{mode}'")
        except Exception as exc:
            logger.error("Mode switch failed: %s", exc)

    def _apply_calibration(self, values: Dict[str, float]) -> None:
        """Push Calibration tab values into config + live components."""
        if not values:
            return
        try:
            if self.config_manager is not None:
                for key, value in values.items():
                    self.config_manager.set(key, value)
                self.config_manager.set("calibration.calibrated", True)
                self.config_manager.save()

            engine = self.engine
            mouse = getattr(engine, "mouse", None)
            if mouse is not None:
                if "cursor.sensitivity" in values:
                    mouse.set_sensitivity(values["cursor.sensitivity"])
                if "cursor.smoothing" in values:
                    mouse.set_smoothing(values["cursor.smoothing"])

            scroll = getattr(engine, "scroll", None)
            if scroll is not None and "scroll.sensitivity" in values:
                scroll.set_sensitivity(values["scroll.sensitivity"])

            recognizer = getattr(engine, "gesture_recognizer", None)
            if recognizer is not None:
                for key, value in values.items():
                    if key.startswith("gestures."):
                        recognizer.config[key.split(".", 1)[1]] = value
                recognizer.reset()

            self.add_log(f"Calibration applied: {len(values)} value(s)")
        except Exception as exc:
            logger.error("Calibration apply failed: %s", exc)
            self.add_log(f"Calibration failed: {exc}")

    def _run_macro(self, name: str) -> None:
        macros = getattr(self.engine, "macros", None)
        if macros is None:
            return
        try:
            if macros.execute(name):
                self.add_log(f"Macro '{name}' executed")
            else:
                self.add_log(f"Macro '{name}' skipped (cooldown or unknown)")
        except Exception as exc:
            logger.error("Macro execution failed: %s", exc)

    def _on_settings_applied(self, data: Dict[str, Any]) -> None:
        self.config = dict(self.config_manager.config) if self.config_manager else dict(data)
        self.calibration_tab = self.calibration_tab  # keep attribute stable
        self.gesture_tab.refresh()
        self.add_log("Settings applied")

    def reload_settings(self) -> None:
        self.settings_tab.reload()
        index = self.tabs.indexOf(self.calibration_tab)
        replacement = CalibrationTab(config=self._current_config())
        replacement.values_changed.connect(self._apply_calibration)
        if index >= 0:
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, replacement, "Calibration")
        else:
            self.tabs.addTab(replacement, "Calibration")
        self.calibration_tab = replacement
        self.tabs.setCurrentWidget(replacement)
        self.add_log("Settings reloaded from disk")

    def _current_config(self) -> Dict[str, Any]:
        if self.config_manager is not None:
            return dict(getattr(self.config_manager, "config", {}))
        return dict(self.config)

    def _refresh_analytics(self) -> None:
        try:
            if self.tabs.currentWidget() is self.analytics_tab:
                self.analytics_tab.refresh()
        except Exception as exc:
            logger.debug("Analytics refresh failed: %s", exc)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About GestureOS",
            "<h3>GestureOS</h3>"
            "<p>AI gesture control for Windows - drive your PC with hand movements.</p>"
            "<p>MediaPipe hand tracking + OpenCV, PySide6 UI.</p>"
            f"<p>Backend: {getattr(getattr(self.engine, 'tracker', None), 'backend', 'n/a')}</p>",
        )

    # ------------------------------------------------------------------ #
    # Public API used by main.py
    # ------------------------------------------------------------------ #
    def update_camera_feed(self, frame, fps: float = 0.0, mode: str = "", gaze: str = "") -> None:
        """Show the annotated frame and refresh the metric readouts."""
        engine = self.engine
        message = ""
        if frame is None:
            message = "Waiting for camera frames..."
        self.camera_feed.set_frame(frame, fps=fps, message=message)

        latency = float(getattr(engine, "processing_time", 0.0) or 0.0)
        self.control_panel.update_metrics(fps=float(fps or 0.0), latency_ms=latency, mode=mode, gaze=gaze)
        self.status_fps.setText(f"FPS: {fps:.0f}" if fps else "FPS: -")
        self.status_latency.setText(f"Latency: {latency:.1f} ms")
        self.status_profile.setText(f"Profile: {self._active_profile()}")

        analytics = getattr(engine, "analytics", None)
        if analytics is not None:
            try:
                analytics.record_fps(float(fps or 0.0))
                if latency:
                    analytics.record_latency(latency)
            except Exception:
                pass

    def update_hand_visualizer(self, landmarks: Optional[Sequence[Any]], handedness: str = "") -> None:
        fingers: List[bool] = []
        engine_hands = getattr(self.engine, "_last_processed_hands", None) or []
        if engine_hands:
            fingers = list(getattr(engine_hands[0], "fingers_up", []) or [])
        self.hand_visualizer.update_hand(landmarks, handedness, fingers)

    def add_log(self, message: str) -> None:
        """Append a line to the log panel (and mirror it to the logger)."""
        self.log_panel.add_log(str(message))
        logger.debug(message)

    def add_log_throttled(self, key: str, message: str, interval: float = 2.0) -> None:
        now = time.time()
        if now - self._log_throttle.get(key, 0.0) >= interval:
            self._log_throttle[key] = now
            self.add_log(message)

    def notify_gesture(self, gesture: str, action: str, confidence: float = 0.0) -> None:
        """Log only when the gesture/action actually changes (avoids log spam)."""
        if gesture == self._last_gesture and action == self._last_action:
            return
        self._last_gesture, self._last_action = gesture, action
        if action and action != "none":
            suffix = f" ({confidence:.2f})" if confidence else ""
            self.add_log(f"{gesture} -> {action}{suffix}")

    def set_backend_info(self, text: str) -> None:
        self.status_backend.setText(text)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the engine cleanly when the window is closed."""
        try:
            self.update_timer.stop()
            self._stats_timer.stop()
            if self.engine is not None and getattr(self.engine, "running", False):
                self.engine.stop()
        except Exception as exc:
            logger.debug("Error during close: %s", exc)
        super().closeEvent(event)


__all__ = ["MainWindow", "DARK_STYLE"]
