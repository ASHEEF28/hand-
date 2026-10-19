"""Reusable PySide6 widgets for the GestureOS main window.

Kept separate from ``main_window.py`` so the window itself stays a composition
root.  Every widget is defensive: it tolerates ``None`` frames, missing engine
modules and repeated updates, because the gesture loop calls into it at ~30 Hz.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# The gesture tables live in gestures/help.py so the CLI can print them without
# importing Qt.  Re-exported here for backwards compatibility.
from gestures.help import DEFAULT_ACTIONS, GESTURE_HELP

logger = logging.getLogger(__name__)

ACCENT = QColor(0, 212, 255)
GOOD = QColor(0, 230, 118)
WARN = QColor(255, 193, 7)
BAD = QColor(255, 82, 82)
MUTED = QColor(160, 160, 176)
BG = QColor(20, 20, 26)

#: Skeleton bones for the hand visualiser (MediaPipe landmark indices).
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
TIPS = (4, 8, 12, 16, 20)


# --------------------------------------------------------------------------- #
# Camera preview
# --------------------------------------------------------------------------- #
class CameraFeedWidget(QLabel):
    """Shows the annotated camera frame, scaled to fit while keeping aspect."""

    frame_received = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #101014; border-radius: 10px; color: #6b6b78;")
        self.setText("Camera idle\n\nPress Start to begin tracking")
        self._pixmap: Optional[QPixmap] = None
        self._frames = 0

    def set_frame(self, frame, fps: float = 0.0, message: str = "") -> None:
        pixmap = frame_to_pixmap(frame)
        if pixmap is None:
            if message:
                self.setText(message)
            return
        self._pixmap = pixmap
        self._frames += 1
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.frame_received.emit()

    def set_message(self, message: str) -> None:
        self._pixmap = None
        self.setText(message)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @property
    def frames(self) -> int:
        return self._frames


def frame_to_pixmap(frame) -> Optional[QPixmap]:
    """BGR numpy array -> QPixmap (``None`` when the frame is unusable)."""
    if frame is None:
        return None
    try:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = frame
    try:
        height, width, channels = rgb.shape
    except ValueError:
        return None
    bytes_per_line = channels * width
    fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
    image = QImage(rgb.data, width, height, bytes_per_line, fmt).copy()
    return QPixmap.fromImage(image)


# --------------------------------------------------------------------------- #
# Hand skeleton visualiser
# --------------------------------------------------------------------------- #
class HandVisualizer(QWidget):
    """Draws the landmark skeleton and per-finger state for one hand."""

    def __init__(self, parent: Optional[QWidget] = None, size: int = 170) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setStyleSheet("background: #14141a; border-radius: 8px;")
        self._landmarks: List[Any] = []
        self._handedness: str = ""
        self._fingers: List[bool] = []

    def update_hand(self, landmarks: Optional[Sequence[Any]], handedness: str = "",
                    fingers_up: Optional[Sequence[bool]] = None) -> None:
        self._landmarks = list(landmarks or [])
        self._handedness = handedness or ""
        self._fingers = list(fingers_up or [])
        self.update()

    def clear(self) -> None:
        self.update_hand(None)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), BG)

        width, height = self.width(), self.height()
        margin = 18
        painter.setPen(QPen(MUTED, 1))
        painter.drawRect(margin // 2, margin // 2, width - margin, height - margin)

        if not self._landmarks:
            painter.setPen(MUTED)
            painter.drawText(self.rect(), Qt.AlignCenter, "no hand")
            painter.end()
            return

        points: List[Tuple[float, float]] = []
        for lm in self._landmarks:
            x = float(getattr(lm, "x", 0.0))
            y = float(getattr(lm, "y", 0.0))
            points.append((margin + (1.0 - x) * (width - 2 * margin) if self._handedness == "Right"
                           else margin + x * (width - 2 * margin),
                           margin + y * (height - 2 * margin)))

        painter.setPen(QPen(ACCENT, 2))
        for a, b in CONNECTIONS:
            if a < len(points) and b < len(points):
                painter.drawLine(int(points[a][0]), int(points[a][1]), int(points[b][0]), int(points[b][1]))

        for index, (px, py) in enumerate(points):
            painter.setPen(Qt.NoPen)
            painter.setBrush(GOOD if index in TIPS else ACCENT)
            radius = 5 if index in TIPS else 3
            painter.drawEllipse(int(px) - radius, int(py) - radius, radius * 2, radius * 2)

        painter.setPen(MUTED)
        painter.drawText(8, height - 6, self._handedness or "hand")
        if self._fingers:
            labels = "T I M R P"
            painter.drawText(width - 62, height - 6,
                             " ".join(c if up else "." for c, up in zip(labels.split(), self._fingers)))
        painter.end()


# --------------------------------------------------------------------------- #
# Control panel
# --------------------------------------------------------------------------- #
class ControlPanel(QWidget):
    """Status readouts plus start / pause / stop controls."""

    start_requested = Signal()
    stop_requested = Signal()
    pause_requested = Signal()
    profile_requested = Signal(str)
    mode_requested = Signal(str)

    def __init__(self, profiles: Optional[Sequence[str]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Control")
        title.setStyleSheet("font-size: 15px; font-weight: 600; color: #e8e8ef;")
        layout.addWidget(title)

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(14, 14)
        self.status_text = QLabel("Stopped")
        header = QHBoxLayout()
        header.addWidget(self.status_dot)
        header.addWidget(self.status_text)
        header.addStretch(1)
        layout.addLayout(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self.labels: Dict[str, QLabel] = {}
        rows = (
            ("camera", "Camera"),
            ("hand", "Hand"),
            ("gesture", "Gesture"),
            ("action", "Action"),
            ("profile", "Profile"),
            ("confidence", "Confidence"),
            ("fps", "FPS"),
            ("latency", "Latency"),
            ("mode", "Mode"),
            ("gaze", "Gaze"),
        )
        for row, (key, caption) in enumerate(rows):
            name = QLabel(caption)
            name.setStyleSheet("color: #8a8a99;")
            value = QLabel("-")
            value.setStyleSheet("color: #e8e8ef; font-weight: 500;")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self.labels[key] = value
        layout.addLayout(grid)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(6)
        layout.addWidget(self.confidence_bar)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        for button in (self.start_button, self.pause_button, self.stop_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.pause_button.clicked.connect(self.pause_requested.emit)

        profile_box = QGroupBox("Profile")
        profile_layout = QVBoxLayout(profile_box)
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(list(profiles or ["default"]))
        self.profile_combo.currentTextChanged.connect(self.profile_requested.emit)
        profile_layout.addWidget(self.profile_combo)
        layout.addWidget(profile_box)

        mode_box = QGroupBox("Cursor mode")
        mode_layout = QVBoxLayout(mode_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["cursor", "scroll"])
        self.mode_combo.currentTextChanged.connect(self.mode_requested.emit)
        mode_layout.addWidget(self.mode_combo)
        layout.addWidget(mode_box)

        layout.addStretch(1)
        self.set_status(False, False)

    # ------------------------------------------------------------------ #
    def set_status(self, camera_running: bool, hand_detected: bool) -> None:
        dot_color = GOOD if camera_running and hand_detected else (WARN if camera_running else BAD)
        self.status_dot.setPixmap(_dot(dot_color))
        self.status_text.setText(
            "Tracking" if camera_running and hand_detected else ("Running" if camera_running else "Stopped")
        )
        self.start_button.setEnabled(not camera_running)
        self.stop_button.setEnabled(camera_running)
        self.pause_button.setEnabled(camera_running)

    def update_status(self, camera_running: bool, hand_detected: bool, gesture: str = "none",
                      action: str = "none", profile: str = "default", confidence: float = 0.0) -> None:
        """Signature used by ``main.py`` each UI tick."""
        self.set_status(camera_running, hand_detected)
        self.labels["camera"].setText("on" if camera_running else "off")
        self.labels["camera"].setStyleSheet(f"color: {'#00e676' if camera_running else '#ff5252'};")
        self.labels["hand"].setText("yes" if hand_detected else "no")
        self.labels["hand"].setStyleSheet(f"color: {'#00e676' if hand_detected else '#8a8a99'};")
        self.labels["gesture"].setText(str(gesture or "none"))
        self.labels["action"].setText(str(action or "none"))
        self.labels["profile"].setText(str(profile or "default"))
        try:
            percent = int(round(clamp01(float(confidence)) * 100))
        except (TypeError, ValueError):
            percent = 0
        self.labels["confidence"].setText(f"{percent}%")
        self.confidence_bar.setValue(percent)

    def update_metrics(self, fps: float = 0.0, latency_ms: float = 0.0, mode: str = "",
                       gaze: str = "") -> None:
        self.labels["fps"].setText(f"{fps:.0f}" if fps else "-")
        self.labels["latency"].setText(f"{latency_ms:.1f} ms" if latency_ms else "-")
        if mode:
            self.labels["mode"].setText(str(mode))
            if self.mode_combo.currentText() != str(mode):
                self.mode_combo.blockSignals(True)
                index = self.mode_combo.findText(str(mode))
                if index >= 0:
                    self.mode_combo.setCurrentIndex(index)
                self.mode_combo.blockSignals(False)
        self.labels["gaze"].setText(str(gaze) if gaze else "-")

    def set_paused(self, paused: bool) -> None:
        self.pause_button.setText("Resume" if paused else "Pause")

    def set_profiles(self, profiles: Sequence[str], active: str = "") -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(list(profiles))
        if active and active in profiles:
            self.profile_combo.setCurrentText(active)
        self.profile_combo.blockSignals(False)


def clamp01(value: float) -> float:
    return 0.0 if value < 0 else (1.0 if value > 1.0 else value)


def _dot(color: QColor) -> QPixmap:
    pixmap = QPixmap(14, 14)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setBrush(color)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(1, 1, 12, 12)
    painter.end()
    return pixmap


# --------------------------------------------------------------------------- #
# Log panel
# --------------------------------------------------------------------------- #
class LogPanel(QPlainTextEdit):
    """Read-only rolling log with timestamps."""

    def __init__(self, max_lines: int = 500, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        self.setStyleSheet(
            "QPlainTextEdit { background: #101014; color: #c9c9d4; border: 1px solid #23232c;"
            " border-radius: 8px; font-family: Consolas, Menlo, monospace; font-size: 12px; }"
        )
        self._count = 0

    def add_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.appendPlainText(f"[{stamp}] {message}")
        self._count += 1

    def add_event(self, kind: str, message: str) -> None:
        self.add_log(f"{kind.upper():<8} {message}")

    @property
    def line_count(self) -> int:
        return self._count


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
class GestureReferenceTab(QWidget):
    """Read-only table of gestures -> actions (live from the recogniser)."""

    def __init__(self, engine: Any = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        layout = QVBoxLayout(self)
        caption = QLabel("How to control your PC")
        caption.setStyleSheet("font-size: 15px; font-weight: 600; color: #e8e8ef;")
        layout.addWidget(caption)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Gesture", "Action", "How to do it"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        self.refresh()

    def refresh(self) -> None:
        action_map = {}
        recognizer = getattr(self.engine, "gesture_recognizer", None) if self.engine else None
        if recognizer is not None:
            action_map = dict(getattr(recognizer, "action_map", {}) or {})

        self.table.setRowCount(len(GESTURE_HELP))
        for row, (gesture, how) in enumerate(GESTURE_HELP.items()):
            action = action_map.get(gesture, DEFAULT_ACTIONS.get(gesture, ""))
            self.table.setItem(row, 0, QTableWidgetItem(gesture))
            self.table.setItem(row, 1, QTableWidgetItem(str(action)))
            self.table.setItem(row, 2, QTableWidgetItem(how))


class CalibrationTab(QWidget):
    """Live tuning of cursor and gesture thresholds."""

    values_changed = Signal(dict)

    FIELDS = (
        ("cursor.sensitivity", "Cursor sensitivity", 0.1, 5.0, 0.1),
        ("cursor.smoothing", "Cursor smoothing", 0.0, 0.95, 0.05),
        ("gestures.pinch_ratio_threshold", "Pinch sensitivity", 0.10, 0.70, 0.01),
        ("gestures.swipe_min_distance", "Swipe distance", 0.02, 0.60, 0.01),
        ("gestures.swipe_cooldown", "Swipe cooldown (s)", 0.0, 3.0, 0.05),
        ("gestures.double_click_interval", "Double click window (s)", 0.05, 1.0, 0.05),
        ("gestures.hold_duration", "Pose hold time (s)", 0.0, 2.0, 0.05),
        ("scroll.sensitivity", "Scroll sensitivity", 0.1, 5.0, 0.1),
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config or {}
        layout = QVBoxLayout(self)
        caption = QLabel("Calibration")
        caption.setStyleSheet("font-size: 15px; font-weight: 600; color: #e8e8ef;")
        layout.addWidget(caption)
        hint = QLabel("Changes apply immediately and are saved to settings.json.")
        hint.setStyleSheet("color: #8a8a99;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self.spinboxes: Dict[str, QDoubleSpinBox] = {}
        for key, label, low, high, step in self.FIELDS:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setDecimals(3)
            spin.setValue(float(_dig(self.config, key, low)))
            spin.valueChanged.connect(lambda _value, k=key: self._emit())
            self.spinboxes[key] = spin
            form.addRow(label, spin)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset defaults")
        self.apply_button.clicked.connect(self._emit)
        self.reset_button.clicked.connect(self.reset_defaults)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(1)

    def _emit(self) -> None:
        self.values_changed.emit(self.values())

    def values(self) -> Dict[str, float]:
        return {key: float(spin.value()) for key, spin in self.spinboxes.items()}

    def reset_defaults(self) -> None:
        defaults = {
            "cursor.sensitivity": 1.5,
            "cursor.smoothing": 0.30,
            "gestures.pinch_ratio_threshold": 0.34,
            "gestures.swipe_min_distance": 0.15,
            "gestures.swipe_cooldown": 0.6,
            "gestures.double_click_interval": 0.35,
            "gestures.hold_duration": 0.35,
            "scroll.sensitivity": 1.0,
        }
        for key, value in defaults.items():
            if key in self.spinboxes:
                self.spinboxes[key].blockSignals(True)
                self.spinboxes[key].setValue(value)
                self.spinboxes[key].blockSignals(False)
        self._emit()


class ProfilesTab(QWidget):
    """Profile picker and editor."""

    profile_selected = Signal(str)

    def __init__(self, engine: Any = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        layout = QVBoxLayout(self)
        caption = QLabel("Profiles")
        caption.setStyleSheet("font-size: 15px; font-weight: 600; color: #e8e8ef;")
        layout.addWidget(caption)

        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_select)
        layout.addWidget(self.list, 1)

        self.description = QLabel("")
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color: #8a8a99;")
        layout.addWidget(self.description)

        form = QFormLayout()
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.1, 5.0)
        self.sensitivity.setSingleStep(0.1)
        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.0, 0.95)
        self.smoothing.setSingleStep(0.05)
        form.addRow("Cursor sensitivity", self.sensitivity)
        form.addRow("Cursor smoothing", self.smoothing)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save profile")
        self.delete_button = QPushButton("Delete")
        self.new_button = QPushButton("New profile")
        buttons.addWidget(self.new_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.save_button.clicked.connect(self.save_profile)
        self.delete_button.clicked.connect(self.delete_profile)
        self.new_button.clicked.connect(self.new_profile)
        self.reload()

    # ------------------------------------------------------------------ #
    @property
    def manager(self):
        return getattr(self.engine, "profiles", None)

    def reload(self, active: str = "") -> None:
        manager = self.manager
        self.list.blockSignals(True)
        self.list.clear()
        if manager is not None:
            for name in manager.list_profiles():
                item = QListWidgetItem(name)
                self.list.addItem(item)
                if name == (active or manager.get_active()):
                    self.list.setCurrentItem(item)
        self.list.blockSignals(False)
        self._show_current()

    def _on_select(self, *_args) -> None:
        self._show_current()

    def _show_current(self) -> None:
        manager = self.manager
        item = self.list.currentItem()
        if manager is None or item is None:
            return
        profile = manager.get_profile(item.text())
        self.description.setText(profile.description or "No description")
        self.sensitivity.blockSignals(True)
        self.smoothing.blockSignals(True)
        self.sensitivity.setValue(float(profile.cursor_sensitivity))
        self.smoothing.setValue(float(profile.cursor_smoothing))
        self.sensitivity.blockSignals(False)
        self.smoothing.blockSignals(False)
        self.delete_button.setEnabled(item.text() != "default")

    def current_name(self) -> str:
        item = self.list.currentItem()
        return item.text() if item else ""

    def save_profile(self) -> None:
        manager = self.manager
        name = self.current_name()
        if manager is None or not name:
            return
        manager.update(name, cursor_sensitivity=self.sensitivity.value(),
                       cursor_smoothing=self.smoothing.value())
        manager.set_active(name)
        self.profile_selected.emit(name)
        self.reload(name)

    def delete_profile(self) -> None:
        manager = self.manager
        name = self.current_name()
        if manager is None or not name:
            return
        manager.delete(name)
        self.reload()

    def new_profile(self) -> None:
        manager = self.manager
        if manager is None:
            return
        base = self.current_name() or "default"
        index = 2
        while True:
            candidate = f"custom_{index}"
            if not manager.has_profile(candidate):
                break
            index += 1
        source = manager.get_profile(base)
        manager.create(candidate, source.to_dict() | {"name": candidate, "description": "Custom profile"})
        self.reload(candidate)


class MacrosTab(QWidget):
    """Macro list, editor and app launcher shortcuts."""

    macro_triggered = Signal(str)

    def __init__(self, engine: Any = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        layout = QVBoxLayout(self)
        caption = QLabel("Macros & app launcher")
        caption.setStyleSheet("font-size: 15px; font-weight: 600; color: #e8e8ef;")
        layout.addWidget(caption)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.macro_list = QListWidget()
        self.macro_list.currentItemChanged.connect(self._show_macro)
        left_layout.addWidget(QLabel("Macros"))
        left_layout.addWidget(self.macro_list)
        self.run_button = QPushButton("Run selected")
        self.run_button.clicked.connect(self.run_selected)
        left_layout.addWidget(self.run_button)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Macro name")
        self.steps_edit = QPlainTextEdit()
        self.steps_edit.setPlaceholderText(json.dumps(
            [{"type": "hotkey", "keys": ["ctrl", "s"]}], indent=2))
        self.steps_edit.setMaximumHeight(180)
        right_layout.addWidget(QLabel("Name"))
        right_layout.addWidget(self.name_edit)
        right_layout.addWidget(QLabel("Steps (JSON)"))
        right_layout.addWidget(self.steps_edit)
        save_row = QHBoxLayout()
        self.save_macro = QPushButton("Save macro")
        self.delete_macro = QPushButton("Delete")
        self.save_macro.clicked.connect(self._save_macro)
        self.delete_macro.clicked.connect(self._delete_macro)
        save_row.addWidget(self.save_macro)
        save_row.addWidget(self.delete_macro)
        right_layout.addLayout(save_row)
        splitter.addWidget(right)
        layout.addWidget(splitter, 1)

        app_box = QGroupBox("Launch an application")
        app_layout = QHBoxLayout(app_box)
        self.app_combo = QComboBox()
        self.app_combo.setEditable(True)
        self.launch_button = QPushButton("Launch")
        self.launch_button.clicked.connect(self._launch_app)
        app_layout.addWidget(self.app_combo, 1)
        app_layout.addWidget(self.launch_button)
        layout.addWidget(app_box)
        self.reload()

    # ------------------------------------------------------------------ #
    @property
    def macros(self):
        return getattr(self.engine, "macros", None)

    @property
    def launcher(self):
        engine_macros = self.macros
        if engine_macros is not None:
            return getattr(engine_macros, "launcher", None)
        return getattr(self.engine, "app_launcher", None)

    def reload(self) -> None:
        self.macro_list.clear()
        if self.macros is not None:
            for name in self.macros.list_macros():
                self.macro_list.addItem(QListWidgetItem(name))
        self.app_combo.clear()
        if self.launcher is not None:
            self.app_combo.addItems(self.launcher.list_apps())

    def _show_macro(self, *_args) -> None:
        item = self.macro_list.currentItem()
        if item is None or self.macros is None:
            return
        macro = self.macros.get_macro(item.text()) or {}
        self.name_edit.setText(item.text())
        self.steps_edit.setPlainText(json.dumps(macro.get("steps", []), indent=2))

    def _save_macro(self) -> None:
        if self.macros is None:
            return
        name = self.name_edit.text().strip()
        if not name:
            return
        try:
            steps = json.loads(self.steps_edit.toPlainText() or "[]")
        except json.JSONDecodeError as exc:
            self.steps_edit.setPlainText(self.steps_edit.toPlainText() + f"\n// invalid JSON: {exc}")
            return
        if isinstance(steps, list) and steps:
            self.macros.add_macro(name, steps)
            self.reload()

    def _delete_macro(self) -> None:
        if self.macros is None:
            return
        name = self.name_edit.text().strip()
        if name and self.macros.remove_macro(name):
            self.reload()

    def run_selected(self) -> None:
        item = self.macro_list.currentItem()
        if item is not None:
            self.macro_triggered.emit(item.text())

    def _launch_app(self) -> None:
        name = self.app_combo.currentText().strip()
        if name and self.launcher is not None:
            self.launcher.launch(name)


class AnalyticsTab(QWidget):
    """Live accuracy / latency / usage numbers."""

    def __init__(self, engine: Any = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        layout = QVBoxLayout(self)
        caption = QLabel("Analytics")
        caption.setStyleSheet("font-size: 15px; font-weight: 600; color: #e8e8ef;")
        layout.addWidget(caption)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setStyleSheet(
            "QPlainTextEdit { background: #101014; color: #c9c9d4; border: 1px solid #23232c;"
            " border-radius: 8px; font-family: Consolas, Menlo, monospace; }")
        layout.addWidget(self.summary, 1)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Gesture", "Count", "Accuracy", "Avg confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(220)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.reset_button = QPushButton("Reset session")
        self.export_button = QPushButton("Export CSV")
        self.refresh_button.clicked.connect(self.refresh)
        self.reset_button.clicked.connect(self._reset)
        self.export_button.clicked.connect(self._export)
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.export_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.refresh()

    @property
    def analytics(self):
        return getattr(self.engine, "analytics", None)

    def refresh(self) -> None:
        analytics = self.analytics
        if analytics is None:
            self.summary.setPlainText("Analytics disabled.")
            return
        data = analytics.summary()
        session = data.get("session", {})
        latency = data.get("latency", {})
        fps = data.get("fps", {})
        lines = [
            f"Session            : {session.get('session_id', '-')}",
            f"Duration           : {session.get('duration_s', 0):.0f} s",
            f"Gestures           : {session.get('gestures', 0)} ({session.get('gestures_per_minute', 0):.1f}/min)",
            f"Lifetime gestures  : {data.get('lifetime_gestures', 0)}",
            f"Latency  avg/p95   : {latency.get('avg', 0):.1f} / {latency.get('p95', 0):.1f} ms",
            f"FPS      avg/max   : {fps.get('avg', 0):.1f} / {fps.get('max', 0):.1f}",
            f"Avg confidence     : {data.get('average_confidence', 0):.2f}",
            "",
            "Top actions:",
        ]
        for action, count in data.get("top_actions", [])[:8]:
            lines.append(f"  {action:<24} {count}")
        self.summary.setPlainText("\n".join(lines))

        accuracy = data.get("accuracy", {})
        counts = data.get("gesture_counts", {})
        gestures = sorted(counts, key=lambda g: counts[g], reverse=True)
        self.table.setRowCount(len(gestures))
        for row, gesture in enumerate(gestures):
            samples = getattr(analytics, "_confidence", {}).get(gesture, [])
            average = sum(samples) / len(samples) if samples else 0.0
            self.table.setItem(row, 0, QTableWidgetItem(gesture))
            self.table.setItem(row, 1, QTableWidgetItem(str(counts.get(gesture, 0))))
            self.table.setItem(row, 2, QTableWidgetItem(f"{accuracy.get(gesture, 0.0) * 100:.0f}%"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{average:.2f}"))

    def _reset(self) -> None:
        if self.analytics is not None:
            self.analytics.reset()
            self.refresh()

    def _export(self) -> None:
        if self.analytics is not None:
            path = self.analytics.export_csv()
            self.summary.appendPlainText(f"\nExported timeline to {path}")


class SettingsTab(QWidget):
    """Raw settings.json editor with validation."""

    settings_applied = Signal(dict)

    def __init__(self, config_manager: Any = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config_manager = config_manager
        layout = QVBoxLayout(self)
        caption = QLabel("Settings (JSON)")
        caption.setStyleSheet("font-size: 15px; font-weight: 600; color: #e8e8ef;")
        layout.addWidget(caption)

        self.editor = QPlainTextEdit()
        self.editor.setStyleSheet(
            "QPlainTextEdit { background: #101014; color: #c9c9d4; border: 1px solid #23232c;"
            " border-radius: 8px; font-family: Consolas, Menlo, monospace; font-size: 12px; }")
        layout.addWidget(self.editor, 1)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8a8a99;")
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.reload_button = QPushButton("Reload from disk")
        self.apply_button = QPushButton("Apply")
        self.reload_button.clicked.connect(self.reload)
        self.apply_button.clicked.connect(self.apply)
        buttons.addWidget(self.reload_button)
        buttons.addWidget(self.apply_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.reload()

    def reload(self) -> None:
        if self.config_manager is None:
            self.editor.setPlainText("{}")
            return
        try:
            self.editor.setPlainText(json.dumps(self.config_manager.config, indent=4))
            self.status_label.setText(f"Loaded {getattr(self.config_manager, 'config_path', 'settings')}")
        except Exception as exc:
            self.status_label.setText(f"Could not load settings: {exc}")

    def apply(self) -> None:
        try:
            data = json.loads(self.editor.toPlainText())
        except json.JSONDecodeError as exc:
            self.status_label.setText(f"Invalid JSON: {exc}")
            return
        if not isinstance(data, dict):
            self.status_label.setText("Settings must be a JSON object")
            return
        if self.config_manager is not None:
            for section, values in data.items():
                if isinstance(values, dict):
                    self.config_manager.update_section(section, values)
            self.config_manager.save()
            self.status_label.setText("Saved. Some values need a restart to take effect.")
        self.settings_applied.emit(data)


def _dig(config: Dict[str, Any], path: str, default: Any) -> Any:
    value: Any = config
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


__all__ = [
    "CameraFeedWidget",
    "HandVisualizer",
    "ControlPanel",
    "LogPanel",
    "GestureReferenceTab",
    "CalibrationTab",
    "ProfilesTab",
    "MacrosTab",
    "AnalyticsTab",
    "SettingsTab",
    "frame_to_pixmap",
    "GESTURE_HELP",
    "DEFAULT_ACTIONS",
]
