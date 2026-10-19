"""First-run onboarding and operation-mode selection.

``main.py`` calls:

    completed = exec_onboarding(self.camera, self.tracker)
    mode = exec_mode_selection()

Both degrade gracefully: if Qt cannot open a display (headless CI, SSH session)
they log and return safe defaults instead of crashing the app.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from utils import try_import

logger = logging.getLogger(__name__)

ONBOARDING_STEPS = (
    "Welcome to GestureOS",
    "Camera check",
    "Hand tracking check",
    "Movement range",
    "All set",
)


def _qt_available() -> bool:
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
        return True
    except Exception as exc:
        logger.debug("PySide6 unavailable: %s", exc)
        return False


def _ensure_app() -> Optional[Any]:
    QtWidgets = try_import("PySide6.QtWidgets")
    if QtWidgets is None:
        return None
    import sys

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    return app


class _CameraCheckPage:
    """Live preview used by the onboarding wizard."""

    def __init__(self, camera: Any, tracker: Any) -> None:
        self.camera = camera
        self.tracker = tracker


def exec_mode_selection(parent: Any = None, default: str = "hand") -> str:
    """Ask which operation mode to use. Returns ``"hand"`` or ``"pen"``."""
    if not _qt_available():
        logger.info("Mode selection skipped (no Qt display) - using '%s'", default)
        return default

    try:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

        app = _ensure_app()
        dialog = QDialog(parent)
        dialog.setWindowTitle("GestureOS - Select mode")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("How do you want to control your PC?"))

        buttons = QDialogButtonBox()
        hand_button = buttons.addButton("Hand tracking", QDialogButtonBox.AcceptRole)
        pen_button = buttons.addButton("Pen / stylus tracking", QDialogButtonBox.ActionRole)
        layout.addWidget(buttons)

        result = {"mode": default}

        def choose(mode: str) -> None:
            result["mode"] = mode
            dialog.accept()

        hand_button.clicked.connect(lambda: choose("hand"))
        pen_button.clicked.connect(lambda: choose("pen"))
        dialog.exec()
        mode = result["mode"]
        logger.info("Mode selected: %s", mode)
        return mode
    except Exception as exc:
        logger.warning("Mode selection failed (%s) - using '%s'", exc, default)
        return default


def exec_onboarding(camera: Any, tracker: Any, parent: Any = None, config: Any = None) -> bool:
    """Run the first-launch wizard.

    Verifies that the camera delivers frames and that MediaPipe can see a hand,
    then records the user's comfortable movement range.  Returns ``True`` when
    the wizard completed (or when it was skipped because Qt is unavailable).
    """
    if not _qt_available():
        logger.info("Onboarding skipped (no Qt display)")
        return False

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (
            QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
        )

        app = _ensure_app()
        dialog = QDialog(parent)
        dialog.setWindowTitle("Welcome to GestureOS")
        dialog.setMinimumSize(720, 520)

        layout = QVBoxLayout(dialog)
        title = QLabel(ONBOARDING_STEPS[0])
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        body = QLabel(_STEP_TEXT[0])
        body.setWordWrap(True)
        preview = QLabel()
        preview.setMinimumSize(480, 300)
        preview.setAlignment(Qt_AlignCenter())
        preview.setStyleSheet("background: #14141a; border-radius: 8px;")
        status = QLabel("Waiting for camera...")
        progress = QProgressBar()
        progress.setRange(0, len(ONBOARDING_STEPS) - 1)
        progress.setValue(0)

        buttons = QHBoxLayout()
        back = QPushButton("Back")
        forward = QPushButton("Next")
        skip = QPushButton("Skip")
        buttons.addWidget(back)
        buttons.addStretch(1)
        buttons.addWidget(skip)
        buttons.addWidget(forward)

        layout.addWidget(title)
        layout.addWidget(body)
        layout.addWidget(preview, 1)
        layout.addWidget(status)
        layout.addWidget(progress)
        layout.addLayout(buttons)

        state = {"step": 0, "completed": False, "hand_seen": False, "frames": 0,
                 "bounds": {"x_min": 1.0, "y_min": 1.0, "x_max": 0.0, "y_max": 0.0}}

        if camera is not None and not camera.is_running():
            camera.start()

        def refresh() -> None:
            frame = None
            hands = []
            if camera is not None:
                frame, _ts = camera.get_latest_frame()
            if frame is not None and tracker is not None:
                try:
                    hands = tracker.process(frame)
                except Exception as exc:
                    status.setText(f"Tracking error: {exc}")
                    hands = []
                state["frames"] += 1
                if hands:
                    state["hand_seen"] = True
                    for hand in hands:
                        tip = hand.landmark(8)
                        if tip is not None:
                            b = state["bounds"]
                            b["x_min"] = min(b["x_min"], tip.x)
                            b["x_max"] = max(b["x_max"], tip.x)
                            b["y_min"] = min(b["y_min"], tip.y)
                            b["y_max"] = max(b["y_max"], tip.y)
                annotated = tracker.draw_landmarks(frame, hands) if tracker is not None else frame
                preview.setPixmap(to_pixmap(annotated, preview.size()))

            step = state["step"]
            if step == 1:
                status.setText("Camera OK" if state["frames"] > 3 else "Waiting for camera frames...")
            elif step == 2:
                status.setText("Hand detected - nice!" if state["hand_seen"] else "Show your hand to the camera")
            elif step == 3:
                bounds = state["bounds"]
                if bounds["x_max"] > bounds["x_min"]:
                    status.setText(
                        "Range captured: x %.2f-%.2f, y %.2f-%.2f"
                        % (bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"])
                    )
                else:
                    status.setText("Move your hand to the corners of your comfortable range")

        def set_step(step: int) -> None:
            state["step"] = max(0, min(step, len(ONBOARDING_STEPS) - 1))
            title.setText(ONBOARDING_STEPS[state["step"]])
            body.setText(_STEP_TEXT[state["step"]])
            progress.setValue(state["step"])
            forward.setText("Finish" if state["step"] == len(ONBOARDING_STEPS) - 1 else "Next")
            back.setEnabled(state["step"] > 0)

        def on_forward() -> None:
            if state["step"] < len(ONBOARDING_STEPS) - 1:
                set_step(state["step"] + 1)
            else:
                finish(True)

        def finish(completed: bool) -> None:
            state["completed"] = completed
            if completed and config is not None:
                bounds = state["bounds"]
                if bounds["x_max"] > bounds["x_min"]:
                    try:
                        region = {
                            "x_min": round(max(0.0, bounds["x_min"] - 0.05), 3),
                            "y_min": round(max(0.0, bounds["y_min"] - 0.05), 3),
                            "x_max": round(min(1.0, bounds["x_max"] + 0.05), 3),
                            "y_max": round(min(1.0, bounds["y_max"] + 0.05), 3),
                        }
                        config.set("cursor.movement_region", region)
                        config.set("cursor_mode.movement_region", region)
                        config.save()
                        logger.info("Onboarding saved movement region %s", region)
                    except Exception as exc:
                        logger.warning("Could not save movement region: %s", exc)
            dialog.accept()

        forward.clicked.connect(on_forward)
        back.clicked.connect(lambda: set_step(state["step"] - 1))
        skip.clicked.connect(lambda: finish(False))

        timer = QTimer(dialog)
        timer.timeout.connect(refresh)
        timer.start(50)
        set_step(0)

        dialog.exec()
        timer.stop()
        completed = bool(state["completed"])
        logger.info("Onboarding finished (completed=%s, hand_seen=%s)", completed, state["hand_seen"])
        return completed
    except Exception as exc:
        logger.warning("Onboarding failed: %s", exc)
        return False


def Qt_AlignCenter():  # noqa: N802 - keeps the Qt import local
    from PySide6.QtCore import Qt

    return Qt.AlignCenter


def to_pixmap(frame, size=None):
    """Convert a BGR numpy frame into a QPixmap scaled to ``size``."""
    from PySide6.QtGui import QImage, QPixmap

    if frame is None:
        return QPixmap()
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(image)
    if size is not None and not size.isEmpty():
        from PySide6.QtCore import Qt

        pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap


_STEP_TEXT = {
    0: (
        "GestureOS lets you control Windows with hand gestures.\n\n"
        "This quick setup takes about 30 seconds: it checks your camera, "
        "confirms hand tracking works and learns how far you like to reach."
    ),
    1: (
        "Step 1 - Camera.\n\nMake sure your webcam is connected and not in use "
        "by another app. You should see yourself in the preview below."
    ),
    2: (
        "Step 2 - Hand tracking.\n\nHold your open hand up to the camera, about "
        "an arm's length away, with even lighting. The skeleton appears when "
        "tracking locks on."
    ),
    3: (
        "Step 3 - Movement range.\n\nWith your index finger extended, move your "
        "hand to the four corners of the area you want to cover. GestureOS maps "
        "that box onto your whole screen, so you never have to stretch."
    ),
    4: (
        "All set!\n\nIndex finger moves the cursor. Pinch thumb + index to "
        "left click, thumb + middle to right click. Hold a pinch to drag.\n\n"
        "Open the Gestures tab any time to see the full list."
    ),
}


__all__ = ["exec_onboarding", "exec_mode_selection", "to_pixmap", "ONBOARDING_STEPS"]
