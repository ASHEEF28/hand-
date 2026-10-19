"""Threaded camera capture for GestureOS.

The capture loop runs on its own thread and keeps a small ring buffer so the
UI / gesture loop always reads the freshest frame without blocking on the
camera driver.

If no webcam can be opened the capture falls back to a *simulated* source
(a synthetic moving image) when ``simulate_on_failure`` is enabled, which keeps
the rest of the pipeline runnable on machines without a camera - handy for CI,
development and the ``--simulate`` CLI flag.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from utils import clamp, deep_merge


DEFAULT_CAMERA_CONFIG: Dict[str, Any] = {
    "device_id": 0,
    "width": 1280,
    "height": 720,
    "fps": 60,
    "buffer_size": 1,
    "auto_exposure": True,
    "exposure": -6,
    "brightness": 0,
    "contrast": 0,
    "saturation": 0,
    "fourcc": "",
    "backend": "any",
    "simulate": False,
    "simulate_on_failure": True,
    "simulate_fps": 30,
    "read_timeout": 1.0,
}

_BACKENDS = {
    "any": None,
    "dshow": "CAP_DSHOW",
    "msmf": "CAP_MSMF",
    "v4l2": "CAP_V4L2",
    "avfoundation": "CAP_AVFOUNDATION",
}


class SimulatedCamera:
    """Generates synthetic BGR frames - no hardware required.

    The image contains a moving gradient, a bouncing marker and a timestamp so
    it is obvious in the UI that no real camera is attached.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1.0, float(fps))
        self._t0 = time.time()
        self._frame_index = 0

    def isOpened(self) -> bool:  # noqa: N802 - mimics cv2.VideoCapture
        return np is not None

    def read(self) -> Tuple[bool, Any]:
        if np is None:
            return False, None
        t = time.time() - self._t0
        w, h = self.width, self.height

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[..., 0] = ((xx / max(w, 1)) * 90 + 20 * math.sin(t * 0.7)).clip(0, 255).astype(np.uint8)
        base[..., 1] = ((yy / max(h, 1)) * 90 + 20 * math.cos(t * 0.5)).clip(0, 255).astype(np.uint8)
        base[..., 2] = 60

        # Bouncing marker
        cx = int(w * (0.5 + 0.35 * math.sin(t * 1.3)))
        cy = int(h * (0.5 + 0.3 * math.cos(t * 1.7)))
        r = max(12, min(w, h) // 24)
        cv2 = _cv2()
        if cv2 is not None:
            cv2.circle(base, (cx, cy), r, (0, 215, 255), -1)
            cv2.putText(
                base,
                "SIMULATED CAMERA - no webcam attached",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        self._frame_index += 1
        return True, base

    def get(self, prop: int) -> float:
        """Mimic ``cv2.VideoCapture.get`` for the properties callers actually use."""
        cv2 = _cv2()
        if cv2 is not None:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.width)
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.height)
            if prop == cv2.CAP_PROP_FPS:
                return float(self.fps)
            if prop in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_POS_FRAMES):
                return float(self._frame_index)
        # Numeric fallbacks when OpenCV is unavailable (3/4/5/7 are the cv2 ids).
        return float({3: self.width, 4: self.height, 5: self.fps,
                      7: self._frame_index}.get(int(prop), 0.0))

    def release(self) -> None:
        return None


def _cv2():
    try:
        import cv2

        return cv2
    except Exception:  # pragma: no cover
        return None


class CameraCapture:
    """Non-blocking camera capture with a background reader thread."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, source: Optional[Any] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_CAMERA_CONFIG), config or {})
        self._explicit_source = source
        self._cap: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._buffer: Deque[Tuple[Any, float]] = deque(maxlen=max(1, int(self.config.get("buffer_size", 1))))
        self._latest: Optional[Tuple[Any, float]] = None
        self._simulated = False
        self._frames_read = 0
        self._fps_window: Deque[float] = deque(maxlen=30)
        self._last_error: str = ""
        self._device_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def simulated(self) -> bool:
        """True when frames come from :class:`SimulatedCamera`."""
        return self._simulated

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._cap is not None:
            cv2 = _cv2()
            if cv2 is not None:
                try:
                    return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                        self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )
                except Exception:
                    pass
        return int(self.config.get("width", 1280)), int(self.config.get("height", 720))

    def measured_fps(self) -> float:
        """Average FPS of the last ~30 captured frames."""
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        if span <= 0:
            return 0.0
        return (len(self._fps_window) - 1) / span

    def frames_read(self) -> int:
        return self._frames_read

    def is_running(self) -> bool:
        return self._running

    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        """Open the camera and start the reader thread."""
        with self._device_lock:
            if self._running:
                return True

            if not self._open_device():
                return False

            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, name="GestureOS-Camera", daemon=True)
            self._thread.start()
            logger.info(
                "Camera started (%s, %dx%d @ %sfps)",
                "simulated" if self._simulated else f"device {self.config.get('device_id')}",
                self.resolution[0],
                self.resolution[1],
                self.config.get("fps"),
            )
            return True

    def _open_device(self) -> bool:
        if self._explicit_source is not None:
            self._cap = self._explicit_source
            self._simulated = isinstance(self._explicit_source, SimulatedCamera)
            return True

        if self.config.get("simulate", False):
            return self._open_simulated("simulation requested in config")

        cv2 = _cv2()
        if cv2 is None:
            logger.error("OpenCV (cv2) is not installed - cannot capture frames")
            self._last_error = "OpenCV not installed"
            return self._maybe_simulate_on_failure()

        device_id = self.config.get("device_id", 0)
        backend_name = str(self.config.get("backend", "any")).lower()
        backend_attr = _BACKENDS.get(backend_name)
        backend = getattr(cv2, backend_attr, None) if backend_attr else None

        try:
            cap = cv2.VideoCapture(device_id, backend) if backend else cv2.VideoCapture(device_id)
        except Exception as exc:
            logger.error("Failed to open camera %s: %s", device_id, exc)
            self._last_error = str(exc)
            return self._maybe_simulate_on_failure()

        if not cap.isOpened():
            cap.release()
            msg = f"Camera device {device_id} could not be opened"
            logger.warning(msg)
            self._last_error = msg
            return self._maybe_simulate_on_failure()

        self._apply_properties(cap, cv2)
        self._cap = cap
        self._simulated = False

        # Verify we can actually read a frame (some drivers open but never deliver).
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self._cap = None
            self._last_error = f"Camera device {device_id} opened but returned no frame"
            logger.warning(self._last_error)
            return self._maybe_simulate_on_failure()

        with self._lock:
            self._latest = (frame, time.time())
            self._buffer.append(self._latest)
        return True

    def _apply_properties(self, cap: Any, cv2: Any) -> None:
        wanted = (
            (cv2.CAP_PROP_FRAME_WIDTH, self.config.get("width", 1280)),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.config.get("height", 720)),
            (cv2.CAP_PROP_FPS, self.config.get("fps", 60)),
            (cv2.CAP_PROP_BUFFERSIZE, self.config.get("buffer_size", 1)),
        )
        for prop, value in wanted:
            try:
                cap.set(prop, float(value))
            except Exception as exc:  # pragma: no cover - driver dependent
                logger.debug("Could not set camera property %s: %s", prop, exc)

        fourcc = str(self.config.get("fourcc") or "")
        if fourcc:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
            except Exception as exc:  # pragma: no cover
                logger.debug("Could not set FOURCC %s: %s", fourcc, exc)

        if self.config.get("auto_exposure", True):
            for prop_name, value in (("CAP_PROP_AUTO_EXPOSURE", 3),):
                prop = getattr(cv2, prop_name, None)
                if prop is not None:
                    try:
                        cap.set(prop, value)
                    except Exception:
                        pass
        else:
            prop = getattr(cv2, "CAP_PROP_EXPOSURE", None)
            if prop is not None:
                try:
                    cap.set(prop, float(self.config.get("exposure", -6)))
                except Exception:
                    pass

        for prop_name, key in (
            ("CAP_PROP_BRIGHTNESS", "brightness"),
            ("CAP_PROP_CONTRAST", "contrast"),
            ("CAP_PROP_SATURATION", "saturation"),
        ):
            prop = getattr(cv2, prop_name, None)
            value = self.config.get(key, 0)
            if prop is not None and value:
                try:
                    cap.set(prop, float(value))
                except Exception:
                    pass

    def _open_simulated(self, reason: str) -> bool:
        logger.warning("Using simulated camera source: %s", reason)
        self._cap = SimulatedCamera(
            width=int(self.config.get("width", 640)),
            height=int(self.config.get("height", 480)),
            fps=float(self.config.get("simulate_fps", self.config.get("fps", 30))),
        )
        self._simulated = True
        self._last_error = reason
        return True

    def _maybe_simulate_on_failure(self) -> bool:
        if self.config.get("simulate_on_failure", True):
            return self._open_simulated(self._last_error or "camera unavailable")
        logger.error("Camera unavailable and simulation disabled - cannot start")
        return False

    def stop(self) -> None:
        """Stop the reader thread and release the device."""
        with self._device_lock:
            if not self._running and self._cap is None:
                return
            self._running = False
            self._frame_event.set()

            thread = self._thread
            self._thread = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        with self._lock:
            cap = self._cap
            self._cap = None
            self._buffer.clear()
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:  # pragma: no cover
                logger.debug("Error releasing camera: %s", exc)
        logger.info("Camera stopped")

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Frame access
    # ------------------------------------------------------------------ #
    def _capture_loop(self) -> None:
        target_fps = clamp(float(self.config.get("fps", 60) or 60), 1.0, 240.0)
        frame_budget = 1.0 / target_fps
        while self._running:
            loop_start = time.perf_counter()
            with self._lock:
                cap = self._cap
            if cap is None:
                break
            try:
                ok, frame = cap.read()
            except Exception as exc:
                logger.error("Camera read error: %s", exc)
                self._last_error = str(exc)
                ok, frame = False, None

            if ok and frame is not None:
                now = time.time()
                with self._lock:
                    self._latest = (frame, now)
                    self._buffer.append((frame, now))
                self._frames_read += 1
                self._fps_window.append(now)
                self._frame_event.set()
            else:
                if self._simulated:
                    time.sleep(frame_budget)
                    continue
                logger.warning("Camera read failed - stopping capture")
                self._last_error = self._last_error or "camera read failed"
                self._running = False
                self._frame_event.set()
                break

            elapsed = time.perf_counter() - loop_start
            sleep_for = frame_budget - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def get_latest_frame(self) -> Tuple[Optional[Any], float]:
        """Return the newest frame without blocking: ``(frame_or_None, timestamp)``."""
        with self._lock:
            if self._latest is None:
                return None, 0.0
            return self._latest

    def read(self, timeout: Optional[float] = None) -> Tuple[Optional[Any], float]:
        """Block until a new frame is available (or ``timeout`` seconds pass)."""
        if timeout is None:
            timeout = float(self.config.get("read_timeout", 1.0))
        if not self._running and self._latest is None:
            return None, 0.0

        self._frame_event.wait(timeout=timeout)
        self._frame_event.clear()
        return self.get_latest_frame()

    def read_buffered(self) -> list:
        """Drain and return everything currently in the ring buffer."""
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
        return frames


__all__ = ["CameraCapture", "SimulatedCamera", "DEFAULT_CAMERA_CONFIG"]
