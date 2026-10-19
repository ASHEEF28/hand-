"""Media, volume, brightness, zoom and rotation controllers.

* :class:`MediaController`      - volume (pycaw on Windows, media keys elsewhere)
  and transport keys (play/pause, next, previous).
* :class:`BrightnessController` - display brightness, either absolute (via
  ``screen-brightness-control``) or driven by pinch distance.
* :class:`ZoomController`       - two-hand pinch zoom -> Ctrl + wheel.
* :class:`RotationController`   - wrist rotation from the palm axis angle.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from utils import IS_WINDOWS, clamp, deep_merge, signed_rotation, try_import
from utils.input_backend import BaseBackend, get_backend

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "use_pycaw": True,
    "volume_step": 5,               # percent per gesture
    "volume_key_step": 2,           # media-key presses per gesture
    "action_cooldown": 0.25,
    "max_volume": 100,
}

DEFAULT_BRIGHTNESS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "use_screen_brightness_control": True,
    "step": 5,
    "min_pinch_distance": 0.02,     # normalised pinch -> 0% brightness
    "max_pinch_distance": 0.22,     # normalised pinch -> 100% brightness
    "smoothing": 0.5,
    "update_cooldown": 0.15,
    "hotkey_fallback": True,
}

DEFAULT_ZOOM_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_distance": 0.08,
    "max_distance": 0.9,
    "step_threshold": 0.03,
    "sensitivity": 1.0,
    "smoothing": 0.4,
    "cooldown": 0.15,
    "max_steps_per_event": 4,
}

DEFAULT_ROTATION_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "rotation_threshold": 15.0,     # degrees of accumulated wrist rotation
    "sensitivity": 1.0,
    "cooldown": 0.4,
    "smoothing": 0.35,
    "reset_on_idle": 0.8,           # seconds without a hand before re-anchoring
}


def _point(value: Any) -> Tuple[float, float]:
    """Accept a Landmark, an ``(x, y)`` tuple or a 3-tuple."""
    if value is None:
        return (0.0, 0.0)
    if hasattr(value, "x") and hasattr(value, "y"):
        return (float(value.x), float(value.y))
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    raise TypeError(f"Cannot interpret {value!r} as a 2D point")


class MediaController:
    """Volume and media transport control."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        config = config or {}
        section = config.get("media", config) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_MEDIA_CONFIG), section or {})
        self.backend: BaseBackend = backend or get_backend()
        self._cooldowns: Dict[str, float] = {}
        self._volume_interface = None
        self._volume_initialized = False
        self._muted = False
        self.history: List[Dict[str, Any]] = []
        if self.config.get("use_pycaw", True):
            self._init_pycaw()

    # ------------------------------------------------------------------ #
    # pycaw (Windows Core Audio)
    # ------------------------------------------------------------------ #
    def _init_pycaw(self) -> None:
        if self._volume_initialized:
            return
        self._volume_initialized = True
        if not IS_WINDOWS:
            return
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            logger.info("pycaw audio endpoint ready")
        except Exception as exc:
            logger.debug("pycaw unavailable (%s) - falling back to media keys", exc)
            self._volume_interface = None

    @property
    def has_native_volume(self) -> bool:
        return self._volume_interface is not None

    def get_volume(self) -> Optional[int]:
        """Master volume as a percentage, or ``None`` when unavailable."""
        if self._volume_interface is None:
            return None
        try:
            return int(round(self._volume_interface.GetMasterVolumeLevelScalar() * 100))
        except Exception as exc:
            logger.debug("get_volume failed: %s", exc)
            return None

    def set_volume(self, percent: float) -> bool:
        percent = clamp(float(percent), 0.0, float(self.config.get("max_volume", 100)))
        if self._volume_interface is None:
            return False
        try:
            self._volume_interface.SetMasterVolumeLevelScalar(percent / 100.0, None)
            self._record("set_volume", percent=percent)
            return True
        except Exception as exc:
            logger.error("set_volume failed: %s", exc)
            return False

    def _gate(self, action: str) -> bool:
        now = time.time()
        cooldown = float(self.config.get("action_cooldown", 0.25))
        last = self._cooldowns.get(action)
        if last is not None and now - last < cooldown:
            return False
        self._cooldowns[action] = now
        return True

    def _record(self, action: str, **payload: Any) -> None:
        self.history.append({"action": action, "t": time.time(), **payload})

    # ------------------------------------------------------------------ #
    # Volume gestures
    # ------------------------------------------------------------------ #
    def volume_up(self, step: Optional[float] = None) -> bool:
        return self._adjust_volume(+1, step)

    def volume_down(self, step: Optional[float] = None) -> bool:
        return self._adjust_volume(-1, step)

    def _adjust_volume(self, direction: int, step: Optional[float] = None) -> bool:
        action = "volume_up" if direction > 0 else "volume_down"
        if not self._gate(action):
            return False

        if self._volume_interface is not None:
            percent = float(step if step is not None else self.config.get("volume_step", 5))
            current = self.get_volume()
            if current is None:
                current = 50
            return self.set_volume(current + direction * percent)

        presses = int(step if step is not None else self.config.get("volume_key_step", 2))
        key = "volumeup" if direction > 0 else "volumedown"
        for _ in range(max(1, presses)):
            self.backend.press(key)
        self._record(action, presses=presses)
        return True

    def mute_toggle(self) -> bool:
        if not self._gate("mute"):
            return False
        if self._volume_interface is not None:
            try:
                current = self._volume_interface.GetMute()
                self._volume_interface.SetMute(0 if current else 1, None)
                self._muted = not current
                self._record("mute_toggle", muted=self._muted)
                return True
            except Exception as exc:
                logger.debug("Native mute failed (%s) - using media key", exc)
        self.backend.press("volumemute")
        self._record("mute_toggle", key="volumemute")
        return True

    def is_muted(self) -> Optional[bool]:
        if self._volume_interface is not None:
            try:
                return bool(self._volume_interface.GetMute())
            except Exception:
                return None
        return self._muted

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def play_pause(self) -> bool:
        if not self._gate("play_pause"):
            return False
        self.backend.press("playpause")
        self._record("play_pause")
        return True

    def next_track(self) -> bool:
        if not self._gate("next"):
            return False
        self.backend.press("nexttrack")
        self._record("next_track")
        return True

    def previous_track(self) -> bool:
        if not self._gate("prev"):
            return False
        self.backend.press("prevtrack")
        self._record("previous_track")
        return True

    def stop(self) -> bool:
        if not self._gate("stop"):
            return False
        self.backend.press("stop")
        self._record("stop")
        return True


class BrightnessController:
    """Display brightness, including continuous pinch-distance control."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        config = config or {}
        section = config.get("brightness", config) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_BRIGHTNESS_CONFIG), section or {})
        self.backend: BaseBackend = backend or get_backend()
        self._sbc = None
        if self.config.get("use_screen_brightness_control", True):
            self._sbc = try_import("screen_brightness_control")
        self._smoothed: Optional[float] = None
        self._last_update = 0.0
        self._last_percent: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    @property
    def has_native_brightness(self) -> bool:
        return self._sbc is not None

    def get_brightness(self) -> Optional[int]:
        if self._sbc is None:
            return self._last_percent
        try:
            values = self._sbc.get_brightness()
            if isinstance(values, (list, tuple)) and values:
                return int(values[0])
            return int(values)
        except Exception as exc:
            logger.debug("get_brightness failed: %s", exc)
            return self._last_percent

    def set_brightness(self, percent: float) -> bool:
        percent = clamp(float(percent), 0.0, 100.0)
        self._last_percent = int(round(percent))
        if self._sbc is None:
            return False
        try:
            self._sbc.set_brightness(self._last_percent)
            self.history.append({"action": "set_brightness", "percent": self._last_percent, "t": time.time()})
            return True
        except Exception as exc:
            logger.error("set_brightness failed: %s", exc)
            return False

    def step(self, delta: float) -> bool:
        """Relative change in percentage points."""
        current = self.get_brightness()
        if current is None:
            current = 50
        target = clamp(current + float(delta), 0.0, 100.0)
        if self.set_brightness(target):
            return True
        if self.config.get("hotkey_fallback", True):
            key = "brightnessup" if delta > 0 else "brightnessdown"
            for _ in range(max(1, int(abs(delta) / max(self.config.get("step", 5), 1)))):
                self.backend.press(key)
            self.history.append({"action": "brightness_key", "key": key, "t": time.time()})
            return True
        return False

    def brightness_up(self, amount: Optional[float] = None) -> bool:
        return self.step(+float(amount if amount is not None else self.config.get("step", 5)))

    def brightness_down(self, amount: Optional[float] = None) -> bool:
        return self.step(-float(amount if amount is not None else self.config.get("step", 5)))

    def update(self, pinch_distance: float) -> Optional[int]:
        """Continuous control: map thumb-index distance onto brightness.

        Called from ``main.py`` with ``hand.gesture_data['pinch_distance']``.
        Returns the percentage applied (or ``None`` while cooling down).
        """
        min_d = float(self.config.get("min_pinch_distance", 0.02))
        max_d = float(self.config.get("max_pinch_distance", 0.22))
        raw = clamp((float(pinch_distance) - min_d) / max(max_d - min_d, 1e-6), 0.0, 1.0) * 100.0

        factor = clamp(float(self.config.get("smoothing", 0.5)), 0.0, 0.95)
        self._smoothed = raw if self._smoothed is None else self._smoothed * factor + raw * (1.0 - factor)

        now = time.time()
        if now - self._last_update < float(self.config.get("update_cooldown", 0.15)):
            return None
        # Skip updates that would not change the reported integer percentage.
        if self._last_percent is not None and abs(self._smoothed - self._last_percent) < 1.0:
            return None
        self._last_update = now
        self.set_brightness(self._smoothed)
        return self._last_percent


class ZoomController:
    """Two-hand pinch zoom.

    ``main.py`` calls :meth:`update` with the two palm centres each frame; the
    controller returns ``'zoom_in'`` / ``'zoom_out'`` / ``None`` and can also
    apply the zoom itself (Ctrl + wheel) when a mouse controller is attached.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mouse: Optional[Any] = None,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        config = config or {}
        section = config.get("zoom", config) if isinstance(config, dict) else {}
        gestures = (config.get("gestures", {}) if isinstance(config, dict) else {}) or {}
        merged = deep_merge(dict(DEFAULT_ZOOM_CONFIG), section or {})
        # zoom.* wins over the legacy gestures.zoom_sensitivity key.
        if "zoom_sensitivity" in gestures and "sensitivity" not in (section or {}):
            merged["sensitivity"] = gestures["zoom_sensitivity"]
        self.config: Dict[str, Any] = merged
        self.mouse = mouse
        self.backend: BaseBackend = backend or get_backend()
        self._smoothed: Optional[float] = None
        self._last_event = 0.0
        self._accumulator = 0.0
        self.level = 1.0
        self.last_direction: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._smoothed = None
        self._accumulator = 0.0

    def distance_between(self, c1: Any, c2: Any) -> float:
        p1, p2 = _point(c1), _point(c2)
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def update(self, center1: Any, center2: Any, apply: bool = True) -> Optional[str]:
        """Track the gap between two palms and emit discrete zoom steps."""
        if not self.config.get("enabled", True):
            return None
        gap = self.distance_between(center1, center2)
        min_d = float(self.config["min_distance"])
        max_d = float(self.config["max_distance"])
        normalized = clamp((gap - min_d) / max(max_d - min_d, 1e-6), 0.0, 1.0)

        factor = clamp(float(self.config.get("smoothing", 0.4)), 0.0, 0.95)
        if self._smoothed is None:
            self._smoothed = normalized
            self.level = normalized
            return None

        previous = self._smoothed
        self._smoothed = previous * factor + normalized * (1.0 - factor)
        delta = (self._smoothed - previous) * float(self.config.get("sensitivity", 1.0))
        self._accumulator += delta

        threshold = float(self.config["step_threshold"])
        direction: Optional[str] = None
        if abs(self._accumulator) >= threshold:
            steps = int(self._accumulator / threshold)
            steps = int(clamp(steps, -self.config["max_steps_per_event"], self.config["max_steps_per_event"]))
            self._accumulator -= steps * threshold
            if steps != 0:
                direction = "zoom_in" if steps > 0 else "zoom_out"
                self.level = clamp(self.level + steps * 0.05, 0.1, 5.0)
                self.last_direction = direction
                self.history.append({"direction": direction, "steps": steps, "t": time.time()})
                if apply:
                    self.apply(direction, steps)
        return direction

    def apply(self, direction: str, steps: int = 1) -> bool:
        """Perform Ctrl + wheel zoom (or record intent on dry-run backends)."""
        if self.mouse is not None and hasattr(self.mouse, "zoom"):
            magnitude = abs(int(steps)) * (120 if direction == "zoom_in" else -120)
            return bool(self.mouse.zoom(magnitude // 120 * 3))
        try:
            self.backend.hotkey("ctrl", "plus" if direction == "zoom_in" else "minus")
            return True
        except Exception as exc:
            logger.error("Zoom apply failed: %s", exc)
            return False

    def set_level(self, level: float) -> None:
        self.level = clamp(float(level), 0.1, 5.0)


class RotationController:
    """Wrist rotation from the palm axis (index MCP -> pinky MCP)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        section = config.get("rotation", config) if isinstance(config, dict) else {}
        gestures = (config.get("gestures", {}) if isinstance(config, dict) else {}) or {}
        merged = deep_merge(dict(DEFAULT_ROTATION_CONFIG), section or {})
        # rotation.* wins over the legacy gestures.rotation_* keys.
        section = section or {}
        if "rotation_threshold" in gestures and "rotation_threshold" not in section:
            merged["rotation_threshold"] = gestures["rotation_threshold"]
        if "rotation_sensitivity" in gestures and "sensitivity" not in section:
            merged["sensitivity"] = gestures["rotation_sensitivity"]
        self.config: Dict[str, Any] = merged
        self._previous_angle: Optional[float] = None
        self._accumulator = 0.0
        self._last_seen = 0.0
        self._last_event = 0.0
        self.total_rotation = 0.0
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._previous_angle = None
        self._accumulator = 0.0
        self.total_rotation = 0.0

    @staticmethod
    def palm_angle(wrist: Any, index_mcp: Any, pinky_mcp: Any) -> float:
        """Angle (degrees) of the palm axis relative to the wrist."""
        _ = _point(wrist)  # wrist is kept in the signature for future tilt use
        p_index = _point(index_mcp)
        p_pinky = _point(pinky_mcp)
        return math.degrees(math.atan2(p_pinky[1] - p_index[1], p_pinky[0] - p_index[0]))

    def update(self, wrist: Any, index_mcp: Any, pinky_mcp: Any) -> Optional[str]:
        """Accumulate wrist rotation; return ``'clockwise'``/``'counter_clockwise'``."""
        if not self.config.get("enabled", True):
            return None

        now = time.time()
        if self._last_seen and now - self._last_seen > float(self.config["reset_on_idle"]):
            self.reset()
        self._last_seen = now

        angle = self.palm_angle(wrist, index_mcp, pinky_mcp)
        if self._previous_angle is None:
            self._previous_angle = angle
            return None

        delta = signed_rotation(self._previous_angle, angle) * float(self.config.get("sensitivity", 1.0))
        self._previous_angle = angle

        # Ignore jitter.
        if abs(delta) < 0.35:
            return None

        self._accumulator += delta
        self.total_rotation += delta

        threshold = float(self.config["rotation_threshold"])
        if abs(self._accumulator) < threshold:
            return None
        if now - self._last_event < float(self.config["cooldown"]):
            return None

        self._last_event = now
        direction = "clockwise" if self._accumulator > 0 else "counter_clockwise"
        magnitude = self._accumulator
        self._accumulator = 0.0
        self.history.append({"direction": direction, "degrees": magnitude, "t": now})
        return direction

    @property
    def accumulated(self) -> float:
        return self._accumulator


__all__ = [
    "MediaController",
    "BrightnessController",
    "ZoomController",
    "RotationController",
    "DEFAULT_MEDIA_CONFIG",
    "DEFAULT_BRIGHTNESS_CONFIG",
    "DEFAULT_ZOOM_CONFIG",
    "DEFAULT_ROTATION_CONFIG",
]
