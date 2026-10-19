"""Cursor movement, clicking and drag & drop.

The controller maps a normalised hand position (0..1 from MediaPipe) onto the
virtual desktop, applying - in order:

1. optional horizontal mirror (the camera preview is a selfie view, so the
   cursor should follow your hand, not the raw sensor),
2. ``movement_region`` remap - the usable area of the frame,
3. ``edge_margin`` inset so screen corners remain reachable but not sticky,
4. ``dead_zone`` - sub-pixel jitter suppression,
5. ``smoothing`` - exponential moving average,
6. ``acceleration`` + ``max_speed`` - fast flicks still cover the screen.

All OS interaction goes through :mod:`utils.input_backend`, so the controller
works (and is testable) without a real display.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from utils import clamp, deep_merge, ema, remap, screen_size, try_import
from utils.input_backend import BaseBackend, get_backend

logger = logging.getLogger(__name__)

DEFAULT_CURSOR_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "sensitivity": 1.5,
    "smoothing": 0.3,
    "interpolation": "cubic",
    "edge_margin": 0.05,
    "dead_zone": 0.02,
    "mirror_x": True,
    "movement_region": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9},
    "multi_monitor": True,
    "edge_protection": True,
    "acceleration": 1.2,
    "max_speed": 3000,
    "double_click_interval": 0.3,
    "drag_button": "left",
    "click_cooldown": 0.05,
}


class MouseController:
    """Moves the OS cursor and performs clicks / drags from hand landmarks."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        config = config or {}
        # main.py passes the whole config dict; accept a bare cursor dict too.
        cursor_cfg = config.get("cursor", config) if isinstance(config, dict) else {}
        cursor_mode = config.get("cursor_mode", {}) if isinstance(config, dict) else {}
        gestures = config.get("gestures", {}) if isinstance(config, dict) else {}

        merged = deep_merge(dict(DEFAULT_CURSOR_CONFIG), cursor_mode or {})
        merged = deep_merge(merged, cursor_cfg or {})
        # The gestures section is only a fallback: an explicit cursor.* value wins.
        if "double_click_interval" in gestures and "double_click_interval" not in (cursor_cfg or {}):
            merged["double_click_interval"] = gestures["double_click_interval"]
        self.config: Dict[str, Any] = merged

        self.backend: BaseBackend = backend or get_backend()
        self.screen_bounds: Tuple[int, int, int, int] = self._detect_screen_bounds()

        self._smoothed: Optional[Tuple[float, float]] = None
        self._last_target: Optional[Tuple[int, int]] = None
        self._dragging = False
        self._lock = threading.Lock()
        self._last_click_at = 0.0
        self._last_action = "none"

        # Double-click coalescing: a single click is delayed by
        # double_click_interval so a following pinch can become a double click.
        self._pending_timer: Optional[threading.Timer] = None
        self._click_count = 0
        self._double_click_count = 0
        self._right_click_count = 0
        self._moves = 0
        self._path: Deque[Tuple[float, int, int]] = deque(maxlen=60)

        self.enabled = bool(self.config.get("enabled", True))
        logger.info(
            "Mouse controller ready (screen=%dx%d at %d,%d, backend=%s)",
            self.screen_bounds[2], self.screen_bounds[3],
            self.screen_bounds[0], self.screen_bounds[1],
            self.backend.name,
        )

    # ------------------------------------------------------------------ #
    # Screen geometry
    # ------------------------------------------------------------------ #
    def _detect_screen_bounds(self) -> Tuple[int, int, int, int]:
        """Virtual desktop bounds ``(left, top, width, height)``."""
        if self.config.get("multi_monitor", True):
            screeninfo = try_import("screeninfo")
            if screeninfo is not None:
                try:
                    monitors = [m for m in screeninfo.get_monitors() if m.width and m.height]
                    if monitors:
                        left = min(int(m.x or 0) for m in monitors)
                        top = min(int(m.y or 0) for m in monitors)
                        right = max(int(m.x or 0) + int(m.width) for m in monitors)
                        bottom = max(int(m.y or 0) + int(m.height) for m in monitors)
                        return left, top, right - left, bottom - top
                except Exception as exc:  # pragma: no cover - platform dependent
                    logger.debug("screeninfo multi-monitor detection failed: %s", exc)

        width, height = self.backend.size() if self.backend.available else screen_size()
        if not width or not height:
            width, height = screen_size()
        return 0, 0, int(width), int(height)

    def refresh_screen_bounds(self) -> Tuple[int, int, int, int]:
        """Re-detect monitors (called after a display change)."""
        self.screen_bounds = self._detect_screen_bounds()
        return self.screen_bounds

    # ------------------------------------------------------------------ #
    # Tuning
    # ------------------------------------------------------------------ #
    def set_sensitivity(self, value: float) -> None:
        self.config["sensitivity"] = clamp(float(value), 0.1, 10.0)

    def set_smoothing(self, value: float) -> None:
        self.config["smoothing"] = clamp(float(value), 0.0, 0.95)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    @property
    def sensitivity(self) -> float:
        return float(self.config.get("sensitivity", 1.5))

    @property
    def smoothing(self) -> float:
        return float(self.config.get("smoothing", 0.3))

    @property
    def last_action(self) -> str:
        return self._last_action

    def stats(self) -> Dict[str, Any]:
        return {
            "moves": self._moves,
            "clicks": self._click_count,
            "double_clicks": self._double_click_count,
            "right_clicks": self._right_click_count,
            "dragging": self._dragging,
            "position": self.position(),
        }

    def position(self) -> Tuple[int, int]:
        try:
            return self.backend.position()
        except Exception:  # pragma: no cover
            return self._last_target or (0, 0)

    # ------------------------------------------------------------------ #
    # Mapping
    # ------------------------------------------------------------------ #
    def normalized_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Map a normalised hand position to absolute screen pixels."""
        left, top, width, height = self.screen_bounds

        x = 1.0 - clamp(norm_x, 0.0, 1.0) if self.config.get("mirror_x", True) else clamp(norm_x, 0.0, 1.0)
        y = clamp(norm_y, 0.0, 1.0)

        region = self.config.get("movement_region") or {}
        x_min = float(region.get("x_min", 0.0))
        x_max = float(region.get("x_max", 1.0))
        y_min = float(region.get("y_min", 0.0))
        y_max = float(region.get("y_max", 1.0))
        if x_max - x_min > 1e-6:
            x = clamp(remap(x, x_min, x_max, 0.0, 1.0), 0.0, 1.0)
        if y_max - y_min > 1e-6:
            y = clamp(remap(y, y_min, y_max, 0.0, 1.0), 0.0, 1.0)

        margin = clamp(float(self.config.get("edge_margin", 0.0)), 0.0, 0.4)
        if margin > 0:
            x = remap(x, 0.0, 1.0, margin, 1.0 - margin)
            y = remap(y, 0.0, 1.0, margin, 1.0 - margin)

        return left + int(round(x * (width - 1))), top + int(round(y * (height - 1)))

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #
    def move_cursor(self, norm_x: float, norm_y: float) -> Optional[Tuple[int, int]]:
        """Move the cursor to a normalised hand position. Returns the target."""
        if not self.enabled or norm_x is None or norm_y is None:
            return None

        target = self.normalized_to_screen(float(norm_x), float(norm_y))

        with self._lock:
            if self._smoothed is None:
                self._smoothed = (float(target[0]), float(target[1]))
            else:
                factor = self.smoothing
                self._smoothed = (
                    ema(self._smoothed[0], float(target[0]), factor),
                    ema(self._smoothed[1], float(target[1]), factor),
                )

            sx, sy = self._smoothed
            if self._last_target is not None:
                dx = sx - self._last_target[0]
                dy = sy - self._last_target[1]
                delta = math.hypot(dx, dy)

                # Dead zone: ignore jitter below the threshold.
                dead_zone_px = float(self.config.get("dead_zone", 0.02)) * max(
                    self.screen_bounds[2], self.screen_bounds[3]
                ) * 0.05
                if delta < max(dead_zone_px, 0.5):
                    return self._last_target

                # Acceleration: larger motions get a boost, capped by max_speed.
                accel = float(self.config.get("acceleration", 1.0))
                if accel > 1.0:
                    boost = 1.0 + (accel - 1.0) * clamp(delta / 120.0, 0.0, 1.0)
                    sx = self._last_target[0] + dx * boost
                    sy = self._last_target[1] + dy * boost

            max_speed = float(self.config.get("max_speed", 3000))
            if self._last_target is not None and max_speed > 0:
                dx = sx - self._last_target[0]
                dy = sy - self._last_target[1]
                delta = math.hypot(dx, dy)
                if delta > max_speed:
                    scale = max_speed / delta
                    sx = self._last_target[0] + dx * scale
                    sy = self._last_target[1] + dy * scale

            x = int(round(clamp(sx, self.screen_bounds[0], self.screen_bounds[0] + self.screen_bounds[2] - 1)))
            y = int(round(clamp(sy, self.screen_bounds[1], self.screen_bounds[1] + self.screen_bounds[3] - 1)))

            if self.config.get("edge_protection", True):
                x = int(clamp(x, self.screen_bounds[0] + 1, self.screen_bounds[0] + self.screen_bounds[2] - 2))
                y = int(clamp(y, self.screen_bounds[1] + 1, self.screen_bounds[1] + self.screen_bounds[3] - 2))

            self._smoothed = (float(x), float(y))
            self._last_target = (x, y)
            self._moves += 1
            self._path.append((time.time(), x, y))

        try:
            self.backend.move_to(x, y)
            self._last_action = "move"
            if self._dragging:
                self._last_action = "drag"
        except Exception as exc:
            logger.error("Cursor move failed: %s", exc)
            return None
        return (x, y)

    def move_cursor_pixels(self, x: int, y: int) -> Tuple[int, int]:
        """Absolute move in screen pixels (used by window dragging)."""
        left, top, width, height = self.screen_bounds
        x = int(clamp(x, left, left + width - 1))
        y = int(clamp(y, top, top + height - 1))
        with self._lock:
            self._smoothed = (float(x), float(y))
            self._last_target = (x, y)
            self._moves += 1
        self.backend.move_to(x, y)
        return x, y

    def velocity(self) -> float:
        """Recent cursor speed in px/s (used by analytics and swipe tuning)."""
        if len(self._path) < 2:
            return 0.0
        t0, x0, y0 = self._path[0]
        t1, x1, y1 = self._path[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / dt

    # ------------------------------------------------------------------ #
    # Clicks
    # ------------------------------------------------------------------ #
    def _click_gate(self) -> bool:
        now = time.time()
        cooldown = float(self.config.get("click_cooldown", 0.05))
        if now - self._last_click_at < cooldown:
            return False
        self._last_click_at = now
        return True

    def left_click(self, double: bool = False) -> bool:
        """Left click.

        Single clicks are delayed by ``double_click_interval`` so that a second
        pinch inside that window is coalesced into a real double click instead
        of firing ``click, click``.
        """
        if not self.enabled:
            return False
        if double:
            self.double_click()
            return True

        with self._lock:
            if self._pending_timer is not None:
                # Second click within the interval -> promote to double click.
                self._pending_timer.cancel()
                self._pending_timer = None
                promote = True
            else:
                promote = False

        if promote:
            return self.double_click()

        interval = float(self.config.get("double_click_interval", 0.3))
        if interval <= 0:
            return self._single_click()

        timer = threading.Timer(interval, self._flush_single_click)
        timer.daemon = True
        with self._lock:
            self._pending_timer = timer
        timer.start()
        return True

    def _flush_single_click(self) -> None:
        with self._lock:
            self._pending_timer = None
        self._single_click()

    def _single_click(self) -> bool:
        if not self._click_gate():
            return False
        try:
            self.backend.click("left", 1)
            self._click_count += 1
            self._last_action = "left_click"
            return True
        except Exception as exc:
            logger.error("Left click failed: %s", exc)
            return False

    def double_click(self) -> bool:
        with self._lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
        if not self._click_gate():
            return False
        try:
            self.backend.click("left", 2)
            self._double_click_count += 1
            self._last_action = "double_click"
            return True
        except Exception as exc:
            logger.error("Double click failed: %s", exc)
            return False

    def right_click(self) -> bool:
        if not self.enabled or not self._click_gate():
            return False
        try:
            self.backend.click("right", 1)
            self._right_click_count += 1
            self._last_action = "right_click"
            return True
        except Exception as exc:
            logger.error("Right click failed: %s", exc)
            return False

    def triple_click(self) -> bool:
        """Triple click - selects the paragraph under the cursor."""
        with self._lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
        if not self._click_gate():
            return False
        try:
            self.backend.click("left", 3)
            self._last_action = "triple_click"
            return True
        except Exception as exc:
            logger.error("Triple click failed: %s", exc)
            return False

    def middle_click(self) -> bool:
        if not self.enabled or not self._click_gate():
            return False
        try:
            self.backend.click("middle", 1)
            self._last_action = "middle_click"
            return True
        except Exception as exc:
            logger.error("Middle click failed: %s", exc)
            return False

    def cancel_pending_click(self) -> None:
        """Drop a queued single click (used when a drag starts)."""
        with self._lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None

    # ------------------------------------------------------------------ #
    # Drag & drop
    # ------------------------------------------------------------------ #
    def start_drag(self, button: Optional[str] = None) -> bool:
        if self._dragging:
            return False
        self.cancel_pending_click()
        button = button or str(self.config.get("drag_button", "left"))
        try:
            self.backend.mouse_down(button)
            self._dragging = True
            self._last_action = "drag_start"
            logger.debug("Drag started (%s button)", button)
            return True
        except Exception as exc:
            logger.error("Drag start failed: %s", exc)
            return False

    def update_drag(self, norm_x: float, norm_y: float) -> Optional[Tuple[int, int]]:
        if not self._dragging:
            return self.move_cursor(norm_x, norm_y)
        # Dragging bypasses smoothing/acceleration for a 1:1 feel.
        target = self.normalized_to_screen(float(norm_x), float(norm_y))
        try:
            self.backend.move_to(*target)
            self._last_target = target
            self._last_action = "drag"
            return target
        except Exception as exc:
            logger.error("Drag move failed: %s", exc)
            return None

    def end_drag(self) -> bool:
        if not self._dragging:
            return False
        button = str(self.config.get("drag_button", "left"))
        try:
            self.backend.mouse_up(button)
        except Exception as exc:
            logger.error("Drag end failed: %s", exc)
        finally:
            self._dragging = False
            self._last_action = "drag_end"
        return True

    def toggle_drag(self) -> bool:
        """Start or stop a drag. Returns the resulting state (True = dragging)."""
        if self._dragging:
            self.end_drag()
        else:
            self.start_drag()
        return self._dragging

    def is_dragging(self) -> bool:
        return self._dragging

    # ------------------------------------------------------------------ #
    # Scrolling
    # ------------------------------------------------------------------ #
    def scroll(self, clicks: int, horizontal: bool = False) -> bool:
        """Scroll by ``clicks`` (negative = down/left)."""
        if clicks == 0:
            return False
        try:
            self.backend.scroll(int(clicks), horizontal=horizontal)
            self._last_action = "scroll"
            return True
        except Exception as exc:
            logger.error("Scroll failed: %s", exc)
            return False

    def zoom(self, steps: int) -> bool:
        """Ctrl + scroll - the near-universal zoom shortcut."""
        if steps == 0:
            return False
        try:
            self.backend.hotkey("ctrl")  # no-op guard for backends without modifier hold
        except Exception:
            pass
        try:
            pg = try_import("pyautogui")
            if pg is not None and self.backend.name == "pyautogui":
                pg.keyDown("ctrl")
                try:
                    pg.scroll(int(steps))
                finally:
                    pg.keyUp("ctrl")
            else:
                # Recording / fallback backends: express intent explicitly.
                self.backend.hotkey("ctrl", "plus" if steps > 0 else "minus")
            self._last_action = "zoom"
            return True
        except Exception as exc:
            logger.error("Zoom failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Keyboard passthrough (used by controllers and macros)
    # ------------------------------------------------------------------ #
    def press(self, key: str) -> bool:
        try:
            self.backend.press(key)
            self._last_action = f"press:{key}"
            return True
        except Exception as exc:
            logger.error("Key press '%s' failed: %s", key, exc)
            return False

    def hotkey(self, *keys: str) -> bool:
        try:
            self.backend.hotkey(*keys)
            self._last_action = "hotkey:" + "+".join(keys)
            return True
        except Exception as exc:
            logger.error("Hotkey %s failed: %s", keys, exc)
            return False

    def write(self, text: str) -> bool:
        try:
            self.backend.write(text)
            self._last_action = "write"
            return True
        except Exception as exc:
            logger.error("Text write failed: %s", exc)
            return False

    def shutdown(self) -> None:
        """Release held buttons and pending timers (called on exit)."""
        self.cancel_pending_click()
        if self._dragging:
            self.end_drag()


__all__ = ["MouseController", "DEFAULT_CURSOR_CONFIG"]
