"""Scroll gesture controller.

While GestureOS is in *scroll mode* (index + middle finger extended) the palm
position drives scrolling.  Movement is accumulated and converted into discrete
wheel clicks so scrolling feels continuous but never over-scrolls.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from utils import clamp, deep_merge
from utils.input_backend import BaseBackend, get_backend

logger = logging.getLogger(__name__)

DEFAULT_SCROLL_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "sensitivity": 1.0,
    "pixels_per_click": 0.02,      # normalised palm travel per wheel click
    "dead_zone": 0.004,            # ignore tiny jitter
    "direction": "natural",        # "natural" (hand down = scroll down) or "inverted"
    "horizontal": True,
    "horizontal_pixels_per_click": 0.03,
    "max_clicks_per_frame": 6,
    "cooldown": 0.016,             # ~60 Hz cap on wheel events
    "axis_lock_ratio": 1.5,        # dominant axis must win by this factor
    "idle_reset": 1.0,             # seconds without movement before re-anchoring
}


class ScrollController:
    """Converts palm motion into mouse-wheel events."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[BaseBackend] = None,
        mouse: Optional[Any] = None,
    ) -> None:
        config = config or {}
        scroll_cfg = config.get("scroll", config) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_SCROLL_CONFIG), scroll_cfg or {})

        # Prefer the shared mouse controller (keeps one input path), else backend.
        self.mouse = mouse
        self.backend: BaseBackend = backend or get_backend()

        self._anchor: Optional[tuple] = None
        self._accumulator_x = 0.0
        self._accumulator_y = 0.0
        self._last_emit = 0.0
        self._last_move = 0.0
        self._active = False
        self.total_vertical = 0
        self.total_horizontal = 0

    # ------------------------------------------------------------------ #
    # Tuning
    # ------------------------------------------------------------------ #
    def set_sensitivity(self, value: float) -> None:
        self.config["sensitivity"] = clamp(float(value), 0.1, 5.0)

    def set_direction(self, direction: str) -> None:
        self.config["direction"] = "inverted" if str(direction).lower().startswith("inv") else "natural"

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        """Re-anchor (called when no hand is visible or scroll mode ends)."""
        self._anchor = None
        self._accumulator_x = 0.0
        self._accumulator_y = 0.0
        self._active = False

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #
    def process(self, palm_y: float, palm_x: float) -> Dict[str, int]:
        """Feed the current palm position (normalised). Returns emitted clicks."""
        result = {"vertical": 0, "horizontal": 0}
        if not self.config.get("enabled", True):
            return result

        now = time.time()
        if self._anchor is None or now - self._last_move > float(self.config["idle_reset"]):
            self._anchor = (float(palm_x), float(palm_y))
            self._accumulator_x = 0.0
            self._accumulator_y = 0.0
            self._active = True
            self._last_move = now
            return result

        ax, ay = self._anchor
        dx = float(palm_x) - ax
        dy = float(palm_y) - ay
        self._last_move = now

        dead_zone = float(self.config["dead_zone"])
        sensitivity = float(self.config.get("sensitivity", 1.0))

        if abs(dx) < dead_zone:
            dx = 0.0
        if abs(dy) < dead_zone:
            dy = 0.0

        if dx == 0.0 and dy == 0.0:
            return result

        inverted = self.config.get("direction") == "inverted"
        self._accumulator_y += dy * sensitivity * (-1.0 if inverted else 1.0)
        self._accumulator_x += dx * sensitivity * (-1.0 if inverted else 1.0)

        if now - self._last_emit < float(self.config["cooldown"]):
            return result

        per_click_y = max(float(self.config["pixels_per_click"]), 1e-4)
        clicks_y = int(self._accumulator_y / per_click_y)
        if clicks_y:
            self._accumulator_y -= clicks_y * per_click_y
            clicks_y = int(clamp(clicks_y, -self.config["max_clicks_per_frame"],
                                 self.config["max_clicks_per_frame"]))
            # Wheel convention: positive clicks scroll up, hand moving down = content down.
            wheel = -clicks_y * 3
            self._emit_scroll(wheel, horizontal=False)
            result["vertical"] = wheel
            self.total_vertical += abs(wheel)

        if self.config.get("horizontal", True):
            per_click_x = max(float(self.config["horizontal_pixels_per_click"]), 1e-4)
            clicks_x = int(self._accumulator_x / per_click_x)
            if clicks_x and abs(dx) * float(self.config["axis_lock_ratio"]) > abs(dy):
                self._accumulator_x -= clicks_x * per_click_x
                clicks_x = int(clamp(clicks_x, -self.config["max_clicks_per_frame"],
                                     self.config["max_clicks_per_frame"]))
                self._emit_scroll(clicks_x * 3, horizontal=True)
                result["horizontal"] = clicks_x * 3
                self.total_horizontal += abs(clicks_x * 3)

        # Re-anchor to the current position so motion is relative, not absolute.
        self._anchor = (float(palm_x), float(palm_y))
        if result["vertical"] or result["horizontal"]:
            self._last_emit = now
        return result

    def _emit_scroll(self, clicks: int, horizontal: bool = False) -> None:
        if clicks == 0:
            return
        try:
            if self.mouse is not None and hasattr(self.mouse, "scroll"):
                self.mouse.scroll(clicks, horizontal=horizontal) if horizontal else self.mouse.scroll(clicks)
            else:
                self.backend.scroll(clicks, horizontal=horizontal)
        except TypeError:
            # MouseController.scroll may not accept `horizontal` if overridden.
            self.backend.scroll(clicks, horizontal=horizontal)
        except Exception as exc:
            logger.error("Scroll emit failed: %s", exc)

    def stats(self) -> Dict[str, Any]:
        return {
            "vertical": self.total_vertical,
            "horizontal": self.total_horizontal,
            "active": self._active,
        }


__all__ = ["ScrollController", "DEFAULT_SCROLL_CONFIG"]
