"""Radial AR overlay drawn on the camera feed.

Renders an in-video menu anchored to the palm, so actions can be picked without
looking away from the screen.  Everything is drawn with OpenCV (no Qt needed),
which keeps it usable in headless mode too.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils import clamp, deep_merge, try_import

logger = logging.getLogger(__name__)

DEFAULT_AR_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "menu_radius": 80,
    "opacity": 0.7,
    "color": (0, 212, 255),
    "highlight_color": (0, 230, 118),
    "text_color": (255, 255, 255),
    "font_scale": 0.5,
    "thickness": 2,
    "show_hud": True,
    "hud_items": ("fps", "mode", "gesture", "latency"),
    "start_angle": -90.0,
}


class AROverlay:
    """Draws a radial menu / HUD around the tracked palm."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_AR_CONFIG), config or {})
        self.visible = bool(self.config.get("enabled", False))
        self.last_items: List[str] = []
        self.last_selected = 0

    # ------------------------------------------------------------------ #
    def set_enabled(self, enabled: bool) -> None:
        self.visible = bool(enabled)

    def _geometry(self, frame, palm_center: Sequence[float]) -> Tuple[int, int, int]:
        height, width = frame.shape[:2]
        px = clamp(float(palm_center[0]), 0.0, 1.0)
        py = clamp(float(palm_center[1]), 0.0, 1.0)
        cx = int(px * width)
        cy = int(py * height)
        radius = int(clamp(self.config.get("menu_radius", 80), 24, min(width, height) // 2))
        return cx, cy, radius

    def render_menu(
        self,
        frame,
        palm_center: Sequence[float],
        items: Sequence[str],
        selected: int = 0,
    ):
        """Draw the radial menu for ``items`` around the palm; returns the frame."""
        cv2 = try_import("cv2")
        if frame is None or cv2 is None or not items:
            return frame


        self.last_items = list(items)
        self.last_selected = int(selected) % len(items)

        cx, cy, radius = self._geometry(frame, palm_center)
        overlay = frame.copy()
        count = len(items)
        start = float(self.config.get("start_angle", -90.0))
        thickness = int(self.config.get("thickness", 2))

        # Ring + dimmed disc
        cv2.circle(overlay, (cx, cy), radius, self.config["color"], thickness)
        cv2.circle(overlay, (cx, cy), max(radius // 4, 8), self.config["color"], -1)

        centers: List[Tuple[int, int]] = []
        for index in range(count):
            angle = math.radians(start + index * (360.0 / count))
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            centers.append((x, y))

            is_selected = index == self.last_selected
            color = self.config["highlight_color"] if is_selected else self.config["color"]
            node_radius = int(radius * (0.30 if is_selected else 0.22))
            cv2.circle(overlay, (x, y), node_radius, color, -1 if is_selected else thickness)
            cv2.line(overlay, (cx, cy), (x, y), color, 1)

            label = str(items[index])
            scale = float(self.config.get("font_scale", 0.5))
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            origin = (x - tw // 2, y + th // 2)
            cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (20, 20, 20) if is_selected else self.config["text_color"], 2, cv2.LINE_AA)
            cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        self.config["text_color"] if is_selected else (20, 20, 20), 1, cv2.LINE_AA)

        opacity = clamp(float(self.config.get("opacity", 0.7)), 0.0, 1.0)
        blended = cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0)

        # Selection marker outside the ring
        if centers:
            sx, sy = centers[self.last_selected]
            cv2.arrowedLine(blended, (cx, cy), (sx, sy), self.config["highlight_color"], 2, tipLength=0.25)
        return blended

    # ------------------------------------------------------------------ #
    def selection_from_finger(self, palm_center: Sequence[float], fingertip: Sequence[float],
                              item_count: int) -> int:
        """Pick the menu sector the index fingertip points at."""
        if item_count <= 0:
            return 0
        dx = float(fingertip[0]) - float(palm_center[0])
        dy = float(fingertip[1]) - float(palm_center[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0
        angle = math.degrees(math.atan2(dy, dx))
        start = float(self.config.get("start_angle", -90.0))
        sector = 360.0 / item_count
        offset = (angle - start + sector / 2.0) % 360.0
        return int(offset // sector) % item_count

    def render_hud(self, frame, stats: Dict[str, Any], origin: Tuple[int, int] = (10, 60)):
        """Draw a compact stats panel in the corner of the frame."""
        cv2 = try_import("cv2")
        if frame is None or cv2 is None or not self.config.get("show_hud", True):
            return frame

        wanted = tuple(self.config.get("hud_items", ("fps", "mode", "gesture", "latency")))
        lines = [f"{key}: {stats.get(key, '-')}" for key in wanted if key in stats or True]
        x, y = origin
        scale = float(self.config.get("font_scale", 0.5))

        width = 0
        for line in lines:
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
            width = max(width, size[0])
        height = len(lines) * 20 + 12

        panel = frame.copy()
        cv2.rectangle(panel, (x - 6, y - 18), (x + width + 10, y - 18 + height), (24, 24, 30), -1)
        opacity = clamp(float(self.config.get("opacity", 0.7)), 0.0, 1.0)
        frame = cv2.addWeighted(panel, opacity, frame, 1.0 - opacity, 0)

        for index, line in enumerate(lines):
            cv2.putText(frame, line, (x, y + index * 20), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        self.config["text_color"], 1, cv2.LINE_AA)
        return frame

    def render_status(self, frame, message: str, color: Tuple[int, int, int] = (0, 212, 255)):
        """Banner across the bottom of the frame (used for errors and hints)."""
        cv2 = try_import("cv2")
        if frame is None or cv2 is None or not message:
            return frame
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, height - 34), (width, height), (18, 18, 24), -1)
        cv2.putText(frame, str(message), (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
        return frame


__all__ = ["AROverlay", "DEFAULT_AR_CONFIG"]
