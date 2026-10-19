"""High level window manager: grab / move / snap plus shell navigation.

``main.py`` talks to this class rather than to the lower level controllers, so
all window behaviour (including the grab-and-move gesture) lives here.

Grabbing works by tracking the hand position ourselves and calling
``MoveWindow`` each frame, which - unlike ``WM_NCLBUTTONDOWN`` - never blocks
the gesture loop.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from utils import IS_WINDOWS, clamp, deep_merge, try_import
from utils.input_backend import BaseBackend, get_backend
from controllers.mouse_controller import MouseController
from controllers.window_controller import DesktopNavigationController, WindowController

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_MANAGER_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "snap_enabled": True,
    "snap_edge_threshold": 0.05,      # normalised distance from a screen edge
    "snap_corner_threshold": 0.12,
    "grab_button_fallback": "left",   # used when Win32 is unavailable
    "min_drag_distance": 4,           # px before a grab becomes a move
    "move_smoothing": 0.35,
    "history_size": 20,
    "action_cooldown": 0.4,
}


class WindowManager:
    """Coordinates cursor, window and desktop-navigation controllers."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mouse: Optional[MouseController] = None,
        window_controller: Optional[WindowController] = None,
        navigation: Optional[DesktopNavigationController] = None,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        config = config or {}
        section = config.get("window_manager", config) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_WINDOW_MANAGER_CONFIG), section or {})
        self.backend: BaseBackend = backend or get_backend()

        self.mouse = mouse or MouseController(config, backend=self.backend)
        self.windows = window_controller or WindowController(config, backend=self.backend)
        self.navigation = navigation or DesktopNavigationController(config, backend=self.backend)

        self._grabbing = False
        self._grab_hwnd: Optional[int] = None
        self._grab_origin: Tuple[int, int] = (0, 0)
        self._grab_window_rect: Optional[Tuple[int, int, int, int]] = None
        self._grab_moved = False
        self._smoothed: Optional[Tuple[float, float]] = None
        self._grab_target: Optional[Tuple[float, float]] = None
        self._cooldowns: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []
        self.max_history = int(self.config.get("history_size", 20))

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def is_grabbing(self) -> bool:
        """``main.py`` reads this as an attribute (no call)."""
        return self._grabbing

    @property
    def grabbed_hwnd(self) -> Optional[int]:
        return self._grab_hwnd

    def _gate(self, action: str) -> bool:
        now = time.time()
        cooldown = float(self.config.get("action_cooldown", 0.4))
        last = self._cooldowns.get(action)
        if last is not None and now - last < cooldown:
            return False
        self._cooldowns[action] = now
        return True

    def _remember(self, action: str, **payload: Any) -> None:
        self.history.append({"action": action, "t": time.time(), **payload})
        if len(self.history) > self.max_history:
            del self.history[0 : len(self.history) - self.max_history]

    # ------------------------------------------------------------------ #
    # Shell navigation (delegated)
    # ------------------------------------------------------------------ #
    def switch(self, direction: str = "next") -> bool:
        ok = self.navigation.switch_window(direction)
        self._remember("switch", direction=direction)
        return ok

    def next_window(self) -> bool:
        return self.switch("next")

    def prev_window(self) -> bool:
        return self.switch("prev")

    def task_view(self) -> bool:
        self._remember("task_view")
        return self.navigation.task_view()

    def show_desktop(self) -> bool:
        self._remember("show_desktop")
        return self.navigation.show_desktop()

    def start_menu(self) -> bool:
        self._remember("start_menu")
        return self.navigation.start_menu()

    def search(self) -> bool:
        return self.navigation.search()

    def open_settings(self) -> bool:
        return self.navigation.settings()

    def open_explorer(self) -> bool:
        self._remember("explorer")
        return self.navigation.open_explorer()

    def virtual_desktop_left(self) -> bool:
        self._remember("desktop_left")
        return self.navigation.virtual_desktop_left()

    def virtual_desktop_right(self) -> bool:
        self._remember("desktop_right")
        return self.navigation.virtual_desktop_right()

    def new_virtual_desktop(self) -> bool:
        return self.navigation.new_virtual_desktop()

    def close_virtual_desktop(self) -> bool:
        return self.navigation.close_virtual_desktop()

    def lock_screen(self) -> bool:
        return self.navigation.lock()

    # ------------------------------------------------------------------ #
    # Window state actions
    # ------------------------------------------------------------------ #
    def minimize(self) -> bool:
        self._remember("minimize")
        return self.windows.minimize()

    def maximize(self) -> bool:
        self._remember("maximize")
        return self.windows.maximize()

    def restore(self) -> bool:
        return self.windows.restore()

    def toggle_maximize(self) -> bool:
        self._remember("toggle_maximize")
        return self.windows.toggle_maximize()

    def close(self) -> bool:
        self._remember("close")
        return self.windows.close()

    def snap(self, direction: str = "left") -> bool:
        self._remember("snap", direction=direction)
        return self.windows.snap(direction)

    # ------------------------------------------------------------------ #
    # Grab / move / drop
    # ------------------------------------------------------------------ #
    def grab(self, position: Optional[Tuple[int, int]] = None, hwnd: Optional[int] = None) -> bool:
        """Start moving the window under ``position`` (screen pixels)."""
        if self._grabbing:
            return False
        if not self._gate("grab"):
            return False

        left, top, width, height = self.mouse.screen_bounds
        if position is None:
            position = self.mouse.position()
        px = int(clamp(position[0], left, left + width - 1))
        py = int(clamp(position[1], top, top + height - 1))

        win32gui = try_import("win32gui") if IS_WINDOWS else None
        target_hwnd = hwnd
        if target_hwnd is None and win32gui is not None:
            try:
                target_hwnd = win32gui.WindowFromPoint((px, py))
                # Walk up to the top-level owner so we move the whole window.
                root = win32gui.GetAncestor(target_hwnd, 2)  # GA_ROOT
                if root:
                    target_hwnd = root
            except Exception as exc:
                logger.debug("WindowFromPoint failed: %s", exc)
                target_hwnd = None

        if target_hwnd is None:
            info = self.windows.get_foreground_window()
            target_hwnd = info.get("hwnd")

        self._grab_hwnd = target_hwnd
        self._grab_origin = (px, py)
        self._grab_moved = False
        self._smoothed = (float(px), float(py))
        self._grab_target = (float(px), float(py))
        self._grab_window_rect = None

        if target_hwnd and win32gui is not None:
            try:
                self._grab_window_rect = win32gui.GetWindowRect(target_hwnd)
            except Exception as exc:
                logger.debug("GetWindowRect failed: %s", exc)

        self._grabbing = True
        if self._grab_window_rect is None:
            # No Win32 handle: fall back to a real mouse drag on the title bar.
            self.mouse.move_cursor_pixels(px, py)
            self.mouse.start_drag(button=str(self.config.get("grab_button_fallback", "left")))
        self._remember("grab", hwnd=target_hwnd, at=(px, py))
        logger.debug("Grabbed window %s at %s", target_hwnd, (px, py))
        return True

    def move(self, position: Tuple[int, int]) -> bool:
        """Drag the grabbed window to ``position`` (screen pixels)."""
        if not self._grabbing:
            return False
        left, top, width, height = self.mouse.screen_bounds
        px = clamp(float(position[0]), left, left + width - 1)
        py = clamp(float(position[1]), top, top + height - 1)

        self._grab_target = (px, py)

        smoothing = clamp(float(self.config.get("move_smoothing", 0.35)), 0.0, 0.9)
        if self._smoothed is None:
            self._smoothed = (px, py)
        else:
            self._smoothed = (
                self._smoothed[0] * smoothing + px * (1 - smoothing),
                self._smoothed[1] * smoothing + py * (1 - smoothing),
            )
        px, py = self._smoothed

        if not self._grab_moved:
            if abs(px - self._grab_origin[0]) + abs(py - self._grab_origin[1]) < float(
                self.config.get("min_drag_distance", 4)
            ):
                return True
            self._grab_moved = True

        win32gui = try_import("win32gui") if IS_WINDOWS else None
        if self._grab_window_rect and self._grab_hwnd and win32gui is not None:
            wl, wt, wr, wb = self._grab_window_rect
            dx = int(px - self._grab_origin[0])
            dy = int(py - self._grab_origin[1])
            try:
                win32gui.MoveWindow(self._grab_hwnd, wl + dx, wt + dy, wr - wl, wb - wt, True)
                return True
            except Exception as exc:
                logger.error("Window move failed: %s", exc)
                return False

        # Fallback path: the OS is doing the dragging via a held mouse button.
        self.mouse.move_cursor_pixels(int(px), int(py))
        return True

    def update_grab(self, norm_x: float, norm_y: float) -> bool:
        """Convenience wrapper taking normalised hand coordinates."""
        target = self.mouse.normalized_to_screen(float(norm_x), float(norm_y))
        return self.move(target)

    def release(self) -> bool:
        """Drop the grabbed window, snapping to an edge when close enough."""
        if not self._grabbing:
            return False
        self._grabbing = False

        # Snap on where the user actually dropped, not on the smoothed cursor.
        snap_direction = self._snap_direction_for(
            self._grab_target or self._smoothed or self._grab_origin
        )
        if self._grab_window_rect is None:
            self.mouse.end_drag()

        hwnd, moved = self._grab_hwnd, self._grab_moved
        self._grab_hwnd = None
        self._grab_target = None
        self._grab_window_rect = None
        self._grab_moved = False
        self._remember("release", hwnd=hwnd, moved=moved, snap=snap_direction)

        if snap_direction and self.config.get("snap_enabled", True) and moved:
            return self.snap(snap_direction)
        return True

    def toggle_grab(self, position: Optional[Tuple[int, int]] = None) -> bool:
        """Grab or drop. Returns the resulting state (True = still grabbing)."""
        if self._grabbing:
            self.release()
        else:
            self.grab(position)
        return self._grabbing

    def _snap_direction_for(self, position: Tuple[float, float]) -> Optional[str]:
        """Which Aero-snap zone (if any) the drop point falls into."""
        if not self.config.get("snap_enabled", True):
            return None
        left, top, width, height = self.mouse.screen_bounds
        x = (float(position[0]) - left) / max(width, 1)
        y = (float(position[1]) - top) / max(height, 1)
        edge = float(self.config["snap_edge_threshold"])
        corner = float(self.config["snap_corner_threshold"])

        near_left, near_right = x <= edge, x >= 1 - edge
        near_top, near_bottom = y <= corner, y >= 1 - edge

        if near_left and near_top:
            return "top_left"
        if near_left and near_bottom:
            return "bottom_left"
        if near_right and near_top:
            return "top_right"
        if near_right and near_bottom:
            return "bottom_right"
        if near_left:
            return "left"
        if near_right:
            return "right"
        if y <= edge:
            return "top"
        return None

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def foreground_window(self) -> Dict[str, Any]:
        return self.windows.get_foreground_window()

    def open_windows(self) -> List[Dict[str, Any]]:
        return self.windows.list_windows()

    def shutdown(self) -> None:
        if self._grabbing:
            self.release()
        self.mouse.shutdown()


__all__ = ["WindowManager", "DEFAULT_WINDOW_MANAGER_CONFIG"]
