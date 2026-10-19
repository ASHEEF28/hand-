"""Window management and desktop navigation.

Two controllers live here:

* :class:`WindowController` - acts on the foreground window (minimize,
  maximize, close, snap, move, resize).
* :class:`DesktopNavigationController` - shell level actions (Task View, Show
  Desktop, Start Menu, virtual desktops, Explorer, Search, lock).

On Windows these use the Win32 API for reliability and fall back to shell
hotkeys.  On other platforms (development machines, CI) every action degrades
to the equivalent hotkey or, when that is impossible, to a recorded no-op so the
gesture pipeline keeps working end to end.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from utils import IS_WINDOWS, deep_merge, try_import
from utils.input_backend import BaseBackend, get_backend

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "use_win32": True,
    "snap_offset": 40,
    "move_step": 60,
    "resize_step": 80,
    "action_cooldown": 0.35,
    "confirm_close": False,
}

DEFAULT_NAV_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "action_cooldown": 0.5,
    "start_menu_key": "win",
    "task_view_hotkey": ("win", "tab"),
    "show_desktop_hotkey": ("win", "d"),
    "explorer_hotkey": ("win", "e"),
    "search_hotkey": ("win", "s"),
    "settings_hotkey": ("win", "i"),
    "lock_hotkey": ("win", "l"),
    "desktop_left_hotkey": ("win", "ctrl", "left"),
    "desktop_right_hotkey": ("win", "ctrl", "right"),
    "new_desktop_hotkey": ("win", "ctrl", "d"),
    "close_desktop_hotkey": ("win", "ctrl", "f4"),
    "alt_tab_hotkey": ("alt", "tab"),
    "alt_shift_tab_hotkey": ("alt", "shift", "tab"),
}


def _win32():
    """Return ``(win32gui, win32con, win32api)`` when available."""
    if not IS_WINDOWS:
        return None, None, None
    return try_import("win32gui"), try_import("win32con"), try_import("win32api")


class WindowController:
    """Controls the foreground window."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        config = config or {}
        section = config.get("window", config) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_WINDOW_CONFIG), section or {})
        self.backend: BaseBackend = backend or get_backend()
        self.win32gui, self.win32con, self.win32api = _win32() if self.config.get("use_win32", True) else (None,) * 3
        self._cooldowns: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Foreground window info
    # ------------------------------------------------------------------ #
    def get_foreground_window(self) -> Dict[str, Any]:
        """``{'hwnd', 'title', 'rect', 'minimized', 'maximized'}`` (best effort)."""
        if self.win32gui and self.win32con:
            try:
                hwnd = self.win32gui.GetForegroundWindow()
                if not hwnd:
                    return {}
                title = self.win32gui.GetWindowText(hwnd)
                rect = self.win32gui.GetWindowRect(hwnd)
                placement = self.win32gui.GetWindowPlacement(hwnd)
                return {
                    "hwnd": hwnd,
                    "title": title,
                    "rect": rect,
                    "minimized": bool(placement[1] & getattr(self.win32con, "SW_SHOWMINIMIZED", 2)),
                    "maximized": placement[1] == getattr(self.win32con, "SW_SHOWMAXIMIZED", 3),
                }
            except Exception as exc:
                logger.debug("Win32 foreground window query failed: %s", exc)
        return {}

    def foreground_title(self) -> str:
        return str(self.get_foreground_window().get("title", "") or "")

    def list_windows(self, visible_only: bool = True) -> List[Dict[str, Any]]:
        """Enumerate top-level windows (Windows only; empty list elsewhere)."""
        if not (self.win32gui and self.win32con):
            return []
        windows: List[Dict[str, Any]] = []

        def _enum(hwnd: int, _: Any) -> None:
            try:
                if not self.win32gui.IsWindowVisible(hwnd) and visible_only:
                    return
                title = self.win32gui.GetWindowText(hwnd)
                if not title.strip():
                    return
                windows.append({"hwnd": hwnd, "title": title, "rect": self.win32gui.GetWindowRect(hwnd)})
            except Exception:
                return

        try:
            self.win32gui.EnumWindows(_enum, None)
        except Exception as exc:
            logger.debug("Window enumeration failed: %s", exc)
        return windows

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _gate(self, action: str) -> bool:
        now = time.time()
        cooldown = float(self.config.get("action_cooldown", 0.35))
        last = self._cooldowns.get(action)
        if last is not None and now - last < cooldown:
            return False
        self._cooldowns[action] = now
        self.history.append({"action": action, "t": now})
        return True

    def _log_intent(self, action: str, detail: str = "") -> None:
        logger.info("Window action '%s' %s (no Win32 backend - hotkey/no-op)", action, detail)

    def minimize(self, hwnd: Optional[int] = None) -> bool:
        if not self._gate("minimize"):
            return False
        if self.win32gui and self.win32con:
            try:
                hwnd = hwnd or self.win32gui.GetForegroundWindow()
                if hwnd:
                    self.win32gui.ShowWindow(hwnd, self.win32con.SW_MINIMIZE)
                    return True
            except Exception as exc:
                logger.error("Minimize failed: %s", exc)
        self.backend.hotkey("win", "down")
        return True

    def maximize(self, hwnd: Optional[int] = None) -> bool:
        if not self._gate("maximize"):
            return False
        if self.win32gui and self.win32con:
            try:
                hwnd = hwnd or self.win32gui.GetForegroundWindow()
                if hwnd:
                    self.win32gui.ShowWindow(hwnd, self.win32con.SW_MAXIMIZE)
                    return True
            except Exception as exc:
                logger.error("Maximize failed: %s", exc)
        self.backend.hotkey("win", "up")
        return True

    def restore(self, hwnd: Optional[int] = None) -> bool:
        if not self._gate("restore"):
            return False
        if self.win32gui and self.win32con:
            try:
                hwnd = hwnd or self.win32gui.GetForegroundWindow()
                if hwnd:
                    self.win32gui.ShowWindow(hwnd, self.win32con.SW_RESTORE)
                    return True
            except Exception as exc:
                logger.error("Restore failed: %s", exc)
        self.backend.hotkey("win", "down")
        return True

    def toggle_maximize(self) -> bool:
        info = self.get_foreground_window()
        return self.restore() if info.get("maximized") else self.maximize()

    def close(self, hwnd: Optional[int] = None) -> bool:
        if not self._gate("close"):
            return False
        if self.win32gui and self.win32con:
            try:
                hwnd = hwnd or self.win32gui.GetForegroundWindow()
                if hwnd:
                    if self.config.get("confirm_close", False):
                        self.backend.hotkey("alt", "f4")
                    else:
                        self.win32gui.PostMessage(hwnd, self.win32con.WM_CLOSE, 0, 0)
                    return True
            except Exception as exc:
                logger.error("Close failed: %s", exc)
        self.backend.hotkey("alt", "f4")
        return True

    def snap(self, direction: str = "left") -> bool:
        """Aero-snap the foreground window (left/right/top/bottom/quadrants)."""
        key = f"snap:{direction}"
        if not self._gate(key):
            return False
        mapping = {
            "left": ("win", "left"),
            "right": ("win", "right"),
            "top": ("win", "up"),
            "bottom": ("win", "down"),
            "top_left": ("win", "left", "up"),
            "top_right": ("win", "right", "up"),
            "bottom_left": ("win", "left", "down"),
            "bottom_right": ("win", "right", "down"),
        }
        keys = mapping.get(direction)
        if not keys:
            logger.warning("Unknown snap direction: %s", direction)
            return False
        self.backend.hotkey(*keys)
        return True

    def move_window(self, dx: int, dy: int) -> bool:
        """Nudge the foreground window by ``dx``/``dy`` pixels."""
        if not self._gate("move"):
            return False
        if self.win32gui and self.win32con:
            try:
                hwnd = self.win32gui.GetForegroundWindow()
                left, top, right, bottom = self.win32gui.GetWindowRect(hwnd)
                self.win32gui.MoveWindow(hwnd, left + int(dx), top + int(dy), right - left, bottom - top, True)
                return True
            except Exception as exc:
                logger.error("Window move failed: %s", exc)
        self._log_intent("move", f"dx={dx} dy={dy}")
        return False

    def resize_window(self, dw: int, dh: int) -> bool:
        if not self._gate("resize"):
            return False
        if self.win32gui and self.win32con:
            try:
                hwnd = self.win32gui.GetForegroundWindow()
                left, top, right, bottom = self.win32gui.GetWindowRect(hwnd)
                self.win32gui.MoveWindow(
                    hwnd, left, top, max(160, right - left + int(dw)), max(120, bottom - top + int(dh)), True
                )
                return True
            except Exception as exc:
                logger.error("Window resize failed: %s", exc)
        self._log_intent("resize", f"dw={dw} dh={dh}")
        return False

    def bring_to_front(self, hwnd: int) -> bool:
        if not (self.win32gui and self.win32con):
            return False
        try:
            if self.win32gui.IsIconic(hwnd):
                self.win32gui.ShowWindow(hwnd, self.win32con.SW_RESTORE)
            self.win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception as exc:
            logger.error("Bring to front failed: %s", exc)
            return False


class DesktopNavigationController:
    """Shell-level navigation (Task View, desktops, Start Menu, Explorer...)."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        config = config or {}
        section = config.get("navigation", config) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_NAV_CONFIG), section or {})
        self.backend: BaseBackend = backend or get_backend()
        self.win32gui, self.win32con, self.win32api = _win32()
        self._cooldowns: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []

    def _gate(self, action: str) -> bool:
        now = time.time()
        cooldown = float(self.config.get("action_cooldown", 0.5))
        last = self._cooldowns.get(action)
        if last is not None and now - last < cooldown:
            return False
        self._cooldowns[action] = now
        self.history.append({"action": action, "t": now})
        return True

    def _hotkey(self, action: str, keys: Tuple[str, ...]) -> bool:
        if not self._gate(action):
            return False
        try:
            self.backend.hotkey(*keys)
            logger.debug("Navigation: %s -> %s", action, "+".join(keys))
            return True
        except Exception as exc:
            logger.error("Navigation action '%s' failed: %s", action, exc)
            return False

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def task_view(self) -> bool:
        return self._hotkey("task_view", tuple(self.config["task_view_hotkey"]))

    def show_desktop(self) -> bool:
        return self._hotkey("show_desktop", tuple(self.config["show_desktop_hotkey"]))

    def start_menu(self) -> bool:
        key = str(self.config.get("start_menu_key", "win"))
        if not self._gate("start_menu"):
            return False
        try:
            self.backend.press(key)
            return True
        except Exception as exc:
            logger.error("Start menu failed: %s", exc)
            return False

    def search(self) -> bool:
        return self._hotkey("search", tuple(self.config["search_hotkey"]))

    def settings(self) -> bool:
        return self._hotkey("settings", tuple(self.config["settings_hotkey"]))

    def open_explorer(self) -> bool:
        return self._hotkey("explorer", tuple(self.config["explorer_hotkey"]))

    def lock(self) -> bool:
        return self._hotkey("lock", tuple(self.config["lock_hotkey"]))

    def virtual_desktop_left(self) -> bool:
        return self._hotkey("desktop_left", tuple(self.config["desktop_left_hotkey"]))

    def virtual_desktop_right(self) -> bool:
        return self._hotkey("desktop_right", tuple(self.config["desktop_right_hotkey"]))

    def new_virtual_desktop(self) -> bool:
        return self._hotkey("desktop_new", tuple(self.config["new_desktop_hotkey"]))

    def close_virtual_desktop(self) -> bool:
        return self._hotkey("desktop_close", tuple(self.config["close_desktop_hotkey"]))

    def switch_window(self, direction: str = "next") -> bool:
        """Alt+Tab (next) / Alt+Shift+Tab (previous)."""
        keys = self.config["alt_tab_hotkey"] if direction == "next" else self.config["alt_shift_tab_hotkey"]
        return self._hotkey(f"alt_tab_{direction}", tuple(keys))

    def cycle_window(self, steps: int = 1) -> bool:
        """Alt+Tab held for ``steps`` positions then released."""
        if not self._gate("cycle"):
            return False
        pg = try_import("pyautogui")
        if pg is None or self.backend.name != "pyautogui":
            return self.switch_window("next" if steps >= 0 else "prev")
        try:
            pg.keyDown("alt")
            for _ in range(abs(int(steps))):
                pg.press("tab" if steps >= 0 else "left")
                time.sleep(0.05)
            pg.keyUp("alt")
            return True
        except Exception as exc:
            logger.error("Window cycling failed: %s", exc)
            return False


__all__ = ["WindowController", "DesktopNavigationController", "DEFAULT_WINDOW_CONFIG", "DEFAULT_NAV_CONFIG"]
