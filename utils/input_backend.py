"""Input backend abstraction (mouse / keyboard).

GestureOS drives the OS through PyAutoGUI on Windows, but every controller goes
through this thin interface so that:

* the app degrades gracefully when PyAutoGUI (or a display) is missing -
  :class:`RecordingBackend` logs intent instead of moving a real cursor;
* the whole control layer is unit testable on any platform.

The backend is resolved once with :func:`get_backend` and shared.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from utils import try_import

logger = logging.getLogger(__name__)


class BaseBackend:
    """Interface implemented by the real and the recording backends."""

    name = "base"
    available = False

    def size(self) -> Tuple[int, int]:
        raise NotImplementedError

    def position(self) -> Tuple[int, int]:
        raise NotImplementedError

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        raise NotImplementedError

    def move_rel(self, dx: int, dy: int) -> None:
        raise NotImplementedError

    def click(self, button: str = "left", clicks: int = 1) -> None:
        raise NotImplementedError

    def mouse_down(self, button: str = "left") -> None:
        raise NotImplementedError

    def mouse_up(self, button: str = "left") -> None:
        raise NotImplementedError

    def scroll(self, clicks: int, horizontal: bool = False) -> None:
        raise NotImplementedError

    def press(self, key: str) -> None:
        raise NotImplementedError

    def hotkey(self, *keys: str) -> None:
        raise NotImplementedError

    def write(self, text: str, interval: float = 0.0) -> None:
        raise NotImplementedError


class PyAutoGUIBackend(BaseBackend):
    """Real input through PyAutoGUI.

    ``FAILSAFE`` is disabled on purpose: GestureOS routinely parks the cursor at
    screen edges, and PyAutoGUI's failsafe would abort the program.
    """

    name = "pyautogui"

    def __init__(self) -> None:
        self._pg = try_import("pyautogui")
        if self._pg is None:
            raise RuntimeError("pyautogui is not installed")
        self._pg.FAILSAFE = False
        self._pg.PAUSE = 0.0
        self.available = True
        self._lock = threading.Lock()

    def size(self) -> Tuple[int, int]:
        with self._lock:
            s = self._pg.size()
        return int(s[0]), int(s[1])

    def position(self) -> Tuple[int, int]:
        with self._lock:
            p = self._pg.position()
        return int(p[0]), int(p[1])

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        with self._lock:
            self._pg.moveTo(int(x), int(y), duration=max(0.0, duration), _pause=False)

    def move_rel(self, dx: int, dy: int) -> None:
        with self._lock:
            self._pg.moveRel(int(dx), int(dy), _pause=False)

    def click(self, button: str = "left", clicks: int = 1) -> None:
        with self._lock:
            self._pg.click(button=button, clicks=int(clicks), _pause=False)

    def mouse_down(self, button: str = "left") -> None:
        with self._lock:
            self._pg.mouseDown(button=button, _pause=False)

    def mouse_up(self, button: str = "left") -> None:
        with self._lock:
            self._pg.mouseUp(button=button, _pause=False)

    def scroll(self, clicks: int, horizontal: bool = False) -> None:
        with self._lock:
            if horizontal and hasattr(self._pg, "hscroll"):
                self._pg.hscroll(int(clicks), _pause=False)
            else:
                self._pg.scroll(int(clicks), _pause=False)

    def press(self, key: str) -> None:
        with self._lock:
            self._pg.press(key, _pause=False)

    def hotkey(self, *keys: str) -> None:
        with self._lock:
            self._pg.hotkey(*keys, _pause=False)

    def write(self, text: str, interval: float = 0.0) -> None:
        with self._lock:
            self._pg.write(text, interval=interval, _pause=False)


class RecordingBackend(BaseBackend):
    """Dry-run backend: records every call instead of touching the OS.

    Used when PyAutoGUI is unavailable (headless CI, non-Windows dev boxes) and
    by the test-suite to assert on emitted input.
    """

    name = "recording"

    def __init__(self, screen: Tuple[int, int] = (1920, 1080), max_events: int = 500) -> None:
        self.available = True
        self._screen = (int(screen[0]), int(screen[1]))
        self._position = [self._screen[0] // 2, self._screen[1] // 2]
        self.events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def _record(self, kind: str, **payload: Any) -> None:
        with self._lock:
            self.events.append({"kind": kind, "t": time.time(), **payload})

    def calls(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            events = list(self.events)
        return [e for e in events if kind is None or e["kind"] == kind]

    def last(self, kind: str) -> Optional[Dict[str, Any]]:
        calls = self.calls(kind)
        return calls[-1] if calls else None

    def count(self, kind: str) -> int:
        return len(self.calls(kind))

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def size(self) -> Tuple[int, int]:
        return self._screen

    def position(self) -> Tuple[int, int]:
        return tuple(self._position)  # type: ignore[return-value]

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        self._position = [int(x), int(y)]
        self._record("move_to", x=int(x), y=int(y))

    def move_rel(self, dx: int, dy: int) -> None:
        self._position = [self._position[0] + int(dx), self._position[1] + int(dy)]
        self._record("move_rel", dx=int(dx), dy=int(dy))

    def click(self, button: str = "left", clicks: int = 1) -> None:
        self._record("click", button=button, clicks=int(clicks))

    def mouse_down(self, button: str = "left") -> None:
        self._record("mouse_down", button=button)

    def mouse_up(self, button: str = "left") -> None:
        self._record("mouse_up", button=button)

    def scroll(self, clicks: int, horizontal: bool = False) -> None:
        self._record("scroll", clicks=int(clicks), horizontal=horizontal)

    def press(self, key: str) -> None:
        self._record("press", key=key)

    def hotkey(self, *keys: str) -> None:
        self._record("hotkey", keys=list(keys))

    def write(self, text: str, interval: float = 0.0) -> None:
        self._record("write", text=text)


_backend_instance: Optional[BaseBackend] = None
_backend_lock = threading.Lock()


def get_backend(force: Optional[str] = None) -> BaseBackend:
    """Return the shared input backend (``pyautogui`` when possible)."""
    global _backend_instance
    with _backend_lock:
        if force == "recording" or (force is None and _backend_instance is not None):
            if force == "recording":
                return RecordingBackend()
            if _backend_instance is not None:
                return _backend_instance
        try:
            _backend_instance = PyAutoGUIBackend()
            logger.info("Input backend: pyautogui")
        except Exception as exc:
            logger.warning("PyAutoGUI unavailable (%s) - using dry-run input backend", exc)
            _backend_instance = RecordingBackend()
        return _backend_instance


def set_backend(backend: Optional[BaseBackend]) -> None:
    """Override the shared backend (tests, ``--dry-run``)."""
    global _backend_instance
    with _backend_lock:
        _backend_instance = backend


__all__ = [
    "BaseBackend",
    "PyAutoGUIBackend",
    "RecordingBackend",
    "get_backend",
    "set_backend",
]
