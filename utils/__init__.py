"""Shared utility helpers for GestureOS.

Small, dependency-free functions used across the vision, gesture and
controller layers.  Everything here is pure Python so it can be unit tested
on any platform (the real app targets Windows).
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import os
import platform
import sys
import time
from typing import Any, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    if low > high:
        low, high = high, low
    return low if value < low else high if value > high else value


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between ``a`` and ``b`` (``t`` is clamped to 0..1)."""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map ``value`` from one range onto another."""
    if abs(in_max - in_min) < 1e-9:
        return out_min
    t = (value - in_min) / (in_max - in_min)
    return out_min + (out_max - out_min) * t


def ema(previous: float, current: float, smoothing: float) -> float:
    """Exponential moving average.

    ``smoothing`` is the weight of the *previous* value: 0.0 = no smoothing,
    1.0 = frozen.  This matches the ``smoothing`` semantics used in
    ``settings.json``.
    """
    smoothing = clamp(smoothing, 0.0, 0.999)
    return previous * smoothing + current * (1.0 - smoothing)


def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Euclidean distance between two 2D/3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def angle_between(vertex: Sequence[float], p1: Sequence[float], p2: Sequence[float]) -> float:
    """Angle (degrees) at ``vertex`` formed by ``p1`` and ``p2``."""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_t = clamp((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_t))


def signed_rotation(prev_angle: float, curr_angle: float) -> float:
    """Shortest signed delta between two angles in degrees (-180..180)."""
    delta = (curr_angle - prev_angle + 180.0) % 360.0 - 180.0
    return delta


def now_ms() -> float:
    """Monotonic clock in milliseconds."""
    return time.perf_counter() * 1000.0


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (returns ``base``)."""
    for key, value in (override or {}).items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def cfg_get(config: Optional[dict], path: str, default: Any = None) -> Any:
    """Read ``"a.b.c"`` style keys out of a nested dict."""
    value: Any = config or {}
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


# --------------------------------------------------------------------------- #
# Optional dependency handling
# --------------------------------------------------------------------------- #
def try_import(module_name: str) -> Optional[Any]:
    """Import ``module_name`` if available, otherwise return ``None``.

    GestureOS runs on Windows with the full stack, but large parts of the code
    (gesture maths, analytics, tests) are useful on other platforms too, so
    every platform-specific dependency is imported through this helper.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on host env
        logger.debug("Optional module %s unavailable: %s", module_name, exc)
        return None


def module_available(module_name: str) -> bool:
    """Cheap availability check that does not import heavy packages.

    ``find_spec`` returns ``None`` for a module that is not installed (it only
    raises for malformed names / missing parents), so both mean "unavailable".
    """
    if not module_name:
        return False
    if module_name in sys.modules:
        return sys.modules[module_name] is not None
    try:
        return importlib.util.find_spec(module_name) is not None  # type: ignore[attr-defined]
    except Exception:
        return False


def project_root() -> str:
    """Absolute path of the GestureOS package root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_path(*parts: str) -> str:
    """Resolve a path relative to the project root, creating parent dirs."""
    path = os.path.join(project_root(), *parts)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def screen_size(fallback: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    """Best-effort primary screen resolution in pixels."""
    screeninfo = try_import("screeninfo")
    if screeninfo:
        try:
            monitors = [m for m in screeninfo.get_monitors() if m.width and m.height]
            if monitors:
                primary = monitors[0]
                return int(primary.width), int(primary.height)
        except Exception as exc:  # pragma: no cover - depends on host env
            logger.debug("screeninfo failed: %s", exc)

    pyautogui = try_import("pyautogui")
    if pyautogui:
        try:
            size = pyautogui.size()
            return int(size[0]), int(size[1])
        except Exception as exc:  # pragma: no cover
            logger.debug("pyautogui.size() failed: %s", exc)

    return fallback


__all__ = [
    "IS_WINDOWS",
    "IS_MACOS",
    "IS_LINUX",
    "clamp",
    "lerp",
    "remap",
    "ema",
    "distance",
    "angle_between",
    "signed_rotation",
    "now_ms",
    "deep_merge",
    "cfg_get",
    "try_import",
    "module_available",
    "project_root",
    "data_path",
    "screen_size",
]
