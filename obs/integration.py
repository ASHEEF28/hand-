"""OBS Studio integration (optional, off by default).

Uses ``obsws-python`` to talk to obs-websocket (OBS 28+, port 4455).  Useful for
streamers: switch scenes with a gesture, toggle the mic, start/stop recording.

Every method is defensive - if OBS is not running the integration simply reports
``False`` instead of raising into the gesture loop.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from utils import deep_merge, module_available, try_import

logger = logging.getLogger(__name__)

DEFAULT_OBS_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "host": "localhost",
    "port": 4455,
    "password": "",
    "reconnect_interval": 5.0,
    "auto_reconnect": True,
    "timeout": 3.0,
    "scene_gestures": {
        "swipe_left": "previous_scene",
        "swipe_right": "next_scene",
        "five_finger_pinch": "toggle_recording",
        "four_finger_pinch": "toggle_streaming",
    },
}


class OBSIntegration:
    """Thin, fault-tolerant wrapper around obs-websocket."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_OBS_CONFIG), config or {})
        self._client = None
        self._lock = threading.Lock()
        self._connected = False
        self._last_error: str = ""
        self._scenes: List[str] = []
        self._scene_index = 0
        self._reconnect_thread: Optional[threading.Thread] = None
        self._running = False
        self.stats: Dict[str, Any] = {"commands": 0, "errors": 0, "connected_at": None}

    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        return module_available("obsws_python")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str:
        return self._last_error

    def connect(self) -> bool:
        """Connect to obs-websocket."""
        with self._lock:
            if self._connected and self._client is not None:
                return True
            obsws = try_import("obsws_python")
            if obsws is None:
                self._last_error = "obsws-python is not installed"
                logger.warning("%s - OBS integration disabled", self._last_error)
                return False
            try:
                self._client = obsws.ReqClient(
                    host=str(self.config.get("host", "localhost")),
                    port=int(self.config.get("port", 4455)),
                    password=str(self.config.get("password", "") or ""),
                    timeout=float(self.config.get("timeout", 3.0)),
                )
                version = self._client.get_version()
                self._connected = True
                self._last_error = ""
                self.stats["connected_at"] = time.time()
                logger.info(
                    "Connected to OBS (obs-websocket %s, host %s:%s)",
                    getattr(version, "obs_websocket_version", "?"),
                    self.config.get("host"), self.config.get("port"),
                )
                self.refresh_scenes()
                self._start_reconnect_watchdog()
                return True
            except Exception as exc:
                self._connected = False
                self._client = None
                self._last_error = str(exc)
                logger.warning("Could not connect to OBS at %s:%s - %s",
                               self.config.get("host"), self.config.get("port"), exc)
                self._start_reconnect_watchdog()
                return False

    def disconnect(self) -> None:
        self._running = False
        with self._lock:
            self._connected = False
            self._client = None
        logger.info("Disconnected from OBS")

    def _start_reconnect_watchdog(self) -> None:
        if not self.config.get("auto_reconnect", True):
            return
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return
        self._running = True
        self._reconnect_thread = threading.Thread(target=self._watchdog, name="GestureOS-OBS", daemon=True)
        self._reconnect_thread.start()

    def _watchdog(self) -> None:
        interval = float(self.config.get("reconnect_interval", 5.0))
        while self._running:
            time.sleep(max(1.0, interval))
            if not self._running:
                break
            if not self._connected:
                logger.debug("Attempting OBS reconnect...")
                self.connect()

    # ------------------------------------------------------------------ #
    def _call(self, name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
        """Invoke an obs-websocket request by name, tolerating failures."""
        with self._lock:
            client = self._client if self._connected else None
        if client is None:
            return None
        method: Optional[Callable[..., Any]] = getattr(client, name, None)
        if method is None:
            logger.debug("obs-websocket request '%s' is not supported", name)
            return None
        try:
            result = method(*args, **kwargs)
            self.stats["commands"] += 1
            return result
        except Exception as exc:
            self.stats["errors"] += 1
            self._last_error = str(exc)
            self._connected = False
            logger.warning("OBS request '%s' failed: %s", name, exc)
            return None

    # ------------------------------------------------------------------ #
    # Scenes
    # ------------------------------------------------------------------ #
    def refresh_scenes(self) -> List[str]:
        response = self._call("get_scene_list")
        scenes = getattr(response, "scenes", None) or []
        self._scenes = [getattr(s, "scene_name", str(s)) for s in scenes]
        return list(self._scenes)

    def scenes(self) -> List[str]:
        return list(self._scenes) or self.refresh_scenes()

    def current_scene(self) -> Optional[str]:
        response = self._call("get_current_program_scene")
        return getattr(response, "scene_name", None) or getattr(response, "current_program_scene_name", None)

    def set_scene(self, name: str) -> bool:
        return self._call("set_current_program_scene", scene_name=name) is not None

    def next_scene(self) -> bool:
        scenes = self.scenes()
        if not scenes:
            return False
        self._scene_index = (self._scene_index + 1) % len(scenes)
        return self.set_scene(scenes[self._scene_index])

    def previous_scene(self) -> bool:
        scenes = self.scenes()
        if not scenes:
            return False
        self._scene_index = (self._scene_index - 1) % len(scenes)
        return self.set_scene(scenes[self._scene_index])

    # ------------------------------------------------------------------ #
    # Recording / streaming / audio
    # ------------------------------------------------------------------ #
    def toggle_recording(self) -> bool:
        return self._call("toggle_record") is not None

    def start_recording(self) -> bool:
        return self._call("start_record") is not None

    def stop_recording(self) -> bool:
        return self._call("stop_record") is not None

    def toggle_streaming(self) -> bool:
        return self._call("toggle_stream") is not None

    def start_streaming(self) -> bool:
        return self._call("start_stream") is not None

    def stop_streaming(self) -> bool:
        return self._call("stop_stream") is not None

    def toggle_mute(self, input_name: Optional[str] = None) -> bool:
        name = input_name or self.config.get("audio_input")
        if not name:
            response = self._call("get_input_list")
            inputs = getattr(response, "inputs", None) or []
            if not inputs:
                return False
            name = getattr(inputs[0], "input_name", None)
        return self._call("toggle_input_mute", input_name=name) is not None

    def set_volume(self, percent: float, input_name: Optional[str] = None) -> bool:
        name = input_name or self.config.get("audio_input")
        if not name:
            return False
        return self._call("set_input_volume", input_name=name, input_volume_mul=max(0.0, float(percent) / 100.0)) is not None

    def trigger_hotkey(self, name: str) -> bool:
        return self._call("trigger_hotkey_by_name", hotkey_name=name) is not None

    # ------------------------------------------------------------------ #
    def handle_gesture(self, action: str) -> bool:
        """Map a gesture action onto an OBS command via ``scene_gestures``."""
        command = str(self.config.get("scene_gestures", {}).get(action, ""))
        if not command:
            return False
        handler = getattr(self, command, None)
        if handler is None or not callable(handler):
            logger.debug("Unknown OBS command '%s' for action '%s'", command, action)
            return False
        try:
            return bool(handler())
        except Exception as exc:
            logger.error("OBS command '%s' failed: %s", command, exc)
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available,
            "connected": self._connected,
            "scenes": self._scenes,
            "current_scene": self.current_scene() if self._connected else None,
            "last_error": self._last_error,
            **self.stats,
        }


__all__ = ["OBSIntegration", "DEFAULT_OBS_CONFIG"]
