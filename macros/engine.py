"""Gesture macros and application launcher.

Macros are ordered step lists stored in ``macros/macros.json``; applications are
name -> executable mappings stored in ``macros/apps.json``.  Both files are
created with useful defaults on first run, and both are editable from the UI.

Macro steps
-----------
``{"type": "hotkey", "keys": ["ctrl", "s"]}``     press a key combination
``{"type": "key", "key": "enter"}``               press one key
``{"type": "text", "text": "hello"}``             type a string
``{"type": "wait", "seconds": 0.5}``              sleep
``{"type": "launch", "app": "calculator"}``       start an application
``{"type": "click", "button": "left", "count": 1}``
``{"type": "scroll", "clicks": -3}``
``{"type": "url", "url": "https://..."}``         open a link
``{"type": "shell", "command": "..."}``           run a command (opt-in)
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
import webbrowser
from typing import Any, Callable, Dict, List, Optional

from utils import IS_WINDOWS, deep_merge, project_root
from utils.input_backend import BaseBackend, get_backend

logger = logging.getLogger(__name__)

DEFAULT_MACROS: Dict[str, Any] = {
    "copy": {"description": "Copy selection", "cooldown": 0.3,
             "steps": [{"type": "hotkey", "keys": ["ctrl", "c"]}]},
    "paste": {"description": "Paste clipboard", "cooldown": 0.3,
              "steps": [{"type": "hotkey", "keys": ["ctrl", "v"]}]},
    "cut": {"description": "Cut selection", "cooldown": 0.3,
            "steps": [{"type": "hotkey", "keys": ["ctrl", "x"]}]},
    "undo": {"description": "Undo", "cooldown": 0.3,
             "steps": [{"type": "hotkey", "keys": ["ctrl", "z"]}]},
    "redo": {"description": "Redo", "cooldown": 0.3,
             "steps": [{"type": "hotkey", "keys": ["ctrl", "y"]}]},
    "select_all": {"description": "Select all", "cooldown": 0.4,
                   "steps": [{"type": "hotkey", "keys": ["ctrl", "a"]}]},
    "save": {"description": "Save document", "cooldown": 0.5,
             "steps": [{"type": "hotkey", "keys": ["ctrl", "s"]}]},
    "find": {"description": "Find in page", "cooldown": 0.5,
             "steps": [{"type": "hotkey", "keys": ["ctrl", "f"]}]},
    "screenshot": {"description": "Capture screen region", "cooldown": 1.0,
                   "steps": [{"type": "hotkey", "keys": ["win", "shift", "s"]}]},
    "fullscreen": {"description": "Toggle fullscreen", "cooldown": 0.6,
                   "steps": [{"type": "key", "key": "f11"}]},
    "close_window": {"description": "Close the foreground window", "cooldown": 0.8,
                     "steps": [{"type": "hotkey", "keys": ["alt", "f4"]}]},
    "reload_page": {"description": "Reload browser tab", "cooldown": 0.6,
                    "steps": [{"type": "key", "key": "f5"}]},
    "new_tab": {"description": "New browser tab", "cooldown": 0.6,
                "steps": [{"type": "hotkey", "keys": ["ctrl", "t"]}]},
    "close_tab": {"description": "Close browser tab", "cooldown": 0.6,
                  "steps": [{"type": "hotkey", "keys": ["ctrl", "w"]}]},
    "emoji_picker": {"description": "Open the emoji panel", "cooldown": 1.0,
                     "steps": [{"type": "hotkey", "keys": ["win", "."]}]},
}

DEFAULT_APPS: Dict[str, Any] = {
    "calculator": {"command": "calc.exe", "windows": "calc.exe"},
    "notepad": {"command": "notepad.exe", "windows": "notepad.exe"},
    "explorer": {"command": "explorer.exe", "windows": "explorer.exe"},
    "terminal": {"command": "wt.exe", "windows": "wt.exe", "fallback": ["cmd.exe", "powershell.exe"]},
    "browser": {"command": "msedge.exe", "windows": "msedge.exe",
                "fallback": ["chrome.exe", "firefox.exe"],
                "linux": "xdg-open http://localhost", "darwin": "open http://localhost"},
    "task_manager": {"command": "taskmgr.exe", "windows": "taskmgr.exe"},
    "paint": {"command": "mspaint.exe", "windows": "mspaint.exe"},
    "settings": {"command": "ms-settings:", "windows": "ms-settings:"},
    "code": {"command": "code", "windows": "code.cmd", "linux": "code", "darwin": "code"},
}


def _resolve_path(*parts: str) -> str:
    path = os.path.join(*parts)
    return path if os.path.isabs(path) else os.path.join(project_root(), path)


class MacroEngine:
    """Executes named macros on a worker thread so the gesture loop never stalls."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[BaseBackend] = None,
        launcher: Optional["ApplicationLauncher"] = None,
    ) -> None:
        config = config or {}
        section = config.get("macros", {}) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(
            {
                "store_path": "macros/macros.json",
                "cooldown": 0.3,
                "allow_shell": False,
                "async": True,
                "max_runtime": 30.0,
            },
            section or {},
        )
        self.backend: BaseBackend = backend or get_backend()
        self.launcher = launcher or ApplicationLauncher(config, backend=self.backend)
        self.macros: Dict[str, Dict[str, Any]] = {}
        self._cooldowns: Dict[str, float] = {}
        self._running: Dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self.execution_log: List[Dict[str, Any]] = []
        self.on_execute: Optional[Callable[[str, bool], None]] = None
        self._load()

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    @property
    def store_path(self) -> str:
        return _resolve_path(str(self.config.get("store_path", "macros/macros.json")))

    def _load(self) -> None:
        path = self.store_path
        if not os.path.exists(path):
            self.macros = json.loads(json.dumps(DEFAULT_MACROS))
            self.save()
            logger.info("Created default macros at %s (%d macros)", path, len(self.macros))
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.macros = data.get("macros", data) if isinstance(data, dict) else {}
            logger.info("Loaded %d macro(s) from %s", len(self.macros), path)
        except Exception as exc:
            logger.error("Could not load macros from %s: %s - using defaults", path, exc)
            self.macros = json.loads(json.dumps(DEFAULT_MACROS))

    def save(self) -> bool:
        path = self.store_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "macros": self.macros}, handle, indent=2)
            return True
        except Exception as exc:
            logger.error("Could not save macros: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def list_macros(self) -> List[str]:
        return sorted(self.macros.keys())

    def get_macro(self, name: str) -> Optional[Dict[str, Any]]:
        return self.macros.get(name)

    def add_macro(self, name: str, steps: List[Dict[str, Any]], description: str = "",
                  cooldown: Optional[float] = None) -> bool:
        name = (name or "").strip()
        if not name or not isinstance(steps, list) or not steps:
            return False
        self.macros[name] = {
            "description": description,
            "cooldown": float(cooldown if cooldown is not None else self.config.get("cooldown", 0.3)),
            "steps": steps,
        }
        return self.save()

    def update_macro(self, name: str, **fields: Any) -> bool:
        if name not in self.macros:
            return False
        self.macros[name].update(fields)
        return self.save()

    def remove_macro(self, name: str) -> bool:
        if name in self.macros:
            del self.macros[name]
            return self.save()
        return False

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute(self, name: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Run a macro by name (asynchronously by default)."""
        if not name:
            return False
        macro = self.macros.get(name)
        if macro is None:
            # Unknown names may be application shortcuts ("launch chrome").
            if self.launcher and self.launcher.knows(name):
                return self.launcher.launch(name)
            logger.debug("Unknown macro '%s'", name)
            return False

        cooldown = float(macro.get("cooldown", self.config.get("cooldown", 0.3)))
        now = time.time()
        last = self._cooldowns.get(name)
        if last is not None and now - last < cooldown:
            return False
        self._cooldowns[name] = now

        steps = list(macro.get("steps", []))
        if self.config.get("async", True):
            thread = threading.Thread(
                target=self._run_steps, args=(name, steps, params or {}),
                name=f"Macro-{name}", daemon=True,
            )
            self._running[name] = thread
            thread.start()
        else:
            self._run_steps(name, steps, params or {})
        return True

    def is_running(self, name: Optional[str] = None) -> bool:
        if name:
            thread = self._running.get(name)
            return bool(thread and thread.is_alive())
        return any(t.is_alive() for t in self._running.values())

    def stop_all(self) -> None:
        self._stop_event.set()

    def _run_steps(self, name: str, steps: List[Dict[str, Any]], params: Dict[str, Any]) -> None:
        started = time.time()
        self._stop_event.clear()
        success = True
        for index, step in enumerate(steps):
            if self._stop_event.is_set():
                logger.info("Macro '%s' cancelled at step %d", name, index)
                success = False
                break
            if time.time() - started > float(self.config.get("max_runtime", 30.0)):
                logger.warning("Macro '%s' exceeded max runtime - aborting", name)
                success = False
                break
            try:
                self._run_step(step, params)
            except Exception as exc:
                logger.error("Macro '%s' step %d failed: %s", name, index, exc)
                success = False
                break

        self.execution_log.append({"macro": name, "t": started, "ok": success, "steps": len(steps)})
        self._running.pop(name, None)
        if self.on_execute:
            try:
                self.on_execute(name, success)
            except Exception as exc:  # pragma: no cover
                logger.debug("Macro callback failed: %s", exc)

    def _run_step(self, step: Dict[str, Any], params: Dict[str, Any]) -> None:
        kind = str(step.get("type", "")).lower()

        if kind == "hotkey":
            keys = step.get("keys") or []
            if keys:
                self.backend.hotkey(*[str(k) for k in keys])
        elif kind == "key":
            self.backend.press(str(step.get("key", "")))
        elif kind == "text":
            text = str(step.get("text", ""))
            if text.startswith("$"):  # parameter substitution: {"text": "$query"}
                text = str(params.get(text[1:], ""))
            self.backend.write(text)
        elif kind == "wait":
            time.sleep(clamp_seconds(step.get("seconds", 0.1)))
        elif kind == "click":
            self.backend.click(str(step.get("button", "left")), int(step.get("count", 1)))
        elif kind == "scroll":
            self.backend.scroll(int(step.get("clicks", -3)), horizontal=bool(step.get("horizontal", False)))
        elif kind == "launch":
            self.launcher.launch(str(step.get("app", "")), step.get("args"))
        elif kind == "url":
            webbrowser.open(str(step.get("url", "")))
        elif kind == "shell":
            if not self.config.get("allow_shell", False):
                logger.warning("Shell steps are disabled (macros.allow_shell=false)")
                return
            command = str(step.get("command", ""))
            if command:
                subprocess.Popen(shlex.split(command), shell=False)
        else:
            logger.warning("Unknown macro step type: %s", kind)


def clamp_seconds(value: Any, maximum: float = 10.0) -> float:
    try:
        return max(0.0, min(float(value), maximum))
    except (TypeError, ValueError):
        return 0.1


class ApplicationLauncher:
    """Launches applications by friendly name."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        config = config or {}
        section = config.get("apps", {}) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(
            {"store_path": "macros/apps.json", "search_start_menu": True, "cooldown": 1.0},
            section or {},
        )
        self.backend = backend or get_backend()
        self.apps: Dict[str, Dict[str, Any]] = {}
        self._cooldowns: Dict[str, float] = {}
        self.launch_log: List[Dict[str, Any]] = []
        self._load()

    @property
    def store_path(self) -> str:
        return _resolve_path(str(self.config.get("store_path", "macros/apps.json")))

    def _load(self) -> None:
        path = self.store_path
        if not os.path.exists(path):
            self.apps = json.loads(json.dumps(DEFAULT_APPS))
            self.save()
            logger.info("Created default app shortcuts at %s (%d apps)", path, len(self.apps))
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.apps = data.get("apps", data) if isinstance(data, dict) else {}
            logger.info("Loaded %d app shortcut(s)", len(self.apps))
        except Exception as exc:
            logger.error("Could not load apps from %s: %s - using defaults", path, exc)
            self.apps = json.loads(json.dumps(DEFAULT_APPS))

    def save(self) -> bool:
        path = self.store_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "apps": self.apps}, handle, indent=2)
            return True
        except Exception as exc:
            logger.error("Could not save app shortcuts: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def list_apps(self) -> List[str]:
        return sorted(self.apps.keys())

    def knows(self, name: str) -> bool:
        return self._normalize(name) in self.apps

    def add_app(self, name: str, command: str, **extra: Any) -> bool:
        name = self._normalize(name)
        if not name or not command:
            return False
        entry = {"command": command}
        entry.update(extra)
        self.apps[name] = entry
        return self.save()

    def remove_app(self, name: str) -> bool:
        name = self._normalize(name)
        if name in self.apps:
            del self.apps[name]
            return self.save()
        return False

    @staticmethod
    def _normalize(name: str) -> str:
        return (name or "").strip().lower().replace(" ", "_")

    # ------------------------------------------------------------------ #
    # Launching
    # ------------------------------------------------------------------ #
    def resolve(self, name: str) -> Optional[str]:
        """Find the executable / URI for an app name."""
        key = self._normalize(name)
        entry = self.apps.get(key)
        candidates: List[str] = []
        if entry:
            platform_key = "windows" if IS_WINDOWS else ("darwin" if os.uname().sysname == "Darwin" else "linux")
            for field in (platform_key, "command"):
                value = entry.get(field)
                if value:
                    candidates.append(str(value))
            candidates.extend(str(c) for c in entry.get("fallback", []) or [])
        else:
            candidates.append(key)

        for candidate in candidates:
            if os.path.isabs(candidate) and os.path.exists(candidate):
                return candidate
            found = shutil.which(candidate)
            if found:
                return found
            if IS_WINDOWS and candidate.lower().startswith(("ms-", "http:", "https:")):
                return candidate
            if candidate.lower().endswith((".exe", ".cmd", ".bat", ".lnk")):
                located = self._search_windows_paths(candidate)
                if located:
                    return located
        return None

    def _search_windows_paths(self, executable: str) -> Optional[str]:
        """Look in the usual Windows install locations and Start Menu."""
        roots = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
            os.environ.get("WINDIR", r"C:\Windows"),
        ]
        if self.config.get("search_start_menu", True):
            roots.append(os.environ.get("ProgramData", r"C:\ProgramData"))
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            direct = os.path.join(root, executable)
            if os.path.exists(direct):
                return direct
        return None

    def launch(self, name: str, args: Optional[List[str]] = None) -> bool:
        """Launch an app by name. Returns True when the launch was attempted."""
        key = self._normalize(name)
        if not key:
            return False

        now = time.time()
        cooldown = float(self.config.get("cooldown", 1.0))
        last = self._cooldowns.get(key)
        if last is not None and now - last < cooldown:
            logger.debug("Launch of '%s' ignored (cooldown)", key)
            return False
        self._cooldowns[key] = now

        target = self.resolve(key)
        self.launch_log.append({"app": key, "target": target, "t": now})

        if not target:
            logger.warning("Could not resolve application '%s'", key)
            return False

        try:
            if IS_WINDOWS and target.lower().startswith(("ms-", "http:", "https:")):
                os.startfile(target)  # type: ignore[attr-defined]
            elif target.lower().startswith(("http:", "https:")):
                webbrowser.open(target)
            else:
                command = [target] + [str(a) for a in (args or [])]
                flags = 0
                if IS_WINDOWS:
                    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                subprocess.Popen(command, creationflags=flags, close_fds=True)
            logger.info("Launched '%s' (%s)", key, target)
            return True
        except Exception as exc:
            logger.error("Failed to launch '%s' (%s): %s", key, target, exc)
            return False


__all__ = ["MacroEngine", "ApplicationLauncher", "DEFAULT_MACROS", "DEFAULT_APPS"]
