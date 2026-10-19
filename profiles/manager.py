"""Gesture profiles (Default, Gaming, Coding, Design, Presentation).

A profile is a bundle of tuning parameters plus an optional gesture -> action
override map.  ``ProfileManager`` loads them from ``profiles/profiles.json``
(merged with ``settings.json``), tracks the active profile and can push its
values into the running controllers.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from utils import deep_merge, project_root

logger = logging.getLogger(__name__)


@dataclass
class Profile:
    """Tuning preset for a working context."""

    name: str = "default"
    description: str = ""
    cursor_sensitivity: float = 1.5
    cursor_smoothing: float = 0.3
    pinch_threshold: float = 0.06
    swipe_threshold: float = 0.15
    scroll_sensitivity: float = 1.0
    gesture_cooldown: float = 0.3
    gestures: Dict[str, str] = field(default_factory=dict)   # gesture -> action overrides
    extra: Dict[str, Any] = field(default_factory=dict)      # free-form controller settings

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Profile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        profile = cls(**kwargs)
        profile.gestures = dict(data.get("gestures", {}) or {})
        profile.extra = {k: v for k, v in (data or {}).items() if k not in known}
        return profile

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def merged_gesture_map(self, base: Dict[str, str]) -> Dict[str, str]:
        """``base`` action map with this profile's overrides applied."""
        merged = dict(base)
        merged.update(self.gestures or {})
        return merged


DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "name": "default",
        "description": "Balanced settings for general use",
        "cursor_sensitivity": 1.5,
        "cursor_smoothing": 0.30,
        "pinch_threshold": 0.06,
        "swipe_threshold": 0.15,
        "scroll_sensitivity": 1.0,
        "gesture_cooldown": 0.30,
    },
    "gaming": {
        "name": "gaming",
        "description": "Fast cursor, minimal smoothing, tight cooldowns",
        "cursor_sensitivity": 2.4,
        "cursor_smoothing": 0.10,
        "pinch_threshold": 0.07,
        "swipe_threshold": 0.12,
        "scroll_sensitivity": 1.4,
        "gesture_cooldown": 0.15,
    },
    "coding": {
        "name": "coding",
        "description": "Precise cursor with medium smoothing",
        "cursor_sensitivity": 1.2,
        "cursor_smoothing": 0.40,
        "pinch_threshold": 0.05,
        "swipe_threshold": 0.18,
        "scroll_sensitivity": 1.2,
        "gesture_cooldown": 0.25,
    },
    "design": {
        "name": "design",
        "description": "Very precise cursor, high smoothing for fine work",
        "cursor_sensitivity": 0.9,
        "cursor_smoothing": 0.60,
        "pinch_threshold": 0.045,
        "swipe_threshold": 0.20,
        "scroll_sensitivity": 0.8,
        "gesture_cooldown": 0.35,
    },
    "presentation": {
        "name": "presentation",
        "description": "Media and slide friendly: bigger gestures, slower cursor",
        "cursor_sensitivity": 1.6,
        "cursor_smoothing": 0.45,
        "pinch_threshold": 0.08,
        "swipe_threshold": 0.12,
        "scroll_sensitivity": 1.0,
        "gesture_cooldown": 0.45,
        "gestures": {
            "swipe_right": "media_next",
            "swipe_left": "media_prev",
            "swipe_up": "volume_up",
            "swipe_down": "volume_down",
        },
    },
}


class ProfileManager:
    """Loads, switches and applies gesture profiles."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        section = config.get("profiles", {}) if isinstance(config, dict) else {}
        self.config: Dict[str, Any] = deep_merge(
            {"store_path": "profiles/profiles.json", "auto_switch": True},
            section or {},
        )
        self.profiles: Dict[str, Profile] = {}
        self.active_name: str = str(section.get("current", "default") if isinstance(section, dict) else "default")
        self.on_change: Optional[Callable[[str, Profile], None]] = None
        self.switch_history: List[Dict[str, Any]] = []
        self._load(config)

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    @property
    def store_path(self) -> str:
        path = str(self.config.get("store_path", "profiles/profiles.json"))
        return path if os.path.isabs(path) else os.path.join(project_root(), path)

    def _load(self, config: Optional[Dict[str, Any]] = None) -> None:
        # 1) built-in defaults
        for name, data in DEFAULT_PROFILES.items():
            self.profiles[name] = Profile.from_dict(data)

        # 2) values embedded in settings.json
        embedded = (config or {}).get("profiles", {}).get("profiles", {}) if isinstance(config, dict) else {}
        for name, data in (embedded or {}).items():
            if isinstance(data, dict):
                if name in self.profiles:
                    merged = self.profiles[name].to_dict()
                    merged.update({k: v for k, v in data.items() if k != "gestures" or v})
                    self.profiles[name] = Profile.from_dict(merged)
                else:
                    self.profiles[name] = Profile.from_dict(data)

        # 3) profiles/profiles.json wins
        path = self.store_path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                stored = data.get("profiles", data) if isinstance(data, dict) else {}
                for name, profile_data in (stored or {}).items():
                    if isinstance(profile_data, dict):
                        self.profiles[name] = Profile.from_dict(profile_data)
                logger.info("Loaded %d profile(s) from %s", len(stored), path)
            except Exception as exc:
                logger.error("Could not load profiles from %s: %s", path, exc)
        else:
            self.save()

        if self.active_name not in self.profiles:
            self.active_name = "default" if "default" in self.profiles else next(iter(self.profiles))

    def save(self) -> bool:
        path = self.store_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "version": 1,
                        "current": self.active_name,
                        "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()},
                    },
                    handle,
                    indent=2,
                )
            return True
        except Exception as exc:
            logger.error("Could not save profiles: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #
    def list_profiles(self) -> List[str]:
        return sorted(self.profiles.keys())

    def get_active(self) -> str:
        """Name of the active profile (``main.py`` compares this each frame)."""
        return self.active_name

    def get_profile(self, name: Optional[str] = None) -> Profile:
        """Return a profile by name, defaulting to the active one."""
        key = (name or self.active_name or "default").lower()
        if key in self.profiles:
            return self.profiles[key]
        if self.active_name in self.profiles:
            return self.profiles[self.active_name]
        return Profile(name=key)

    def has_profile(self, name: str) -> bool:
        return (name or "").lower() in self.profiles

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def set_active(self, name: str) -> bool:
        key = (name or "").lower()
        if key not in self.profiles:
            logger.warning("Unknown profile '%s' (available: %s)", name, ", ".join(self.list_profiles()))
            return False
        if key == self.active_name:
            return True
        previous = self.active_name
        self.active_name = key
        self.switch_history.append({"from": previous, "to": key, "t": time.time()})
        logger.info("Profile switched: %s -> %s", previous, key)
        if self.on_change:
            try:
                self.on_change(key, self.profiles[key])
            except Exception as exc:  # pragma: no cover
                logger.error("Profile change callback failed: %s", exc)
        self.save()
        return True

    def create(self, name: str, data: Optional[Dict[str, Any]] = None) -> Profile:
        key = (name or "").strip().lower()
        if not key:
            raise ValueError("Profile name must not be empty")
        payload = {"name": key, "description": f"Custom profile {key}"}
        payload.update(data or {})
        profile = Profile.from_dict(payload)
        self.profiles[key] = profile
        self.save()
        return profile

    def update(self, name: str, **fields: Any) -> bool:
        key = (name or "").lower()
        profile = self.profiles.get(key)
        if profile is None:
            return False
        for field_name, value in fields.items():
            if hasattr(profile, field_name):
                setattr(profile, field_name, value)
            else:
                profile.extra[field_name] = value
        return self.save()

    def delete(self, name: str) -> bool:
        key = (name or "").lower()
        if key in ("default",) or key not in self.profiles:
            return False
        del self.profiles[key]
        if self.active_name == key:
            self.active_name = "default"
        return self.save()

    # ------------------------------------------------------------------ #
    # Applying to live controllers
    # ------------------------------------------------------------------ #
    def apply(
        self,
        mouse: Optional[Any] = None,
        recognizer: Optional[Any] = None,
        scroll: Optional[Any] = None,
        config_manager: Optional[Any] = None,
    ) -> Profile:
        """Push the active profile into the running components."""
        profile = self.get_profile()

        if mouse is not None:
            try:
                mouse.set_sensitivity(profile.cursor_sensitivity)
                mouse.set_smoothing(profile.cursor_smoothing)
            except Exception as exc:
                logger.debug("Could not apply cursor settings: %s", exc)

        if scroll is not None and hasattr(scroll, "set_sensitivity"):
            try:
                scroll.set_sensitivity(profile.scroll_sensitivity)
            except Exception as exc:
                logger.debug("Could not apply scroll settings: %s", exc)

        if recognizer is not None:
            try:
                base_map = dict(getattr(recognizer, "action_map", {}) or {})
                recognizer.set_action_map(profile.merged_gesture_map(_strip_overrides(base_map)))
            except Exception as exc:
                logger.debug("Could not apply gesture map: %s", exc)

        if config_manager is not None:
            try:
                config_manager.update_section("calibration", {
                    "cursor_sensitivity": profile.cursor_sensitivity,
                    "smoothing": profile.cursor_smoothing,
                    "pinch_threshold": profile.pinch_threshold,
                    "swipe_threshold": profile.swipe_threshold,
                })
                config_manager.set("profiles.current", profile.name)
            except Exception as exc:
                logger.debug("Could not persist profile settings: %s", exc)

        return profile


def _strip_overrides(action_map: Dict[str, str]) -> Dict[str, str]:
    """Keep the default map clean before applying profile overrides."""
    from gestures.recognizer import DEFAULT_ACTION_MAP

    return dict(DEFAULT_ACTION_MAP) if not action_map else dict(action_map)


__all__ = ["Profile", "ProfileManager", "DEFAULT_PROFILES"]
