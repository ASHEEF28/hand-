"""Configuration management for GestureOS."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "camera": {
        "device_id": 0,
        "width": 1280,
        "height": 720,
        "fps": 60,
        "buffer_size": 1,
        "auto_exposure": True,
        "exposure": -6,
        "brightness": 0,
        "contrast": 0,
        "saturation": 0
    },
    "tracking": {
        "max_hands": 2,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
        "model_complexity": 1,
        "smooth_landmarks": True,
        "smoothing_factor": 0.5,
        "min_presence_confidence": 0.5
    },
    "cursor": {
        "enabled": True,
        "sensitivity": 1.5,
        "smoothing": 0.3,
        "interpolation": "cubic",
        "edge_margin": 0.05,
        "dead_zone": 0.02,
        "movement_region": {
            "x_min": 0.1,
            "y_min": 0.1,
            "x_max": 0.9,
            "y_max": 0.9
        },
        "multi_monitor": True,
        "edge_protection": True,
        "acceleration": 1.2,
        "max_speed": 3000
    },
    "gestures": {
        "pinch_threshold": 0.04,
        "double_pinch_threshold": 0.3,
        "hold_threshold": 0.5,
        "swipe_threshold": 0.15,
        "swipe_velocity_threshold": 500,
        "swipe_cooldown": 0.5,
        "pinch_cooldown": 0.3,
        "hold_cooldown": 0.2,
        "pinch_hold_threshold": 0.5,
        "swipe_min_distance": 0.15,
        "swipe_max_time": 0.5,
        "rotation_threshold": 15.0,
        "pinch_distance_threshold": 0.04,
        "two_hand_distance_threshold": 0.1,
        "zoom_sensitivity": 1.0,
        "rotation_sensitivity": 1.0,
        "gesture_cooldown": 0.3,
        "hold_duration": 0.5,
        "double_click_interval": 0.3,
        "swipe_min_velocity": 500,
        "pinch_max_distance": 0.05
    },
    "cursor_mode": {
        "finger": "index",
        "smoothing_factor": 0.3,
        "sensitivity": 1.5,
        "acceleration": 1.2,
        "dead_zone": 0.02,
        "edge_margin": 0.05,
        "interpolation": "cubic",
        "multi_monitor": True,
        "edge_protection": True,
        "movement_region": {
            "x_min": 0.1,
            "y_min": 0.1,
            "x_max": 0.9,
            "y_max": 0.9
        }
    },
    "profiles": {
        "current": "default",
        "profiles": {
            "default": {
                "name": "Default",
                "description": "Default gesture mappings",
                "gestures": {}
            },
            "gaming": {
                "name": "Gaming",
                "description": "Gaming optimized gestures",
                "gestures": {}
            },
            "coding": {
                "name": "Coding",
                "description": "Developer optimized gestures",
                "gestures": {}
            },
            "design": {
                "name": "Design",
                "description": "Design optimized gestures",
                "gestures": {}
            },
            "presentation": {
                "name": "Presentation",
                "description": "Presentation optimized gestures",
                "gestures": {}
            }
        }
    },
    "ui": {
        "theme": "dark",
        "show_landmarks": True,
        "show_fps": True,
        "show_mode": True,
        "show_gesture": True,
        "show_camera_status": True,
        "show_profile": True,
        "window_width": 1280,
        "window_height": 720,
        "camera_feed_width": 640,
        "camera_feed_height": 360
    },
    "operation_mode": {
        "mode": "hand",
        "show_on_startup": True
    },
    "pen_tracker": {
        "mode": "color",
        "target_color": "red",
        "max_pens": 1,
        "min_detection_confidence": 0.5,
        "smooth_landmarks": True,
        "smoothing_factor": 0.6
    },
    "pen_recognizer": {
        "pen_gestures": {
            "tap_threshold": 0.02,
            "double_tap_interval": 0.3,
            "hold_threshold": 0.5,
            "swipe_threshold": 0.1,
            "swipe_velocity_threshold": 800,
            "swipe_max_time": 0.5
        }
    },
    "pen": {
        "enabled": True,
        "mode": "absolute",
        "sensitivity": 1.5,
        "smoothing": 0.3,
        "pressure_sensitivity": 1.0,
        "double_click_distance": 5,
        "double_click_interval": 0.3,
        "tap_to_click": True,
        "barrel_as_right_click": True,
        "pen_only_mode": False,
        "show_pen_cursor": True
    },
    "calibration": {
        "cursor_sensitivity": 1.5,
        "smoothing": 0.3,
        "pinch_threshold": 0.04,
        "swipe_threshold": 0.15,
        "calibrated": False
    },
    "advanced": {
        "log_level": "INFO",
        "log_file": "gestureos.log",
        "max_log_size": 10485760,
        "log_backup_count": 5,
        "performance_mode": False,
        "low_latency_mode": True,
        "cpu_affinity": [],
        "priority": "high",
        "auto_save_settings": True,
        "check_updates": True,
        "telemetry": False
    },
    "voice": {
        "enabled": False,
        "model_size": "tiny.en",
        "device": "cpu",
        "compute_type": "int8",
        "language": "en",
        "command_cooldown": 1.0,
        "listen_timeout": 3.0
    },
    "network": {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 8765,
        "heartbeat_interval": 5.0
    },
    "obs": {
        "enabled": False,
        "host": "localhost",
        "port": 4455,
        "password": "",
        "reconnect_interval": 5.0
    },
    "gaze": {
        "enabled": False,
        "smoothing": 0.5,
        "min_detection_confidence": 0.5,
        "gaze_actions": {
            "left": "swipe_left",
            "right": "swipe_right",
            "up": "volume_up",
            "down": "volume_down"
        }
    },
    "background_subtraction": {
        "enabled": False,
        "method": "mog2",
        "history": 500,
        "var_threshold": 16,
        "detect_shadows": True,
        "learning_rate": 0.01
    },
    "gesture_classifier": {
        "enabled": False,
        "model_path": "models/gesture_classifier.pkl",
        "confidence_threshold": 0.6,
        "feature_version": 1
    },
    "gesture_learner": {
        "enabled": True,
        "match_threshold": 0.8,
        "history_frames": 15
    },
    "gesture_history": {
        "enabled": True,
        "max_size": 50
    },
    "anomaly_detection": {
        "enabled": True,
        "z_score_threshold": 3.0,
        "window_size": 100
    },
    "analytics": {
        "enabled": True,
        "save_interval": 10,
        "track_accuracy": True
    },
    "onboarding": {
        "completed": False,
        "show_on_startup": True
    },
    "ar_overlay": {
        "enabled": False,
        "menu_radius": 80,
        "opacity": 0.7
    }
}

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "settings.json"
        self.config = DEFAULT_CONFIG.copy()
        self._load_config()
    
    def _load_config(self):
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                self._merge_config(self.config, loaded)
                logger.info(f"Loaded config from {self.config_path}")
            else:
                self._save_config()
                logger.info(f"Created default config at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = DEFAULT_CONFIG.copy()
    
    def _merge_config(self, base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _save_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        if self.config.get('advanced', {}).get('auto_save_settings', True):
            self._save_config()
    
    def get_section(self, section: str) -> dict:
        return self.config.get(section, {})
    
    def update_section(self, section: str, values: dict):
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)
        if self.config.get('advanced', {}).get('auto_save_settings', True):
            self._save_config()
    
    def save(self):
        self._save_config()
    
    def reset_to_defaults(self):
        self.config = DEFAULT_CONFIG.copy()
        self._save_config()

_config_instance = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance







