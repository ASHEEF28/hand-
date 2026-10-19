"""GestureOS - AI Gesture Control System Main Application"""
import os

os.environ['MPLBACKEND'] = 'Agg'  # Suppress matplotlib Qt backend (conflicts with PySide6)

import sys
import time
import signal
import logging
import threading
from typing import Optional, List, Tuple, Any
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from utils import cfg_get, try_import
from camera.capture import CameraCapture
from vision.hand_tracker import Hand, HandTracker, HandLandmark, HandTrackingUnavailable
from vision.gaze_tracker import GazeTracker
from vision.background_subtractor import BackgroundSubtractor
from gestures.recognizer import GestureRecognizer, GestureType, GestureEvent
from gestures.classifier import GestureClassifier
from gestures.gesture_learner import GestureLearner
from controllers.mouse_controller import MouseController
from controllers.scroll_controller import ScrollController
from controllers.window_controller import WindowController, DesktopNavigationController
from controllers.media_controller import MediaController, BrightnessController, ZoomController, RotationController
from window_manager.manager import WindowManager
from macros.engine import MacroEngine, ApplicationLauncher
from profiles.manager import ProfileManager, Profile
from voice.assistant import VoiceAssistant
from network.server import GestureNetworkServer
from obs.integration import OBSIntegration
from analytics.tracker import GestureAnalytics
from utils.gesture_history import GestureHistory
from utils.anomaly_detector import AnomalyDetector

# The PySide6 UI is imported lazily inside run_with_ui() so that headless and
# CI runs do not require Qt (or a display) to be installed.

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE = 'logs/gestureos.log'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GestureOS")


def setup_file_logging(path: Optional[str] = DEFAULT_LOG_FILE) -> Optional[str]:
    """Also log to a rotating file (the console disappears when run.bat closes).

    Returns the path used, or ``None`` when file logging is disabled/unavailable.
    """
    if not path:
        return None
    try:
        from logging.handlers import RotatingFileHandler

        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)
        return path
    except OSError as exc:
        logger.warning(f"File logging disabled ({path}): {exc}")
        return None


class GestureOSEngine:
    """Core engine orchestrating all GestureOS modules."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        operation_mode: str = "hand",
        simulate: bool = False,
        dry_run: bool = False,
        tracker: Optional[Any] = None,
        camera: Optional[Any] = None,
    ):
        self.config = get_config(config_path)
        self.running = False
        self.paused = False
        # "pen" is a roadmap feature: the flag and config sections exist, but no
        # pen tracker/recogniser is implemented.  Fall back to hand tracking
        # instead of starting a mode that can never produce a frame.
        self.operation_mode = str(operation_mode or "hand").lower()
        if self.operation_mode != "hand":
            logger.warning(
                f"Operation mode '{self.operation_mode}' is not implemented yet "
                "(see the Roadmap section of README.md) - falling back to hand tracking"
            )
            self.operation_mode = "hand"
        self.simulate = bool(simulate)
        self.dry_run = bool(dry_run)
        self._injected_tracker = tracker
        self._injected_camera = camera
        if self.dry_run:
            # Route every mouse/keyboard action to the recording backend so the
            # app can be exercised without touching the real desktop.  An already
            # installed recorder (tests, a second engine) is reused as-is.
            from utils.input_backend import RecordingBackend, get_backend, set_backend

            existing = get_backend()
            if isinstance(existing, RecordingBackend):
                self._input_backend = existing
            else:
                self._input_backend = RecordingBackend()
                set_backend(self._input_backend)
            logger.warning("Dry-run mode: input actions are recorded, not executed")
        else:
            self._input_backend = None
        self.current_mode = "cursor"
        self.current_gesture = "none"
        self.current_action = "none"
        self.gesture_confidence = 0.0
        self.active_profile_name = "default"

        self.camera: Optional[CameraCapture] = None
        self.tracker: Optional[HandTracker] = None
        self.gaze_tracker: Optional[GazeTracker] = None
        self.background_subtractor: Optional[BackgroundSubtractor] = None
        self.gesture_recognizer: Optional[GestureRecognizer] = None
        self.gesture_classifier: Optional[GestureClassifier] = None
        self.gesture_learner: Optional[GestureLearner] = None
        self.mouse: Optional[MouseController] = None
        self.scroll: Optional[ScrollController] = None
        self.window_ctl: Optional[WindowController] = None
        self.window_mgr: Optional[WindowManager] = None
        self.desktop_nav: Optional[DesktopNavigationController] = None
        self.media: Optional[MediaController] = None
        self.brightness: Optional[BrightnessController] = None
        self.zoom: Optional[ZoomController] = None
        self.rotation: Optional[RotationController] = None
        self.macros: Optional[MacroEngine] = None
        self.app_launcher: Optional[ApplicationLauncher] = None
        self.profiles: Optional[ProfileManager] = None
        self.voice: Optional[VoiceAssistant] = None
        self.network: Optional[GestureNetworkServer] = None
        self.obs: Optional[OBSIntegration] = None
        self.analytics: Optional[GestureAnalytics] = None
        self.gesture_history: Optional[GestureHistory] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.ar_overlay: Optional[Any] = None

        self.ui_app = None
        self.ui_window: Optional[Any] = None

        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0

        self.processing_time = 0
        self._last_processed_hands: List[Hand] = []
        self._last_gaze_data: dict = {}
        self._last_voice_command: Optional[dict] = None
        self._voice_thread: Optional[threading.Thread] = None
        self._network_thread: Optional[threading.Thread] = None
        self._context_app = ""
        self._last_context_update = 0.0
        self._hands_detected_frames = 0
        self._total_frames = 0
        self._ar_menu_active = False
        self._ar_menu_items: List[str] = []
        self._ar_selected = 0

        self._init_modules()

    def _init_modules(self):
        config_dict = self.config.config

        # Camera + hand tracker are needed in every mode.
        camera_config = dict(config_dict.get('camera', {}))
        if self.simulate:
            camera_config['simulate'] = True
            logger.warning("Simulation mode: using a synthetic camera source")
        self.camera = self._injected_camera or CameraCapture(camera_config)

        if self._injected_tracker is not None:
            # Dependency injection: a synthetic detector (demo mode, tests).
            self.tracker = self._injected_tracker
        else:
            self.tracker = HandTracker(config_dict.get('tracking', {}))
            try:
                # Fail fast (with a helpful message) if MediaPipe cannot load.
                self.tracker._ensure_initialized()
            except HandTrackingUnavailable as exc:
                logger.error("Hand tracking unavailable: %s", exc)
                raise SystemExit(str(exc)) from exc

        self._init_hand_modules(config_dict)

        logger.info(
            f"All modules initialized for {self.operation_mode} mode "
            f"(tracker backend={getattr(self.tracker, 'backend', 'unknown')})"
        )

    def _init_hand_modules(self, config_dict: dict):
        """Initialize modules for hand tracking mode."""
        self.gesture_recognizer = GestureRecognizer(config_dict)

        bg_config = config_dict.get('background_subtraction', {})
        if bg_config.get('enabled', False):
            self.background_subtractor = BackgroundSubtractor(bg_config)

        gaze_config = config_dict.get('gaze', {})
        if gaze_config.get('enabled', False):
            self.gaze_tracker = GazeTracker(gaze_config)

        # Input layer first - the other controllers build on it.
        self.mouse = MouseController(config_dict)
        self.scroll = ScrollController(config_dict, mouse=self.mouse)
        self.window_ctl = WindowController(config_dict)
        self.desktop_nav = DesktopNavigationController(config_dict)
        self.window_mgr = WindowManager(config_dict, mouse=self.mouse,
                                        window_controller=self.window_ctl,
                                        navigation=self.desktop_nav)
        self.media = MediaController(config_dict)
        self.brightness = BrightnessController(config_dict)
        self.zoom = ZoomController(config_dict, mouse=self.mouse)
        self.rotation = RotationController(config_dict)
        self.app_launcher = ApplicationLauncher(config_dict)
        self.macros = MacroEngine(config_dict, launcher=self.app_launcher)
        self.profiles = ProfileManager(config_dict)

        gc_config = config_dict.get('gesture_classifier', {})
        if gc_config.get('enabled', False):
            self.gesture_classifier = GestureClassifier(gc_config)

        gl_config = config_dict.get('gesture_learner', {})
        if gl_config.get('enabled', True):
            self.gesture_learner = GestureLearner(gl_config)

        voice_config = config_dict.get('voice', {})
        if voice_config.get('enabled', False):
            self.voice = VoiceAssistant(voice_config)

        net_config = config_dict.get('network', {})
        if net_config.get('enabled', False):
            self.network = GestureNetworkServer(net_config)
            self.network._host = net_config.get('host', '0.0.0.0')
            self.network._port = net_config.get('port', 8765)

        obs_config = config_dict.get('obs', {})
        if obs_config.get('enabled', False):
            self.obs = OBSIntegration(obs_config)

        analytics_config = config_dict.get('analytics', {})
        if analytics_config.get('enabled', True):
            self.analytics = GestureAnalytics(analytics_config)

        gh_config = config_dict.get('gesture_history', {})
        if gh_config.get('enabled', True):
            self.gesture_history = GestureHistory(max_size=gh_config.get('max_size', 50))

        ad_config = config_dict.get('anomaly_detection', {})
        if ad_config.get('enabled', True):
            self.anomaly_detector = AnomalyDetector(ad_config)

        ar_config = config_dict.get('ar_overlay', {})
        if ar_config.get('enabled', False):
            try:
                from ui.ar_overlay import AROverlay

                self.ar_overlay = AROverlay(ar_config)
            except Exception as exc:
                logger.warning(f"AR overlay unavailable: {exc}")

        self._register_gesture_callbacks()

    def _register_gesture_callbacks(self):
        """Wire cross-module callbacks (profiles -> controllers, macros -> UI)."""
        if self.profiles is not None:
            def _on_profile_change(name: str, profile) -> None:
                logger.info(f"Profile '{name}' active: sensitivity={profile.cursor_sensitivity}, "
                            f"smoothing={profile.cursor_smoothing}")
                self.active_profile_name = name
                self._apply_profile()

            self.profiles.on_change = _on_profile_change
            self._apply_profile()
            self.active_profile_name = self.profiles.get_active()

        if self.macros is not None:
            self.macros.on_execute = lambda name, ok: logger.info(
                f"Macro '{name}' {'completed' if ok else 'failed'}")

        if self.gesture_recognizer is not None:
            logger.info(f"Gesture map loaded: {len(self.gesture_recognizer.action_map)} mappings")

    def _do_action(self, action: str, event: GestureEvent) -> None:
        """Dispatch one recognised gesture to the OS controllers.

        (This used to be shadowed by a duplicated ``_check_cooldown`` definition,
        which meant no gesture ever reached the controllers.)
        """
        if not action:
            return

        self.current_gesture = event.gesture.value if event is not None else "none"
        self.current_action = action
        self.gesture_confidence = float(getattr(event, "confidence", 0.0) or 0.0)

        if self.paused:
            return

        hand = getattr(event, "hand", None)
        duration = 0.0
        if hand is not None and hasattr(event, "timestamp"):
            duration = max(0.0, time.time() - float(event.timestamp))

        if self.anomaly_detector:
            if self.anomaly_detector.is_anomaly(action, self.gesture_confidence or 1.0, duration):
                logger.debug(f"Anomaly filtered: {action}")
                return

        now = time.time()

        profile = self.profiles.get_active() if self.profiles else "default"
        if profile != self.active_profile_name:
            self.active_profile_name = profile

        if action == "left_click":
            self.mouse.left_click(double=False)
        elif action == "right_click":
            self.mouse.right_click()
        elif action == "double_click":
            self.mouse.left_click(double=True)
        elif action == "hold_pinch":
            if not self.mouse.is_dragging():
                self.mouse.start_drag()
            else:
                self.mouse.end_drag()
        elif action == "scroll_mode":
            self.current_mode = "scroll" if self.current_mode == "cursor" else "cursor"
            if self.scroll:
                self.scroll.reset()
            logger.info(f"Mode switched to {self.current_mode}")
        elif action == "swipe_left":
            self.window_mgr.switch("next")
        elif action == "swipe_right":
            self.window_mgr.switch("prev")
        elif action == "swipe_up":
            self.window_mgr.task_view()
        elif action == "swipe_down":
            self.window_mgr.show_desktop()
        elif action == "show_desktop":
            self.window_mgr.show_desktop()
        elif action == "start_menu":
            self.window_mgr.start_menu()
        elif action == "task_view":
            self.window_mgr.task_view()
        elif action == "virtual_desktop_left":
            self.window_mgr.virtual_desktop_left()
        elif action == "virtual_desktop_right":
            self.window_mgr.virtual_desktop_right()
        elif action == "grab_window":
            index_tip = hand.landmarks[HandLandmark.INDEX_TIP.value] if hand and len(hand.landmarks) > 8 else None
            position = self._hand_to_screen(index_tip.x, index_tip.y) if index_tip else None
            if self.window_mgr.is_grabbing:
                self.window_mgr.release()
            else:
                self.window_mgr.grab(position)
        elif action == "release_window":
            if self.window_mgr.is_grabbing:
                self.window_mgr.release()
        elif action == "close_window":
            self.window_mgr.close()
        elif action == "maximize_window":
            self.window_mgr.toggle_maximize()
        elif action == "minimize_window":
            self.window_mgr.minimize()
        elif action == "snap_left":
            self.window_mgr.snap("left")
        elif action == "snap_right":
            self.window_mgr.snap("right")
        elif action == "open_explorer":
            self.window_mgr.open_explorer()
        elif action == "open_search":
            self.desktop_nav.search()
        elif action == "open_settings":
            self.desktop_nav.settings()
        elif action == "screenshot":
            if self.macros:
                self.macros.execute("screenshot")
        elif action == "volume_up":
            self.media.volume_up()
        elif action == "volume_down":
            self.media.volume_down()
        elif action == "mute_toggle":
            self.media.mute_toggle()
        elif action == "media_next":
            self.media.next_track()
        elif action == "media_prev":
            self.media.previous_track()
        elif action == "media_play_pause":
            self.media.play_pause()
        elif action in ("rotate_cw", "rotate_ccw"):
            # Rotation is app specific, so it is routed through the macro engine:
            # define a "rotate_cw" / "rotate_ccw" macro to bind it to your app.
            if self.macros and self.macros.get_macro(action):
                self.macros.execute(action)
            else:
                degrees = getattr(event, "data", {}).get("degrees", 0.0)
                logger.debug(f"{action} ({degrees:.1f} deg) - no macro bound")
        elif action in ("zoom_in", "zoom_out"):
            if self.zoom:
                self.zoom.apply(action, steps=max(1, int(abs(getattr(event, "data", {}).get("delta", 0.0)) / 0.03)))
        elif action in ("brightness_up", "brightness_down"):
            if self.brightness:
                data = getattr(event, "data", {}) or {}
                gesture_data = getattr(hand, "gesture_data", {}) or {}
                if "delta" in data:
                    # Two-hand pinch apart/together: stepwise change.
                    steps = max(1, int(abs(float(data["delta"])) / 0.02))
                    amount = steps * float(self.brightness.config.get("step", 5))
                    if action == "brightness_up":
                        self.brightness.brightness_up(amount)
                    else:
                        self.brightness.brightness_down(amount)
                elif gesture_data:
                    # Single hand: continuous control from the pinch distance.
                    pinch_dist = gesture_data.get('pinch_distance', 0.04)
                    if action == "brightness_down":
                        pinch_dist = max(0.01, pinch_dist - 0.02)
                    self.brightness.update(pinch_dist)
        elif action == "ar_menu_toggle":
            self._ar_menu_active = not self._ar_menu_active
            if self._ar_menu_active:
                self._ar_menu_items = ["Left Click", "Right Click", "Scroll", "Volume", "Close"]
                self._ar_selected = 0
        elif action == "ar_menu_select":
            if self._ar_menu_active and self._ar_menu_items:
                selected = self._ar_menu_items[self._ar_selected % len(self._ar_menu_items)]
                self._execute_ar_action(selected)
                self._ar_menu_active = False
        else:
            if self.macros:
                self.macros.execute(action)

        # NOTE: `is not None` - GestureHistory defines __len__, so an *empty*
        # history is falsy and `if self.gesture_history:` would never record the
        # first action.
        if self.gesture_history is not None:
            self.gesture_history.push(action, lambda: None, {
                "gesture": self.current_gesture,
                "confidence": self.gesture_confidence,
                "timestamp": now,
            })

        if self.analytics:
            self.analytics.record_gesture(
                self.current_gesture, action, self.gesture_confidence,
                self.processing_time, True,
            )

        if self.obs and self.obs.is_connected:
            try:
                self.obs.handle_gesture(action)
            except Exception as exc:
                logger.debug(f"OBS gesture handling failed: {exc}")

        if self.ui_window is not None:
            try:
                self.ui_window.notify_gesture(self.current_gesture, action, self.gesture_confidence)
            except Exception:
                pass

        if self.network and hasattr(self.network, 'broadcast'):
            try:
                self.network.broadcast({
                    'gesture': event.gesture.value,
                    'action': action,
                    'confidence': event.confidence,
                    'mode': self.current_mode
                })
            except Exception:
                pass

    def _execute_ar_action(self, action_name: str):
        action_map = {
            "Left Click": "left_click",
            "Right Click": "right_click",
            "Scroll": "scroll_mode",
            "Volume": "volume_up",
            "Close": "close_window"
        }
        mapped = action_map.get(action_name)
        if mapped:
            dummy_event = GestureEvent(GestureType.NONE, None, 1.0, time.time())
            self._do_action(mapped, dummy_event)

    def _hand_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        if self.mouse:
            screen_x = int(norm_x * self.mouse.screen_bounds[2])
            screen_y = int(norm_y * self.mouse.screen_bounds[3])
            return screen_x, screen_y
        return int(norm_x * 1920), int(norm_y * 1080)

    #: Foreground window title fragment -> profile name.
    CONTEXT_PROFILES = {
        'chrome': 'default', 'firefox': 'default', 'edge': 'default', 'browser': 'default',
        'code': 'coding', 'visual studio': 'coding', 'vscode': 'coding', 'pycharm': 'coding',
        'photoshop': 'design', 'illustrator': 'design', 'figma': 'design', 'blender': 'design',
        'minecraft': 'gaming', 'fortnite': 'gaming', 'steam': 'gaming',
        'powerpoint': 'presentation', 'zoom': 'presentation', 'teams': 'presentation',
    }

    def _update_context(self) -> None:
        """Auto-switch profile based on the foreground window (throttled)."""
        if self.profiles is None or not self.config.get('profiles.auto_switch', True):
            return

        now = time.time()
        if now - self._last_context_update < 1.5:
            return
        self._last_context_update = now

        try:
            win32gui = try_import('win32gui')
            if win32gui is None:
                return
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return
            title = (win32gui.GetWindowText(hwnd) or "").lower()
            if not title or title == self._context_app:
                return
            self._context_app = title

            for key, profile in self.CONTEXT_PROFILES.items():
                if key in title:
                    if self.profiles.has_profile(profile) and self.profiles.get_active() != profile:
                        old = self.profiles.get_active()
                        if self.profiles.set_active(profile):
                            logger.info(f"Auto-switched profile: {old} -> {profile} ({title[:40]})")
                    break
        except Exception as exc:
            logger.debug(f"Context update failed: {exc}")

    def start(self):
        """Start the engine."""
        logger.info("Starting GestureOS...")

        if self.camera is not None and not self.camera.start():
            logger.error("Failed to start camera")
            return False

        if self.voice:
            try:
                if self.voice.is_available():
                    self.voice.start()
                    self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
                    self._voice_thread.start()
            except Exception as e:
                logger.warning(f"Voice assistant start failed: {e}")

        if self.network:
            try:
                self.network.set_callback(self._on_network_command)
                self._network_thread = threading.Thread(target=self._network_loop, daemon=True)
                self._network_thread.start()
            except Exception as e:
                logger.warning(f"Network server start failed: {e}")

        if self.obs:
            try:
                self.obs.connect()
            except Exception as e:
                logger.warning(f"OBS connection failed: {e}")

        self.running = True
        logger.info(f"GestureOS started successfully in {self.operation_mode} mode")
        return True

    def _voice_loop(self):
        while self.running and self.voice:
            try:
                event = self.voice.get_event(timeout=0.1)
                if event:
                    command = event.get('action', {})
                    if command and isinstance(command, dict):
                        self._on_voice_command(command)
            except Exception:
                pass
            time.sleep(0.05)

    #: Voice command -> internal action name (``__x__`` entries are handled below).
    VOICE_ACTIONS = {
        'click': 'left_click',
        'right_click': 'right_click',
        'double_click': 'double_click',
        'volume_up': 'volume_up',
        'volume_down': 'volume_down',
        'mute': 'mute_toggle',
        'play_pause': 'media_play_pause',
        'next_track': 'media_next',
        'prev_track': 'media_prev',
        'alt_tab': 'swipe_left',
        'task_view': 'task_view',
        'show_desktop': 'show_desktop',
        'start_menu': 'start_menu',
        'minimize': 'minimize_window',
        'maximize': 'maximize_window',
        'close': 'close_window',
        'screenshot': 'screenshot',
        'zoom_in': 'zoom_in',
        'zoom_out': 'zoom_out',
        # The voice grammar says zoom_up/zoom_down; keep both spellings.
        'zoom_up': 'zoom_in',
        'zoom_down': 'zoom_out',
        'play': 'media_play_pause',
        'stop': 'media_play_pause',
        'brightness_up': 'brightness_up',
        'brightness_down': 'brightness_down',
    }

    def _on_voice_command(self, command: dict):
        action = str(command.get('action', '') or '')
        params = command.get('params', {}) or {}
        self._last_voice_command = command
        logger.info(f"Voice command: {action} {params}")

        # Commands that need their parameters handled specially.
        if action == 'scroll':
            direction = str(params.get('direction', 'down'))
            clicks = int(params.get('clicks', 3))
            if self.mouse:
                self.mouse.scroll(-clicks if direction == 'down' else clicks,
                                  horizontal=direction in ('left', 'right'))
            return
        if action in ('scroll_up', 'scroll_down', 'scroll_left', 'scroll_right'):
            if self.mouse:
                clicks = int(params.get('clicks', 3))
                horizontal = action in ('scroll_left', 'scroll_right')
                positive = action in ('scroll_up', 'scroll_right')
                self.mouse.scroll(clicks if positive else -clicks, horizontal=horizontal)
            return
        if action == 'select_paragraph':
            if self.mouse:
                self.mouse.triple_click()
            return
        if action == 'open':
            if self.app_launcher:
                self.app_launcher.launch(str(params.get('app', '')))
            return
        if action == 'search':
            query = str(params.get('query', ''))
            if query and self.desktop_nav:
                self.desktop_nav.search()
                time.sleep(0.6)
                if self.mouse:
                    self.mouse.write(query)
                    self.mouse.press('enter')
            return
        if action == 'type':
            if self.mouse:
                self.mouse.write(str(params.get('text', '')))
            return
        if action == 'pause':
            self.pause()
            return
        if action == 'resume':
            self.resume()
            return

        mapped = self.VOICE_ACTIONS.get(action)
        if mapped is None:
            logger.debug(f"Unhandled voice command: {action}")
            return

        event = GestureEvent(GestureType.NONE, None, 1.0, time.time())
        event.data = dict(params)
        self._do_action(mapped, event)

    def _on_network_command(self, command: dict) -> bool:
        """Handle a command from a remote client (phone / web dashboard)."""
        action = str(command.get('command', '') or '')
        params = command.get('params', {}) or {}
        logger.info(f"Network command: {action} {params}")

        if not action:
            return False

        # Introspection commands for dashboards.
        if action == 'status':
            return bool(self.network.broadcast(self.snapshot()) if self.network else False)
        if action == 'set_profile':
            profile = str(params.get('profile', ''))
            if self.profiles and self.profiles.set_active(profile):
                return True
            return False
        if action == 'set_mode':
            mode = str(params.get('mode', ''))
            if mode in ('cursor', 'scroll'):
                self.current_mode = mode
                if self.scroll:
                    self.scroll.reset()
                return True
            return False
        if action == 'pause':
            self.pause()
            return True
        if action == 'resume':
            self.resume()
            return True
        if action == 'run_macro':
            return bool(self.macros and self.macros.execute(str(params.get('macro', ''))))
        if action == 'launch_app':
            return bool(self.app_launcher and self.app_launcher.launch(str(params.get('app', ''))))
        if action == 'scroll':
            return bool(self.mouse and self.mouse.scroll(int(params.get('clicks', 3))))
        if action == 'key_press':
            keys = [str(k) for k in (params.get('keys', []) or [])]
            if not keys or not self.mouse:
                return False
            return self.mouse.press(keys[0]) if len(keys) == 1 else self.mouse.hotkey(*keys)
        if action == 'type_text':
            return bool(self.mouse and self.mouse.write(str(params.get('text', ''))))
        if action == 'learn_gesture':
            name = str(params.get('name', ''))
            if self.gesture_learner and name:
                self.gesture_learner.start_recording(name)
                return True
            return False

        mapped = self.VOICE_ACTIONS.get(action, action)
        event = GestureEvent(GestureType.NONE, None, 1.0, time.time())
        event.data = dict(params)
        self._do_action(mapped, event)
        return True

    def snapshot(self) -> dict:
        """Current engine state - used by the UI, the network server and logs."""
        return {
            'running': self.running,
            'paused': self.paused,
            'mode': self.current_mode,
            'operation_mode': self.operation_mode,
            'gesture': self.current_gesture,
            'action': self.current_action,
            'confidence': round(self.gesture_confidence, 3),
            'profile': self.active_profile_name,
            'fps': self.fps,
            'latency_ms': round(self.processing_time, 2),
            'hands': len(self._last_processed_hands),
            'frames': self._total_frames,
            'camera': {
                'running': bool(self.camera and self.camera.is_running()),
                'simulated': bool(self.camera and self.camera.simulated),
                'measured_fps': round(self.camera.measured_fps(), 1) if self.camera else 0.0,
            },
            'tracker_backend': getattr(self.tracker, 'backend', 'unknown'),
            'ar_menu': {'active': self._ar_menu_active, 'selected': self._ar_selected,
                        'items': list(self._ar_menu_items)},
            'voice': bool(self.voice and self.voice.is_running),
            'network': bool(self.network and self.network.is_running),
            'obs': bool(self.obs and self.obs.is_connected),
            'anomalies': self.anomaly_detector.stats()['filtered'] if self.anomaly_detector else 0,
        }

    def _network_loop(self):
        self.network.start(self.network._host, self.network._port)

    def stop(self):
        """Stop the engine."""
        logger.info("Stopping GestureOS...")
        self.running = False

        if self.camera:
            self.camera.stop()
        if self.tracker:
            self.tracker.close()

        if self.mouse and self.mouse.is_dragging():
            self.mouse.end_drag()
        if self.window_mgr:
            try:
                self.window_mgr.shutdown()
            except Exception as exc:
                logger.debug(f"Window manager shutdown error: {exc}")
        if self.mouse:
            try:
                self.mouse.shutdown()
            except Exception as exc:
                logger.debug(f"Mouse controller shutdown error: {exc}")
        if self.macros:
            try:
                self.macros.stop_all()
            except Exception:
                pass
        if self.voice:
            try:
                self.voice.stop()
            except Exception:
                pass
        if self.network:
            try:
                self.network.stop()
            except Exception:
                pass
        if self.obs:
            try:
                self.obs.disconnect()
            except Exception:
                pass
        if self.analytics:
            try:
                self.analytics.save()
            except Exception:
                pass

        if self._network_thread and self._network_thread.is_alive():
            self._network_thread.join(timeout=2.0)

        logger.info("GestureOS stopped")

    def pause(self):
        self.paused = True
        logger.info("GestureOS paused")

    def resume(self):
        self.paused = False
        logger.info("GestureOS resumed")

    def toggle_pause(self):
        if self.paused:
            self.resume()
        else:
            self.pause()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame

        return self._process_hand_frame(frame)

    def _process_hand_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for hand tracking mode."""
        start_time = time.time()

        if self.background_subtractor:
            frame = self.background_subtractor.apply_background_removal(frame)

        hands = self.tracker.process(frame)
        self._last_processed_hands = hands
        self._total_frames += 1
        if hands:
            self._hands_detected_frames += 1

        if self.gaze_tracker:
            self._last_gaze_data = self.gaze_tracker.process(frame)

        gestures = self.gesture_recognizer.process_hands(
            hands, frame_size=(frame.shape[1], frame.shape[0]) if getattr(frame, "shape", None) else None
        )

        if self.gesture_classifier and hands:
            for hand in hands:
                label = self.gesture_classifier.predict_if_confident(hand)
                if label:
                    hand.gesture_data['classifier_label'] = label
                    logger.debug(f"Classifier: {label}")

        if self.gesture_learner and hands:
            for hand in hands:
                match = self.gesture_learner.match(hand)
                if match and match[1] > 0.8:
                    if self._check_cooldown("learned_gesture", 0.5):
                        if self.macros:
                            self.macros.execute(match[0])

        if not self.paused:
            self._execute_actions(hands, gestures)

        self._update_context()

        annotated = self.tracker.draw_landmarks(frame, hands)

        if self.gaze_tracker and self._last_gaze_data.get('face_detected'):
            annotated = self.gaze_tracker.draw_gaze(annotated, self._last_gaze_data)

        self._add_overlay(annotated, hands)

        if self.ar_overlay and self._ar_menu_active and hands:
            hand = hands[0]
            palm = hand.palm_center
            annotated = self.ar_overlay.render_menu(
                annotated, palm, self._ar_menu_items, self._ar_selected
            )

        self.processing_time = (time.time() - start_time) * 1000
        self.frame_count += 1
        self.fps_counter += 1

        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = now
            if self.analytics:
                self.analytics.record_fps(self.fps)

        return annotated

    def _check_cooldown(self, key: str, duration: float) -> bool:
        now = time.time()
        if not hasattr(self, '_cooldowns'):
            self._cooldowns = {}
        if key in self._cooldowns and now - self._cooldowns[key] < duration:
            return False
        self._cooldowns[key] = now
        return True

    def _execute_actions(self, hands: List[Hand], gestures: List[GestureEvent]):
        """Move the cursor / scroll, then dispatch every recognised gesture."""
        if not hands:
            if self.scroll:
                self.scroll.reset()
            if self.rotation:
                self.rotation.reset()
            if self.zoom:
                self.zoom.reset()
            self.current_gesture = "none"
            self.current_action = "none"
            self.gesture_confidence = 0.0
            return

        hand = hands[0]

        # Apply profile tuning only when the active profile actually changes.
        if self.profiles:
            active = self.profiles.get_active()
            if active != self.active_profile_name:
                self._apply_profile()

        index_tip = hand.landmarks[HandLandmark.INDEX_TIP.value] if len(hand.landmarks) > 8 else None

        if self.window_mgr is not None and self.window_mgr.is_grabbing:
            # A window grab owns the cursor while it is active.
            if index_tip is not None:
                self.window_mgr.update_grab(index_tip.x, index_tip.y)
        elif self.current_mode == "cursor":
            if index_tip is not None and self.mouse:
                self.mouse.move_cursor(index_tip.x, index_tip.y)
                if self.mouse.is_dragging():
                    self.mouse.update_drag(index_tip.x, index_tip.y)
        elif self.current_mode == "scroll":
            palm = hand.palm_center
            self.scroll.process(palm[1], palm[0])

        # Keep the UI's "Gesture" readout live even between discrete events.
        if self.gesture_recognizer is not None:
            pose = self.gesture_recognizer.current_pose()
            if pose and pose != "none":
                self.current_gesture = pose

        if self.rotation and len(hand.landmarks) > HandLandmark.PINKY_MCP.value:
            wrist = hand.landmarks[HandLandmark.WRIST.value]
            idx_mcp = hand.landmarks[HandLandmark.INDEX_MCP.value]
            pnk_mcp = hand.landmarks[HandLandmark.PINKY_MCP.value]
            rot_dir = self.rotation.update(wrist, idx_mcp, pnk_mcp)
            if rot_dir == 'clockwise' and self._check_cooldown("rot_cw", 0.3):
                self._do_action("rotate_cw", GestureEvent(GestureType.WRIST_ROTATE_CW, hand, 0.8, time.time(),
                                                          {"degrees": self.rotation.total_rotation}))
            elif rot_dir == 'counter_clockwise' and self._check_cooldown("rot_ccw", 0.3):
                self._do_action("rotate_ccw", GestureEvent(GestureType.WRIST_ROTATE_CCW, hand, 0.8, time.time(),
                                                           {"degrees": self.rotation.total_rotation}))

        if self.zoom and len(hands) == 2:
            # The recogniser emits the zoom events; this only tracks the level.
            self.zoom.update(hands[0].palm_center, hands[1].palm_center, apply=False)

        self._dispatch_gestures(hands, gestures)

    def _dispatch_gestures(self, hands: List[Hand], gestures: List[GestureEvent]) -> None:
        """Turn recognised gesture events into actions (with per-action cooldowns)."""
        if not gestures or self.gesture_recognizer is None:
            return

        hand = hands[0] if hands else None

        # While the AR menu is open, the index fingertip selects an entry and a
        # middle-finger pinch confirms it.
        if self._ar_menu_active and self.ar_overlay is not None and hand is not None:
            index_tip = hand.landmarks[HandLandmark.INDEX_TIP.value] if len(hand.landmarks) > 8 else None
            if index_tip is not None and self._ar_menu_items:
                self._ar_selected = self.ar_overlay.selection_from_finger(
                    hand.palm_center, (index_tip.x, index_tip.y), len(self._ar_menu_items)
                )

        for event in gestures:
            action = self.gesture_recognizer.action_for(event)
            if not action:
                continue

            if self._ar_menu_active and event.gesture == GestureType.MIDDLE_PINCH:
                action = "ar_menu_select"

            cooldown = self._action_cooldown(action)
            if not self._check_cooldown(action, cooldown):
                continue

            try:
                self._do_action(action, event)
            except Exception as exc:
                logger.error(f"Action '{action}' failed: {exc}")
                if self.analytics:
                    self.analytics.record_error(action, str(exc))

    def _apply_profile(self) -> bool:
        """Push the active profile's tuning into the live controllers."""
        if self.profiles is None:
            return False
        try:
            self.profiles.apply(mouse=self.mouse, recognizer=self.gesture_recognizer,
                                scroll=self.scroll, config_manager=self.config)
            self.active_profile_name = self.profiles.get_active()
            return True
        except Exception as exc:
            logger.error(f"Could not apply profile: {exc}")
            return False

    def _action_cooldown(self, action: str) -> float:
        """Per-action cooldown so one gesture cannot machine-gun the OS."""
        gestures_cfg = self.config.get_section('gestures') or {}
        overrides = {
            "left_click": float(gestures_cfg.get('pinch_cooldown', 0.25)),
            "right_click": float(gestures_cfg.get('pinch_cooldown', 0.25)),
            "double_click": 0.15,
            "hold_pinch": float(gestures_cfg.get('hold_cooldown', 0.2)),
            "scroll_mode": 0.6,
            "grab_window": 0.5,
            "ar_menu_toggle": 0.8,
            "ar_menu_select": 0.4,
        }
        if action in overrides:
            return overrides[action]
        if action.startswith("swipe") or action.startswith("media") or action.startswith("volume"):
            return float(gestures_cfg.get('swipe_cooldown', 0.6))
        if action.startswith("brightness") or action.startswith("zoom") or action.startswith("rotate"):
            return 0.25
        return float(gestures_cfg.get('gesture_cooldown', 0.3))

    def _add_overlay(self, frame: np.ndarray, hands: List[Hand]):
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0, 180), -1)

        profile_name = self.profiles.get_active() if self.profiles else "default"
        gesture_conf_str = f" ({self.gesture_confidence:.2f})" if self.gesture_confidence > 0 else ""

        overlay_texts = [
            f"FPS: {self.fps}",
            f"Mode: {self.current_mode.upper()}",
            f"Gesture: {self.current_gesture}{gesture_conf_str}",
            f"Action: {self.current_action}",
            f"Profile: {profile_name}",
            f"Hands: {len(hands)}",
            f"Latency: {self.processing_time:.1f}ms"
        ]

        if self._last_gaze_data.get('face_detected'):
            overlay_texts.append(f"Gaze: {self._last_gaze_data.get('gaze_direction', '?')}")

        if self._last_voice_command:
            v_action = self._last_voice_command.get('action', '')
            overlay_texts.append(f"Voice: {v_action}")

        x_offset = 10
        for text in overlay_texts:
            cv2.putText(frame, text, (x_offset, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 212, 255), 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            x_offset += text_size + 15

        if self.paused:
            cv2.putText(frame, "PAUSED", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 82, 82), 3)

        status_color = (0, 230, 118) if hands else (160, 160, 176)
        cv2.circle(frame, (w - 20, 20), 8, status_color, -1)
        return frame

    def run_headless(self, show_preview: Optional[bool] = None):
        """Run without the Qt UI.

        An OpenCV preview window is shown when a display is available; on a
        headless machine (CI, SSH) it is skipped automatically and status is
        logged once a second instead.
        """
        if not self.start():
            return

        if show_preview is None:
            show_preview = self._display_available()
        if show_preview:
            logger.info("Running headless with OpenCV preview. Press 'q' to quit, 'p' to pause.")
        else:
            logger.info("Running headless (no display). Press Ctrl+C to stop.")

        last_report = time.time()
        try:
            while self.running:
                frame, _timestamp = self.camera.read()
                if frame is None:
                    time.sleep(0.002)
                    continue

                processed = self.process_frame(frame)

                if show_preview:
                    try:
                        cv2.imshow("GestureOS", processed)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('p'):
                            self.toggle_pause()
                    except cv2.error as exc:
                        logger.warning(f"Preview unavailable ({exc}) - continuing without a window")
                        show_preview = False

                if time.time() - last_report >= 1.0:
                    last_report = time.time()
                    snapshot = self.snapshot()
                    logger.info(
                        f"fps={snapshot['fps']} hands={snapshot['hands']} "
                        f"mode={snapshot['mode']} gesture={snapshot['gesture']} "
                        f"action={snapshot['action']} latency={snapshot['latency_ms']}ms"
                    )
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
            if show_preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    @staticmethod
    def _display_available() -> bool:
        """True when an OpenCV window can plausibly be shown."""
        if sys.platform.startswith('win') or sys.platform == 'darwin':
            return True
        return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

    def run_with_ui(self):
        if not self.start():
            return

        try:
            from PySide6.QtWidgets import QApplication

            from ui.main_window import MainWindow
            from ui.onboarding import exec_onboarding
        except Exception as exc:
            logger.error(
                "The PySide6 UI could not be loaded (%s).\n"
                "Install it with: pip install PySide6   - or run with --headless", exc
            )
            self.run_headless()
            return

        self.ui_app = QApplication.instance() or QApplication(sys.argv)

        if self.operation_mode == "hand":
            if self.config.config.get('onboarding', {}).get('show_on_startup', True):
                onboard_cfg = self.config.config.get('onboarding', {})
                if not onboard_cfg.get('completed', False):
                    try:
                        completed = exec_onboarding(self.camera, self.tracker, config=self.config)
                        if completed:
                            self.config.set('onboarding.completed', True)
                            self.config.save()
                    except Exception as e:
                        logger.warning(f"Onboarding error: {e}")

        self.ui_window = MainWindow(self.config.config, self)
        self.ui_window.set_backend_info(f"tracker: {getattr(self.tracker, 'backend', 'unknown')}")
        self.ui_window.show()
        self.ui_app.aboutToQuit.connect(self.stop)

        def ui_update():
            if not self.running:
                return
            if getattr(self, '_processing_frame', False):
                return
            self._processing_frame = True
            try:
                if self.operation_mode == "hand":
                    frame, ts = self.camera.get_latest_frame()
                else:
                    # In pen mode, create a blank frame for display
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    ts = time.time()
                if frame is not None:
                    processed = self.process_frame(frame)
                    self.ui_window.update_camera_feed(
                        processed, self.fps, self.current_mode,
                        self._last_gaze_data.get('gaze_direction', '')
                    )
                    if self._last_processed_hands:
                        h = self._last_processed_hands[0]
                        self.ui_window.update_hand_visualizer(h.landmarks, h.handedness)
                    profile_name = self.profiles.get_active() if self.profiles else "default"
                    gesture_conf = self.gesture_confidence
                    self.ui_window.control_panel.update_status(
                        self.camera.is_running(),
                        len(self._last_processed_hands) > 0,
                        self.current_gesture,
                        self.current_action,
                        profile_name,
                        gesture_conf
                    )
                if self.frame_count % 30 == 0:
                    log_msg = (f"FPS: {self.fps} | Mode: {self.current_mode} | "
                               f"Gesture: {self.current_gesture} | Latency: {self.processing_time:.1f}ms")
                    if self._last_gaze_data.get('face_detected'):
                        log_msg += f" | Gaze: {self._last_gaze_data.get('gaze_direction', '?')}"
                    self.ui_window.add_log(log_msg)
            except Exception:
                pass
            finally:
                self._processing_frame = False

        self.ui_window.update_timer.timeout.connect(ui_update)

        def sigint_handler(signum, frame):
            self.stop()
            self.ui_app.quit()

        signal.signal(signal.SIGINT, sigint_handler)

        try:
            self.ui_app.exec()
        except KeyboardInterrupt:
            self.stop()

    def run(self, use_ui: bool = True):
        if use_ui:
            self.run_with_ui()
        else:
            self.run_headless()


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="GestureOS",
        description="GestureOS - control your PC with hand gestures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py                      normal run with the PySide6 UI\n"
            "  python main.py --headless           no UI, OpenCV preview window\n"
            "  python main.py --check              verify the installation and exit\n"
            "  python main.py --list-gestures      print the gesture -> action table\n"
            "  python main.py --demo               scripted hands + dry-run input (no webcam)\n"
            "  python main.py --demo --frames 120  run 120 frames headless, then exit\n"
        ),
    )
    parser.add_argument('--config', type=str, help='Path to a settings.json file')
    parser.add_argument('--profile', type=str, help='Start with this profile (default, gaming, coding, ...)')
    parser.add_argument('--mode', choices=('hand', 'pen'), default='hand',
                        help='Operation mode (pen mode is a roadmap feature)')

    run_mode = parser.add_argument_group('run modes')
    run_mode.add_argument('--headless', '--no-ui', action='store_true', dest='headless',
                          help='Run without the PySide6 UI (OpenCV preview window)')
    run_mode.add_argument('--simulate', action='store_true',
                          help='Use a synthetic camera source instead of a webcam')
    run_mode.add_argument('--synthetic-hands', action='store_true',
                          help='Replace MediaPipe with a scripted hand source')
    run_mode.add_argument('--demo', action='store_true',
                          help='Full demo: --simulate --synthetic-hands --dry-run --headless')
    run_mode.add_argument('--dry-run', action='store_true',
                          help='Record mouse/keyboard actions instead of sending them to the OS')
    run_mode.add_argument('--frames', type=int, default=0,
                          help='Process N frames then exit (0 = run forever); implies --headless')
    run_mode.add_argument('--camera', type=int, help='Camera device id (overrides settings.json)')
    run_mode.add_argument('--speed', type=float, default=1.0,
                          help='Play the synthetic hand script this many times faster '
                               '(with --synthetic-hands/--demo; 4 = four times real time)')

    info = parser.add_argument_group('diagnostics')
    info.add_argument('--check', action='store_true', help='Check dependencies and exit')
    info.add_argument('--list-gestures', action='store_true', help='Print the gesture table and exit')
    info.add_argument('--log-file', default=DEFAULT_LOG_FILE,
                      help=f"Rotating log file (default: {DEFAULT_LOG_FILE}; '' to disable)")
    info.add_argument('--log-level', default=None, choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
                      help='Override the log level')
    info.add_argument('--version', action='store_true', help='Print the version and exit')
    return parser


def print_gesture_table(config=None) -> None:
    """Print every gesture, its action and how to perform it (no Qt required)."""
    from gestures.help import gesture_rows

    recognizer = GestureRecognizer(config.config if config else {})
    rows = gesture_rows(recognizer.action_map)
    width = max(len(gesture) for gesture, _a, _h in rows) + 2
    print(f"{'GESTURE'.ljust(width)}{'ACTION'.ljust(28)}HOW")
    print("-" * (width + 28 + 46))
    for gesture, action, help_text in rows:
        print(f"{gesture.ljust(width)}{(action or '-').ljust(28)}{help_text}")


def check_installation(config_path=None) -> int:
    """Verify dependencies and modules; returns a process exit code."""
    from utils import module_available

    required = ('cv2', 'numpy')
    recommended = ('mediapipe', 'PySide6', 'pyautogui')
    optional = ('win32gui', 'pycaw', 'screen_brightness_control', 'screeninfo',
                'faster_whisper', 'sounddevice', 'websockets', 'obsws_python',
                'sklearn', 'joblib', 'yaml', 'psutil')

    problems = 0
    print("GestureOS installation check")
    print(f"  Python      : {sys.version.split()[0]} ({sys.platform})")
    print(f"  Platform    : {'Windows' if sys.platform.startswith('win') else sys.platform}")

    for name, group in ((required, 'required'), (recommended, 'recommended'), (optional, 'optional')):
        print(f"  {group.capitalize()} packages:")
        for module in name:
            available = module_available(module)
            mark = "ok" if available else ("MISSING" if group == 'required' else "not installed")
            if group == 'required' and not available:
                problems += 1
            print(f"    {module:<28} {mark}")

    print("  Internal modules:")
    root = Path(__file__).parent
    skipped_dirs = {'tests', 'test', '.git', '.venv', 'venv', '__pycache__', 'build', 'dist'}
    modules = set()
    for path in root.rglob('*.py'):
        relative = path.relative_to(root)
        if '__' in str(path) or skipped_dirs & set(relative.parts):
            continue          # dunder files and the dev-only test-suite
        modules.add('.'.join(relative.with_suffix('').parts))
    for module in sorted(modules):
        try:
            __import__(module)
            print(f"    {module:<28} ok")
        except Exception as exc:
            problems += 1
            print(f"    {module:<28} FAIL: {type(exc).__name__}: {exc}")

    try:
        config = get_config(config_path)
        print(f"  Settings    : {config.config_path} ({len(config.config)} sections)")
    except Exception as exc:
        problems += 1
        print(f"  Settings    : FAIL {exc}")

    if problems:
        print(f"\n{problems} problem(s) found - see the MISSING/FAIL lines above.")
        return 1
    print("\nAll required components are available.")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    setup_file_logging(getattr(args, 'log_file', DEFAULT_LOG_FILE))

    if args.version:
        try:
            from __init__ import __version__  # type: ignore
        except Exception:
            __version__ = "1.0.0"
        print(f"GestureOS {__version__}")
        return 0

    if args.check:
        return check_installation(args.config)

    if args.list_gestures:
        print_gesture_table(get_config(args.config))
        return 0

    simulate = args.simulate or args.demo
    dry_run = args.dry_run or args.demo
    synthetic = args.synthetic_hands or args.demo
    headless = args.headless or args.demo or args.frames > 0

    # The synthetic hand source is built first: --synthetic-hands / --demo must
    # work even when MediaPipe is missing or has no model downloaded.
    tracker = None
    if synthetic:
        from vision.synthetic_hands import SyntheticHandDetector

        try:
            fps = float(cfg_get(get_config(args.config).config, 'camera.fps', 30) or 30)
        except Exception:
            fps = 30.0
        tracker = SyntheticHandDetector(fps=fps, time_scale=float(args.speed or 1.0))
        logger.warning("Synthetic hand source active - MediaPipe is bypassed "
                       f"(speed x{tracker.time_scale:g})")

    engine = GestureOSEngine(
        config_path=args.config,
        operation_mode=args.mode,
        simulate=simulate,
        dry_run=dry_run,
        tracker=tracker,
    )

    if args.camera is not None:
        engine.config.set('camera.device_id', int(args.camera))
        if engine.camera is not None:
            engine.camera.config['device_id'] = int(args.camera)

    if args.profile and engine.profiles is not None:
        if engine.profiles.set_active(args.profile):
            logger.info(f"Using profile '{args.profile}'")
        else:
            logger.warning(f"Unknown profile '{args.profile}' - using '{engine.profiles.get_active()}'")

    if args.frames > 0:
        return run_frames(engine, args.frames)

    try:
        engine.run(use_ui=not headless)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
        return 1
    finally:
        engine.stop()
    return 0


def run_frames(engine: "GestureOSEngine", count: int) -> int:
    """Process exactly ``count`` frames headless, then report a summary.

    Used by the demo mode and by CI: no UI, no webcam, no real input needed.
    """
    if not engine.start():
        logger.error("Engine failed to start")
        return 1

    logger.info(f"Processing {count} frame(s) headless...")
    processed = 0
    stalled_since = time.time()
    try:
        while processed < count and engine.running:
            frame, _timestamp = engine.camera.read(timeout=1.0)
            if frame is None:
                # A dead camera must not spin forever: give up after 5 s.
                if time.time() - stalled_since > 5.0:
                    logger.error("No frames received for 5 s - is the camera in use "
                                 "or unplugged? Try --simulate.")
                    break
                time.sleep(0.005)
                continue
            stalled_since = time.time()
            engine.process_frame(frame)
            processed += 1
            if processed % max(1, count // 10) == 0:
                snapshot = engine.snapshot()
                logger.info(
                    f"  {processed}/{count} frames | fps={snapshot['fps']} | "
                    f"gesture={snapshot['gesture']} | action={snapshot['action']} | "
                    f"mode={snapshot['mode']} | latency={snapshot['latency_ms']}ms"
                )
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        engine.stop()

    snapshot = engine.snapshot()
    print("\nRun summary")
    print(f"  frames processed : {processed}")
    print(f"  frames with hands: {engine._hands_detected_frames}")
    print(f"  tracker backend  : {snapshot['tracker_backend']}")
    print(f"  camera simulated : {snapshot['camera']['simulated']}")
    print(f"  last gesture     : {snapshot['gesture']} -> {snapshot['action']}")
    print(f"  latency          : {snapshot['latency_ms']} ms")
    if engine.analytics:
        summary = engine.analytics.summary()
        print(f"  gestures logged  : {summary['session']['gestures']}")
        print(f"  top actions      : {summary['top_actions'][:5]}")
    if engine.anomaly_detector:
        stats = engine.anomaly_detector.stats()
        print(f"  anomalies blocked: {stats['filtered']} of {stats['seen']}")
    if engine._input_backend is not None:
        kinds = {}
        for event in engine._input_backend.calls():
            kinds[event['kind']] = kinds.get(event['kind'], 0) + 1
        print(f"  dry-run input    : {kinds}")
    tracker = engine.tracker
    if getattr(tracker, 'backend', '') == 'synthetic':
        state = 'finished' if tracker.finished else ('looping' if tracker.loop else 'still running')
        print(f"  synthetic hands  : {tracker.hands_emitted} emitted, script {state}")
        if not tracker.loop and not tracker.finished:
            print("  hint             : raise --frames (or --speed) to see the whole routine")
    if engine.gesture_history is not None and len(engine.gesture_history):
        print(f"  action history   : {engine.gesture_history.counts()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
