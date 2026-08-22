"""GestureOS - AI Gesture Control System Main Application"""
import os
os.environ['MPLBACKEND'] = 'Agg'  # Suppress matplotlib Qt backend (conflicts with PySide6)
import sys
import os
import time
import signal
import logging
import threading
import math
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, ConfigManager
from camera.capture import CameraCapture
from vision.hand_tracker import HandTracker, HandLandmark
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
from ui.ar_overlay import AROverlay
from ui.onboarding import exec_onboarding, exec_mode_selection
from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GestureOS")


class GestureOSEngine:
    """Core engine orchestrating all GestureOS modules."""

    def __init__(self, config_path: Optional[str] = None, operation_mode: str = "hand"):
        self.config = get_config(config_path)
        self.running = False
        self.paused = False
        self.operation_mode = operation_mode  # "hand" or "pen"
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
        self.ar_overlay: Optional[AROverlay] = None

        self.ui_app = None
        self.ui_window: Optional[MainWindow] = None

        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0

        self.processing_time = 0
        self._last_processed_hands: List[Hand] = []
        self._last_gaze_data: dict = {}
        self._voice_thread: Optional[threading.Thread] = None
        self._network_thread: Optional[threading.Thread] = None
        self._context_app = ""
        self._ar_menu_active = False
        self._ar_menu_items: List[str] = []
        self._ar_selected = 0

        self._init_modules()

    def _init_modules(self):
        config_dict = self.config.config

        # Always initialize camera and hand tracker for both modes
        self.camera = CameraCapture(config_dict.get('camera', {}))
        self.tracker = HandTracker(config_dict.get('tracking', {}))

        self._init_hand_modules(config_dict)

        logger.info(f"All modules initialized for {self.operation_mode} mode")

    def _init_hand_modules(self, config_dict: dict):
        """Initialize modules for hand tracking mode."""
        self.gesture_recognizer = GestureRecognizer(config_dict)

        bg_config = config_dict.get('background_subtraction', {})
        if bg_config.get('enabled', False):
            self.background_subtractor = BackgroundSubtractor(bg_config)

        gaze_config = config_dict.get('gaze', {})
        if gaze_config.get('enabled', False):
            self.gaze_tracker = GazeTracker(gaze_config)

        self.mouse = MouseController(config_dict)
        self.scroll = ScrollController(config_dict)
        self.window_mgr = WindowManager(config_dict)
        self.window_ctl = WindowController(config_dict)
        self.desktop_nav = DesktopNavigationController(config_dict)
        self.media = MediaController(config_dict)
        self.brightness = BrightnessController(config_dict)
        self.zoom = ZoomController(config_dict)
        self.rotation = RotationController(config_dict)
        self.macros = MacroEngine(config_dict)
        self.app_launcher = ApplicationLauncher(config_dict)
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
            self.network = GestureNetworkServer()
            self.network._host = net_config.get('host', '0.0.0.0')
            self.network._port = net_config.get('port', 8765)

        obs_config = config_dict.get('obs', {})
        if obs_config.get('enabled', False):
            self.obs = OBSIntegration(obs_config)

        analytics_config = config_dict.get('analytics', {})
        if analytics_config.get('enabled', True):
            self.analytics = GestureAnalytics()

        gh_config = config_dict.get('gesture_history', {})
        if gh_config.get('enabled', True):
            self.gesture_history = GestureHistory(max_size=gh_config.get('max_size', 50))

        ad_config = config_dict.get('anomaly_detection', {})
        if ad_config.get('enabled', True):
            self.anomaly_detector = AnomalyDetector(ad_config)

        ar_config = config_dict.get('ar_overlay', {})
        if ar_config.get('enabled', False):
            self.ar_overlay = AROverlay(ar_config)

        self._register_gesture_callbacks()

    def _register_gesture_callbacks(self):
        """Register callbacks for hand gestures."""
        if self.gesture_recognizer:
            pass

    def _check_cooldown(self, key: str, duration: float) -> bool:
        self.current_gesture = event.gesture.value
        self.current_action = action
        self.gesture_confidence = event.confidence

        if self.paused:
            return

        if self.anomaly_detector:
            duration = time.time() - event.timestamp if hasattr(event, 'timestamp') else 0
            if self.anomaly_detector.is_anomaly(action, event.confidence, duration):
                logger.debug(f"Anomaly filtered: {action}")
                return

        hand = event.hand
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
            if hand and len(hand.landmarks) > 8:
                idx_tip = hand.landmarks[HandLandmark.INDEX_TIP.value]
                sx, sy = self._hand_to_screen(idx_tip.x, idx_tip.y)
                if not self.window_mgr.is_grabbing:
                    self.window_mgr.grab((sx, sy))
                else:
                    self.window_mgr.release()
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
        elif action == "rotate_cw":
            logger.debug("Rotate CW via wrist gesture")
        elif action == "rotate_ccw":
            logger.debug("Rotate CCW via wrist gesture")
        elif action == "zoom_in":
            logger.debug("Zoom in via two-hand gesture")
        elif action == "zoom_out":
            logger.debug("Zoom out via two-hand gesture")
        elif action == "brightness_up":
            if self.brightness and hand.gesture_data:
                pinch_dist = hand.gesture_data.get('pinch_distance', 0.04)
                self.brightness.update(pinch_dist)
        elif action == "brightness_down":
            if self.brightness and hand.gesture_data:
                pinch_dist = hand.gesture_data.get('pinch_distance', 0.04)
                self.brightness.update(max(0.01, pinch_dist - 0.02))
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

        if self.gesture_history:
            self.gesture_history.push(action, lambda: None, {})

        if self.analytics:
            self.analytics.record_gesture(event.gesture.value, action, event.confidence, 0.1, True)

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

    def _update_context(self):
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd).lower()
            app_map = {
                'chrome': 'browsing', 'firefox': 'browsing', 'edge': 'browsing',
                'code': 'coding', 'visual studio': 'coding', 'vscode': 'coding',
                'photoshop': 'design', 'illustrator': 'design', 'figma': 'design',
                'game': 'gaming', 'minecraft': 'gaming', 'fortnite': 'gaming'
            }
            for key, profile in app_map.items():
                if key in title:
                    if self.profiles and self.profiles.get_active() != profile:
                        old = self.profiles.get_active()
                        self.profiles.set_active(profile)
                        logger.info(f"Auto-switched profile: {old} -> {profile}")
                    break
        except Exception:
            pass

    def start(self):
        """Start the engine."""
        logger.info("Starting GestureOS...")

        if self.operation_mode == "hand":
            if not self.camera.start():
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

    def _on_voice_command(self, command: dict):
        action = command.get('action', '')
        params = command.get('params', {})
        self._last_voice_command = command
        logger.info(f"Voice command: {action} {params}")

        if action == 'click':
            self.mouse.left_click()
        elif action == 'right_click':
            self.mouse.right_click()
        elif action == 'double_click':
            self.mouse.left_click(double=True)
        elif action == 'scroll':
            direction = params.get('direction', 'down')
            self.mouse.scroll(3 if direction == 'down' else -3)
        elif action == 'volume_up':
            self.media.volume_up()
        elif action == 'volume_down':
            self.media.volume_down()
        elif action == 'mute':
            self.media.mute_toggle()
        elif action == 'open':
            app = params.get('app', '')
            self.app_launcher.launch(app)
        elif action == 'type':
            text = params.get('text', '')
            import pyautogui
            pyautogui.write(text)
        elif action == 'alt_tab':
            self.window_mgr.switch("next")
        elif action == 'minimize':
            self.window_mgr.minimize()
        elif action == 'close':
            self.window_mgr.close()
        elif action == 'pause':
            self.pause()
        elif action == 'resume':
            self.resume()

    def _on_network_command(self, command: dict):
        action = command.get('command', '')
        params = command.get('params', {})
        logger.info(f"Network command: {action}")

        if action == 'click':
            self.mouse.left_click()
        elif action == 'right_click':
            self.mouse.right_click()
        elif action == 'scroll':
            self.mouse.scroll(params.get('clicks', 3))
        elif action == 'volume_up':
            self.media.volume_up()
        elif action == 'volume_down':
            self.media.volume_down()
        elif action == 'mute':
            self.media.mute_toggle()
        elif action == 'play_pause':
            self.media.play_pause()
        elif action == 'next_track':
            self.media.next_track()
        elif action == 'prev_track':
            self.media.previous_track()
        elif action == 'key_press':
            import pyautogui
            keys = params.get('keys', [])
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
        elif action == 'type_text':
            import pyautogui
            pyautogui.write(params.get('text', ''))

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

        if self.gaze_tracker:
            self._last_gaze_data = self.gaze_tracker.process(frame)

        gestures = self.gesture_recognizer.process_hands(hands)

        if self.gesture_classifier and hands:
            for hand in hands:
                pred = self.gesture_classifier.predict(hand)
                if pred and pred[1] > 0.6:
                    pass

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
        if not hands:
            self.scroll.reset()
            self.current_gesture = "none"
            self.current_action = "none"
            return

        hand = hands[0]
        profile = self.profiles.get_profile() if self.profiles else None
        if profile:
            self.mouse.set_sensitivity(profile.cursor_sensitivity)
            self.mouse.set_smoothing(profile.cursor_smoothing)

        if self.current_mode == "cursor":
            index_tip = hand.landmarks[8]
            self.mouse.move_cursor(index_tip.x, index_tip.y)
            if self.mouse.is_dragging():
                self.mouse.update_drag(index_tip.x, index_tip.y)
        elif self.current_mode == "scroll":
            palm = hand.palm_center
            self.scroll.process(palm[1], palm[0])

        if self.rotation and hands:
            hand_r = hands[0]
            wrist = hand_r.landmarks[HandLandmark.WRIST.value]
            idx_mcp = hand_r.landmarks[HandLandmark.INDEX_MCP.value]
            pnk_mcp = hand_r.landmarks[HandLandmark.PINKY_MCP.value]
            rot_dir = self.rotation.update(wrist, idx_mcp, pnk_mcp)
            if rot_dir == 'clockwise':
                if self._check_cooldown("rot_cw", 0.3):
                    self._do_action("rotate_cw", GestureEvent(GestureType.WRIST_ROTATE_CW, hand_r, 0.8, time.time()))
            elif rot_dir == 'counter_clockwise':
                if self._check_cooldown("rot_ccw", 0.3):
                    self._do_action("rotate_ccw", GestureEvent(GestureType.WRIST_ROTATE_CCW, hand_r, 0.8, time.time()))

        if self.zoom and len(hands) == 2:
            h1_center = hands[0].palm_center
            h2_center = hands[1].palm_center
            self.zoom.update(h1_center, h2_center)

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

    def run_headless(self):
        if not self.start():
            return

        logger.info("Running in headless mode. Press Ctrl+C to stop.")

        try:
            while self.running:
                frame, timestamp = self.camera.read()
                if frame is not None:
                    processed = self.process_frame(frame)
                    cv2.imshow("GestureOS", processed)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.toggle_pause()
                else:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def run_with_ui(self):
        if not self.start():
            return

        from PySide6.QtWidgets import QApplication
        self.ui_app = QApplication(sys.argv)

        if self.operation_mode == "hand":
            if self.config.config.get('onboarding', {}).get('show_on_startup', True):
                onboard_cfg = self.config.config.get('onboarding', {})
                if not onboard_cfg.get('completed', False):
                    try:
                        completed = exec_onboarding(self.camera, self.tracker)
                        if completed:
                            self.config.set('onboarding.completed', True)
                            self.config.save()
                    except Exception as e:
                        logger.warning(f"Onboarding error: {e}")

        self.ui_window = MainWindow(self.config.config, self)
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GestureOS - AI Gesture Control System")
    parser.add_argument('--headless', action='store_true', help='Run without UI')
    parser.add_argument('--config', type=str, help='Path to config file')

    args = parser.parse_args()

    # Use default mode (hand)
    operation_mode = "hand"

    engine = GestureOSEngine(config_path=args.config, operation_mode=operation_mode)

    try:
        engine.run(use_ui=not args.headless)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
    