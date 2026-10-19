"""Human-readable gesture documentation.

Lives outside :mod:`ui` on purpose: the CLI (``python main.py --list-gestures``)
and the tests print this table without importing PySide6.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Fallback action labels used when no live recognizer is available.
DEFAULT_ACTIONS = {
    "pinch": "left_click",
    "double_pinch": "double_click",
    "middle_pinch": "right_click",
    "hold_pinch": "hold_pinch (drag)",
    "scroll_mode": "scroll_mode",
    "swipe_left": "swipe_left (Alt+Tab)",
    "swipe_right": "swipe_right (Alt+Shift+Tab)",
    "swipe_up": "swipe_up (Task View)",
    "swipe_down": "swipe_down (Show Desktop)",
    "three_finger_swipe_left": "media_prev",
    "three_finger_swipe_right": "media_next",
    "three_finger_swipe_up": "volume_up",
    "three_finger_swipe_down": "volume_down",
    "thumbs_up": "media_play_pause",
    "thumbs_down": "mute_toggle",
    "fist": "grab_window",
    "four_finger_pinch": "show_desktop",
    "five_finger_pinch": "start_menu",
    "two_hand_zoom_in": "zoom_in (Ctrl + wheel)",
    "two_hand_zoom_out": "zoom_out (Ctrl + wheel)",
    "brightness_up": "brightness_up",
    "brightness_down": "brightness_down",
    "wrist_rotate_cw": "rotate_cw",
    "wrist_rotate_ccw": "rotate_ccw",
    "rock_sign": "ar_menu_toggle",
}

GESTURE_HELP = {
    "pinch": "Thumb + index fingertip together",
    "double_pinch": "Two quick pinches",
    "middle_pinch": "Thumb + middle fingertip together",
    "hold_pinch": "Keep pinching for ~0.5 s (toggles drag & drop)",
    "scroll_mode": "Index + middle extended, other fingers curled",
    "swipe_left": "Open hand, sweep left",
    "swipe_right": "Open hand, sweep right",
    "swipe_up": "Open hand, sweep up",
    "swipe_down": "Open hand, sweep down",
    "three_finger_swipe_left": "Index + middle + ring, sweep left",
    "three_finger_swipe_right": "Index + middle + ring, sweep right",
    "three_finger_swipe_up": "Index + middle + ring, sweep up",
    "three_finger_swipe_down": "Index + middle + ring, sweep down",
    "thumbs_up": "Thumb up, fingers curled (hold)",
    "thumbs_down": "Thumb down, fingers curled (hold)",
    "fist": "Closed fist (hold) - grabs the window under the cursor",
    "four_finger_pinch": "Thumb touches index + middle + ring, pinky out",
    "five_finger_pinch": "All fingertips gathered on the thumb",
    "two_hand_zoom_in": "Two open palms, move apart",
    "two_hand_zoom_out": "Two open palms, move together",
    "brightness_up": "Two pinched hands, move apart",
    "brightness_down": "Two pinched hands, move together",
    "wrist_rotate_cw": "Rotate your wrist clockwise",
    "wrist_rotate_ccw": "Rotate your wrist counter-clockwise",
    "rock_sign": "Index + pinky extended - opens the AR menu",
}


def gesture_rows(action_map: Optional[Dict[str, str]] = None) -> List[Tuple[str, str, str]]:
    """``(gesture, action, how_to_perform)`` rows for tables and CLI output."""
    mapping: Dict[str, Any] = dict(action_map or {})
    rows: List[Tuple[str, str, str]] = []
    for gesture, how in GESTURE_HELP.items():
        action = str(mapping.get(gesture) or DEFAULT_ACTIONS.get(gesture, ""))
        rows.append((gesture, action, how))
    return rows


__all__ = ["DEFAULT_ACTIONS", "GESTURE_HELP", "gesture_rows"]
