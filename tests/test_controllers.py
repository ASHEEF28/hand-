"""Tests for the OS control layer: mouse, scroll, window, media, zoom."""
from __future__ import annotations

import math
import time

import pytest

from controllers.media_controller import (
    BrightnessController,
    MediaController,
    RotationController,
    ZoomController,
)
from controllers.mouse_controller import MouseController
from controllers.scroll_controller import ScrollController
from controllers.window_controller import DesktopNavigationController, WindowController
from window_manager.manager import WindowManager


class TestMouseMapping:
    def test_screen_bounds_shape(self, mouse):
        left, top, width, height = mouse.screen_bounds
        assert (left, top) == (0, 0)
        assert width == 1920 and height == 1080

    def test_mirror_maps_right_hand_to_right_screen(self, mouse):
        """MediaPipe sees a mirrored world, so x is flipped by default."""
        assert mouse.normalized_to_screen(0.0, 0.5)[0] == 1919
        assert mouse.normalized_to_screen(1.0, 0.5)[0] == 0
        assert mouse.normalized_to_screen(0.5, 0.5) == (960, 540)

    def test_mirror_can_be_disabled(self, config, backend):
        config["cursor"]["mirror_x"] = False
        mouse = MouseController(config, backend=backend)
        assert mouse.normalized_to_screen(0.0, 0.0) == (0, 0)
        assert mouse.normalized_to_screen(1.0, 1.0) == (1919, 1079)

    def test_movement_region_expands_to_full_screen(self, config, backend):
        """A smaller region means less hand travel for the same cursor travel."""
        config["cursor"]["movement_region"] = {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75}
        mouse = MouseController(config, backend=backend)
        # region corners land on screen corners (x is mirrored)
        assert mouse.normalized_to_screen(0.25, 0.25) == (1919, 0)
        assert mouse.normalized_to_screen(0.75, 0.75) == (0, 1079)
        # the middle of the region is still the middle of the screen
        assert mouse.normalized_to_screen(0.5, 0.5) == (960, 540)
        # leaving the region clamps instead of wrapping
        assert mouse.normalized_to_screen(0.0, 0.0) == (1919, 0)
        assert mouse.normalized_to_screen(1.0, 1.0) == (0, 1079)

    def test_edge_margin_keeps_cursor_off_the_bezel(self, config, backend):
        config["cursor"]["edge_margin"] = 0.1
        mouse = MouseController(config, backend=backend)
        x, y = mouse.normalized_to_screen(0.0, 0.0)
        assert x < 1919 and y > 0

    def test_out_of_range_input_is_clamped(self, mouse):
        assert mouse.normalized_to_screen(-5.0, 9.0) == (1919, 1079)
        assert mouse.normalized_to_screen(2.0, -1.0) == (0, 0)


class TestMouseMovement:
    def test_move_cursor_reaches_target(self, config, backend):
        mouse = MouseController(config, backend=backend)
        target = mouse.move_cursor(0.5, 0.5)
        assert target == (960, 540)
        last = backend.last("move_to")
        assert (last["x"], last["y"]) == (960, 540)

    def test_smoothing_lags_behind_a_jump(self, config, backend):
        config["cursor"]["smoothing"] = 0.8
        mouse = MouseController(config, backend=backend)
        mouse.move_cursor(0.5, 0.5)
        moved = mouse.move_cursor(0.0, 0.5)     # jump to the far edge
        assert moved[0] > 200                   # heavily smoothed, not at the edge yet

    def test_dead_zone_ignores_jitter(self, config, backend):
        config["cursor"]["dead_zone"] = 0.5     # huge dead zone
        mouse = MouseController(config, backend=backend)
        mouse.move_cursor(0.5, 0.5)
        before = backend.count("move_to")
        mouse.move_cursor(0.501, 0.501)
        assert backend.count("move_to") == before

    def test_max_speed_caps_a_huge_jump(self, config, backend):
        config["cursor"]["max_speed"] = 100
        config["cursor"]["smoothing"] = 0.0
        mouse = MouseController(config, backend=backend)
        mouse.move_cursor(0.5, 0.5)
        mouse.move_cursor(0.0, 0.0)
        x, y = backend.last("move_to")["x"], backend.last("move_to")["y"]
        assert math.hypot(x - 960, y - 540) <= 101

    def test_disabled_controller_does_nothing(self, config, backend):
        config["cursor"]["enabled"] = False
        mouse = MouseController(config, backend=backend)
        assert mouse.move_cursor(0.5, 0.5) is None
        assert mouse.left_click(double=True) is False
        assert backend.calls() == []

    def test_sensitivity_and_smoothing_setters_clamp(self, mouse):
        mouse.set_sensitivity(99)
        assert mouse.sensitivity == 10.0
        mouse.set_smoothing(-3)
        assert mouse.smoothing == 0.0

    def test_velocity_is_reported(self, config, backend):
        mouse = MouseController(config, backend=backend)
        mouse.move_cursor(0.1, 0.5)
        mouse.move_cursor(0.9, 0.5)
        assert mouse.velocity() >= 0.0


class TestClicks:
    def test_single_click_is_delayed_then_sent(self, config, backend):
        config["gestures"]["double_click_interval"] = 0.05
        mouse = MouseController(config, backend=backend)
        assert mouse.left_click() is True
        assert backend.count("click") == 0        # waiting to see if it is a double
        time.sleep(0.12)
        assert backend.count("click") == 1
        assert backend.last("click")["clicks"] == 1
        mouse.shutdown()

    def test_double_click_is_coalesced(self, config, backend):
        config["gestures"]["double_click_interval"] = 0.5
        mouse = MouseController(config, backend=backend)
        mouse.left_click()
        mouse.left_click()                        # within the interval
        assert backend.count("click") == 1
        assert backend.last("click")["clicks"] == 2
        assert mouse.stats()["double_clicks"] == 1
        mouse.shutdown()

    def test_explicit_double_click(self, mouse, backend):
        assert mouse.left_click(double=True) is True
        assert backend.last("click")["clicks"] == 2

    def test_right_and_middle_click(self, mouse, backend):
        mouse.right_click()
        assert backend.last("click")["button"] == "right"
        mouse.middle_click()
        assert backend.last("click")["button"] == "middle"
        assert mouse.stats()["right_clicks"] == 1

    def test_click_cooldown(self, config, backend):
        config["cursor"]["click_cooldown"] = 1.0
        mouse = MouseController(config, backend=backend)
        assert mouse.right_click() is True
        assert mouse.right_click() is False       # still cooling down
        mouse.shutdown()

    def test_zero_interval_clicks_immediately(self, config, backend):
        config["cursor"]["double_click_interval"] = 0.0
        config["gestures"]["double_click_interval"] = 0.0
        mouse = MouseController(config, backend=backend)
        mouse.left_click()
        assert backend.count("click") == 1
        mouse.shutdown()


class TestDragging:
    def test_drag_lifecycle(self, config, backend):
        mouse = MouseController(config, backend=backend)
        assert mouse.is_dragging() is False
        assert mouse.start_drag() is True
        assert mouse.is_dragging() is True
        assert mouse.start_drag() is False        # already dragging
        mouse.update_drag(0.7, 0.3)
        assert backend.last("move_to")["x"] < 960  # mirrored: 0.7 -> left of centre
        assert mouse.end_drag() is True
        assert mouse.is_dragging() is False
        assert [c["kind"] for c in backend.calls()] == ["mouse_down", "move_to", "mouse_up"]
        mouse.shutdown()

    def test_toggle_drag(self, config, backend):
        mouse = MouseController(config, backend=backend)
        assert mouse.toggle_drag() is True
        assert mouse.is_dragging() is True
        assert mouse.toggle_drag() is False
        assert mouse.is_dragging() is False

    def test_starting_a_drag_cancels_a_pending_click(self, config, backend):
        config["gestures"]["double_click_interval"] = 0.5
        mouse = MouseController(config, backend=backend)
        mouse.left_click()                        # queued, not yet sent
        mouse.start_drag()
        time.sleep(0.05)
        assert backend.count("click") == 0
        assert backend.count("mouse_down") == 1
        mouse.shutdown()

    def test_shutdown_releases_held_button(self, config, backend):
        mouse = MouseController(config, backend=backend)
        mouse.start_drag()
        mouse.shutdown()
        assert mouse.is_dragging() is False
        assert backend.count("mouse_up") == 1


class TestScrolling:
    def test_scroll_sends_wheel_events(self, mouse, backend):
        assert mouse.scroll(-3) is True
        assert backend.last("scroll")["clicks"] == -3
        assert mouse.scroll(0) is False

    def test_scroll_controller_accumulates(self, config, backend):
        scroll = ScrollController(config, backend=backend)
        scroll.process(0.5, 0.5)                  # anchor
        total = 0
        for i in range(1, 8):
            total += scroll.process(0.5 + i * 0.02, 0.5)["vertical"]
        assert total != 0
        assert backend.count("scroll") > 0

    def test_dead_zone_ignores_tiny_motion(self, config, backend):
        scroll = ScrollController(config, backend=backend)
        scroll.process(0.5, 0.5)
        result = scroll.process(0.5005, 0.5)
        assert result == {"vertical": 0, "horizontal": 0}

    def test_reset_reanchors(self, config, backend):
        scroll = ScrollController(config, backend=backend)
        scroll.process(0.5, 0.5)
        scroll.process(0.9, 0.5)
        scroll.reset()
        assert scroll.active is False
        assert scroll.process(0.9, 0.5) == {"vertical": 0, "horizontal": 0}

    def test_inverted_direction(self, config, backend):
        config["scroll"]["direction"] = "inverted"
        scroll = ScrollController(config, backend=backend)
        scroll.process(0.5, 0.5)
        natural_down = 0
        for i in range(1, 6):
            natural_down += scroll.process(0.5 + i * 0.03, 0.5)["vertical"]
        assert natural_down > 0                   # inverted: hand down scrolls up

    def test_horizontal_scroll(self, config, backend):
        config["scroll"]["horizontal"] = True
        config["scroll"]["horizontal_pixels_per_click"] = 0.01
        scroll = ScrollController(config, backend=backend)
        scroll.process(0.5, 0.5)
        horizontal = 0
        for i in range(1, 8):
            horizontal += scroll.process(0.5, 0.5 + i * 0.03)["horizontal"]
        assert horizontal != 0
        assert scroll.stats()["horizontal"] > 0

    def test_disabled_controller(self, config, backend):
        config["scroll"]["enabled"] = False
        scroll = ScrollController(config, backend=backend)
        scroll.process(0.5, 0.5)
        assert scroll.process(0.9, 0.9) == {"vertical": 0, "horizontal": 0}
        assert backend.calls() == []


class TestMedia:
    def test_volume_uses_media_keys_without_pycaw(self, config, backend):
        media = MediaController(config, backend=backend)
        assert media.has_native_volume is False
        media.volume_up()
        assert backend.last("press")["key"] == "volumeup"
        media.volume_down()
        assert backend.last("press")["key"] == "volumedown"

    def test_transport_keys(self, config, backend):
        media = MediaController(config, backend=backend)
        media.play_pause()
        media.next_track()
        media.previous_track()
        media.stop()
        keys = [c["key"] for c in backend.calls("press")]
        assert keys == ["playpause", "nexttrack", "prevtrack", "stop"]

    def test_mute_toggle(self, config, backend):
        media = MediaController(config, backend=backend)
        assert media.mute_toggle() is True
        assert backend.last("press")["key"] == "volumemute"

    def test_cooldown_suppresses_spam(self, config, backend):
        config["media"]["action_cooldown"] = 5.0
        media = MediaController(config, backend=backend)
        assert media.volume_up() is True
        assert media.volume_up() is False
        assert backend.count("press") == config["media"]["volume_key_step"]

    def test_volume_step_presses_multiple_times(self, config, backend):
        config["media"]["volume_key_step"] = 4
        media = MediaController(config, backend=backend)
        media.volume_up()
        assert backend.count("press") == 4

    def test_set_volume_without_native_backend(self, config, backend):
        media = MediaController(config, backend=backend)
        assert media.set_volume(42) is False
        assert media.get_volume() is None


class TestBrightness:
    def test_step_falls_back_to_hotkeys(self, config, backend):
        brightness = BrightnessController(config, backend=backend)
        if brightness.has_native_brightness:
            pytest.skip("screen-brightness-control is installed on this machine")
        assert brightness.step(10) is True
        assert backend.last("press")["key"] == "brightnessup"
        brightness.step(-10)
        assert backend.last("press")["key"] == "brightnessdown"

    def test_update_maps_pinch_distance_to_percent(self, config, backend):
        brightness = BrightnessController(config, backend=backend)
        brightness.config["smoothing"] = 0.0        # no lag for this assertion
        low = brightness.update(0.02)
        brightness._last_update = 0.0               # bypass the update cooldown
        high = brightness.update(0.22)
        assert low == 0
        assert high == 100

    def test_update_smooths_and_skips_noop(self, config, backend):
        brightness = BrightnessController(config, backend=backend)
        first = brightness.update(0.10)
        second = brightness.update(0.10)           # no meaningful change
        assert first is not None
        assert second is None

    def test_set_brightness_clamps(self, config, backend):
        brightness = BrightnessController(config, backend=backend)
        brightness.set_brightness(250)
        assert brightness._last_percent == 100
        brightness.set_brightness(-20)
        assert brightness._last_percent == 0


class TestZoom:
    def test_zoom_in_and_out(self, config, backend):
        zoom = ZoomController(config, mouse=None, backend=backend)
        zoom.update((0.4, 0.5), (0.6, 0.5))        # prime
        results = []
        for gap in (0.25, 0.32, 0.40, 0.50, 0.42, 0.34):
            results.append(zoom.update((0.5 - gap / 2, 0.5), (0.5 + gap / 2, 0.5)))
        assert "zoom_in" in results
        assert "zoom_out" in results

    def test_apply_uses_ctrl_hotkey_without_mouse(self, config, backend):
        zoom = ZoomController(config, mouse=None, backend=backend)
        assert zoom.apply("zoom_in") is True
        assert backend.last("hotkey")["keys"] == ["ctrl", "plus"]
        zoom.apply("zoom_out")
        assert backend.last("hotkey")["keys"] == ["ctrl", "minus"]

    def test_apply_uses_mouse_zoom(self, config, backend):
        mouse = MouseController(config, backend=backend)
        zoom = ZoomController(config, mouse=mouse, backend=backend)
        zoom.apply("zoom_in", steps=2)
        assert any(c["kind"] == "hotkey" and "ctrl" in c["keys"] for c in backend.calls())

    def test_no_apply_when_disabled(self, config, backend):
        config["zoom"]["enabled"] = False
        zoom = ZoomController(config, backend=backend)
        zoom.update((0.4, 0.5), (0.6, 0.5))
        assert zoom.update((0.2, 0.5), (0.8, 0.5)) is None

    def test_reset_clears_state(self, config, backend):
        zoom = ZoomController(config, backend=backend)
        zoom.update((0.4, 0.5), (0.6, 0.5))
        zoom.reset()
        assert zoom._smoothed is None

    def test_distance_between_accepts_landmarks(self, config, backend):
        from vision.hand_tracker import Landmark

        zoom = ZoomController(config, backend=backend)
        assert zoom.distance_between((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
        assert zoom.distance_between(Landmark(0, 0), Landmark(0, 0.5)) == pytest.approx(0.5)


class TestRotation:
    def _hand_at(self, degrees):
        wrist = (0.5, 0.8)
        index = (0.45, 0.55)
        rad = math.radians(degrees)
        pinky = (0.45 + 0.2 * math.cos(rad), 0.55 + 0.2 * math.sin(rad))
        return wrist, index, pinky

    def test_clockwise_rotation_detected(self, config):
        rotation = RotationController(config)
        directions = []
        for degrees in range(0, 45, 5):
            directions.append(rotation.update(*self._hand_at(degrees)))
        assert "clockwise" in directions

    def test_counter_clockwise_rotation_detected(self, config):
        rotation = RotationController(config)
        directions = []
        for degrees in range(0, -45, -5):
            directions.append(rotation.update(*self._hand_at(degrees)))
        assert "counter_clockwise" in directions

    def test_jitter_is_ignored(self, config):
        rotation = RotationController(config)
        results = []
        for degrees in (0, 0.2, -0.2, 0.15, -0.15, 0.1):
            results.append(rotation.update(*self._hand_at(degrees)))
        assert all(r is None for r in results)

    def test_threshold_is_configurable(self, config):
        config["rotation"]["rotation_threshold"] = 90.0
        rotation = RotationController(config)
        results = [rotation.update(*self._hand_at(d)) for d in range(0, 40, 5)]
        assert all(r is None for r in results)

    def test_idle_resets_accumulator(self, config):
        config["rotation"]["reset_on_idle"] = 0.01
        rotation = RotationController(config)
        rotation.update(*self._hand_at(0))
        rotation.update(*self._hand_at(10))
        time.sleep(0.03)
        assert rotation.update(*self._hand_at(20)) is None   # re-anchored
        assert rotation.total_rotation == pytest.approx(0.0, abs=1e-6)

    def test_disabled(self, config):
        config["rotation"]["enabled"] = False
        rotation = RotationController(config)
        assert all(rotation.update(*self._hand_at(d)) is None for d in range(0, 60, 10))


class TestWindowControllers:
    def test_foreground_window_is_empty_off_windows(self, config, backend):
        window = WindowController(config, backend=backend)
        if window.win32gui is None:
            assert window.get_foreground_window() == {}
            assert window.list_windows() == []
        else:
            pytest.skip("running on Windows with pywin32")

    def test_minimize_uses_hotkey_without_win32(self, config, backend):
        window = WindowController(config, backend=backend)
        assert window.minimize() is True
        if window.win32gui is None:
            assert backend.last("hotkey")["keys"] == ["win", "down"]

    def test_window_action_cooldowns(self, config, backend):
        config["window"]["action_cooldown"] = 5.0
        window = WindowController(config, backend=backend)
        assert window.close() is True
        assert window.close() is False

    @pytest.mark.parametrize("direction,keys", [
        ("left", ["win", "left"]),
        ("right", ["win", "right"]),
        ("top", ["win", "up"]),
        ("bottom", ["win", "down"]),
        ("top_left", ["win", "left", "up"]),
    ])
    def test_snap_directions(self, config, backend, direction, keys):
        window = WindowController(config, backend=backend)
        assert window.snap(direction) is True
        assert backend.last("hotkey")["keys"] == keys

    def test_unknown_snap_direction(self, config, backend):
        window = WindowController(config, backend=backend)
        assert window.snap("sideways") is False

    def test_move_and_resize_without_win32(self, config, backend):
        window = WindowController(config, backend=backend)
        if window.win32gui is not None:
            pytest.skip("running on Windows with pywin32")
        assert window.move_window(10, 10) is False
        assert window.resize_window(10, 10) is False


class TestDesktopNavigation:
    @pytest.mark.parametrize("method,keys", [
        ("task_view", ["win", "tab"]),
        ("show_desktop", ["win", "d"]),
        ("open_explorer", ["win", "e"]),
        ("search", ["win", "s"]),
        ("settings", ["win", "i"]),
        ("lock", ["win", "l"]),
        ("virtual_desktop_left", ["win", "ctrl", "left"]),
        ("virtual_desktop_right", ["win", "ctrl", "right"]),
        ("new_virtual_desktop", ["win", "ctrl", "d"]),
    ])
    def test_shell_hotkeys(self, config, backend, method, keys):
        nav = DesktopNavigationController(config, backend=backend)
        assert getattr(nav, method)() is True
        assert backend.last("hotkey")["keys"] == keys

    def test_start_menu_presses_win(self, config, backend):
        nav = DesktopNavigationController(config, backend=backend)
        assert nav.start_menu() is True
        assert backend.last("press")["key"] == "win"

    def test_switch_window_directions(self, config, backend):
        nav = DesktopNavigationController(config, backend=backend)
        nav.switch_window("next")
        assert backend.last("hotkey")["keys"] == ["alt", "tab"]
        nav.switch_window("prev")
        assert backend.last("hotkey")["keys"] == ["alt", "shift", "tab"]

    def test_hotkeys_are_configurable(self, config, backend):
        config["navigation"]["task_view_hotkey"] = ["ctrl", "alt", "t"]
        nav = DesktopNavigationController(config, backend=backend)
        nav.task_view()
        assert backend.last("hotkey")["keys"] == ["ctrl", "alt", "t"]


class TestWindowManager:
    def test_grab_move_release_cycle(self, config, backend):
        manager = WindowManager(config, backend=backend)
        assert manager.is_grabbing is False
        assert manager.grab((500, 400)) is True
        assert manager.is_grabbing is True
        assert manager.move((700, 500)) is True
        assert manager.release() is True
        assert manager.is_grabbing is False
        assert [h["action"] for h in manager.history] == ["grab", "release"]

    def test_grab_twice_is_rejected(self, config, backend):
        manager = WindowManager(config, backend=backend)
        manager.grab((500, 400))
        assert manager.grab((600, 400)) is False
        manager.release()

    def test_move_without_grab_is_noop(self, config, backend):
        manager = WindowManager(config, backend=backend)
        assert manager.move((100, 100)) is False

    def test_release_without_grab_is_noop(self, config, backend):
        manager = WindowManager(config, backend=backend)
        assert manager.release() is False

    def test_toggle_grab(self, config, backend):
        manager = WindowManager(config, backend=backend)
        assert manager.toggle_grab((300, 300)) is True
        assert manager.is_grabbing is True
        assert manager.toggle_grab() is False
        assert manager.is_grabbing is False

    def test_small_movement_does_not_snap(self, config, backend):
        manager = WindowManager(config, backend=backend)
        manager.grab((960, 540))
        manager.move((964, 542))
        manager.release()
        assert not any(c["kind"] == "hotkey" for c in backend.calls())

    def test_drop_at_left_edge_snaps_left(self, config, backend):
        manager = WindowManager(config, backend=backend)
        manager.grab((960, 540))
        for x in range(900, 20, -40):
            manager.move((x, 540))
        manager.move((5, 540))
        manager.release()
        assert any(c.get("keys") == ["win", "left"] for c in backend.calls("hotkey"))

    def test_drop_at_top_right_snaps_to_quadrant(self, config, backend):
        manager = WindowManager(config, backend=backend)
        manager.grab((500, 500))
        manager.move((1900, 40))
        manager.release()
        assert any(c.get("keys") == ["win", "right", "up"] for c in backend.calls("hotkey"))

    def test_snap_detection_helper(self, config, backend):
        manager = WindowManager(config, backend=backend)
        assert manager._snap_direction_for((1915, 540)) == "right"
        assert manager._snap_direction_for((5, 540)) == "left"
        assert manager._snap_direction_for((960, 1070)) is None
        assert manager._snap_direction_for((960, 5)) == "top"

    def test_snap_can_be_disabled(self, config, backend):
        config["window_manager"]["snap_enabled"] = False
        manager = WindowManager(config, backend=backend)
        manager.grab((500, 500))
        manager.move((5, 540))
        manager.release()
        assert not any(c["kind"] == "hotkey" for c in backend.calls())

    def test_navigation_is_delegated(self, config, backend):
        manager = WindowManager(config, backend=backend)
        assert manager.task_view() is True
        assert manager.show_desktop() is True
        assert manager.start_menu() is True
        assert manager.switch("next") is True
        assert manager.virtual_desktop_right() is True
        assert manager.minimize() is True
        assert manager.close() is True
        actions = [h["action"] for h in manager.history]
        assert {"task_view", "show_desktop", "start_menu", "switch"} <= set(actions)

    def test_shutdown_releases_grab(self, config, backend):
        manager = WindowManager(config, backend=backend)
        manager.grab((500, 500))
        manager.shutdown()
        assert manager.is_grabbing is False
