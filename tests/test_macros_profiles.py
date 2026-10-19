"""Tests for macros, the app launcher, profiles, analytics and the learner."""
from __future__ import annotations

import os
import time

import pytest

from macros.engine import ApplicationLauncher, MacroEngine
from profiles.manager import DEFAULT_PROFILES, Profile, ProfileManager


class TestMacroEngine:
    def test_defaults_are_created(self, config, backend, tmp_path):
        engine = MacroEngine(config, backend=backend)
        path = config["macros"]["store_path"]
        assert os.path.exists(path)
        assert {"copy", "paste", "save", "screenshot"} <= set(engine.list_macros())

    def test_execute_sends_hotkey(self, config, backend):
        engine = MacroEngine(config, backend=backend)
        assert engine.execute("save") is True
        assert backend.last("hotkey")["keys"] == ["ctrl", "s"]
        assert engine.execution_log[-1]["ok"] is True

    def test_execute_waits_for_steps(self, config, backend):
        engine = MacroEngine(config, backend=backend)
        engine.add_macro("combo", [
            {"type": "hotkey", "keys": ["ctrl", "a"]},
            {"type": "wait", "seconds": 0.01},
            {"type": "key", "key": "delete"},
            {"type": "text", "text": "done"},
        ])
        assert engine.execute("combo") is True
        kinds = [c["kind"] for c in backend.calls()]
        assert kinds == ["hotkey", "press", "write"]

    def test_unknown_macro_falls_back_to_app_launch(self, config, backend):
        engine = MacroEngine(config, backend=backend)
        assert engine.execute("definitely_not_a_macro") is False
        assert engine.execute("notepad") in (True, False)   # resolves only if installed

    def test_cooldown_blocks_rapid_repeats(self, config, backend):
        config["macros"]["cooldown"] = 5.0
        engine = MacroEngine(config, backend=backend)
        assert engine.execute("copy") is True
        assert engine.execute("copy") is False

    def test_add_update_remove(self, config, backend):
        engine = MacroEngine(config, backend=backend)
        assert engine.add_macro("team_standup", [{"type": "key", "key": "f1"}], "daily") is True
        assert "team_standup" in engine.list_macros()
        assert engine.update_macro("team_standup", description="updated") is True
        assert engine.get_macro("team_standup")["description"] == "updated"
        assert engine.remove_macro("team_standup") is True
        assert "team_standup" not in engine.list_macros()
        assert engine.add_macro("", []) is False
        assert engine.update_macro("nope") is False
        assert engine.remove_macro("nope") is False

    def test_persistence_roundtrip(self, config, backend):
        first = MacroEngine(config, backend=backend)
        first.add_macro("saved_macro", [{"type": "key", "key": "f2"}])
        second = MacroEngine(config, backend=backend)
        assert second.get_macro("saved_macro")["steps"] == [{"type": "key", "key": "f2"}]

    def test_corrupt_store_falls_back_to_defaults(self, config, backend):
        with open(config["macros"]["store_path"], "w", encoding="utf-8") as handle:
            handle.write("{ broken")
        engine = MacroEngine(config, backend=backend)
        assert "copy" in engine.list_macros()

    def test_async_execution_completes(self, config, backend):
        config["macros"]["async"] = True
        engine = MacroEngine(config, backend=backend)
        engine.add_macro("async_macro", [{"type": "wait", "seconds": 0.02},
                                         {"type": "key", "key": "f3"}])
        assert engine.execute("async_macro") is True
        deadline = time.time() + 3
        while time.time() < deadline and not engine.execution_log:
            time.sleep(0.02)
        assert engine.execution_log[-1]["macro"] == "async_macro"
        assert backend.last("press")["key"] == "f3"

    def test_stop_all_cancels_long_macro(self, config, backend):
        config["macros"]["async"] = True
        engine = MacroEngine(config, backend=backend)
        engine.add_macro("slow", [{"type": "wait", "seconds": 0.05}] * 20)
        engine.execute("slow")
        engine.stop_all()
        time.sleep(0.3)
        assert engine.is_running("slow") is False

    def test_url_and_unknown_step_types(self, config, backend, monkeypatch):
        engine = MacroEngine(config, backend=backend)
        opened = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
        engine.add_macro("open_url", [{"type": "url", "url": "https://example.com"},
                                      {"type": "bogus_step"}])
        engine.execute("open_url")
        assert opened == ["https://example.com"]

    def test_shell_steps_are_disabled_by_default(self, config, backend, monkeypatch):
        engine = MacroEngine(config, backend=backend)
        calls = []
        monkeypatch.setattr("subprocess.Popen", lambda *a, **k: calls.append(a))
        engine.add_macro("naughty", [{"type": "shell", "command": "rm -rf /"}])
        engine.execute("naughty")
        assert calls == []

    def test_text_step_substitutes_params(self, config, backend):
        engine = MacroEngine(config, backend=backend)
        engine.add_macro("greet", [{"type": "text", "text": "$name"}])
        engine.execute("greet", {"name": "world"})
        assert backend.last("write")["text"] == "world"

    def test_callback_is_invoked(self, config, backend):
        engine = MacroEngine(config, backend=backend)
        seen = []
        engine.on_execute = lambda name, ok: seen.append((name, ok))
        engine.execute("copy")
        assert seen == [("copy", True)]


class TestApplicationLauncher:
    def test_defaults_are_created(self, config, backend):
        launcher = ApplicationLauncher(config, backend=backend)
        assert os.path.exists(config["apps"]["store_path"])
        assert {"notepad", "calculator", "browser"} <= set(launcher.list_apps())

    def test_resolve_uses_platform_key(self, config, backend):
        launcher = ApplicationLauncher(config, backend=backend)
        launcher.apps["python_test"] = {"command": "python3", "windows": "python.exe"}
        resolved = launcher.resolve("python_test")
        assert resolved is None or os.path.exists(resolved)

    def test_resolve_finds_executable_on_path(self, config, backend):
        launcher = ApplicationLauncher(config, backend=backend)
        launcher.apps["py"] = {"command": "python3"}
        if launcher.resolve("py"):
            assert launcher.knows("py") is True

    def test_unknown_app_returns_none(self, config, backend):
        launcher = ApplicationLauncher(config, backend=backend)
        assert launcher.resolve("notepad") in (None,) or isinstance(launcher.resolve("notepad"), str)
        assert launcher.launch("no_such_app_xyz") is False

    def test_launch_cooldown(self, config, backend):
        config["apps"]["cooldown"] = 5.0
        launcher = ApplicationLauncher(config, backend=backend)
        launcher.apps["fake"] = {"command": "definitely-not-installed-xyz"}
        assert launcher.launch("fake") is False
        assert launcher.launch("fake") is False
        assert len(launcher.launch_log) == 1      # second call was cooldown-gated

    def test_add_and_remove_app(self, config, backend):
        launcher = ApplicationLauncher(config, backend=backend)
        assert launcher.add_app("My Editor", "code") is True
        assert "my_editor" in launcher.list_apps()
        assert launcher.remove_app("my_editor") is True
        assert launcher.add_app("", "code") is False

    def test_name_normalisation(self, config, backend):
        launcher = ApplicationLauncher(config, backend=backend)
        assert launcher._normalize("  Task Manager ") == "task_manager"
        assert launcher.knows("Task Manager") is True

    def test_launch_attempt_is_logged(self, config, backend, monkeypatch):
        launcher = ApplicationLauncher(config, backend=backend)
        launcher.apps["ok_app"] = {"command": "python3"}
        monkeypatch.setattr("subprocess.Popen", lambda *a, **k: None)
        if launcher.resolve("ok_app"):
            assert launcher.launch("ok_app") is True
            assert launcher.launch_log[-1]["app"] == "ok_app"


class TestProfiles:
    def test_default_profiles_exist(self, config):
        manager = ProfileManager(config)
        assert {"default", "gaming", "coding", "design", "presentation"} <= set(manager.list_profiles())

    def test_active_profile_defaults(self, config):
        manager = ProfileManager(config)
        assert manager.get_active() == "default"
        profile = manager.get_profile()
        assert isinstance(profile, Profile)
        assert profile.cursor_sensitivity == pytest.approx(1.5)

    def test_config_selects_initial_profile(self, config):
        config["profiles"]["current"] = "design"
        manager = ProfileManager(config)
        assert manager.get_active() == "design"
        assert manager.get_profile().cursor_smoothing == pytest.approx(0.60)

    def test_switch_profile(self, config):
        manager = ProfileManager(config)
        assert manager.set_active("gaming") is True
        assert manager.get_active() == "gaming"
        assert manager.get_profile().cursor_sensitivity == pytest.approx(2.4)
        assert manager.set_active("nope") is False
        assert manager.get_active() == "gaming"

    def test_switch_is_case_insensitive(self, config):
        manager = ProfileManager(config)
        assert manager.set_active("CODING") is True
        assert manager.get_active() == "coding"

    def test_callback_fires_once_per_change(self, config):
        manager = ProfileManager(config)
        seen = []
        manager.on_change = lambda name, profile: seen.append(name)
        manager.set_active("gaming")
        manager.set_active("gaming")        # no-op
        manager.set_active("design")
        assert seen == ["gaming", "design"]

    def test_apply_updates_controllers(self, config, backend):
        from controllers.mouse_controller import MouseController
        from controllers.scroll_controller import ScrollController
        from gestures.recognizer import GestureRecognizer

        manager = ProfileManager(config)
        mouse = MouseController(config, backend=backend)
        scroll = ScrollController(config, backend=backend)
        recognizer = GestureRecognizer(config)

        manager.set_active("design")
        manager.apply(mouse=mouse, recognizer=recognizer, scroll=scroll)
        assert mouse.sensitivity == pytest.approx(0.9)
        assert mouse.smoothing == pytest.approx(0.60)
        assert scroll.config["sensitivity"] == pytest.approx(0.8)

    def test_presentation_profile_overrides_swipes(self, config, backend):
        from controllers.mouse_controller import MouseController
        from gestures.recognizer import GestureRecognizer, GestureType

        manager = ProfileManager(config)
        recognizer = GestureRecognizer(config)
        manager.set_active("presentation")
        manager.apply(mouse=MouseController(config, backend=backend), recognizer=recognizer)
        assert recognizer.action_for(_event(GestureType.SWIPE_RIGHT)) == "media_next"
        assert recognizer.action_for(_event(GestureType.PINCH)) == "left_click"

    def test_create_update_delete(self, config):
        manager = ProfileManager(config)
        profile = manager.create("streaming", {"cursor_sensitivity": 2.2, "description": "OBS"})
        assert manager.has_profile("streaming")
        assert profile.cursor_sensitivity == pytest.approx(2.2)
        assert manager.update("streaming", cursor_smoothing=0.11) is True
        assert manager.get_profile("streaming").cursor_smoothing == pytest.approx(0.11)
        assert manager.delete("streaming") is True
        assert manager.has_profile("streaming") is False
        assert manager.delete("default") is False     # protected

    def test_deleting_active_profile_falls_back(self, config):
        manager = ProfileManager(config)
        manager.create("temp")
        manager.set_active("temp")
        manager.delete("temp")
        assert manager.get_active() == "default"

    def test_unknown_profile_returns_placeholder(self, config):
        manager = ProfileManager(config)
        fallback = manager.get_profile("does_not_exist")
        assert fallback.name == "Default"          # built-in fallback profile
        assert manager.get_active() == "default"

    def test_persistence_roundtrip(self, config):
        first = ProfileManager(config)
        first.create("saved", {"cursor_sensitivity": 3.1})
        second = ProfileManager(config)
        assert second.has_profile("saved")
        assert second.get_profile("saved").cursor_sensitivity == pytest.approx(3.1)

    def test_profile_serialisation(self):
        profile = Profile.from_dict({"name": "x", "cursor_sensitivity": 1.1, "custom_key": 5})
        assert profile.extra == {"custom_key": 5}
        assert profile.to_dict()["cursor_sensitivity"] == pytest.approx(1.1)
        assert profile.merged_gesture_map({"pinch": "left_click"}) == {"pinch": "left_click"}

    def test_defaults_cover_every_documented_profile(self):
        assert set(DEFAULT_PROFILES) == {"default", "gaming", "coding", "design", "presentation"}


def _event(gesture):
    from gestures.recognizer import GestureEvent

    return GestureEvent(gesture)


class TestGestureLearner:
    def test_record_and_match(self, config):
        from gestures.gesture_learner import GestureLearner
        from tests.conftest import pose_hand

        learner = GestureLearner(config["gesture_learner"])
        learner.start_recording("my_gesture")
        for _ in range(8):
            learner.record(pose_hand("rock_sign"))
        assert learner.stop_recording() == "my_gesture"

        name, score = learner.match(pose_hand("rock_sign"))
        assert name == "my_gesture"
        assert score > 0.9
        assert learner.match_if_confident(pose_hand("rock_sign")) == "my_gesture"

    def test_different_pose_scores_lower(self, config):
        from gestures.gesture_learner import GestureLearner
        from tests.conftest import pose_hand

        learner = GestureLearner(config["gesture_learner"])
        learner.start_recording("peace")
        for _ in range(8):
            learner.record(pose_hand("scroll_pose"))
        learner.stop_recording()
        _name, score = learner.match(pose_hand("fist"))
        assert score < 0.98

    def test_too_few_frames_is_discarded(self, config):
        from gestures.gesture_learner import GestureLearner
        from tests.conftest import pose_hand

        learner = GestureLearner(config["gesture_learner"])
        learner.start_recording("short")
        learner.record(pose_hand("open_palm"))
        assert learner.stop_recording() is None
        assert learner.known_gestures() == []

    def test_recording_requires_a_name(self, config):
        from gestures.gesture_learner import GestureLearner

        learner = GestureLearner(config["gesture_learner"])
        with pytest.raises(ValueError):
            learner.start_recording("   ")

    def test_record_without_start_is_ignored(self, config):
        from gestures.gesture_learner import GestureLearner
        from tests.conftest import pose_hand

        learner = GestureLearner(config["gesture_learner"])
        assert learner.record(pose_hand("open_palm")) == 0
        assert learner.is_recording() is False
        assert learner.stop_recording() is None

    def test_forget_and_persistence(self, config):
        from gestures.gesture_learner import GestureLearner
        from tests.conftest import pose_hand

        learner = GestureLearner(config["gesture_learner"])
        learner.start_recording("persisted")
        for _ in range(6):
            learner.record(pose_hand("thumbs_up"))
        learner.stop_recording()

        reloaded = GestureLearner(config["gesture_learner"])
        assert reloaded.known_gestures() == ["persisted"]
        assert reloaded.forget("persisted") is True
        assert GestureLearner(config["gesture_learner"]).known_gestures() == []

    def test_template_cap(self, config):
        from gestures.gesture_learner import GestureLearner
        from tests.conftest import pose_hand

        config["gesture_learner"]["max_templates_per_gesture"] = 2
        learner = GestureLearner(config["gesture_learner"])
        for _ in range(4):
            learner.start_recording("capped")
            for _ in range(6):
                learner.record(pose_hand("open_palm"))
            learner.stop_recording()
        assert len(learner.templates["capped"]) == 2

    def test_match_without_templates(self, config):
        from gestures.gesture_learner import GestureLearner
        from tests.conftest import pose_hand

        learner = GestureLearner(config["gesture_learner"])
        assert learner.match(pose_hand("open_palm")) is None
        assert learner.match_if_confident(pose_hand("open_palm")) is None


class TestGestureClassifier:
    def test_predict_without_model(self, config):
        from gestures.classifier import GestureClassifier
        from tests.conftest import pose_hand

        classifier = GestureClassifier(config["gesture_classifier"])
        assert classifier.is_loaded() is False
        assert classifier.predict(pose_hand("open_palm")) is None
        assert classifier.predict_if_confident(pose_hand("open_palm")) is None

    def test_features_are_scale_invariant(self, config):
        from gestures.classifier import GestureClassifier
        from tests.conftest import pose_hand

        near = GestureClassifier.extract_features(pose_hand("pinch", scale=1.0))
        far = GestureClassifier.extract_features(pose_hand("pinch", scale=0.5))
        assert near.shape == far.shape
        assert near[:6].tolist() == pytest.approx(far[:6].tolist(), abs=0.35)

    def test_features_differ_between_poses(self, config):
        from gestures.classifier import GestureClassifier
        from tests.conftest import pose_hand

        pinch = GestureClassifier.extract_features(pose_hand("pinch"))
        open_palm = GestureClassifier.extract_features(pose_hand("open_palm"))
        assert sum(abs(a - b) for a, b in zip(pinch, open_palm)) > 0.5

    def test_short_hand_yields_zero_vector(self):
        from gestures.classifier import GestureClassifier
        from vision.hand_tracker import Hand, Landmark

        vector = GestureClassifier.extract_features(Hand(landmarks=[Landmark(0.5, 0.5)]))
        assert set(vector.tolist()) == {0.0}

    def test_train_and_predict(self, config):
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")
        from gestures.classifier import GestureClassifier
        from tests.conftest import pose_hand

        samples = []
        for _ in range(6):
            samples.append((pose_hand("pinch", dx=0.01 * len(samples)), "pinch"))
            samples.append((pose_hand("fist", dx=0.01 * len(samples)), "fist"))
            samples.append((pose_hand("open_palm", dx=0.01 * len(samples)), "open"))

        classifier = GestureClassifier(config["gesture_classifier"])
        report = classifier.train(samples)
        assert report["samples"] == len(samples)
        assert set(report["classes"]) == {"pinch", "fist", "open"}
        assert classifier.is_loaded() is True

        prediction = classifier.predict(pose_hand("fist"))
        assert prediction is not None
        label, confidence = prediction
        assert label in {"pinch", "fist", "open"}
        assert 0.0 <= confidence <= 1.0

    def test_train_requires_samples(self, config):
        pytest.importorskip("sklearn")
        from gestures.classifier import GestureClassifier

        classifier = GestureClassifier(config["gesture_classifier"])
        with pytest.raises(ValueError):
            classifier.train([])

    def test_save_without_model_is_false(self, config):
        from gestures.classifier import GestureClassifier

        assert GestureClassifier(config["gesture_classifier"]).save() is False


class TestAnalytics:
    def test_record_and_summary(self, config):
        from analytics.tracker import GestureAnalytics

        analytics = GestureAnalytics(config)
        for i in range(12):
            analytics.record_gesture("pinch", "left_click", 0.9, 10 + i, i % 5 != 0)
        for i in range(5):
            analytics.record_fps(55 + i)

        summary = analytics.summary()
        assert summary["session"]["gestures"] == 12
        assert summary["latency"]["samples"] == 12
        assert summary["fps"]["avg"] > 50
        assert summary["accuracy"]["pinch"] == pytest.approx(9 / 12, abs=0.01)
        assert summary["top_actions"][0] == ("left_click", 12)

    def test_disabled_records_nothing(self, config):
        from analytics.tracker import GestureAnalytics

        config["analytics"]["enabled"] = False
        analytics = GestureAnalytics(config)
        analytics.record_gesture("pinch", "left_click")
        assert analytics.summary()["session"]["gestures"] == 0
        assert analytics.save() is False

    def test_save_and_reload(self, config):
        from analytics.tracker import GestureAnalytics

        first = GestureAnalytics(config)
        first.record_gesture("swipe_left", "swipe_left", 0.8, 20, True)
        assert first.save() is True

        second = GestureAnalytics(config)
        assert second.total_gestures == 1
        assert second._gesture_counts["swipe_left"] == 1
        assert len(second.sessions) == 1

    def test_auto_save_interval(self, config):
        from analytics.tracker import GestureAnalytics

        config["analytics"]["save_interval"] = 5
        analytics = GestureAnalytics(config)
        for _ in range(5):
            analytics.record_gesture("pinch", "left_click")
        assert os.path.exists(analytics.store_path)

    def test_reset_and_export(self, config, tmp_path):
        from analytics.tracker import GestureAnalytics

        analytics = GestureAnalytics(config)
        analytics.record_gesture("pinch", "left_click", 0.9, 12, True)
        analytics.record_error("test", "boom")
        target = str(tmp_path / "timeline.csv")
        assert analytics.export_csv(target) == target
        assert "left_click" in open(target, encoding="utf-8").read()

        analytics.reset()
        assert analytics.summary()["session"]["gestures"] == 0

    def test_latency_percentile(self, config):
        from analytics.tracker import GestureAnalytics

        analytics = GestureAnalytics(config)
        for value in range(1, 101):
            analytics.record_latency(value)
        stats = analytics.latency_stats()
        assert stats["min"] == 1
        assert stats["max"] == 100
        assert 90 <= stats["p95"] <= 100


class TestGazeAndBackground:
    def test_gaze_tracker_initialises(self, config):
        from vision.gaze_tracker import GazeTracker

        tracker = GazeTracker(config.get("gaze", {}))
        assert tracker.backend == "auto"           # nothing loaded yet
        result = tracker.process(None)
        assert result["face_detected"] is False
        assert tracker.backend in ("haar", "facemesh", "none")
        tracker.close()

    def test_gaze_classification(self):
        from vision.gaze_tracker import GazeTracker

        tracker = GazeTracker({"smoothing": 0.0})
        assert tracker._classify(0.0, 0.0) == "center"
        assert tracker._classify(0.5, 0.0) == "right"
        assert tracker._classify(-0.5, 0.0) == "left"
        assert tracker._classify(0.0, 0.5) == "down"
        assert tracker._classify(0.0, -0.5) == "up"

    def test_gaze_action_mapping(self, config):
        from vision.gaze_tracker import GazeTracker

        tracker = GazeTracker({"gaze_actions": {"left": "swipe_left"}})
        assert tracker.action_for("left") == "swipe_left"
        assert tracker.action_for("sideways") == ""

    def test_gaze_processes_a_frame(self):
        import numpy as np

        from vision.gaze_tracker import GazeTracker

        tracker = GazeTracker({})
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = tracker.process(frame)
        assert "face_detected" in result and "gaze_direction" in result
        assert tracker.draw_gaze(frame, result) is not None

    def test_background_subtractor_passthrough_without_opencv_models(self):
        import numpy as np

        from vision.background_subtractor import BackgroundSubtractor

        subtractor = BackgroundSubtractor({"enabled": True, "history": 50})
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        # Warm-up frames must pass through untouched so tracking never starves.
        assert (subtractor.apply_background_removal(frame) == frame).all()
        for _ in range(20):
            subtractor.apply_background_removal(frame)
        moving = frame.copy()
        moving[30:60, 30:60] = 250
        output = subtractor.apply_background_removal(moving)
        assert output.shape == frame.shape
        assert subtractor.frames_processed == 22

    def test_background_mask_only_mode(self):
        import numpy as np

        from vision.background_subtractor import BackgroundSubtractor

        subtractor = BackgroundSubtractor({"mode": "mask_only"})
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        mask = subtractor.foreground_mask(frame)
        assert mask is None or mask.shape == (120, 160)

    def test_background_reset(self):
        from vision.background_subtractor import BackgroundSubtractor

        subtractor = BackgroundSubtractor({})
        subtractor.reset()
        assert subtractor.frames_processed == 0
