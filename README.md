# GestureOS — AI Gesture Control System

Control your PC with hand gestures through a webcam. GestureOS reads your hand with
MediaPipe, turns landmarks into gestures, and drives the real mouse, keyboard and
window manager — with a live PySide6 dashboard, profiles, macros, voice commands and
a dry-run mode so you can try everything without a webcam.

```
webcam ──> camera/capture.py ──> vision/hand_tracker.py ──> gestures/recognizer.py
                                                                     │
                        controllers/* ── window_manager ── macros ───┘
                                    │
                                    └──> PyAutoGUI / pywin32 (or a recording backend in --dry-run)
```

---

## Quick start

```bash
pip install -r requirements.txt      # or double-click install_deps.bat
python main.py --check               # verify the installation
python main.py                       # start with the full UI
```

No webcam, no display, or just curious? Run the built-in demo — a scripted synthetic
hand performs the whole gesture repertoire while all input is recorded instead of
executed:

```bash
python main.py --demo --frames 300 --speed 4
```

### Command line

| Flag | What it does |
|------|--------------|
| *(none)* | Start with the PySide6 UI (falls back to headless if Qt is missing) |
| `--headless`, `--no-ui` | Run without the UI; logs gesture/action/FPS once per second |
| `--demo` | `--simulate --synthetic-hands --dry-run --headless` — the full scripted demo |
| `--frames N` | Process N frames, print a run summary, exit (implies `--headless`) |
| `--simulate` | Use the synthetic camera source instead of a webcam |
| `--synthetic-hands` | Bypass MediaPipe with a scripted hand (works with no model installed) |
| `--speed X` | Play the synthetic hand script X times faster than real time |
| `--dry-run` | Record every mouse/keyboard action instead of sending it to the OS |
| `--profile NAME` | Start with a profile (`default`, `gaming`, `coding`, `design`, `presentation`) |
| `--mode {hand,pen}` | Operation mode — **pen mode is a roadmap feature**, not implemented |
| `--camera ID` | Camera device id (overrides `settings.json`) |
| `--check` | Dependency + import + settings report; exit code 0/1 |
| `--list-gestures` | Print the gesture → action table (no Qt required) |
| `--log-file PATH` | Rotating log file (default `logs/gestureos.log`, `""` disables) |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--version` | Print the version and exit |

Examples:

```bash
python main.py --check                       # is my install complete?
python main.py --list-gestures               # what can I do with my hands?
python main.py --demo --frames 120           # CI-friendly smoke run
python main.py --headless --profile gaming   # no UI, gaming tuning
python main.py --camera 1 --log-level DEBUG  # second webcam, verbose logs
```

---

## Gestures

`python main.py --list-gestures` prints this table live from your current mapping.

| Gesture | Action | How |
|---------|--------|-----|
| `pinch` | `left_click` | Thumb + index fingertip together |
| `double_pinch` | `double_click` | Two quick pinches |
| `middle_pinch` | `right_click` | Thumb + middle fingertip together |
| `hold_pinch` | `hold_pinch` | Keep pinching for ~0.5 s (toggles drag & drop) |
| `scroll_mode` | `scroll_mode` | Index + middle extended, other fingers curled |
| `swipe_left` | `swipe_left` | Open hand, sweep left |
| `swipe_right` | `swipe_right` | Open hand, sweep right |
| `swipe_up` | `swipe_up` | Open hand, sweep up |
| `swipe_down` | `swipe_down` | Open hand, sweep down |
| `three_finger_swipe_left` | `media_prev` | Index + middle + ring, sweep left |
| `three_finger_swipe_right` | `media_next` | Index + middle + ring, sweep right |
| `three_finger_swipe_up` | `volume_up` | Index + middle + ring, sweep up |
| `three_finger_swipe_down` | `volume_down` | Index + middle + ring, sweep down |
| `thumbs_up` | `media_play_pause` | Thumb up, fingers curled (hold) |
| `thumbs_down` | `mute_toggle` | Thumb down, fingers curled (hold) |
| `fist` | `grab_window` | Closed fist (hold) — grabs the window under the cursor |
| `four_finger_pinch` | `show_desktop` | Thumb touches index + middle + ring, pinky out |
| `five_finger_pinch` | `start_menu` | All fingertips gathered on the thumb |
| `two_hand_zoom_in` | `zoom_in` | Two open palms, move apart |
| `two_hand_zoom_out` | `zoom_out` | Two open palms, move together |
| `brightness_up` | `brightness_up` | Two pinched hands, move apart |
| `brightness_down` | `brightness_down` | Two pinched hands, move together |
| `wrist_rotate_cw` | `rotate_cw` | Rotate your wrist clockwise |
| `wrist_rotate_ccw` | `rotate_ccw` | Rotate your wrist counter-clockwise |
| `rock_sign` | `ar_menu_toggle` | Index + pinky extended — opens the AR menu |

Extra actions the engine understands: `open_explorer`, `open_search`, `open_settings`,
`screenshot`, `task_view`, `virtual_desktop_left/right`, `minimize_window`,
`maximize_window`, `close_window`, `snap_left/right`, `release_window`, `ar_menu_select`,
plus every macro name in `macros/macros.json`.

Detection is deliberately conservative: a pose must be held for `gestures.hold_duration`
(0.35 s) and then respects a per-gesture cooldown, so a hand passing through frame does
not machine-gun your desktop. Pinch detection is **scale invariant** — the thumb–index
distance is divided by hand size, so it works whether you sit 30 cm or 2 m away.

---

## Installation

### Requirements

- Python 3.10–3.12 (3.11+ recommended)
- Windows 10/11 for the full feature set; macOS and Linux run the tracking, UI and
  generic input paths (window management, pycaw volume and some hotkeys are Windows-only)
- A webcam — or none at all, see `--simulate` / `--demo`

### Setup

```bash
git clone https://github.com/ASHEEF28/hand-.git GestureOS
cd GestureOS
python -m venv .venv
.venv\Scripts\activate           # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py --check
```

`install_deps.bat` does the venv-less equivalent on Windows; `run.bat` starts the app.

### About MediaPipe

`requirements.txt` pins **`mediapipe>=0.10.14,<1.0`** on purpose. MediaPipe 1.x removed
the bundled `mp.solutions.hands` API and its model files. `vision/hand_tracker.py`
supports both backends:

1. **`solutions`** (MediaPipe 0.10.x) — works offline, zero setup. Preferred.
2. **`tasks`** (MediaPipe 1.x) — needs `models/hand_landmarker.task`, downloadable from
   <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task>.
   Put the file in `models/` and set `tracking.backend = "tasks"` if you must use 1.x.

If neither is available, startup fails fast with that exact message instead of crashing
mid-frame — and `--synthetic-hands` still lets you exercise the whole pipeline.

### Optional extras

Everything below degrades gracefully; `python main.py --check` shows what is missing.

| Package | Enables |
|---------|---------|
| `pywin32`, `pycaw`, `comtypes` | Native window management and volume (Windows) |
| `screen-brightness-control`, `screeninfo` | Real brightness control, multi-monitor bounds |
| `faster-whisper`, `sounddevice`, `soundfile` | Voice commands ("GestureOS, volume up") |
| `websockets` | The network control server (phone/web remote) |
| `obsws-python` | OBS scene/recording control from gestures |
| `scikit-learn`, `joblib` | The trainable gesture classifier |
| `pyyaml`, `psutil` | Macro import/export, resource-aware frame pacing |

---

## The UI

`python main.py` opens a six-tab dashboard:

- **Gestures** — live camera feed with landmark overlay, hand visualiser, control panel
  (start/stop/pause, profile, mode, FPS/latency/confidence) and a scrolling log.
- **Calibration** — live-tune cursor sensitivity, smoothing, pinch sensitivity, swipe
  distance and cooldowns; values are written to `settings.json` and applied immediately.
- **Profiles** — switch, create, edit and delete profiles.
- **Macros** — the 15 built-in macros plus an editor (JSON step list) and an app launcher
  (9 shortcuts).
- **Analytics** — session gesture counts, accuracy per gesture, FPS and latency
  percentiles, error log.
- **Settings** — edit the raw JSON with validation.

An onboarding wizard runs on first launch (camera check, hand detection, mode choice);
it is skipped when `onboarding.completed` is true or when running headless.

The **AR overlay** draws an in-frame radial menu on `rock_sign`; point at an item and
close your fist (or `ar_menu_select`) to run it.

---

## Configuration

All settings live in `settings.json` (35 sections), created from `config.DEFAULT_CONFIG`
on first run and merged — never clobbered — on later runs. Edit it directly or through
the Settings/Calibration tabs.

| Section | Highlights |
|---------|-----------|
| `camera` | `device_id`, `width`/`height`/`fps`, `backend`, `simulate`, `simulate_on_failure` |
| `tracking` | `max_hands`, detection/tracking confidence, `backend` (`auto`/`solutions`/`tasks`), `model_path`, smoothing |
| `cursor` | `sensitivity`, `smoothing`, `mirror_x`, `edge_margin`, `dead_zone`, `movement_region`, `max_speed`, `double_click_interval` |
| `gestures` | `pinch_ratio_threshold`, `pinch_release_ratio`, `hold_duration`, all cooldowns, `swipe_min_distance`/`swipe_max_time`, `two_hand_delta_threshold`, `action_map` |
| `scroll`, `media`, `brightness`, `zoom`, `rotation` | Per-controller sensitivity, dead zones, cooldowns, hotkey fallbacks |
| `window`, `navigation`, `window_manager` | Snap thresholds, grab behaviour, hotkey maps |
| `macros`, `apps`, `profiles` | Store paths, cooldowns, `allow_shell` (default **false**), auto-switch |
| `voice`, `network`, `obs` | Wake word, model size, host/port, OBS scene gestures |
| `gaze`, `background_subtraction`, `gesture_classifier`, `gesture_learner` | Experimental modules, all off by default |
| `analytics`, `gesture_history`, `anomaly_detection` | Rolling stats, replay history, spam filtering |
| `ui`, `ar_overlay`, `onboarding`, `advanced` | Refresh rate, log lines, overlay style, first-run state |

Runtime data (regenerated automatically, git-ignored):

```
macros/macros.json      macro definitions (15 built in)
macros/apps.json        app launcher shortcuts (9 built in)
profiles/profiles.json  profiles (5 built in)
gestures/learned.json   templates for user-taught gestures
analytics/analytics.json rolling session statistics
models/                 optional MediaPipe Tasks model
```

### Profiles

| Profile | Cursor | Notable mapping changes |
|---------|--------|------------------------|
| `default` | sensitivity 1.5, smoothing 0.30 | Balanced |
| `gaming` | 2.4 / 0.15 | Fast, twitchy, minimal smoothing |
| `coding` | 1.2 / 0.45 | Precise, heavily smoothed |
| `design` | 0.9 / 0.60 | Very precise, drag-friendly |
| `presentation` | 1.0 / 0.40 | Swipes become media next/prev, volume up/down |

`profiles.auto_switch` (default on) switches profile from the foreground window title,
using `GestureOSEngine.CONTEXT_PROFILES` in `main.py`:

| Window title contains | Profile |
|-----------------------|---------|
| `code`, `visual studio`, `vscode`, `pycharm` | `coding` |
| `photoshop`, `illustrator`, `figma`, `blender` | `design` |
| `minecraft`, `fortnite`, `steam` | `gaming` |
| `powerpoint`, `zoom`, `teams` | `presentation` |
| `chrome`, `firefox`, `edge`, `browser` | `default` |

### Macros

A macro is a named list of steps: `hotkey`, `key`, `text`, `wait`, `click`, `scroll`,
`launch`, `url` and (disabled unless `macros.allow_shell` is true) `shell`. Bind one to a
gesture with `gestures.action_map`, run it from the Macros tab, from voice
("run macro save"), or from the network API.

```json
{
  "standup": {
    "description": "Focus the meeting window and mute",
    "steps": [
      {"type": "hotkey", "keys": ["alt", "tab"]},
      {"type": "wait", "seconds": 0.4},
      {"type": "key", "key": "m"}
    ]
  }
}
```

---

## Voice, network and OBS

- **Voice** (`voice.enabled`): faster-whisper transcribes, `parse_command` maps ~28
  phrases ("click", "scroll down", "volume up", "open notepad", "type hello world",
  "pause", "resume"). Set `voice.wake_word` to require a prefix.
- **Network** (`network.enabled`): a WebSocket server on port 8765 accepting
  `status`, `click`, `set_profile`, `set_mode`, `pause`, `resume`, `run_macro`,
  `launch_app`, `scroll`, `key_press`, `type_text` and `learn_gesture`. Point a phone or
  a web dashboard at it.
- **OBS** (`obs.enabled`): gestures can switch scenes and toggle recording/streaming
  through obs-websocket (default port 4455).

---

## Project structure

```
main.py                    GestureOSEngine + CLI (entry point)
config.py                  DEFAULT_CONFIG and ConfigManager
settings.json              your live settings (generated, merged on upgrade)
check_syntax.py            static syntax + import-graph checker
camera/capture.py          threaded camera ring buffer, SimulatedCamera
vision/hand_tracker.py     MediaPipe landmarks -> Hand objects (solutions + tasks)
vision/synthetic_hands.py  scripted hand source for demos/tests (--synthetic-hands)
vision/gaze_tracker.py     experimental gaze estimation (off by default)
vision/background_subtractor.py  MOG2/KNN foreground separation (off by default)
gestures/recognizer.py     poses, pinches, swipes, two-hand gestures -> events
gestures/classifier.py     optional sklearn classifier
gestures/gesture_learner.py teach-your-own-gesture templates
gestures/help.py           the gesture table (used by the UI *and* the CLI)
controllers/               mouse, scroll, window, media, brightness, zoom, rotation
window_manager/manager.py  grab / move / snap / release, desktop navigation
macros/engine.py           macro engine + application launcher
profiles/manager.py        profile storage and live application
voice/assistant.py         wake word, transcription, command grammar
network/server.py          WebSocket control server
obs/integration.py         OBS scene/recording control
analytics/tracker.py       FPS, latency, accuracy, errors, CSV export
utils/                     math helpers, input backends, history, anomaly detection
ui/                        main_window, widgets, ar_overlay, onboarding
tests/                     430+ tests, no webcam/display/MediaPipe required
```

### Input backends

`utils/input_backend.py` abstracts every OS interaction:

- `PyAutoGUIBackend` — the real thing (used when pyautogui imports).
- `RecordingBackend` — records calls instead of executing them (`--dry-run`, tests).

Controllers never talk to the OS directly, which is why the whole app is testable
headless.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                       # 430+ tests, ~17 s
pytest -q tests/test_engine.py
python check_syntax.py       # syntax + import graph + unused imports
```

The suite needs no webcam, no display and no MediaPipe:

- hands come from `vision/synthetic_hands.py` (`tests/conftest.py::pose_hand`),
- input goes to a `RecordingBackend`,
- Qt runs on the offscreen platform plugin,
- all on-disk stores are redirected into `tmp_path`.

`tests/test_engine.py` runs the real engine end to end (camera → tracker → recogniser →
controllers) over a scripted demo and asserts which input events came out.
`tests/test_vision.py::TestRealMediaPipe` exercises the actual MediaPipe backend when
`mediapipe<1.0` is installed, and skips otherwise.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No usable MediaPipe hand-landmark backend found` | `pip install 'mediapipe>=0.10.14,<1.0'`, or download the Tasks model into `models/`, or run `--synthetic-hands` |
| `ImportError: libGL.so.1` (Linux) | `pip install opencv-python-headless` instead of `opencv-python`/`opencv-contrib-python`, or `sudo apt install libgl1` |
| Camera not detected | Check `camera.device_id`, close other camera apps, or run `--simulate` |
| Low FPS | Lower `camera.width`/`height`, set `tracking.model_complexity: 0`, raise `ui.refresh_fps` interval |
| Clicks fire when I don't want them | Raise `gestures.pinch_ratio_threshold` (less trigger-happy) or `gestures.hold_duration` in the Calibration tab |
| Pinch never triggers | Lower `gestures.pinch_ratio_threshold`, or set `gestures.use_absolute_pinch: true` with a tuned `gestures.pinch_threshold` |
| Cursor moves the wrong way | Toggle `cursor.mirror_x` |
| Cursor only covers part of the screen | Widen `cursor.movement_region` (0–1 on each axis) |
| Nothing happens at all | Run `--dry-run` and watch the log; run `python main.py --check` |
| Actions repeat too often | Raise `gestures.pose_cooldown`, or tune `anomaly_detection.burst_size` |

Logs go to the console and to a rotating `logs/gestureos.log` (1 MB × 3, disable with
`--log-file ""`); `--log-level DEBUG` adds per-frame detail.

---

## Roadmap

- **Pen mode** (`--mode pen`): the flag and the `pen_tracker`/`pen_recognizer` config
  sections exist, but no pen tracker or recogniser is implemented yet — the app logs a
  warning and falls back to hand tracking instead of starting a mode that could never
  produce a frame. Hand mode is the supported path.
- Gaze-assisted targeting (`vision/gaze_tracker.py` exists, off by default, unvalidated).
- Per-app gesture profiles learned from usage (`gesture_learner` + `classifier` are the
  groundwork).
- A web dashboard client for the network server.

## License

MIT License.
