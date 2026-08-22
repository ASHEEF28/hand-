# GestureOS - AI Gesture Control System

Control your Windows PC naturally using hand gestures via webcam. GestureOS replaces traditional mouse and keyboard interactions with intuitive hand movements.

## Features

- **Cursor Control** - Index finger controls cursor with smoothing, acceleration, and multi-monitor support
- **Mouse Actions** - Left click, right click, double click, drag & drop via pinch gestures
- **Scrolling** - Index + Middle fingers up for vertical/horizontal scrolling
- **Swipe Detection** - Swipe left/right/up/down for Alt+Tab, Task View, Show Desktop
- **Window Control** - Grab, move, snap, maximize, minimize, close windows
- **Desktop Navigation** - Task View, Virtual Desktops, Start Menu, File Explorer
- **Media Controls** - Play/Pause, Next/Previous, Volume Up/Down, Mute
- **Brightness Control** - Adjust brightness via pinch distance
- **Zoom Control** - Two-hand pinch for zoom in/out
- **Rotation Detection** - Wrist rotation gestures
- **Gesture Macros** - Assign custom workflows to gestures
- **Application Launcher** - Launch apps via gestures
- **Custom Gesture Editor** - Create/edit/delete gesture mappings
- **Profiles** - Gaming, Coding, Design, Presentation presets
- **Modern UI** - PySide6 interface with live camera feed and real-time stats

## Tech Stack

- Python 3.11+
- OpenCV - Camera capture and image processing
- MediaPipe - Hand landmark detection
- PyAutoGUI - Mouse and keyboard automation
- PySide6 - Modern desktop UI
- PyWin32 - Windows window management
- Pycaw - Audio volume control
- screen-brightness-control - Display brightness

## Installation

### Requirements
- Python 3.11 or higher
- Webcam
- Windows 10/11

### Setup

1. Clone or download GestureOS:
```
git clone <repo-url>
cd GestureOS
```

2. Install dependencies:
```
pip install -r requirements.txt
```
Or double-click `install_deps.bat`

3. Run GestureOS:
```
python main.py
```
Or double-click `run.bat`

## Usage

### Quick Start
1. Launch GestureOS - camera feed appears with hand tracking overlay
2. Hold your hand in front of the camera
3. Index finger controls cursor movement
4. Use pinch gestures for clicks:

### Gesture Reference

| Gesture | Action |
|---------|--------|
| Thumb + Index pinch | Left Click |
| Thumb + Middle pinch | Right Click |
| Double pinch (fast) | Double Click |
| Hold pinch | Drag & Drop toggle |
| Index + Middle up | Toggle scroll mode |
| Swipe Left | Alt+Tab (next) |
| Swipe Right | Alt+Shift+Tab (prev) |
| Swipe Up | Task View |
| Swipe Down | Show Desktop |
| Four finger pinch | Show Desktop |
| Five finger pinch | Start Menu |

### Profiles
Switch between profiles via the UI:
- **Default** - Balanced settings for general use
- **Gaming** - Faster cursor, less smoothing
- **Coding** - Precise cursor, medium smoothing
- **Design** - Very precise cursor, high smoothing
- **Presentation** - Medium settings

### Calibration
Use the Calibration tab to adjust:
- Cursor sensitivity
- Smoothing factor
- Pinch threshold
- Swipe threshold

## Project Structure

```
GestureOS/
  main.py                  - Application entry point
  config.py                - Configuration manager
  settings.json            - Default settings
  camera/
    capture.py             - Camera capture module
  vision/
    hand_tracker.py        - MediaPipe hand tracking
  gestures/
    recognizer.py          - Gesture recognition engine
  controllers/
    mouse_controller.py    - Cursor & click control
    scroll_controller.py    - Scroll gestures
    window_controller.py   - Window management
    media_controller.py     - Media/brightness/zoom/rotation
  window_manager/
    manager.py             - Window grab/move/resize
  macros/
    engine.py              - Macro & app launcher
  profiles/
    manager.py             - Profile management
  ui/
    main_window.py          - PySide6 UI
  utils/
    __init__.py             - Utility functions
  assets/
  requirements.txt
```

## Configuration

Settings are stored in JSON format. Access via the UI Settings panel or directly edit:
- `config/settings.json` - All GestureOS settings
- `macros/macros.json` - Custom macro definitions
- `macros/apps.json` - Application shortcuts
- `profiles/profiles.json` - Profile configurations

## Performance

- Target: 60 FPS
- Max latency: < 30ms
- Multi-threaded camera capture
- Optimized hand tracking with MediaPipe

## Troubleshooting

- **Camera not detected**: Check device ID in settings, ensure no other app uses camera
- **Low FPS**: Reduce camera resolution, enable `reduce_resolution` in settings
- **False positives**: Increase pinch/swipe thresholds in calibration
- **Tracking loss**: Improve lighting, ensure clear hand visibility

## License

MIT License - see LICENSE file
