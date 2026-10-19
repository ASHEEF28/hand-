"""OS control layer: mouse, scroll, window, desktop, media, brightness, zoom."""
from controllers.mouse_controller import MouseController
from controllers.scroll_controller import ScrollController
from controllers.window_controller import DesktopNavigationController, WindowController
from controllers.media_controller import (
    BrightnessController,
    MediaController,
    RotationController,
    ZoomController,
)

__all__ = [
    "MouseController",
    "ScrollController",
    "WindowController",
    "DesktopNavigationController",
    "MediaController",
    "BrightnessController",
    "ZoomController",
    "RotationController",
]
