"""The meeting overlay.

The first tests check the web side and always run.
The UI tests open real windows and take real screenshots, so they only run on a Mac when asked:

  PRECUE_UI_TESTS=1 python -m pytest tests/test_overlay.py -v

They need Screen Recording permission for your terminal
(System Settings > Privacy & Security > Screen Recording).
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np
import pytest

from app.overlay import HEIGHT, MARGIN, TITLE, WIDTH, stop_overlay
from app.screen import LiveScreen

ALERT_RGB = (176, 32, 32)  # the alert card colour in web/overlay.html


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------- web side (always runs) ----------------

def test_overlay_page_is_served_and_safe():
    screen = LiveScreen(port=free_port())
    page = urllib.request.urlopen(screen.url + "/overlay").read().decode()
    assert "/events?live=1" in page  # the pop-up asks for new events only
    assert "innerHTML" not in page.replace("never innerHTML", "")  # speech text is never run as code


def test_live_events_skip_the_history_but_the_full_page_gets_it():
    screen = LiveScreen(port=free_port())
    screen.send({"type": "alert", "text": "old alert"})

    full = urllib.request.urlopen(screen.url + "/events", timeout=3)
    assert "old alert" in full.readline().decode()

    live = urllib.request.urlopen(screen.url + "/events?live=1", timeout=3)
    assert wait_for(lambda: len(screen.clients) == 2)
    screen.send({"type": "alert", "text": "new alert"})
    assert "new alert" in live.readline().decode()  # the first thing it gets is the NEW alert


# ---------------- real windows and screenshots (macOS, opt-in) ----------------

ui = pytest.mark.skipif(
    sys.platform != "darwin" or os.getenv("PRECUE_UI_TESTS") != "1",
    reason="opens real windows: run with PRECUE_UI_TESTS=1 on a Mac",
)


def wait_for(condition, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.1)
    return False


class Overlay:
    """Starts the real overlay process against a real LiveScreen."""

    def __init__(self, capturable=False, events_before=()):
        self.screen = LiveScreen(port=free_port())
        for event in events_before:  # things that happened before the overlay opened
            self.screen.send(event)
        command = [sys.executable, "-m", "app.overlay", self.screen.url] + (["--capturable"] if capturable else [])
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.ready_line = self.process.stdout.readline().strip()  # waits until the window is set up
        assert wait_for(lambda: len(self.screen.clients) == 1), "the overlay page never connected"

    def alert(self, text="You said 100 crore, the document says ₹10 crore (2. Revenue)."):
        self.screen.send({"type": "alert", "text": text, "time": "10:00:00"})
        time.sleep(1.0)  # let the page draw it

    def close(self):
        stop_overlay(self.process)
        self.screen.server.shutdown()


@pytest.fixture
def overlay():
    o = Overlay()
    yield o
    o.close()


@pytest.fixture
def visible_overlay():
    o = Overlay(capturable=True)  # the control: same window, but screenshots may see it
    yield o
    o.close()


def window_info():
    import Quartz

    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID):
        if w.get("kCGWindowName") == TITLE:
            return w
    return None


def screenshot_with_screencapture():
    """The macOS screenshot tool. Returns the screen as an (height, width, 3) array."""
    from PIL import Image

    path = f"/tmp/precue_test_{os.getpid()}.png"
    subprocess.run(["screencapture", "-x", path], check=True)
    image = np.asarray(Image.open(path).convert("RGB")).astype(int)
    os.remove(path)
    return image


def screenshot_with_screencapturekit():
    """ScreenCaptureKit: the API that Zoom, Teams and Chrome (Google Meet) use to share your screen."""
    import Quartz
    import ScreenCaptureKit as SCK

    done, result = threading.Event(), {}

    def got_image(image, error):
        result["image"], result["error"] = image, error
        done.set()

    def got_content(content, error):
        if error:
            result["error"] = error
            done.set()
            return
        display = content.displays()[0]
        result["shareable_windows"] = [w.title() for w in content.windows()]
        content_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
        config = SCK.SCStreamConfiguration.alloc().init()
        config.setWidth_(display.width() * 2)
        config.setHeight_(display.height() * 2)
        SCK.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(content_filter, config, got_image)

    SCK.SCShareableContent.getShareableContentWithCompletionHandler_(got_content)
    assert done.wait(15), "ScreenCaptureKit did not answer"
    assert result.get("image") is not None, f"ScreenCaptureKit failed: {result.get('error')}"

    image = result["image"]
    width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
    data = bytes(Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image)))
    row = Quartz.CGImageGetBytesPerRow(image)
    data = data[: height * row]  # the buffer may have padding at the end
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, row)[:, : width * 4].reshape(height, width, 4)
    rgb = pixels[..., [2, 1, 0]].astype(int)  # BGRA -> RGB
    return rgb, result["shareable_windows"]


def alert_pixels(image):
    """How many pixels have the alert card colour, in the overlay's corner of the screenshot."""
    from AppKit import NSScreen

    scale = image.shape[1] / NSScreen.mainScreen().frame().size.width  # 2 on a Retina screen
    bounds = window_info()["kCGWindowBounds"]
    top, left = int(bounds["Y"] * scale), int(bounds["X"] * scale)
    corner = image[top: top + int(HEIGHT * scale), left: left + int(WIDTH * scale)]
    close = np.abs(corner - np.array(ALERT_RGB)).max(axis=2) < 20
    return int(close.sum())


@ui
def test_window_is_set_up_as_a_hidden_overlay(overlay):
    assert "hidden from screen sharing=True" in overlay.ready_line, overlay.ready_line
    assert "click-through=True" in overlay.ready_line

    info = window_info()
    assert info is not None, "overlay window not found"
    assert info["kCGWindowSharingState"] == 0  # macOS: this window may not be captured
    assert info["kCGWindowLayer"] >= 25  # above normal windows (layer 0)
    assert info["kCGWindowIsOnscreen"]
    assert info["kCGWindowOwnerPID"] == overlay.process.pid


@ui
def test_window_sits_in_the_top_right_corner(overlay):
    from AppKit import NSScreen

    screen_width = NSScreen.mainScreen().frame().size.width
    bounds = window_info()["kCGWindowBounds"]
    assert bounds["Width"] == WIDTH and bounds["Height"] == HEIGHT
    assert abs(bounds["X"] - (screen_width - WIDTH - MARGIN)) <= 2
    assert 0 < bounds["Y"] < 80  # just below the menu bar


@ui
def test_control_alert_is_really_drawn(visible_overlay):
    """Without this, a 'hidden' result could just mean nothing was drawn at all."""
    assert alert_pixels(screenshot_with_screencapture()) == 0  # nothing before an alert
    visible_overlay.alert()
    assert alert_pixels(screenshot_with_screencapture()) > 5000


@ui
def test_alert_is_hidden_from_screenshots(overlay):
    overlay.alert()
    assert alert_pixels(screenshot_with_screencapture()) == 0


@ui
def test_alert_is_hidden_from_screencapturekit(overlay):
    """The real test for meetings: ScreenCaptureKit is what screen sharing apps use."""
    overlay.alert()
    image, _ = screenshot_with_screencapturekit()
    assert alert_pixels(image) == 0
    # Sharing ONE window (e.g. your slides) sends only that window, so the overlay is never in it.
    # Sharing the whole screen is the case that matters, and that is what this test checks.


@ui
def test_control_alert_is_seen_by_screencapturekit(visible_overlay):
    visible_overlay.alert()
    image, _ = screenshot_with_screencapturekit()
    assert alert_pixels(image) > 5000


@ui
def test_alert_fades_away(visible_overlay):
    visible_overlay.alert()
    assert alert_pixels(screenshot_with_screencapture()) > 5000
    time.sleep(11.5)  # alerts stay 10 seconds, then fade
    assert alert_pixels(screenshot_with_screencapture()) == 0


@ui
def test_old_alerts_are_not_shown_when_the_overlay_opens():
    old = {"type": "alert", "text": "an alert from before the overlay opened"}
    o = Overlay(capturable=True, events_before=[old])
    try:
        time.sleep(1.0)
        assert alert_pixels(screenshot_with_screencapture()) == 0
    finally:
        o.close()


@ui
def test_overlay_does_not_steal_focus():
    from AppKit import NSWorkspace

    before = NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier()
    o = Overlay()
    try:
        o.alert()
        after = NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier()
        assert after == before, "the overlay took the focus away from the app you were using"
    finally:
        o.close()


@ui
def test_start_overlay_as_the_app_does():
    from app.overlay import start_overlay

    screen = LiveScreen(port=free_port())
    process = start_overlay(screen.url)
    try:
        assert wait_for(lambda: len(screen.clients) == 1, timeout=10), "overlay page never connected"
        assert wait_for(lambda: window_info() is not None)
        assert window_info()["kCGWindowSharingState"] == 0
    finally:
        stop_overlay(process)
    assert wait_for(lambda: window_info() is None)


@pytest.fixture
def fullscreen_meeting():
    """A solid green app in full-screen mode, like full-screen Zoom or a slideshow."""
    from tests.fullscreen_app import GREEN

    process = subprocess.Popen([sys.executable, "-m", "tests.fullscreen_app"], stdout=subprocess.PIPE, text=True)
    process.stdout.readline()
    time.sleep(3)  # the full-screen animation
    image = screenshot_with_screencapture()
    green = (np.abs(image - np.array(GREEN)).max(axis=2) < 20).mean()
    assert green > 0.8, f"the fake meeting is not full screen ({green:.0%} of the screen is green)"
    yield
    stop_overlay(process)
    time.sleep(2)  # macOS leaves the full-screen space


@ui
def test_alert_shows_over_a_fullscreen_app(fullscreen_meeting):
    o = Overlay(capturable=True)
    try:
        o.alert()
        assert alert_pixels(screenshot_with_screencapture()) > 5000, "the alert is hidden behind the full-screen app"
    finally:
        o.close()


@ui
def test_alert_over_a_fullscreen_app_is_still_hidden_from_screen_sharing(fullscreen_meeting):
    o = Overlay()
    try:
        o.alert()
        image, _ = screenshot_with_screencapturekit()
        assert alert_pixels(image) == 0
    finally:
        o.close()


@ui
def test_window_closes_with_the_app():
    o = Overlay()
    o.close()
    assert wait_for(lambda: window_info() is None)
