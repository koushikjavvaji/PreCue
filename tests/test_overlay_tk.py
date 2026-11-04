"""The Windows/Linux overlay (app/overlay_tk.py) and how PreCue picks an overlay.

The window tests open real windows, so they only run when asked:
  PRECUE_UI_TESTS=1 python -m pytest tests/test_overlay_tk.py
On a Mac they run the same tkinter overlay (with --tk). On Windows they also check the
Windows-only settings for real (GitHub Actions runs them on Windows).
"""

import argparse
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

import scripts.live_copilot as live
from app import overlay_tk
from app.overlay import stop_overlay
from app.screen import LiveScreen

ALERT_RGB = (176, 32, 32)  # "#b02020", the alert card colour


# ---------------- which overlay is used where ----------------

@pytest.mark.parametrize("platform, flags, expected", [
    ("darwin", [], True),
    ("darwin", ["--no-overlay"], False),
    ("win32", [], True),
    ("win32", ["--no-overlay"], False),
    ("linux", [], False),  # Linux cannot hide the pop-up from screen sharing: browser tab by default
    ("linux", ["--overlay"], True),
])
def test_use_overlay(monkeypatch, platform, flags, expected):
    monkeypatch.setattr(live.sys, "platform", platform)
    args = argparse.Namespace(no_overlay="--no-overlay" in flags, overlay="--overlay" in flags)
    assert live.use_overlay(args) is expected


# ---------------- the Windows settings, with a stand-in for Windows' user32.dll ----------------

class FakeUser32:
    """Records the calls PreCue makes to Windows' window library."""

    def __init__(self, affinity_works=True):
        self.calls = []
        self.style = 0
        self.affinity_works = affinity_works

    def GetParent(self, window_id):
        return 4242  # the window handle

    def GetWindowLongW(self, hwnd, index):
        return self.style

    def SetWindowLongW(self, hwnd, index, style):
        self.calls.append(("style", hwnd, index))
        self.style = style

    def SetLayeredWindowAttributes(self, hwnd, colour, alpha, flags):
        self.calls.append(("layered", hwnd, alpha, flags))

    def SetWindowDisplayAffinity(self, hwnd, affinity):
        self.calls.append(("affinity", hwnd, affinity))
        return 1 if self.affinity_works or affinity == overlay_tk.WDA_MONITOR else 0


def test_windows_overlay_is_hidden_click_through_and_never_takes_focus():
    user32 = FakeUser32()
    assert overlay_tk.windows_make_overlay(1, capturable=False, user32=user32) is True
    for flag in [overlay_tk.WS_EX_LAYERED, overlay_tk.WS_EX_TRANSPARENT,
                 overlay_tk.WS_EX_NOACTIVATE, overlay_tk.WS_EX_TOOLWINDOW]:
        assert user32.style & flag
    assert ("affinity", 4242, overlay_tk.WDA_EXCLUDEFROMCAPTURE) in user32.calls
    assert ("layered", 4242, 255, overlay_tk.LWA_ALPHA) in user32.calls  # visible, not a blank layered window


def test_old_windows_falls_back_to_a_black_box(capsys):
    user32 = FakeUser32(affinity_works=False)
    assert overlay_tk.windows_make_overlay(1, capturable=False, user32=user32) is False
    assert ("affinity", 4242, overlay_tk.WDA_MONITOR) in user32.calls
    assert "cannot hide the overlay" in capsys.readouterr().out


def test_capturable_windows_overlay_is_not_hidden():
    user32 = FakeUser32()
    assert overlay_tk.windows_make_overlay(1, capturable=True, user32=user32) is False
    assert not any(call[0] == "affinity" for call in user32.calls)


# ---------------- the real tkinter window (opt-in) ----------------

ui = pytest.mark.skipif(os.getenv("PRECUE_UI_TESTS") != "1", reason="opens real windows: PRECUE_UI_TESTS=1")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(capturable):
    screen = LiveScreen(port=free_port())
    command = [sys.executable, "-m", "app.overlay", screen.url, "--tk"] + (["--capturable"] if capturable else [])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    ready = process.stdout.readline().strip()
    end = time.time() + 10
    while not screen.clients and time.time() < end:
        time.sleep(0.1)
    assert screen.clients, "the overlay never connected to the live screen"
    return screen, process, ready


def screenshot():
    from PIL import ImageGrab

    return np.asarray(ImageGrab.grab().convert("RGB")).astype(int)


def alert_pixels(image):
    """Alert-coloured pixels in the top-right quarter of the screen (where the overlay is)."""
    height, width, _ = image.shape
    corner = image[: height // 2, width // 2:]
    return int((np.abs(corner - np.array(ALERT_RGB)).max(axis=2) < 20).sum())


@ui
def test_tk_overlay_shows_an_alert_then_hides():
    screen, process, _ = start(capturable=True)
    try:
        time.sleep(1)
        assert alert_pixels(screenshot()) == 0  # hidden while there is nothing to show
        screen.send({"type": "alert", "text": "You said 100 crore, the document says ₹10 crore (2. Revenue)."})
        time.sleep(1.5)
        assert alert_pixels(screenshot()) > 3000
        time.sleep(10)  # alerts stay 10 seconds
        assert alert_pixels(screenshot()) == 0
    finally:
        stop_overlay(process)


@ui
def test_tk_overlay_ignores_old_alerts():
    screen = LiveScreen(port=free_port())
    screen.send({"type": "alert", "text": "an alert from before the overlay opened"})
    process = subprocess.Popen([sys.executable, "-m", "app.overlay", screen.url, "--tk", "--capturable"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        process.stdout.readline()
        time.sleep(2)
        assert alert_pixels(screenshot()) == 0
    finally:
        stop_overlay(process)


@ui
@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only settings")
def test_windows_overlay_is_really_hidden_from_capture():
    import ctypes

    screen, process, ready = start(capturable=False)
    try:
        assert "hidden from screen sharing=True" in ready, ready
        screen.send({"type": "alert", "text": "secret alert"})
        time.sleep(1.5)
        hwnd = ctypes.windll.user32.FindWindowW(None, overlay_tk.TITLE)
        affinity = ctypes.c_uint()
        ctypes.windll.user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity))
        assert affinity.value == overlay_tk.WDA_EXCLUDEFROMCAPTURE
        assert alert_pixels(screenshot()) == 0  # a screenshot does not see it
    finally:
        stop_overlay(process)
