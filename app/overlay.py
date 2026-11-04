"""The overlay: a small floating window for alerts, on top of the meeting.

This file is the macOS overlay. Windows and Linux use overlay_tk.py instead;
start_overlay() and main() pick the right one.

During a meeting you look at Zoom, Meet or your slides, not at a browser tab.
So alerts and suggestions pop up in a small window in the top-right corner:

  - always on top of every app, also full-screen slides and full-screen Zoom
  - transparent and click-through: you cannot hit it by mistake, clicks go to the app below
  - HIDDEN FROM SCREEN SHARING: macOS leaves it out of screenshots and screen
    capture, so the people in the meeting never see it (only you do)
  - it never takes the focus away from your meeting app

How: macOS's own window tools (AppKit), used from Python through PyObjC.
  - a "non-activating panel": the kind of window macOS lets float over
    other apps' full-screen windows without stealing the focus
  - inside it, a web view (WebKit) that shows <url>/overlay (web/overlay.html)

It runs as its own small process, because macOS window code must run on a
program's main thread, and PreCue's main thread is busy listening to the mic:

  live_copilot.py --starts--> python -m app.overlay <url>

Run it alone for a look:  python -m app.overlay http://127.0.0.1:8765
"""

import subprocess
import sys

WIDTH = 380
HEIGHT = 260  # room for two cards; the empty part is transparent
MARGIN = 16  # distance from the screen edges
TITLE = "PreCue Overlay"


def start_overlay(url):
    """Start the overlay window in its own process. Returns the process (give it to stop_overlay at the end)."""
    return subprocess.Popen([sys.executable, "-m", "app.overlay", url])


def stop_overlay(process):
    """Close the overlay window. Ask nicely first; if it is stuck, force it, so no window is left behind."""
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    if sys.platform == "darwin" and "--tk" not in sys.argv:  # --tk: try the Windows/Linux overlay on a Mac
        mac_overlay()
    else:
        from app.overlay_tk import main as tk_main

        tk_main()


def mac_overlay():
    # Imported here: only the overlay process needs the macOS window libraries.
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSColor,
        NSPanel,
        NSStatusWindowLevel,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
        NSWindowSharingNone,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSURL, NSURLRequest
    from PyObjCTools import AppHelper
    from WebKit import WKWebView, WKWebViewConfiguration

    url = sys.argv[1]
    capturable = "--capturable" in sys.argv  # for tests only: lets a screenshot see the overlay

    # A background app: no Dock icon, never becomes the active app. Must be set before any window.
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        top_right_frame(), NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
        NSBackingStoreBuffered, False,
    )
    panel.setTitle_(TITLE)  # not shown (no title bar), but names the window for the system
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.clearColor())  # transparent: only the cards are visible
    panel.setHasShadow_(False)  # the cards have their own shadow
    panel.setLevel_(NSStatusWindowLevel)  # above normal windows
    panel.setIgnoresMouseEvents_(True)  # click-through
    panel.setHidesOnDeactivate_(False)  # panels normally hide when their app is not active; ours never is
    panel.setCollectionBehavior_(  # on every desktop, also over other apps' full-screen windows
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorFullScreenAuxiliary
        | NSWindowCollectionBehaviorStationary
    )
    if not capturable:
        panel.setSharingType_(NSWindowSharingNone)  # left out of screenshots and screen sharing

    web = WKWebView.alloc().initWithFrame_configuration_(
        panel.contentView().bounds(), WKWebViewConfiguration.alloc().init()
    )
    web.setValue_forKey_(False, "drawsBackground")  # let the transparent window show through
    panel.setContentView_(web)
    web.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(f"{url}/overlay")))

    panel.orderFrontRegardless()  # show it without making PreCue the active app

    # Read the settings back from macOS, so the log (and the tests) show what really happened.
    print(f"Overlay ready: hidden from screen sharing={panel.sharingType() == NSWindowSharingNone}, "
          f"level={panel.level()}, click-through={bool(panel.ignoresMouseEvents())}", flush=True)

    AppHelper.runEventLoop(installInterrupt=True)  # keep the window alive until PreCue stops us


def top_right_frame():
    """The window's place: the top-right of the main screen, just below the menu bar.

    macOS measures from the BOTTOM-left corner of the screen, so "top" means a big y.
    """
    from AppKit import NSScreen
    from Foundation import NSMakeRect

    visible = NSScreen.mainScreen().visibleFrame()  # the screen without the menu bar and Dock
    x = visible.origin.x + visible.size.width - WIDTH - MARGIN
    y = visible.origin.y + visible.size.height - HEIGHT - MARGIN
    return NSMakeRect(x, y, WIDTH, HEIGHT)


if __name__ == "__main__":
    main()
