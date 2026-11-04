"""A stand-in for a full-screen meeting app (Zoom, a slideshow): one solid green window in full-screen mode.
Used by tests/test_overlay.py. Run alone: python -m tests.fullscreen_app"""

from AppKit import (
    NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreBuffered, NSColor, NSMakeRect,
    NSWindow, NSWindowCollectionBehaviorFullScreenPrimary, NSWindowStyleMaskResizable, NSWindowStyleMaskTitled,
)
from PyObjCTools import AppHelper

GREEN = (0, 160, 90)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(100, 100, 600, 400), NSWindowStyleMaskTitled | NSWindowStyleMaskResizable, NSBackingStoreBuffered, False)
    window.setTitle_("Fake Meeting")
    window.setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(GREEN[0] / 255, GREEN[1] / 255, GREEN[2] / 255, 1))
    window.setCollectionBehavior_(NSWindowCollectionBehaviorFullScreenPrimary)
    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)
    AppHelper.callLater(0.5, window.toggleFullScreen_, None)  # like pressing the green full-screen button
    print("fake meeting started", flush=True)
    AppHelper.runEventLoop(installInterrupt=True)


if __name__ == "__main__":
    main()
