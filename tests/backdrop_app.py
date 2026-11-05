"""A plain grey panel over the top-right quarter of the screen, for the UI tests only.

The tests look for alert-red pixels where the pop-up appears. Without this, whatever is on
your screen there (a video thumbnail, a red icon) would count too. The panel sits above your
apps but below both pop-ups (macOS and Tk), never takes the focus and lets clicks through.
Run alone: python -m tests.backdrop_app"""

from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSColor, NSMakeRect, NSPanel,
    NSScreen, NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
)
from PyObjCTools import AppHelper

GREY = (128, 128, 128)
LEVEL = 1  # above normal windows (0), below the pop-ups (Tk "topmost" is 3, the macOS pop-up 25)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon, never the active app
    screen = NSScreen.mainScreen().frame()
    width, height = screen.size.width / 2, screen.size.height / 2
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(screen.origin.x + width, screen.origin.y + height, width, height),
        NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel, NSBackingStoreBuffered, False)
    panel.setTitle_("PreCue Test Backdrop")
    panel.setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(GREY[0] / 255, GREY[1] / 255, GREY[2] / 255, 1))
    panel.setLevel_(LEVEL)
    panel.setIgnoresMouseEvents_(True)
    panel.setHidesOnDeactivate_(False)
    # On every desktop, but not over full-screen apps: the full-screen tests bring their own plain background.
    panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary)
    panel.orderFrontRegardless()
    print("backdrop ready", flush=True)
    AppHelper.runEventLoop(installInterrupt=True)


if __name__ == "__main__":
    main()
