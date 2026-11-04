"""The overlay on Windows and Linux: a small always-on-top window with the alert cards.

macOS uses overlay.py (Apple's own window tools). Windows and Linux use this file,
built with tkinter, which comes with Python.

  - a small borderless window in the top-right corner, always on top
  - it is hidden while there is nothing to show, and appears with each alert or suggestion
  - each card disappears after a few seconds
  - Windows: hidden from screen sharing (SetWindowDisplayAffinity, Windows 10 2004 or newer),
    click-through, and never takes the focus
  - Linux: there is no way to hide a window from screen sharing. The meeting would see it
    if you share your whole screen, so on Linux the overlay is only used when asked for.

The cards come from the live screen's event stream (<url>/events?live=1), read in a
background thread; tkinter itself may only be used from the main thread, so the
thread hands each event over through a queue.
"""

import json
import queue
import sys
import threading
import time
import urllib.request

WIDTH = 380
MARGIN = 16
TITLE = "PreCue Overlay"
COLOURS = {"alert": "#b02020", "suggest": "#1d54aa"}  # the same colours as web/overlay.html
LABELS = {"alert": "CHECK THIS", "suggest": "NEXT POINT"}
SHOW_SECONDS = {"alert": 10, "suggest": 8}
MAX_CARDS = 2

# Windows settings (see the Windows docs for SetWindowDisplayAffinity and extended window styles)
WDA_EXCLUDEFROMCAPTURE = 0x11  # leave the window out of screenshots and screen sharing
WDA_MONITOR = 0x01  # older Windows: show the window as a black box in captures
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020  # with LAYERED: clicks go through to the window below
WS_EX_TOOLWINDOW = 0x00000080  # no taskbar button
WS_EX_NOACTIVATE = 0x08000000  # never takes the focus
LWA_ALPHA = 0x02


class TkOverlay:
    def __init__(self, url, capturable=False):
        import tkinter as tk

        self.tk = tk
        self.events = queue.Queue()  # filled by the reader thread, emptied by the window
        self.cards = []  # (frame, time it goes away)

        self.root = tk.Tk()
        self.root.title(TITLE)
        self.root.overrideredirect(True)  # no title bar or border
        self.root.attributes("-topmost", True)
        self.root.configure(bg="black")
        x = self.root.winfo_screenwidth() - WIDTH - MARGIN
        self.root.geometry(f"+{x}+{MARGIN + 30}")
        self.root.withdraw()  # hidden until there is something to show

        self.hidden_from_capture = False
        if sys.platform == "win32":
            self.root.update_idletasks()  # the window must exist before Windows settings can change
            self.hidden_from_capture = windows_make_overlay(self.root.winfo_id(), capturable)

        threading.Thread(target=read_events, args=(url, self.events), daemon=True).start()
        self.root.after(100, self._tick)
        print(f"Overlay ready: hidden from screen sharing={self.hidden_from_capture}", flush=True)

    def run(self):
        self.root.mainloop()

    def _tick(self):
        """Every 0.1 s: show new events, remove old cards, hide the window when it is empty."""
        while not self.events.empty():
            self.show(self.events.get())
        now = time.time()
        for frame, ends_at in list(self.cards):
            if now >= ends_at:
                frame.destroy()
                self.cards.remove((frame, ends_at))
        if not self.cards:
            self.root.withdraw()
        self.root.after(100, self._tick)

    def show(self, event):
        if event.get("type") not in LABELS:
            return
        tk = self.tk
        colour = COLOURS[event["type"]]
        frame = tk.Frame(self.root, bg=colour, padx=14, pady=10)
        tk.Label(frame, text=LABELS[event["type"]], bg=colour, fg="white", font=("Helvetica", 9, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(frame, text=event.get("text", ""), bg=colour, fg="white", font=("Helvetica", 12),
                 wraplength=WIDTH - 28, justify="left", anchor="w").pack(fill="x")
        if self.cards:
            frame.pack(fill="x", pady=(0, 8), before=self.cards[-1][0])  # newest on top
        else:
            frame.pack(fill="x", pady=(0, 8))
        self.cards.append((frame, time.time() + SHOW_SECONDS[event["type"]]))
        while len(self.cards) > MAX_CARDS:
            oldest, _ = self.cards.pop(0)
            oldest.destroy()
        self.root.deiconify()
        self.root.attributes("-topmost", True)  # stay above a window that became active meanwhile


def read_events(url, events):
    """Read the live screen's event stream forever, reconnecting if it drops."""
    while True:
        try:
            with urllib.request.urlopen(f"{url}/events?live=1", timeout=60) as stream:
                for line in stream:
                    if line.startswith(b"data: "):
                        events.put(json.loads(line[6:]))
        except OSError:
            time.sleep(1)  # PreCue is starting or stopping; try again


def windows_make_overlay(window_id, capturable, user32=None):
    """Windows only: hide from screen sharing, click-through, no focus. Returns True if hidden from capture.

    user32 can be replaced in tests; normally it is Windows' own user32.dll.
    """
    if user32 is None:
        import ctypes

        user32 = ctypes.windll.user32
    hwnd = user32.GetParent(window_id)  # tkinter's frame window, the one Windows manages
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)  # a layered window is invisible until it has an opacity
    if capturable:
        return False
    if user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
        return True
    # Windows older than 10 (version 2004): the best it can do is a black box in captures.
    user32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR)
    print("Warning: this Windows version cannot hide the overlay from screen sharing; "
          "the meeting may see a black box. Update Windows, or share one window instead of the screen.")
    return False


def main():
    url = sys.argv[1]
    TkOverlay(url, capturable="--capturable" in sys.argv).run()


if __name__ == "__main__":
    main()
