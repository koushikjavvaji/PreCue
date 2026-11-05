"""PreCue app: everything from one page in your browser. No commands to remember.

  python -m scripts.app          (or double-click PreCue.command on a Mac, PreCue.bat on Windows)

It starts a small web server on this computer only (http://127.0.0.1:8765) and opens it:
  1. add your three free keys once (saved in .env, readable by you only)
  2. drop in the document for your meeting
  3. press Start: the pop-up appears over your meeting and whispers go to your earphones

The page talks to app/session.py; the meeting itself is the same as scripts/live_copilot.py.
"""

import argparse
import encodings.idna  # noqa: F401  (loaded now: two threads loading it at once on their first web request can freeze)
import json
import time
import urllib.request
import webbrowser

from dotenv import load_dotenv

from app.errors import PreCueError
from app.screen import LiveScreen
from app.session import PreCueApp


def main():
    parser = argparse.ArgumentParser(description="PreCue app: start a meeting from your browser.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args = parser.parse_args()

    load_dotenv()
    url = f"http://127.0.0.1:{args.port}"
    if already_running(url):  # double-clicked again: show the PreCue that is already open
        print(f"PreCue is already running at {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return
    app = PreCueApp()  # starts loading the search model in the background right away
    screen = LiveScreen(port=args.port, api=app)
    app.screen = screen
    print(f"PreCue is running at {screen.url}  (press Ctrl+C here to quit)")
    if not args.no_browser:
        webbrowser.open(screen.url)
    try:
        while True:
            time.sleep(1)
    finally:
        app.stop()


def already_running(url):
    """True if PreCue already answers at this address."""
    try:
        with urllib.request.urlopen(f"{url}/api/state", timeout=2) as response:
            return "session" in json.load(response)
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    try:
        main()
    except PreCueError as error:
        raise SystemExit(f"Error: {error}") from None
    except KeyboardInterrupt:
        print("\nPreCue stopped.")
