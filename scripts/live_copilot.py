"""PreCue live: listens to you in a meeting, checks your numbers, alerts and suggests.

Run from the project folder:
  python -m scripts.live_copilot samples/sample_report.txt
  python -m scripts.live_copilot samples/sample_report.txt --no-voice     (screen only)
  python -m scripts.live_copilot samples/sample_report.txt --no-screen    (terminal + whispers)
  python -m scripts.live_copilot samples/sample_report.txt --no-overlay   (browser tab instead of pop-ups)

Alerts pop up in a small floating window on top of your meeting (overlay.py on macOS,
overlay_tk.py on Windows), hidden from screen sharing. On Linux the pop-up cannot be hidden
from screen sharing, so it opens a browser tab unless you add --overlay. The full page with the transcript is at the
printed "Live screen" address.

The easy way is the app (python -m scripts.app): the same meeting, started from a web page.
The meeting loop itself is in app/meeting.py.
"""

import argparse
import os
import queue
import sys
import webbrowser

from dotenv import load_dotenv

from app.copilot import Copilot
from app.errors import PreCueError
from app.index import MODES
from app.llm import get_live_llm, model_name
from app.meeting import LiveMeeting, listen
from app.mic import open_mic
from app.overlay import start_overlay, stop_overlay
from app.prepare import prepare
from app.screen import LiveScreen
from app.stt import DeepgramSTT
from app.voice import Whisperer, output_device_name


def use_overlay(args):
    """The pop-up window on macOS and Windows. On Linux only when asked for:
    Linux cannot hide a window from screen sharing, so the meeting could see it."""
    if args.no_overlay:
        return False
    if sys.platform.startswith("linux"):
        if args.overlay:
            print("WARNING: on Linux the pop-up is visible if you share your whole screen. "
                  "Share one window (e.g. your slides) instead.")
        return args.overlay
    return True


def main():
    parser = argparse.ArgumentParser(description="PreCue live meeting copilot.")
    parser.add_argument("file", help="the document for this meeting (.txt, .md, .pdf, .docx or .pptx)")
    parser.add_argument("--mode", choices=MODES, default="plain")
    parser.add_argument("--no-voice", action="store_true", help="no whispers, screen only")
    parser.add_argument("--no-screen", action="store_true", help="no web page and no pop-ups")
    parser.add_argument("--no-overlay", action="store_true", help="open a browser tab instead of the pop-up window")
    parser.add_argument("--overlay", action="store_true",
                        help="Linux: use the pop-up window (it is visible if you share your whole screen)")
    args = parser.parse_args()

    load_dotenv()
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key or deepgram_key == "your_key_here":
        raise PreCueError("Add your Deepgram key to the .env file (DEEPGRAM_API_KEY=...).")

    print("Preparing the document...")
    retriever, facts = prepare(args.file, args.mode)
    llm = get_live_llm()
    copilot = Copilot(retriever, facts, llm)
    print(f"Ready: {len(retriever.chunks)} chunks, {len(facts)} facts, live LLM {model_name(llm)}")

    screen = None
    overlay = None
    if not args.no_screen:
        screen = LiveScreen()
        print(f"Live screen: {screen.url}")
        if use_overlay(args):
            overlay = start_overlay(screen.url)
        else:
            webbrowser.open(screen.url)

    whisperer = None
    if not args.no_voice:
        try:
            volume = float(os.getenv("WHISPER_VOLUME", "0.6"))
        except ValueError:
            raise PreCueError("WHISPER_VOLUME in .env must be a number from 0 to 1, e.g. 0.6.") from None
        whisperer = Whisperer(deepgram_key, os.getenv("WHISPER_DEVICE"), volume)
        device = output_device_name(whisperer.device)
        print(f"Whispers play on: {device}")
        if "speaker" in device.lower():
            print("WARNING: whispers will play on your speakers, so the meeting may hear them. "
                  "Plug in earphones and set WHISPER_DEVICE in .env (or use --no-voice).")

    meeting = LiveMeeting(copilot, screen, whisperer)
    audio_queue = queue.Queue()
    print("\nSpeak now. Press Ctrl+C to stop.\n")
    try:
        with open_mic(audio_queue):
            listen(DeepgramSTT(deepgram_key), audio_queue, meeting)
    finally:
        if whisperer:
            whisperer.stop()  # end a whisper cleanly instead of cutting the audio device off
        if overlay:
            stop_overlay(overlay)  # close the pop-up window with the app


if __name__ == "__main__":
    try:
        main()
    except PreCueError as error:
        raise SystemExit(f"Error: {error}")
    except KeyboardInterrupt:
        print("\nStopped.")
