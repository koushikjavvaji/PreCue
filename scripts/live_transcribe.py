"""Speak into your mic and see the live text. Press Ctrl+C to stop.

Run from the project folder:  python -m scripts.live_transcribe
"""

import os
import queue

from dotenv import load_dotenv

from app.errors import PreCueError
from app.mic import open_mic
from app.stt import DeepgramSTT


def main():
    load_dotenv()  # reads DEEPGRAM_API_KEY from the .env file
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise SystemExit("Error: add your Deepgram key to the .env file (DEEPGRAM_API_KEY=...).")

    audio_queue = queue.Queue()  # the mic puts audio here, Deepgram takes it from here
    stt = DeepgramSTT(api_key)

    print("Listening... speak now (Ctrl+C to stop)\n")

    with open_mic(audio_queue):
        for event in stt.listen(audio_queue):
            if event["type"] == "speech_started":
                print("[speech started]")
            elif event["type"] == "speech_ended":
                print("[paused]")
            elif event["is_final"]:
                print("FINAL  :", event["text"])
            else:
                print("partial:", event["text"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except PreCueError as error:
        raise SystemExit(f"Error: {error}") from None  # clear message, exit code 1, no ugly traceback
