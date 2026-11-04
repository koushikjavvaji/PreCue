"""The live parts: speech-to-text events, the meeting loop, reconnects, the screen and whispers.
Deepgram, the mic and the speakers are replaced by fakes."""

import json
import queue
import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
from websockets.exceptions import ConnectionClosed

import app.stt as stt
import app.voice as voice
import scripts.live_copilot as live
from app.errors import ConnectionLostError, PreCueError
from app.screen import LiveScreen


def wait_for(condition, timeout=3):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


# ---------------- speech-to-text ----------------

def results(text, is_final=True, speech_final=False, from_finalize=False):
    return {"type": "Results", "is_final": is_final, "speech_final": speech_final,
            "from_finalize": from_finalize, "channel": {"alternatives": [{"transcript": text}]}}


def test_deepgram_messages_become_events():
    deepgram = stt.DeepgramSTT("key")
    assert deepgram._to_event({"type": "SpeechStarted"}) == {"type": "speech_started"}
    assert deepgram._to_event({"type": "UtteranceEnd"}) == {"type": "speech_ended"}
    assert deepgram._to_event(results("hi", speech_final=True))["speech_final"] is True
    assert deepgram._to_event(results("99%", from_finalize=True))["speech_final"] is True  # our Finalize ends it
    assert deepgram._to_event(results("hi"))["speech_final"] is False
    assert deepgram._to_event(results("")) is None
    assert deepgram._to_event({"type": "Metadata"}) is None


class FakeConnection:
    """Records what we send. Closes itself after `limit` messages, which ends the sender thread."""

    def __init__(self, limit):
        self.sent = []
        self.limit = limit

    def send(self, message):
        self.sent.append(message)
        if len(self.sent) >= self.limit:
            raise ConnectionClosed(None, None)


def tone(seconds, loudness):
    samples = int(stt.SAMPLE_RATE * seconds)
    return (np.sin(np.arange(samples) / 5) * loudness).astype(np.int16).tobytes()


def test_keepalive_is_sent_when_no_audio(monkeypatch):
    monkeypatch.setattr(stt, "KEEPALIVE_SECONDS", 0.05)
    connection = FakeConnection(limit=2)
    stt.DeepgramSTT("key")._send_audio(connection, queue.Queue())
    assert json.loads(connection.sent[0]) == {"type": "KeepAlive"}


def test_finalize_is_sent_once_after_speech_then_silence():
    audio = queue.Queue()
    for piece in [tone(0.1, 3000)] * 5 + [tone(0.1, 0)] * 20:  # 0.5 s of speech, 2 s of silence
        audio.put(piece)
    connection = FakeConnection(limit=26)  # 25 pieces + 1 Finalize
    stt.DeepgramSTT("key")._send_audio(connection, audio)
    finalizes = [i for i, m in enumerate(connection.sent) if m == json.dumps({"type": "Finalize"})]
    assert len(finalizes) == 1
    assert finalizes[0] == 5 + 8  # after 0.8 s of silence


def test_no_finalize_for_silence_alone():
    audio = queue.Queue()
    for piece in [tone(0.1, 0)] * 20:
        audio.put(piece)
    connection = FakeConnection(limit=20)
    stt.DeepgramSTT("key")._send_audio(connection, audio)
    assert json.dumps({"type": "Finalize"}) not in connection.sent


def test_quiet_room_noise_counts_as_silence():
    assert stt.is_silent(tone(0.1, 100))
    assert not stt.is_silent(tone(0.1, 3000))


# ---------------- the meeting loop ----------------

class FakeCopilot:
    def __init__(self, delay=0, crash_on=None):
        self.events = []
        self.delay = delay
        self.crash_on = crash_on

    def handle(self, event, text="", late=False):
        time.sleep(self.delay)
        self.events.append((event, text, late))
        if self.crash_on and self.crash_on in text:
            raise ValueError("boom")
        if "100" in text:
            return {"action": "alert", "message": "fix it", "spoken": "it's 10"}
        if event == "pause":
            return {"action": "suggest", "message": "Next: X", "spoken": "Next, X."}
        return {"action": "silent", "message": "", "spoken": ""}


class FakeScreen:
    def __init__(self):
        self.sent = []

    def send(self, event):
        self.sent.append(event)


class FakeWhisperer:
    device = None

    def __init__(self, speaking=False):
        self.said = []
        self.speaking = speaking

    def say(self, text):
        self.said.append(text)

    def is_speaking(self):
        return self.speaking


def final(text, speech_final=False):
    return {"type": "transcript", "text": text, "is_final": True, "speech_final": speech_final}


@pytest.fixture(autouse=True)
def earphones(monkeypatch):
    monkeypatch.setattr(live, "output_device_name", lambda device: "AirPods")


def test_pieces_become_one_sentence_and_alert_is_whispered():
    copilot, screen, whisperer = FakeCopilot(), FakeScreen(), FakeWhisperer()
    meeting = live.LiveMeeting(copilot, screen, whisperer)
    meeting.on_speech({"type": "speech_started"})
    meeting.on_speech({"type": "transcript", "text": "revenue", "is_final": False, "speech_final": False})
    meeting.on_speech(final("Our revenue was"))
    meeting.on_speech(final("100 crore.", speech_final=True))
    assert wait_for(lambda: whisperer.said)
    assert copilot.events[0] == ("sentence", "Our revenue was 100 crore.", False)
    assert whisperer.said == ["it's 10"]
    assert {"partial", "final", "alert"} <= {event["type"] for event in screen.sent}


def test_speech_ended_closes_the_sentence_once():
    copilot = FakeCopilot()
    meeting = live.LiveMeeting(copilot, None, None)
    meeting.on_speech(final("hello there"))
    meeting.on_speech({"type": "speech_ended"})
    meeting.on_speech({"type": "speech_ended"})
    assert wait_for(lambda: copilot.events)
    time.sleep(0.2)
    assert [e for e in copilot.events if e[0] == "sentence"] == [("sentence", "hello there", False)]


def test_pause_is_reported_once(monkeypatch):
    monkeypatch.setattr(live, "PAUSE_SECONDS", 0.5)
    copilot, whisperer = FakeCopilot(), FakeWhisperer()
    meeting = live.LiveMeeting(copilot, None, whisperer)
    meeting.on_speech(final("hello", speech_final=True))
    assert wait_for(lambda: any(e[0] == "pause" for e in copilot.events))
    time.sleep(1.2)
    assert sum(e[0] == "pause" for e in copilot.events) == 1
    assert whisperer.said == ["Next, X."]


def test_slow_llm_marks_waiting_sentences_late(monkeypatch):
    monkeypatch.setattr(live, "PAUSE_SECONDS", 30)
    monkeypatch.setattr(live, "MAX_DELAY_SECONDS", 0.5)
    copilot = FakeCopilot(delay=0.8)
    meeting = live.LiveMeeting(copilot, None, None)
    for text in ["one 1", "two 2", "three 3"]:
        meeting.on_speech(final(text, speech_final=True))
    assert wait_for(lambda: len(copilot.events) >= 3, timeout=6)
    late = [e[2] for e in copilot.events]
    assert late[0] is False and late[-1] is True


def test_a_bug_in_one_check_does_not_stop_the_meeting():
    copilot, screen = FakeCopilot(crash_on="crash"), FakeScreen()
    meeting = live.LiveMeeting(copilot, screen, None)
    meeting.on_speech(final("crash now", speech_final=True))
    meeting.on_speech(final("Revenue 100", speech_final=True))
    assert wait_for(lambda: any(e["type"] == "alert" for e in screen.sent))
    assert any("Internal error" in e.get("text", "") for e in screen.sent)


def test_mic_is_ignored_during_whisper_only_on_speakers(monkeypatch):
    monkeypatch.setattr(live, "output_device_name", lambda device: "MacBook Air Speakers")
    copilot = FakeCopilot()
    meeting = live.LiveMeeting(copilot, None, FakeWhisperer(speaking=True))
    meeting.on_speech(final("our own whisper", speech_final=True))
    time.sleep(0.3)
    assert copilot.events == []

    monkeypatch.setattr(live, "output_device_name", lambda device: "AirPods")
    assert not live.LiveMeeting(copilot, None, FakeWhisperer(speaking=True)).echo_guard


# ---------------- reconnecting ----------------

def test_reconnect_waits_longer_each_time_then_gives_up(monkeypatch):
    waits = []
    monkeypatch.setattr(live.time, "sleep", lambda seconds: waits.append(seconds) if seconds != 0.5 else None)

    class AlwaysDown:
        def listen(self, audio_queue):
            raise ConnectionLostError("network down")
            yield

    audio = queue.Queue()
    audio.put(b"old audio")
    with pytest.raises(PreCueError, match="Gave up"):
        live.listen(AlwaysDown(), audio, live.LiveMeeting(FakeCopilot(), None, None))
    assert waits == [2, 4, 8, 16, 30]
    assert audio.empty()  # audio from the outage is dropped


def test_reconnect_count_resets_after_a_working_connection(monkeypatch):
    monkeypatch.setattr(live.time, "sleep", lambda seconds: None)

    class Flaky:
        attempts = 0

        def listen(self, audio_queue):
            Flaky.attempts += 1
            if Flaky.attempts > 20:
                raise KeyboardInterrupt  # stop the test
            yield {"type": "speech_started"}  # works for a moment...
            raise ConnectionLostError("blip")  # ...then drops

    with pytest.raises(KeyboardInterrupt):  # 20 short drops and it never gave up
        live.listen(Flaky(), queue.Queue(), live.LiveMeeting(FakeCopilot(), None, None))


# ---------------- the live screen ----------------

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_screen_serves_page_streams_events_and_replays_history():
    screen = LiveScreen(port=free_port())
    assert "EventSource" in urllib.request.urlopen(screen.url + "/").read().decode()
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(screen.url + "/other")

    screen.send({"type": "final", "text": "said before the page opened"})
    screen.send({"type": "partial", "text": "not replayed"})
    stream = urllib.request.urlopen(screen.url + "/events", timeout=3)
    assert "said before the page opened" in stream.readline().decode()
    stream.readline()  # blank line between events

    screen.send({"type": "alert", "text": "₹ <b>not html</b>"})
    assert json.loads(stream.readline()[6:])["text"] == "₹ <b>not html</b>"

    stream.close()
    for _ in range(3):
        screen.send({"type": "final", "text": "x"})  # writing to a closed page removes it
        time.sleep(0.2)
    assert wait_for(lambda: not screen.clients)


def test_screen_refuses_other_host_names():
    """DNS rebinding: a website that points its own name at 127.0.0.1 must not read the transcript."""
    screen = LiveScreen(port=free_port())
    port = screen.server.server_address[1]
    for path in ["/", "/events", "/overlay"]:
        request = urllib.request.Request(screen.url + path, headers={"Host": f"evil.example:{port}"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3)
        assert error.value.code == 403
    for host in [f"127.0.0.1:{port}", f"localhost:{port}"]:
        request = urllib.request.Request(screen.url + "/", headers={"Host": host})
        assert urllib.request.urlopen(request, timeout=3).status == 200


def test_slow_page_cannot_fill_the_memory():
    import app.screen as screen_module

    screen = LiveScreen(port=free_port())
    stalled = queue.Queue(maxsize=screen_module.MAX_WAITING)
    screen.clients.append(stalled)  # a page that never reads
    for i in range(screen_module.MAX_WAITING + 100):
        screen.send({"type": "final", "text": str(i)})
    assert stalled.qsize() == screen_module.MAX_WAITING


def test_screen_port_in_use_gives_clear_error():
    port = free_port()
    LiveScreen(port=port)
    with pytest.raises(PreCueError, match="port"):
        LiveScreen(port=port)


# ---------------- whispers ----------------

def test_whisper_device_is_found_by_part_of_its_name(monkeypatch):
    monkeypatch.setattr(voice.sd, "query_devices", lambda *args: [
        {"name": "MacBook Microphone", "max_output_channels": 0},
        {"name": "My AirPods Pro", "max_output_channels": 2},
    ])
    assert voice.find_output_device("airpods") == 1
    assert voice.find_output_device(None) is None
    with pytest.raises(PreCueError, match="AirPods"):
        voice.find_output_device("Bose")


def start_whisperer(speak):
    whisperer = voice.Whisperer.__new__(voice.Whisperer)  # skip __init__: no device, no key
    whisperer.stopping, whisperer.queue, whisperer.quiet_until = False, queue.Queue(), 0
    whisperer._speak = speak
    whisperer.thread = threading.Thread(target=whisperer._play_forever, daemon=True)
    whisperer.thread.start()
    return whisperer


def test_failed_whisper_does_not_stop_the_next_one():
    spoken = []

    def speak(text):
        spoken.append(text)
        if text == "bad":
            raise PreCueError("text-to-speech down")

    whisperer = start_whisperer(speak)
    whisperer.say("bad")
    whisperer.say("good")
    whisperer.say("")  # empty text is ignored
    assert wait_for(lambda: spoken == ["bad", "good"])
    whisperer.stop()
    assert not whisperer.thread.is_alive()


def test_whisper_that_waited_too_long_is_dropped(monkeypatch):
    monkeypatch.setattr(voice, "MAX_WAIT_SECONDS", 0.3)
    spoken = []

    def slow_speak(text):
        spoken.append(text)
        time.sleep(0.6)  # text-to-speech is slow today

    whisperer = start_whisperer(slow_speak)
    whisperer.say("first")
    whisperer.say("second")  # waits 0.6 s behind "first": too late
    whisperer.say("third")
    time.sleep(1.5)
    assert spoken == ["first"]
    whisperer.stop()
