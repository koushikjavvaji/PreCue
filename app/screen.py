"""The live screen: a small web page that shows the transcript, alerts and suggestions.

How it works:
  - Python's built-in web server runs in a background thread on
    http://127.0.0.1:8765 (127.0.0.1 = only this computer can open it).
  - "/" sends the page (web/index.html).
  - "/events" stays open and pushes every new event to the page as it
    happens (Server-Sent Events: the browser keeps listening, the server writes).
  - A page opened late still sees the recent history (we replay it).

No extra libraries: only the Python standard library and plain JavaScript.
"""

import json
import queue
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.errors import PreCueError

PAGE = Path(__file__).parent.parent / "web" / "index.html"
HISTORY = 200  # events replayed to a page that opens late
PING_SECONDS = 15  # a small "still here" message so the browser keeps the connection


class LiveScreen:
    def __init__(self, port=8765):
        self.clients = []  # one queue per open browser tab
        self.history = deque(maxlen=HISTORY)
        self.lock = threading.Lock()  # clients and history are used from several threads
        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), self._make_handler())
        except OSError as error:
            raise PreCueError(f"Could not start the live screen on port {port}: {error}") from error
        self.url = f"http://127.0.0.1:{port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def send(self, event):
        """Show an event on every open page. event is a dict, e.g. {"type": "alert", "text": "..."}."""
        with self.lock:
            if event["type"] != "partial":  # partial text changes all the time; no need to replay it
                self.history.append(event)
            for client in self.clients:
                client.put(event)

    def _make_handler(self):
        screen = self  # the handler class below needs to reach this LiveScreen

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self._send_page()
                elif self.path == "/events":
                    self._stream_events()
                else:
                    self.send_error(404)

            def _send_page(self):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream_events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                client = queue.Queue()
                with screen.lock:
                    for event in screen.history:  # catch up on what happened before the page opened
                        client.put(event)
                    screen.clients.append(client)
                try:
                    while True:
                        try:
                            event = client.get(timeout=PING_SECONDS)
                            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the page was closed
                finally:
                    with screen.lock:
                        screen.clients.remove(client)

            def log_message(self, *args):
                pass  # do not print a line for every request

        return Handler
