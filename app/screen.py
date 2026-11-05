"""The live screen: a small web page that shows the transcript, alerts and suggestions.

How it works:
  - Python's built-in web server runs in a background thread on
    http://127.0.0.1:8765 (127.0.0.1 = only this computer can open it).
  - "/" sends the full page (web/index.html): transcript, alerts, suggestions.
  - "/overlay" sends the small pop-up page (web/overlay.html) that the
    floating overlay window shows during the meeting (see overlay.py).
  - "/events" stays open and pushes every new event to the page as it
    happens (Server-Sent Events: the browser keeps listening, the server writes).
  - A page opened late still sees the recent history (we replay it).
    "/events?live=1" skips the history: the pop-up must never show old alerts.

The app (scripts/app.py) adds its own page at "/" and an "/api/..." for it (app/session.py):
saving keys, uploading the document, starting and stopping the meeting.

Security: the transcript is private, so only pages opened as http://127.0.0.1:<port>
or http://localhost:<port> are served. A website could otherwise point its own name at
127.0.0.1 ("DNS rebinding") and read the transcript from inside your browser.
Changes (POST) are only accepted from PreCue's own page: the browser's "Origin" must be
exactly this server, and the body must be JSON or a file upload. So a website you visit
cannot save keys or start the microphone behind your back (no "cross-site request forgery").

No extra libraries: only the Python standard library and plain JavaScript.
"""

import json
import queue
import socket
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.errors import PreCueError

WEB = Path(__file__).parent.parent / "web"
PAGES = {"/": WEB / "index.html", "/overlay": WEB / "overlay.html"}
HISTORY = 200  # events replayed to a page that opens late
PING_SECONDS = 15  # a small "still here" message so the browser keeps the connection
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "  # the pages' own code only
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
MAX_WAITING = 500  # events waiting for one slow page; more are dropped instead of filling the memory
MAX_JSON_BYTES = 64 * 1024
POST_TYPES = ("application/json", "application/octet-stream")  # never a type a plain web form can send


class PrivateServer(ThreadingHTTPServer):
    """The web server, with its port kept for PreCue alone.

    Python's server allows "address reuse". On Windows that lets another program bind the
    same port and take over the connections (and read the transcript), so there we ask
    Windows for exclusive use of the port instead.
    """

    allow_reuse_address = sys.platform != "win32"

    def server_bind(self):
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class LiveScreen:
    def __init__(self, port=8765, api=None):
        """api: the app's handler (app/session.py). With it, "/" is the app page and /api/ works."""
        self.api = api
        self.pages = {**PAGES, "/": WEB / "app.html", "/live": WEB / "index.html"} if api else PAGES
        self.clients = []  # one queue per open browser tab
        self.history = deque(maxlen=HISTORY)
        self.lock = threading.Lock()  # clients and history are used from several threads
        try:
            self.server = PrivateServer(("127.0.0.1", port), self._make_handler())
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
                if not client.full():  # a page that stopped reading must not grow without limit
                    client.put(event)

    def clear(self):
        """Forget the history (a new meeting starts with an empty page)."""
        with self.lock:
            self.history.clear()

    def _make_handler(self):
        screen = self  # the handler class below needs to reach this LiveScreen

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if not self._from_this_computer():
                    self.send_error(403)
                    return
                url = urlparse(self.path)  # "/events?live=1" -> path "/events", query "live=1"
                if url.path in screen.pages:
                    self._send_page(screen.pages[url.path])
                elif url.path.startswith("/api/") and screen.api:
                    self._send_json(*screen.api.get(url.path))
                elif url.path == "/events":
                    live_only = parse_qs(url.query).get("live") == ["1"]
                    self._stream_events(replay=not live_only)
                else:
                    self.send_error(404)

            def do_POST(self):
                if not screen.api or not self.path.startswith("/api/"):
                    self.send_error(404)
                    return
                kind = self.headers.get("Content-Type", "").split(";")[0].strip()
                if not self._from_this_computer() or not self._from_our_page() or kind not in POST_TYPES:
                    self.send_error(403)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    size = -1
                limit = screen.api.MAX_UPLOAD_BYTES if kind == "application/octet-stream" else MAX_JSON_BYTES
                if size < 0 or size > limit:
                    self._send_json(413, {"error": f"Too big: the limit is {limit // (1024 * 1024) or 1} MB."})
                    self.close_connection = True
                    return
                body = self.rfile.read(size)
                if kind == "application/json":
                    try:
                        body = json.loads(body or b"{}")
                    except ValueError:
                        self._send_json(400, {"error": "Not valid JSON."})
                        return
                self._send_json(*screen.api.post(self.path, body, self.headers))

            def _from_our_page(self):
                """True if the request comes from a page of this server (the browser sets Origin)."""
                port = self.server.server_address[1]
                return self.headers.get("Origin") in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")

            def _send_json(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self._send_safety_headers()
                self.end_headers()
                self.wfile.write(body)

            def _send_safety_headers(self):
                """Even if some text were ever shown as HTML, the page could not load or send anything:
                no outside scripts, images or connections (only our own /events), no forms,
                and no other website can show the page inside a frame."""
                self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")

            def _from_this_computer(self):
                """True if the page was opened as 127.0.0.1 or localhost (blocks DNS rebinding)."""
                port = self.server.server_address[1]
                return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

            def _send_page(self, page):
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._send_safety_headers()
                self.end_headers()
                self.wfile.write(body)

            def _stream_events(self, replay):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self._send_safety_headers()
                self.end_headers()

                client = queue.Queue(maxsize=MAX_WAITING)
                with screen.lock:
                    if replay:
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
