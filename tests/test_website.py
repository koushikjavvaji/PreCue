"""The public website (docs/): its backend function api/release.py and its Vercel settings.
GitHub is replaced by a fake, so these run offline."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

DOCS = Path(__file__).parent.parent / "docs"
spec = importlib.util.spec_from_file_location("release", DOCS / "api" / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

REPO = {"default_branch": "main", "stargazers_count": 42}
COMMIT = {"sha": "84c57ea1234567", "commit": {"committer": {"date": "2026-11-05T08:33:17Z"}}}


def fake_github(url):
    return COMMIT if url.endswith("/commits/main") else REPO


def test_release_info_from_github():
    assert release.release_info(fake_github) == {
        "stars": 42, "updated": "2026-11-05T08:33:17Z", "version": "84c57ea",
        "download": "https://github.com/koushikjavvaji/PreCue/archive/refs/heads/main.zip",
        "repo": "https://github.com/koushikjavvaji/PreCue",
    }


def test_answer_is_cached_at_the_edge():
    status, _, cache = release.answer(fake_github)
    assert status == 200 and "s-maxage=600" in cache  # GitHub sees a few requests an hour, not one per visitor


@pytest.mark.parametrize("problem", [urllib.error.URLError("offline"), KeyError("sha"), TimeoutError()])
def test_github_problems_give_a_clear_error_that_is_not_cached(problem):
    def broken(url):
        raise problem

    status, body, cache = release.answer(broken)
    assert status == 502 and body["error"] and cache == "no-store"


def test_handler_serves_json(monkeypatch):
    monkeypatch.setattr(release, "fetch_json", fake_github)
    server = HTTPServer(("127.0.0.1", 0), release.handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/release", timeout=5) as response:
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert json.load(response)["version"] == "84c57ea"
    server.server_close()


def test_site_sends_security_headers():
    headers = {h["key"]: h["value"] for h in json.loads((DOCS / "vercel.json").read_text())["headers"][0]["headers"]}
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert "connect-src 'self'" in headers["Content-Security-Policy"]  # the page only talks to its own backend
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "microphone=()" in headers["Permissions-Policy"]


def test_site_page_is_safe_and_uses_the_backend():
    page = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'fetch("api/release")' in page
    assert "innerHTML" not in page  # text from the backend is set with textContent only
    assert "http://" not in page.replace("http://www.w3.org", "")  # every link is https
