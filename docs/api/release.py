"""The website's backend: GET /api/release -> what the download section shows.

  {"stars": 12, "updated": "2026-11-05T08:37:58Z", "version": "84c57ea",
   "download": "https://github.com/.../main.zip", "repo": "https://github.com/..."}

It runs as a Vercel serverless function (Python, standard library only).
GitHub allows 60 requests an hour without a token, so the answer is cached at Vercel's
edge for 10 minutes (and an older copy is served while a new one is fetched): one visitor
or a million, GitHub sees a few requests an hour. GITHUB_TOKEN (optional, read-only)
raises that limit.

The page works without this: if it fails, the page keeps its normal text.
"""

import json
import os
import ssl
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

REPO = "koushikjavvaji/PreCue"
API = f"https://api.github.com/repos/{REPO}"
CACHE = "public, s-maxage=600, stale-while-revalidate=3600"

try:  # Python from python.org on macOS has no trusted certificates; certifi has them
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # on Vercel the system's certificates are used
    SSL_CONTEXT = ssl.create_default_context()


def fetch_json(url):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "precue-website"}
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=5, context=SSL_CONTEXT) as response:
        return json.load(response)


def release_info(fetch=fetch_json):
    """The latest version of PreCue, from GitHub. fetch is replaced in tests."""
    repo = fetch(API)
    branch = repo["default_branch"]
    commit = fetch(f"{API}/commits/{branch}")
    return {
        "stars": int(repo["stargazers_count"]),
        "updated": commit["commit"]["committer"]["date"],
        "version": commit["sha"][:7],
        "download": f"https://github.com/{REPO}/archive/refs/heads/{branch}.zip",
        "repo": f"https://github.com/{REPO}",
    }


def answer(fetch=fetch_json):
    """(status, body, cache header) for the request."""
    try:
        return 200, release_info(fetch), CACHE
    except (urllib.error.URLError, OSError, KeyError, TypeError, ValueError):
        return 502, {"error": "GitHub could not be reached. Try again later."}, "no-store"


class handler(BaseHTTPRequestHandler):  # Vercel looks for a class with this name
    def do_GET(self):
        status, body, cache = answer(fetch_json)  # looked up now, so tests can replace fetch_json
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)
