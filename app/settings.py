"""The API keys: read them, save them in .env, and test that they work.

The app's setup page uses this, so nobody has to edit .env by hand.

  - Keys are saved in the project's .env file, readable by your user account only (0o600).
  - The page never gets a key back, only whether it is set and whether it works.
  - Testing a key asks the provider for its list of models (or projects). That is free:
    it uses no tokens and no free-tier quota.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from app.stt import SSL_CONTEXT

ENV_FILE = Path(__file__).parent.parent / ".env"

KEYS = {
    "DEEPGRAM_API_KEY": {
        "name": "Deepgram", "use": "Hears you and whispers back",
        "link": "https://console.deepgram.com/signup",
        "check": "https://api.deepgram.com/v1/projects", "header": lambda key: {"Authorization": f"Token {key}"},
    },
    "GEMINI_API_KEY": {
        "name": "Gemini", "use": "Reads your document before the meeting",
        "link": "https://aistudio.google.com/apikey",
        "check": "https://generativelanguage.googleapis.com/v1beta/models", "header": lambda key: {"x-goog-api-key": key},
    },
    "GROQ_API_KEY": {
        "name": "Groq", "use": "Checks your numbers live, in about 2 seconds",
        "link": "https://console.groq.com/keys",
        "check": "https://api.groq.com/openai/v1/models", "header": lambda key: {"Authorization": f"Bearer {key}"},
    },
}
MAX_KEY_LENGTH = 300


def key_is_set(name):
    value = os.getenv(name, "")
    return bool(value) and value != "your_key_here"


def key_status():
    """For the page: each key's name, use, link and whether it is set. Never the key itself."""
    return [{"id": name, "name": info["name"], "use": info["use"], "link": info["link"], "set": key_is_set(name)}
            for name, info in KEYS.items()]


def save_keys(new_keys, env_file=None):
    """Write the given keys into .env (other lines stay as they are) and use them right away.

    new_keys: {"GROQ_API_KEY": "gsk_..."}. Only the known key names are accepted.
    env_file: for tests. ENV_FILE is read here, at call time, so tests can point it elsewhere.
    """
    env_file = env_file or ENV_FILE
    clean = {}
    for name, value in new_keys.items():
        value = str(value).strip()
        if name not in KEYS:
            raise ValueError(f"Unknown setting: {name}")
        if not value or len(value) > MAX_KEY_LENGTH or any(c.isspace() or c in "\"'#" for c in value):
            raise ValueError(f"That does not look like a {KEYS[name]['name']} key.")
        clean[name] = value

    lines = env_file.read_text().splitlines() if env_file.exists() else []
    for name, value in clean.items():
        line = f"{name}={value}"
        for i, old in enumerate(lines):
            if old.split("=", 1)[0].strip() == name:
                lines[i] = line
                break
        else:
            lines.append(line)

    # Write a private temporary file, then swap it in: .env is never half-written or readable by others.
    temporary = env_file.with_name(env_file.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        file.write("\n".join(lines) + "\n")
    os.replace(temporary, env_file)
    os.environ.update(clean)


def check_key(name):
    """Ask the provider whether the key works. Returns (works, message)."""
    if not key_is_set(name):
        return False, "Not set"
    info = KEYS[name]
    request = urllib.request.Request(info["check"], headers=info["header"](os.environ[name]))
    try:
        with urllib.request.urlopen(request, context=SSL_CONTEXT, timeout=8) as response:
            json.load(response)
        return True, "Connected"
    except urllib.error.HTTPError as error:
        if error.code in (400, 401, 403):
            return False, "Key rejected"
        if error.code == 429:
            return True, "Connected (busy right now)"  # the key is fine, the free tier is just busy
        return False, f"{info['name']} answered HTTP {error.code}"
    except (OSError, ValueError):
        return False, "No connection"
