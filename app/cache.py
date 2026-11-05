"""Small JSON cache in data/cache, shared by every slow step.

Slow steps (LLM calls, embedding a whole document) save their results here.
The file name is a fingerprint of everything the result depends on, so if
the document, the model or a setting changes, a new file is used and the
old result is never wrongly reused.

Delete the data/cache folder at any time to start fresh.
"""

import hashlib
import json
from pathlib import Path

CACHE_FOLDER = Path("data/cache")


def cache_file(kind, *parts):
    """Return the cache path for this kind of result, e.g. cache_file("facts", model, document)."""
    fingerprint = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    return CACHE_FOLDER / f"{kind}_{fingerprint}.json"


def read_json(path):
    """Return the saved data, or None if there is no (readable) file."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # a broken cache is not fatal; we just do the work again


def write_json(path, data):
    """Save data. The cache holds your documents' text, so only your user account may read it."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)  # all at once: a crash mid-write never leaves a half-written file


def forget_document(text, folder=None):
    """Delete every cache file that holds this document's text (when you remove the document).

    File names are fingerprints, so we look inside: a file that contains a few long, distinctive
    lines of the document belongs to it. Returns how many files were deleted.
    """
    folder = folder or CACHE_FOLDER
    lines = [line.strip() for line in text.splitlines()]
    snippets = [line for line in lines if len(line) >= 30 and '"' not in line and "\\" not in line][:3]
    if not snippets or not folder.exists():
        return 0
    removed = 0
    for path in folder.glob("*.json"):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(snippet in content for snippet in snippets):
            path.unlink(missing_ok=True)
            removed += 1
    return removed
