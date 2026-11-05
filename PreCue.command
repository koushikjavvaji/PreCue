#!/bin/bash
# PreCue for macOS: double-click this file to start. A page opens in your browser.
# The first time, it sets itself up (a few minutes: it downloads the libraries PreCue needs).

cd "$(dirname "$0")" || exit 1

# Find Python 3.11 or newer.
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PYTHON="$candidate"; break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "PreCue needs Python 3.11 or newer. Get it from https://www.python.org/downloads/ and double-click again."
  read -r -p "Press Enter to close."; exit 1
fi

# Install (or update) the libraries, only when requirements.txt changed since last time.
STAMP=".venv/.precue-requirements"
if [ ! -x .venv/bin/python ] || ! cmp -s requirements.txt "$STAMP"; then
  echo "Setting up PreCue. The first time takes a few minutes..."
  "$PYTHON" -m venv .venv \
    && .venv/bin/python -m pip install --quiet --upgrade pip \
    && .venv/bin/python -m pip install --quiet -r requirements.txt \
    && cp requirements.txt "$STAMP" \
    || { echo "Setup failed. Check your internet connection and try again."; read -r -p "Press Enter to close."; exit 1; }
fi

[ -f .env ] || cp .env.example .env
chmod 600 .env  # your keys: readable by you only

exec .venv/bin/python -m scripts.app
