#!/bin/bash
# PreCue for macOS: double-click this file to start. A page opens in your browser.
# You do not need Python: the first start gets uv, a small installer, which downloads Python
# and the libraries PreCue needs (a few minutes, once). After that it opens in seconds.

cd "$(dirname "$0")" || exit 1
UV_VERSION="0.13.6"

fail() {
  echo "$1"
  read -r -p "Press Enter to close."
  exit 1
}

# uv: installs Python and PreCue's libraries. Use the one on this Mac, or get it once.
UV="$(command -v uv || true)"
[ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
if [ -z "$UV" ]; then
  echo "Getting uv $UV_VERSION, the installer PreCue uses (once)..."
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh \
    || fail "Could not get uv. Check your internet connection and try again."
  UV="$HOME/.local/bin/uv"
fi

# Install (or update) the libraries, only when requirements.txt changed since last time.
STAMP=".venv/.precue-requirements"
if [ ! -x .venv/bin/python ] || ! cmp -s requirements.txt "$STAMP"; then
  echo "Setting up PreCue. The first time takes a few minutes..."
  if [ ! -x .venv/bin/python ]; then
    "$UV" venv --python 3.12 .venv || fail "Could not set up Python. Check your internet connection and try again."
  fi
  "$UV" pip install --python .venv/bin/python -r requirements.txt \
    || fail "Could not install PreCue's libraries. Check your internet connection and try again."
  cp requirements.txt "$STAMP"
fi

[ -f .env ] || cp .env.example .env
chmod 600 .env  # your keys: readable by you only

exec .venv/bin/python -m scripts.app ${PRECUE_NO_BROWSER:+--no-browser}
