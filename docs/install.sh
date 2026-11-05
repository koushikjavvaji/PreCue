#!/bin/sh
# Installs (or updates) PreCue in ~/PreCue and starts it. On a Mac or Linux, paste into Terminal:
#   curl -LsSf https://getprecue.vercel.app/install.sh | sh
# Updating keeps your keys (.env), your installed libraries (.venv) and your reports (data/).
# Downloaded this way, macOS does not ask "are you sure you want to open it?" every time.
set -eu

DIR="$HOME/PreCue"
ZIP="https://github.com/koushikjavvaji/PreCue/archive/refs/heads/main.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading PreCue..."
curl -LsSf "$ZIP" -o "$TMP/precue.zip"
unzip -q "$TMP/precue.zip" -d "$TMP"
mkdir -p "$DIR"
# Copy the program; never overwrite your keys, your libraries or your reports.
(cd "$TMP/PreCue-main" && tar --exclude ./.env --exclude ./.venv --exclude ./data -cf - .) | (cd "$DIR" && tar -xf -)
chmod +x "$DIR/PreCue.command"

echo "PreCue is in $DIR. Next time, double-click PreCue.command there. Starting it now..."
exec "$DIR/PreCue.command" < /dev/null
