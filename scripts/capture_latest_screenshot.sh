#!/usr/bin/env bash
set -euo pipefail

URL="${1:-https://namuan.github.io/post-automation-world/post_automation_world.html}"
OUTPUT="${2:-assets/post-automation-world.webp}"
BROWSER="${BROWSER:-webkit}"
WIDTH="${WIDTH:-1440}"
HEIGHT="${HEIGHT:-960}"
WAIT_MS="${WAIT_MS:-5500}"
QUALITY="${QUALITY:-78}"
SESSION="post-auto-shot-$$"
TMP_DIR=".screenshot-tmp"
TMP_PNG="$TMP_DIR/post-automation-world.png"

cleanup() {
  playwright-cli -s="$SESSION" close >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command playwright-cli
require_command cwebp

mkdir -p "$(dirname "$OUTPUT")" "$TMP_DIR"

echo "Opening $URL"
playwright-cli -s="$SESSION" open --browser "$BROWSER" "$URL" >/dev/null
playwright-cli -s="$SESSION" resize "$WIDTH" "$HEIGHT" >/dev/null
playwright-cli -s="$SESSION" eval "new Promise((resolve, reject) => {
  const started = Date.now();
  const tick = () => {
    if (document.getElementById('playPause') && document.getElementById('tourSkip')) return resolve(true);
    if (Date.now() - started > 10000) return reject(new Error('Simulation controls did not load'));
    setTimeout(tick, 100);
  };
  tick();
})" >/dev/null

echo "Starting simulation and waiting ${WAIT_MS}ms"
playwright-cli -s="$SESSION" eval "(() => {
  const set = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  document.getElementById('tourSkip')?.click();
  const onboarding = document.getElementById('onboarding');
  if (onboarding) onboarding.hidden = true;
  set('aiLevy', 0.14);
  set('robotTax', 0.12);
  set('ubi', 0.08);
  set('speed', 2.4);
  const start = document.getElementById('playPause');
  if (start && start.textContent.trim() !== 'Pause') start.click();
  return document.title;
})()" >/dev/null

playwright-cli -s="$SESSION" eval "new Promise(resolve => setTimeout(resolve, $WAIT_MS))" >/dev/null

echo "Capturing PNG"
playwright-cli -s="$SESSION" screenshot --filename "$TMP_PNG" >/dev/null
if [ ! -s "$TMP_PNG" ]; then
  echo "Screenshot capture failed: $TMP_PNG was not written" >&2
  exit 1
fi

echo "Compressing to $OUTPUT"
cwebp -quiet -q "$QUALITY" "$TMP_PNG" -o "$OUTPUT"

BYTES="$(wc -c < "$OUTPUT" | tr -d ' ')"
echo "Wrote $OUTPUT (${BYTES} bytes)"
