#!/bin/bash
# One-click setup: double-click in Finder (opens Terminal) or run ./setup.command.
#
# Creates armtranscribe/.venv with everything from requirements.txt and downloads the default models
# (ASR: NVIDIA FastConformer, MT: NLLB-200 1.3B + 600M fallback, speaker embeddings: ECAPA) into armtranscribe/models/. Safe to re-run: the venv is
# reused, models already in models/ are skipped. This and scripts/download_models.py are the
# only steps that use the network; afterwards the transcriber runs offline.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/terminal.sh  # closes this window at the end when double-clicked in Finder
trap 'echo; echo "Setup failed (see the error above). Fix it and run setup.command again."' ERR

# Finder-launched Terminals may not have Homebrew on PATH, so also look in its usual places
PY=""
for c in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
  if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11 not found. Install it with Homebrew:  brew install python@3.11"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "== Creating .venv ($("$PY" --version))"
  "$PY" -m venv .venv
fi

echo "== Installing dependencies (first run downloads ~2 GB)"
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt

echo "== Downloading the default models (first run: ~8 GB, NLLB is converted locally)"
.venv/bin/python scripts/download_models.py
# the faster NLLB the app switches to while it falls behind (--mt-fast)
.venv/bin/python scripts/download_models.py nllb-600m-ct2

echo "== Audio devices"
devices=$(.venv/bin/python -m app --list-devices)
echo "$devices"
if ! grep -q "BlackHole" <<<"$devices"; then
  echo
  echo "BlackHole is not installed: it is needed to capture the other side of a call."
  echo "  brew install blackhole-2ch   (then restart the Mac), and set up a Multi-Output Device (see README)."
fi

echo
echo "Done. Start the transcriber: double-click run.command (microphone + BlackHole), or from this folder:"
echo "  cd \"$(pwd)\" && source .venv/bin/activate && python -m app --system BlackHole"

if [ "$OPENED_FROM_FINDER" = 1 ]; then
  echo
  read -r -n 1 -s -p "Press any key to close this window." || true
  echo
  close_window_on_exit
fi
