#!/bin/bash
# Start the transcriber: double-click in Finder (opens Terminal) or run ./run.command [options].
# Default: system default microphone ("Me") + BlackHole 2ch (the other side of the call).
# Extra options are passed on, e.g.  ./run.command --mic "External Microphone" --debug
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Not installed yet: run setup.command first."
  exit 1
fi
exec .venv/bin/python -m app --system BlackHole "$@"
