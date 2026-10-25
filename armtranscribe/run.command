#!/bin/bash
# Start the transcriber: double-click in Finder (opens Terminal) or run ./run.command [options].
# Default: system default microphone ("Me") + BlackHole 2ch (the other side of the call).
# Extra options are passed on, e.g.  ./run.command --mic "External Microphone" --debug
# Ctrl+C stops the session; then Enter starts a new one, any other key quits.
cd "$(dirname "$0")"
source scripts/terminal.sh  # closes this window on quit when double-clicked in Finder
if [ ! -x .venv/bin/python ]; then
  echo "Not installed yet: run setup.command first."
  exit 1
fi

quit() {
  echo
  close_window_on_exit
  exit 0
}

while true; do
  .venv/bin/python -m app --system BlackHole "$@"
  echo
  trap quit INT  # Ctrl+C at the prompt below quits too
  read -r -n 1 -s -p "Press Enter to start a new session, any other key to quit. " key || key=q
  trap - INT     # while the app runs, Ctrl+C belongs to it
  echo
  [ -z "$key" ] || quit
done
