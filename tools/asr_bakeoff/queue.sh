#!/bin/bash
# Run candidates one process each: queue.sh <file with "label backend model python" lines>
cd "$(dirname "$0")"
while read -r label backend model py; do
  [ -z "$label" ] || [ "${label:0:1}" = "#" ] && continue
  [ -f "results/$label.json" ] && { echo "skip $label (done)"; continue; }
  echo "== $label"; start=$(date +%s)
  HF_HUB_OFFLINE=1 "$py" run.py "$label" "$backend" "$model" 2>&1 | grep -v -iE "warn|Loading weights|redirects|^\s*$" | tail -8
  echo "   $(( $(date +%s) - start )) s"
done < "$1"
