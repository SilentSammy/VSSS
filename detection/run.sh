#!/bin/bash
# Configures the network and launches camera capture, viewing, and vision.
# Usage: ./run.sh [extra args passed to main.py]
cd "$(dirname "$0")" || exit 1

if ! NETWORK_INFO=$(bash ./configure_lab_network.sh --json); then
  echo "Lab network configuration failed; vision will not start." >&2
  exit 1
fi
BROADCAST=$(python3 - "$NETWORK_INFO" <<'PY'
import ipaddress
import json
import sys

try:
    result = json.loads(sys.argv[1])
    if result.get("success") is not True:
        raise ValueError(result.get("error") or "network is not ready")
    print(ipaddress.IPv4Address(result["broadcast"]))
except (ValueError, KeyError, TypeError, AttributeError) as error:
    print(f"Invalid network configuration result: {error}", file=sys.stderr)
    sys.exit(1)
PY
) || exit 1

source ../.venv/bin/activate || exit 1

# An explicit setup (or a help request) skips the interactive menu.
SELECT_SETUP=true
for arg in "$@"; do
  case "$arg" in
    --setup|--setup=*|--help|-h) SELECT_SETUP=false ;;
  esac
done

if "$SELECT_SETUP"; then
  # Query the actual setups so names and print dimensions stay in sync.
  SETUP_DATA=$(python - <<'PY'
import sys
sys.path.insert(0, "..")
from detection.vision_setup import DEFAULT_SETUP, VISION_SETUPS

for name, setup in VISION_SETUPS.items():
    width, height = setup.board.get_print_dimensions()
    default = "default" if setup is DEFAULT_SETUP else ""
    print(f"{name}\t{width * 100:g} x {height * 100:g} cm\t{default}")
PY
  ) || exit 1

  SETUP_NAMES=()
  DEFAULT_CHOICE=1
  echo "Select a vision setup (print width x height):"
  while IFS=$'\t' read -r setup_name dimensions default; do
    [ -n "$setup_name" ] || continue
    SETUP_NAMES+=("$setup_name")
    index=${#SETUP_NAMES[@]}
    if [ "$default" = "default" ]; then
      DEFAULT_CHOICE=$index
      printf '  %d) %s - %s (default)\n' "$index" "$setup_name" "$dimensions"
    else
      printf '  %d) %s - %s\n' "$index" "$setup_name" "$dimensions"
    fi
  done <<< "$SETUP_DATA"

  if [ "${#SETUP_NAMES[@]}" -eq 0 ]; then
    echo "No vision setups are available." >&2
    exit 1
  fi
  while true; do
    read -r -p "Choose a setup (press Enter for default): " choice || choice=$DEFAULT_CHOICE
    choice=${choice:-$DEFAULT_CHOICE}
    # Compare as strings to avoid interpreting user input as shell arithmetic.
    SELECTED_SETUP=
    for index in "${!SETUP_NAMES[@]}"; do
      if [ "$choice" = "$((index + 1))" ]; then
        SELECTED_SETUP=${SETUP_NAMES[$index]}
        break
      fi
    done
    [ -n "$SELECTED_SETUP" ] && break
    echo "Enter an option number from 1 to ${#SETUP_NAMES[@]}."
  done
  echo "Using vision setup: $SELECTED_SETUP"
  set -- --setup "$SELECTED_SETUP" "$@"
fi

echo "Using broadcast address: $BROADCAST"
echo "Starting camera streaming and vision. Press Ctrl+C to stop both."
# Replace Bash with the supervisor so terminal signals reach it directly.
# Both child processes inherit this terminal; the supervisor owns their cleanup.
exec python -u -B camera_stream.py --run-vision --broadcast "$BROADCAST" "$@"
