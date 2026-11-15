#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
yes=false
vision=false
for arg in "$@"; do
  case "$arg" in --yes) yes=true ;; --vision) vision=true ;; *) echo "Unknown option: $arg" >&2; exit 2 ;; esac
done
if ! "$yes"; then
  read -r -p "Install VSSS client dependencies in .venv? [Y/n] " answer
  [[ ${answer:-y} =~ ^[Yy]([Ee][Ss])?$ ]] || exit 0
  if ! "$vision"; then
    read -r -p "Also install the optional vision service? [y/N] " answer
    [[ ${answer:-n} =~ ^[Yy]([Ee][Ss])?$ ]] && vision=true
  fi
fi
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -e .
if "$vision"; then bash aruco_gps/setup.sh --yes; fi
