#!/usr/bin/env bash
# Update the currently checked-out branch without discarding local work.
set -euo pipefail
cd "${1:-$(dirname "$0")}" || exit 1
git pull --ff-only
