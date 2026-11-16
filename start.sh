#!/usr/bin/env bash
# One command to set everything up (first time only) and open the Neural Flight Hub.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: creating .venv and installing requirements…"
  if command -v uv >/dev/null 2>&1; then
    uv venv && uv pip install -r requirements.txt
  else
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  fi
fi
exec .venv/bin/python hub.py "$@"
