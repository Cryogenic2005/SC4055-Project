#!/usr/bin/env bash
# One-shot environment setup. Works on Linux, macOS, WSL and Colab.
# Usage:  bash setup.sh            (creates .venv and installs pinned deps)
#         bash setup.sh --no-venv  (Colab: install into the current interpreter)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" != "--no-venv" ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
# swig must exist before gymnasium[box2d] builds box2d-py.
python -m pip install "swig>=4.2"
python -m pip install -r requirements.txt

echo
echo "Running smoke test..."
python scripts/smoke_test.py
