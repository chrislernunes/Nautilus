#!/usr/bin/env bash
# setup.sh — One-shot setup for Nautilus BTC
# ─────────────────────────────────────────────────────────────────────────────
# Creates a virtual environment, installs Python deps, builds C++ module.
# Tested on: Ubuntu 22.04+, macOS 13+ (Apple Silicon / Intel), Debian 12.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="python3"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         NAUTILUS BTC  —  Setup & Installation            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Check Python version ──────────────────────────────────────────────────────
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJ=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MIN=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJ" -lt 3 ] || ([ "$PY_MAJ" -eq 3 ] && [ "$PY_MIN" -lt 11 ]); then
  echo "ERROR: Python 3.11+ required (found $PY_VER)"
  echo "  Ubuntu: sudo apt install python3.11 python3.11-venv"
  echo "  macOS:  brew install python@3.11"
  exit 1
fi
echo "✓ Python $PY_VER"

# ── Create virtual environment ────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  $PYTHON -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
echo "✓ venv activated: $VENV_DIR"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
"$PIP" install --upgrade pip wheel setuptools --quiet

# ── Install Python requirements ───────────────────────────────────────────────
echo ""
echo "Installing Python requirements..."
"$PIP" install -r requirements.txt --quiet
echo "✓ Python dependencies installed"

# ── Build C++ layer ───────────────────────────────────────────────────────────
echo ""
echo "Building C++ ingestion layer..."
bash build_cpp.sh || {
  echo ""
  echo "⚠  C++ build failed — system will use Python fallback mode."
  echo "   This is fully functional but ~10x lower throughput."
  echo "   To fix: ensure cmake, g++17, pybind11 are installed."
}

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p logs models

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Activate venv:    source .venv/bin/activate"
echo "  Run dashboard:    python main.py"
echo "  Backtest only:    python main.py --backtest"
echo "  WF optimisation:  python main.py --optimise"
echo "  Python-only mode: python main.py --no-cpp"
echo "═══════════════════════════════════════════════════════════"
