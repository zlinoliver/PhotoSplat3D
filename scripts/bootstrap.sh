#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
REQUIRED_PY_MAJOR=3
REQUIRED_PY_MINOR=10
RECOMMENDED_PY_MINOR=13

error() {
  echo "[ERROR] $*"
}

warn() {
  echo "[WARN]  $*"
}

info() {
  echo "[INFO]  $*"
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  error "Python not found: $PYTHON_BIN"
  echo "Install Python 3 first, or set PYTHON_BIN=/path/to/python3 and retry."
  exit 1
fi

read -r PY_MAJOR PY_MINOR PY_MICRO < <("$PYTHON_BIN" -c "import sys; print(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)")

if (( PY_MAJOR < REQUIRED_PY_MAJOR || (PY_MAJOR == REQUIRED_PY_MAJOR && PY_MINOR < REQUIRED_PY_MINOR) )); then
  error "Python ${REQUIRED_PY_MAJOR}.${REQUIRED_PY_MINOR}+ is required. Current: ${PY_MAJOR}.${PY_MINOR}.${PY_MICRO}"
  exit 1
fi

if (( PY_MAJOR == REQUIRED_PY_MAJOR && PY_MINOR < RECOMMENDED_PY_MINOR )); then
  warn "Detected Python ${PY_MAJOR}.${PY_MINOR}.${PY_MICRO}. Recommended: Python ${RECOMMENDED_PY_MINOR}.x"
fi

if ! "$PYTHON_BIN" -c "import venv" >/dev/null 2>&1; then
  error "Python module 'venv' is missing."
  echo "Install Python with venv support and retry (Ubuntu example: sudo apt install python3-venv)."
  exit 1
fi

if ! "$PYTHON_BIN" -c "import tkinter" >/dev/null 2>&1; then
  error "Python module 'tkinter' is missing, GUI cannot start."
  echo "Install Python that includes Tk support, then retry."
  exit 1
fi

info "Project root: $ROOT_DIR"
info "Using Python: $PYTHON_BIN (${PY_MAJOR}.${PY_MINOR}.${PY_MICRO})"

if [ ! -d "$VENV_DIR" ]; then
  info "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  info "Reusing existing virtual environment: $VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  error "pip is not available in virtual environment: $VENV_DIR"
  exit 1
fi

echo "==> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing dependencies"
pip install -r "$ROOT_DIR/requirements.txt"

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  sharp gui"
