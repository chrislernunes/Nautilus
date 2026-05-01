#!/usr/bin/env bash
# build_cpp.sh — Build the C++ pybind11 module
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./build_cpp.sh            # Release build
#   ./build_cpp.sh --debug    # Debug build
#   ./build_cpp.sh --clean    # Clean and rebuild
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/cpp"
BUILD_DIR="$CPP_DIR/build"
BUILD_TYPE="Release"

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --debug)  BUILD_TYPE="Debug" ;;
    --clean)  rm -rf "$BUILD_DIR" && echo "Clean done." ;;
    --help)
      echo "Usage: $0 [--debug] [--clean] [--help]"
      exit 0 ;;
  esac
done

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Nautilus BTC — C++ Layer Build                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Build type : $BUILD_TYPE"
echo "  Source dir : $CPP_DIR"
echo "  Build dir  : $BUILD_DIR"
echo ""

# ── Check prerequisites ───────────────────────────────────────────────────────
check_tool() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: '$1' not found. Please install it."
    exit 1
  fi
}

check_tool cmake
check_tool python3
check_tool c++

echo "✓ cmake  $(cmake --version | head -1)"
echo "✓ python $(python3 --version)"
echo "✓ c++    $(c++ --version | head -1)"

# ── Detect pybind11 cmake dir ─────────────────────────────────────────────────
PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || true)
if [ -z "$PYBIND11_DIR" ]; then
  echo "ERROR: pybind11 not found. Run: pip install pybind11"
  exit 1
fi
echo "✓ pybind11 cmake dir: $PYBIND11_DIR"

# ── Check optional deps ───────────────────────────────────────────────────────
echo ""
echo "Optional dependencies:"
if pkg-config --exists libwebsockets 2>/dev/null; then
  echo "  ✓ libwebsockets found (native C++ WS enabled)"
else
  echo "  ○ libwebsockets NOT found (Python WS bridge will be used)"
  echo "    → Install: sudo apt install libwebsockets-dev"
  echo "    → macOS:   brew install libwebsockets"
fi

if pkg-config --exists simdjson 2>/dev/null || \
   [ -f "/usr/include/simdjson.h" ] || \
   [ -f "/usr/local/include/simdjson.h" ]; then
  echo "  ✓ simdjson found (zero-copy JSON enabled)"
else
  echo "  ○ simdjson NOT found (built-in parser will be used)"
fi

# ── CMake configure ───────────────────────────────────────────────────────────
echo ""
echo "Configuring..."
mkdir -p "$BUILD_DIR"
cmake \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -S "$CPP_DIR" \
  -B "$BUILD_DIR" \
  2>&1 | grep -E "(Found|Warning|Error|nautilus|pybind|LWS|simdjson|Build type)"

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo "Building..."
NPROC=$(python3 -c "import os; print(os.cpu_count())")
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --parallel "$NPROC"

# ── Verify output ─────────────────────────────────────────────────────────────
SO_PATH="$SCRIPT_DIR/python/core/nautilus_cpp"
# Find the actual .so / .pyd file
SO_FILE=$(find "$SCRIPT_DIR/python/core" -name "nautilus_cpp*.so" -o \
                                          -name "nautilus_cpp*.pyd" 2>/dev/null | head -1)

echo ""
if [ -n "$SO_FILE" ]; then
  echo "✓ Build successful!"
  echo "  Output: $SO_FILE"
  # Quick smoke test
  python3 -c "
import sys
sys.path.insert(0, '$(dirname $SO_FILE)')
import nautilus_cpp
print(f'  Module version : {nautilus_cpp.__version__}')
print(f'  LWS compiled   : {nautilus_cpp.__built_with_lws__}')
"
else
  echo "✗ Build failed — .so not found in python/core/"
  exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  C++ build complete. Run: python main.py"
echo "═══════════════════════════════════════════════════════════"
