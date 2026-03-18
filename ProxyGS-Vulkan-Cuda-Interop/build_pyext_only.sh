#!/usr/bin/env bash
set -euo pipefail

# One-command build for vk2torch_ext (Python extension only).
# Usage:
#   ./build_pyext_only.sh [build_dir]
#
# Environment overrides:
#   PYTHON_EXECUTABLE  (default: $HOME/miniconda3/envs/3dgs/bin/python)
#   CMAKE_BIN          (default: <python_dir>/cmake, fallback: cmake in PATH)
#   VULKAN_SDK_PREFIX  (default: $HOME/VulkanSDK/1.4.321.1/x86_64)

BUILD_DIR="${1:-build-py}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$HOME/miniconda3/envs/3dgs/bin/python}"
VULKAN_SDK_PREFIX="${VULKAN_SDK_PREFIX:-$HOME/VulkanSDK/1.4.321.1/x86_64}"

PYTHON_DIR="$(dirname "$PYTHON_EXECUTABLE")"
DEFAULT_CMAKE_BIN="$PYTHON_DIR/cmake"
CMAKE_BIN="${CMAKE_BIN:-$DEFAULT_CMAKE_BIN}"
if [[ ! -x "$CMAKE_BIN" ]]; then
  CMAKE_BIN="cmake"
fi

echo "[build_pyext_only] cmake: $CMAKE_BIN"
echo "[build_pyext_only] python: $PYTHON_EXECUTABLE"
echo "[build_pyext_only] build dir: $BUILD_DIR"
echo "[build_pyext_only] vulkan prefix: $VULKAN_SDK_PREFIX"

"$CMAKE_BIN" -S . -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE=toolchains/system_no_conda.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
  -DCMAKE_PREFIX_PATH="$VULKAN_SDK_PREFIX" \
  -DUSE_DLSS=OFF \
  -DBUILD_PYTHON_EXT=ON \
  -DBUILD_APP=OFF \
  -DDOWNLOAD_DEFAULT_SCENE=OFF

"$CMAKE_BIN" --build "$BUILD_DIR" --config Release -j"$(nproc)"

echo "[build_pyext_only] build complete:"
echo "  $BUILD_DIR/_bin/Release/vk2torch_ext*.so"
