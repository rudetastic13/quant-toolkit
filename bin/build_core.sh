#!/usr/bin/env bash
#
# Build the finance._core C++ extension modules and install them in-tree.
# Usage:
#     bin/build_core.sh
#
# Requires the pricing conda env to be active (python, pybind11, cmake, ninja).
# The .so lands in src/finance/_core/, importable under the PYTHONPATH=src flow.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$(command -v python)" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build
cmake --install build --prefix .

# expose the compilation database for clangd/CLion
ln -sf build/compile_commands.json compile_commands.json
