# C++ extension modules for the finance package (finance._core.*).
#
# This build is intentionally decoupled from the Python packaging backend
# (setuptools): `cmake --install` drops the ABI-tagged extension into
# src/finance/_core/ so the PYTHONPATH=src dev flow picks it up directly.
# Use bin/build_core.sh for a one-command configure/build/install.
cmake_minimum_required(VERSION 3.26)
project(rates_lib_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(datemath src/finance/_core/cpp/datemath.cpp)
target_compile_options(datemath PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>)

install(TARGETS datemath DESTINATION src/finance/_core)
