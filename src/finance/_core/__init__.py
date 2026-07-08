"""Private compiled extension modules (C++ via pybind11).

The extensions are built out-of-band by bin/build_core.sh (CMake + ninja) and
installed in-tree next to this file. The package degrades gracefully when the
extension has not been built: importing finance._core always succeeds and
is_available() reports whether the compiled module is usable.
"""

try:
    from finance._core import datemath
except ImportError:  # extension not built on this machine/platform
    datemath = None


def is_available() -> bool:
    return datemath is not None
