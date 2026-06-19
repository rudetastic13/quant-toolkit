"""xlwings access with a no-op fallback.

When xlwings is installed (the trader's Windows machine) ``xw`` is the real module and the
``q*`` functions register as UDFs. When it isn't (WSL dev box, CI), ``xw`` is a shim whose
decorators pass functions through unchanged, so ``quant_toolkit_xl`` still imports and the
pure logic in ``api`` stays unit-testable without Excel.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised only where xlwings is installed
    import xlwings as xw  # type: ignore

    HAS_XLWINGS = True
except Exception:  # ImportError, or import-time failures off-Windows
    HAS_XLWINGS = False

    class _Shim:
        """Pass-through stand-ins for the xlwings UDF decorators."""

        def func(self, f=None, **_kwargs):
            return f if f is not None else (lambda g: g)

        def sub(self, f=None, **_kwargs):
            return f if f is not None else (lambda g: g)

        def arg(self, *_args, **_kwargs):
            return lambda f: f

        def ret(self, *_args, **_kwargs):
            return lambda f: f

    xw = _Shim()  # type: ignore
