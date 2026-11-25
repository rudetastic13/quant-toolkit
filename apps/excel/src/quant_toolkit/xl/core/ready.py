"""Tell the launcher this Excel session has the worksheet functions registered.

Loaded last (see ``MODULES``), so every other module has been imported and registered by the
time it runs. It writes the session's marker file (``core.session``), which ``quant_toolkit_xl``
waits for before opening a workbook, and removes it when xlOil shuts down. It registers no
functions.
"""
from __future__ import annotations

import os

import xloil as xlo

from quant_toolkit.xl.core import session

_MARKER = session.marker(os.getpid())


def _announce() -> None:
    """Write the marker file."""
    _MARKER.parent.mkdir(parents=True, exist_ok=True)
    _MARKER.touch()


def _forget() -> None:
    """Remove the marker file."""
    _MARKER.unlink(missing_ok=True)


# Only inside Excel: imported from a plain interpreter (the installer's import check), the
# xlOil calls below would start an Excel of their own.
if session.in_excel():
    # Run on Excel's main thread, queued behind the registrations of the modules loaded before this one.
    xlo.excel_callback(_announce)
    xlo.event.PyBye += _forget
