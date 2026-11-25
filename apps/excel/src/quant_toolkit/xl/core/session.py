"""The "functions are registered" marker an Excel session leaves for the launcher.

xlOil imports the worksheet modules after the add-in itself has loaded, a second or more later.
A workbook opened in that gap evaluates ``QuantToolkit.*`` cells to ``#NAME?``, and Excel keeps
that result until the formula is re-entered: a recalculation does not clear it. So the launcher
(``quant_toolkit_xl``) loads the add-in, waits for the marker, and only then opens the workbook.

The marker is an empty file named after the Excel process, written by ``quant_toolkit.xl.core.ready``
and removed when xlOil shuts down. This module imports nothing from xlOil, so the launcher can
use it outside Excel.
"""
from __future__ import annotations

import ctypes
import os
import tempfile
from pathlib import Path

_PREFIX = "ready-"


def in_excel() -> bool:
    """Whether this Python is running inside Excel.

    Returns
    -------
    bool
        True when the host process is ``EXCEL.EXE``, as it is for modules xlOil loads. False in
        a plain interpreter, where the worksheet modules are also imported (the installer checks
        that each one imports, and they are exercised from Python).
    """
    buffer = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer))
    return Path(buffer.value).name.upper() == "EXCEL.EXE"


def marker_dir() -> Path:
    """The folder the markers live in.

    Returns
    -------
    Path
        ``%LOCALAPPDATA%\\quant_toolkit\\xl``, or the same under the temp folder if
        ``%LOCALAPPDATA%`` is not set.
    """
    return Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "quant_toolkit" / "xl"


def marker(pid: int) -> Path:
    """The marker file for one Excel process.

    Parameters
    ----------
    pid : int
        The process id of ``EXCEL.EXE``.

    Returns
    -------
    Path
        ``<marker_dir>\\ready-<pid>``; it exists while that session has the functions registered.
    """
    return marker_dir() / f"{_PREFIX}{pid}"


def marked_pids() -> list[int]:
    """The process ids that currently have a marker.

    Returns
    -------
    list of int
        One per marker file, in no particular order. A crashed Excel leaves its marker behind,
        so the caller checks each id against the running processes.
    """
    pids = []
    for path in marker_dir().glob(f"{_PREFIX}*"):
        suffix = path.name[len(_PREFIX) :]
        if suffix.isdigit():
            pids.append(int(suffix))
    return pids
