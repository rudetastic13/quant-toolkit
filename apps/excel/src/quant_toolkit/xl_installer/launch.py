"""``quant_toolkit_xl`` — load the add-in into Excel for this session and open workbooks.

With the default (on-demand) install nothing loads when Excel starts by itself. This command
loads ``xlOil.xll`` into Excel, waits until the ``QuantToolkit.*`` functions are registered, and
then opens the workbooks named on the command line::

    quant_toolkit_xl                      # Excel with the add-in loaded
    quant_toolkit_xl book.xlsx other.xlsx # the same, then the workbooks

If Excel is already running the add-in is loaded into that session; if that session already has
it, the workbooks are simply opened. Nothing persists: close Excel and the add-in is gone.

Excel is started the ordinary way (``EXCEL.EXE <file>``), not through COM automation, so the
other add-ins you have installed load as usual.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from quant_toolkit.xl.core import session
from quant_toolkit.xl_installer.cli import ADDIN_NAME, Paths, addin_dir

_APP_PATH = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\excel.exe"


def excel_exe() -> Path:
    """Where ``EXCEL.EXE`` is installed.

    Returns
    -------
    Path

    Raises
    ------
    SystemExit
        If Excel is not registered under ``App Paths``.
    """
    import winreg

    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            return Path(winreg.QueryValue(hive, _APP_PATH))
        except OSError:
            continue
    raise SystemExit("FAIL  EXCEL.EXE is not registered (desktop Excel not installed?)")


def excel_pids() -> set[int]:
    """The process ids of the running ``EXCEL.EXE`` instances.

    Returns
    -------
    set of int
        Empty if Excel is not running or ``tasklist`` cannot be run.
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq EXCEL.EXE", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        return set()
    pids = set()
    for line in out.splitlines():
        fields = [f.strip('"') for f in line.split('","')]
        if len(fields) > 1 and fields[0].upper() == "EXCEL.EXE" and fields[1].isdigit():
            pids.add(int(fields[1]))
    return pids


def ready_pids() -> set[int]:
    """The running Excel sessions that have the worksheet functions registered.

    Returns
    -------
    set of int
        Process ids with a marker file. Markers left by sessions that are gone (a crashed
        Excel) are deleted on the way.
    """
    running = excel_pids()
    ready = set()
    for pid in session.marked_pids():
        if pid in running:
            ready.add(pid)
        else:
            session.marker(pid).unlink(missing_ok=True)
    return ready


def wait_until_ready(timeout: float) -> bool:
    """Wait for an Excel session to report its functions registered.

    Parameters
    ----------
    timeout : float
        Seconds to wait.

    Returns
    -------
    bool
        False if no session reported in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_pids():
            return True
        time.sleep(0.2)
    return False


def main(argv: list[str] | None = None) -> int:
    """Load the add-in into Excel and open the given workbooks.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` if omitted.

    Returns
    -------
    int
        Exit code: 0 on success, 1 if a workbook is missing or the functions did not register
        within the timeout (the workbooks are then not opened, since they would show ``#NAME?``).
    """
    parser = argparse.ArgumentParser(
        prog="quant_toolkit_xl", description="Load the quant-toolkit add-in into Excel and open workbooks."
    )
    parser.add_argument("workbooks", nargs="*", type=Path, help="workbooks to open once the functions are registered")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for the functions (default 120)")
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        raise SystemExit(f"FAIL  Windows required (this is {sys.platform})")
    workbooks = [w.resolve() for w in args.workbooks]
    missing = [w for w in workbooks if not w.is_file()]
    if missing:
        print("\n".join(f"FAIL  no such workbook: {w}" for w in missing))
        return 1
    paths = Paths.discover()
    if not paths.ini.exists():
        print(f"FAIL  {paths.ini} not found: run quant_toolkit_xl_installer install first")
        return 1
    excel = excel_exe()

    if ready_pids():
        print("ok    Excel already has the add-in loaded")
    else:
        # A copy in XLSTART loads by itself when Excel starts; otherwise hand Excel the add-in to open.
        target = [] if paths.xll.exists() and not excel_pids() else [str(addin_dir() / ADDIN_NAME)]
        print(f"run   {excel.name} {' '.join(target)}".rstrip())
        subprocess.Popen([str(excel), *target])  # noqa: S603 - the registered Excel and our own add-in
        print("      waiting for the functions to register (answer Excel's security prompt if it shows one)")
        if not wait_until_ready(args.timeout):
            print(f"FAIL  no Excel session registered the functions within {args.timeout:g}s; workbooks not opened")
            print(f"      check {paths.ini.with_suffix('.log')} and run quant_toolkit_xl_installer check")
            return 1
        print("ok    functions registered")

    if workbooks:
        print(f"run   {excel.name} {' '.join(w.name for w in workbooks)}")
        subprocess.Popen([str(excel), *map(str, workbooks)])  # noqa: S603
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
