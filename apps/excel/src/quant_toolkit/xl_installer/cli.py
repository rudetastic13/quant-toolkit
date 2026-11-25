"""``quant_toolkit_xl_installer`` — point xlOil at this env and our modules (prototype).

Commands
--------
check
    Report on the prerequisites: Windows, ``%APPDATA%``, a registered Excel, xlOil importable,
    Excel not running, and the current install state.
install
    ``check``, then write ``%APPDATA%\\xlOil\\xlOil.ini``: point it at this Python and tag our
    modules into ``[xlOil_Python] LoadModules`` (replacing any set a previous run tagged).

    By default the add-in is **on demand**: nothing is put in Excel's startup folder, so Excel
    starts without it and ``quant_toolkit_xl`` loads it for one session. A copy left in
    ``XLSTART`` by an earlier install is removed. ``--startup`` is the persistent alternative:
    it runs ``xloil install``, which copies ``xlOil.xll`` into ``XLSTART`` so every Excel
    session loads it. ``--trust`` adds the add-in's folder to Excel's trusted locations, so
    opening it on demand does not raise Excel's security prompt.
uninstall
    Remove our tagged modules from the ini and our trusted location, then ``xloil remove``
    (drops any ``.xll`` from ``XLSTART``; xlOil leaves the ini in place).

Every ini edit is backed up to ``xlOil.ini.bak`` first and can be previewed with ``--dry-run``.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from quant_toolkit import xl
from quant_toolkit.xl_installer import ini

DEFAULT_MODULES = xl.module_names()  # promoted in quant_toolkit/xl/__init__.py
ADDIN_NAME = "xlOil.xll"
TRUSTED_LOCATION = "QuantToolkit"  # our subkey under Excel's "Trusted Locations"


def addin_dir() -> Path:
    """The folder xlOil's binaries are installed in.

    Returns
    -------
    Path
        ``<env>\\share\\xloil`` (or ``%XLOIL_BIN_DIR%``), holding ``xlOil.xll``, its DLLs and the
        template ini. This is the copy of the add-in loaded on demand.
    """
    return Path(os.environ.get("XLOIL_BIN_DIR") or Path(sys.prefix) / "share" / "xloil")


@dataclass(frozen=True)
class Paths:
    """The per-user locations the installer reads and writes.

    Attributes
    ----------
    appdata : Path
        ``%APPDATA%``.
    xlstart : Path
        Excel's per-user startup folder, ``%APPDATA%\\Microsoft\\Excel\\XLSTART``; an add-in
        placed here loads whenever Excel starts.
    ini : Path
        xlOil's settings file, ``%APPDATA%\\xlOil\\xlOil.ini``.
    """
    appdata: Path
    xlstart: Path
    ini: Path

    @property
    def xll(self) -> Path:
        """Where ``xloil install`` puts the add-in: ``XLSTART\\xlOil.xll``."""
        return self.xlstart / ADDIN_NAME

    @classmethod
    def discover(cls) -> Paths:
        """The paths for the current user.

        Returns
        -------
        Paths

        Raises
        ------
        SystemExit
            If ``%APPDATA%`` is not set (not a Windows user profile).
        """
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise SystemExit("FAIL  %APPDATA% is not set; this tool needs a Windows user profile")
        root = Path(appdata)
        return cls(root, root / "Microsoft" / "Excel" / "XLSTART", root / "xlOil" / "xlOil.ini")


# -- checks -------------------------------------------------------------------------------------


def _excel_version() -> str | None:
    """The registered Excel version.

    Returns
    -------
    str or None
        The ProgID Excel registers as current (``"Excel.Application.16"``), or ``None`` if
        desktop Excel is not installed or the registry cannot be read.
    """
    try:
        import winreg

        return winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, r"Excel.Application\CurVer")
    except OSError:
        return None


def _excel_running() -> bool:
    """Whether an ``EXCEL.EXE`` process is running.

    Returns
    -------
    bool
        False if ``tasklist`` cannot be run, so a failed check never blocks an install.
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq EXCEL.EXE", "/NH"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return False
    return "EXCEL.EXE" in out.upper()


def run_checks(paths: Paths, *, startup: bool = False, create_xlstart: bool = False) -> bool:
    """Print one line per prerequisite and report whether an install can go ahead.

    Parameters
    ----------
    paths : Paths
    startup : bool
        The add-in is to go in ``XLSTART``, so that folder must exist.
    create_xlstart : bool
        Create the ``XLSTART`` folder if it is missing (Excel creates it on first start).

    Returns
    -------
    bool
        True when nothing blocking failed. A running Excel is a warning, not a failure.

    Notes
    -----
    Lines are prefixed ``ok``, ``warn``, ``info`` or ``FAIL``. The checks are: Windows,
    ``XLSTART`` exists (startup installs only), Excel is registered, ``xloil`` imports and its
    add-in file is present, the Python bitness, whether Excel is running, and the current
    mode (on demand or startup), trusted location and ini.
    """
    ok = True

    def report(status: str, msg: str) -> None:
        """Print one status line; a ``FAIL`` marks the whole run as failed."""
        nonlocal ok
        ok &= status != "FAIL"
        print(f"{status:<5} {msg}")

    if sys.platform != "win32":
        report("FAIL", f"Windows required (this is {sys.platform}); xlOil runs inside desktop Excel")
        return False
    report("ok", f"APPDATA   {paths.appdata}")

    if paths.xlstart.is_dir():
        report("ok", f"XLSTART   {paths.xlstart}")
    elif create_xlstart:
        paths.xlstart.mkdir(parents=True, exist_ok=True)
        report("ok", f"XLSTART   {paths.xlstart} (created)")
    elif startup:
        report("FAIL", f"XLSTART   {paths.xlstart} missing: start Excel once, or pass --create-xlstart")

    version = _excel_version()
    report("ok" if version else "FAIL", f"Excel     {version or 'not registered (desktop Excel not installed?)'}")

    try:
        xloil = importlib.import_module("xloil")
        report("ok", f"xloil     {getattr(xloil, '__version__', '?')} in {sys.prefix}")
    except ImportError:
        report("FAIL", "xloil     not importable: pip install -e apps/excel")
    source = addin_dir() / ADDIN_NAME
    report("ok" if source.exists() else "FAIL", f"addin     {source}{'' if source.exists() else ' missing'}")

    report("ok", f"python    {sys.version.split()[0]} {'64' if sys.maxsize > 2**32 else '32'}-bit (must match Excel)")

    if _excel_running():
        report("warn", "Excel is running: close it before install/uninstall, changes load on next start")

    if paths.xll.exists():
        report("info", f"mode      startup: {paths.xll} loads with every Excel session")
    else:
        report("info", "mode      on demand: nothing in XLSTART; quant_toolkit_xl loads the add-in for a session")
    trusted = _trusted_location()
    report("info", f"trusted   {trusted or 'no (Excel may ask before loading the add-in on demand)'}")
    if paths.ini.exists():
        text = paths.ini.read_text(encoding="utf-8")
        report("info", f"ini       {paths.ini}")
        report("info", f"          LoadModules={ini.load_modules(text)}  ours={ini.tagged_modules(text)}")
    else:
        report("info", f"ini       {paths.ini} not created yet")
    return ok


# -- ini editing --------------------------------------------------------------------------------


def _check_importable(modules: list[str]) -> list[str]:
    """Import each module in this interpreter and report the ones that fail.

    Parameters
    ----------
    modules : list of str
        Fully qualified module names about to be written to ``LoadModules``.

    Returns
    -------
    list of str
        The names that raised on import; empty if all imported. A module that fails here would
        fail inside Excel too, where the error is only visible in xlOil's log.
    """
    bad = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - report every failure, any kind
            print(f"FAIL  {name} does not import in this env: {type(e).__name__}: {e}")
            bad.append(name)
    return bad


def _write_ini(paths: Paths, new: str, *, dry_run: bool) -> None:
    """Write new ini text, showing the change and keeping a backup.

    Parameters
    ----------
    paths : Paths
    new : str
        The full new contents of the ini.
    dry_run : bool
        Print the diff but write nothing.

    Notes
    -----
    Does nothing if ``new`` equals what is on disk. Otherwise prints a unified diff, copies the
    current file to ``xlOil.ini.bak`` and writes ``new``.
    """
    old = paths.ini.read_text(encoding="utf-8") if paths.ini.exists() else ""
    if new == old:
        print(f"ok    {paths.ini} already up to date")
        return
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), str(paths.ini), "(new)", lineterm="", n=1)
    print("\n".join(diff))
    if dry_run:
        print("dry-run: ini not written")
        return
    if not old:
        paths.ini.parent.mkdir(parents=True, exist_ok=True)
        paths.ini.write_text(new, encoding="utf-8")
        print(f"ok    created {paths.ini}")
        return
    shutil.copy2(paths.ini, paths.ini.with_suffix(".ini.bak"))
    paths.ini.write_text(new, encoding="utf-8")
    print(f"ok    wrote {paths.ini} (backup: {paths.ini.with_suffix('.ini.bak').name})")


def _configured_ini(paths: Paths) -> str:
    """The ini text pointing xlOil at this interpreter, without touching ``XLSTART``.

    Parameters
    ----------
    paths : Paths

    Returns
    -------
    str
        The current ini, or xlOil's template if there is none yet, with ``XLOIL_PATH`` and
        ``PYTHONEXECUTABLE`` set for this env.

    Raises
    ------
    SystemExit
        If the settings could not be found in the text to set them.

    Notes
    -----
    This is the ini half of ``xloil install``, done with xlOil's own helper so the result is the
    same; the other half (copying the add-in into ``XLSTART``) is what an on-demand install skips.
    """
    from xloil import command_line  # private helper, pinned with xloil==0.22.*

    source = paths.ini if paths.ini.exists() else addin_dir() / paths.ini.name
    text, ok = command_line._write_python_path_to_ini(  # noqa: SLF001
        source.read_text(encoding="utf-8"), addin_dir(), comment_reg_keys=True
    )
    if not ok:
        raise SystemExit(f"FAIL  could not set the Python paths in {source}")
    return text


# -- trusted location ---------------------------------------------------------------------------


def _trusted_key() -> str | None:
    """The registry path of our entry in Excel's trusted locations.

    Returns
    -------
    str or None
        ``Software\\Microsoft\\Office\\<version>\\Excel\\Security\\Trusted Locations\\QuantToolkit``
        under ``HKEY_CURRENT_USER``, or ``None`` if Excel is not registered.
    """
    version = _excel_version()
    if not version:
        return None
    number = version.rsplit(".", 1)[-1]
    return rf"Software\Microsoft\Office\{number}.0\Excel\Security\Trusted Locations\{TRUSTED_LOCATION}"


def _trusted_location() -> str | None:
    """The folder our trusted-location entry names.

    Returns
    -------
    str or None
        The ``Path`` value of the entry, or ``None`` if we have not added one.
    """
    key = _trusted_key()
    if key is None:
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
            return winreg.QueryValueEx(handle, "Path")[0]
    except OSError:
        return None


def _set_trusted(folder: Path | None, *, dry_run: bool) -> None:
    """Add, update or remove our trusted-location entry.

    Parameters
    ----------
    folder : Path or None
        The folder to trust, or ``None`` to remove the entry.
    dry_run : bool
        Report the change without making it.

    Notes
    -----
    Excel loads add-ins from a trusted location without its security prompt. The entry is per
    user and covers that one folder, not its subfolders. Group policy can disable user trusted
    locations, in which case the entry has no effect and the prompt still shows.
    """
    key = _trusted_key()
    current = _trusted_location()
    wanted = None if folder is None else str(folder) + os.sep
    if key is None or current == wanted:
        return
    import winreg

    action = f"trust {wanted}" if wanted else f"stop trusting {current}"
    if dry_run:
        print(f"dry-run: would {action} (HKCU\\{key})")
        return
    if wanted is None:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
    else:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as handle:
            winreg.SetValueEx(handle, "Path", 0, winreg.REG_SZ, wanted)
            winreg.SetValueEx(handle, "Description", 0, winreg.REG_SZ, "quant-toolkit Excel add-in (xlOil)")
    print(f"ok    {action}")


def _xloil(*args: str) -> None:
    """Run xlOil's own command line with this interpreter.

    Parameters
    ----------
    *args : str
        Arguments for ``python -m xloil.command_line``, e.g. ``"install"`` or ``"remove"``.

    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with an error.
    """
    cmd = [sys.executable, "-m", "xloil.command_line", *args]
    print("run  ", " ".join(cmd))
    subprocess.run(cmd, check=True)


# -- commands -----------------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """``check``: report prerequisites and current state, changing nothing.

    Parameters
    ----------
    args : argparse.Namespace
        Uses ``create_xlstart``.

    Returns
    -------
    int
        Exit code: 0 if every prerequisite passed, 1 otherwise.
    """
    return 0 if run_checks(Paths.discover(), startup=False, create_xlstart=args.create_xlstart) else 1


def cmd_install(args: argparse.Namespace) -> int:
    """``install``: check, point the ini at this env and tag our modules into it.

    Parameters
    ----------
    args : argparse.Namespace
        Uses ``modules`` (default: ``quant_toolkit.xl.MODULES``), ``startup`` (also copy the
        add-in into ``XLSTART``), ``trust`` (add the add-in folder to Excel's trusted
        locations), ``create_xlstart``, ``force`` (tag modules even if they fail to import
        here) and ``dry_run``.

    Returns
    -------
    int
        Exit code: 0 on success, 1 if a prerequisite failed, a module did not import, or a
        startup copy could not be removed because Excel has it open.

    Notes
    -----
    Safe to re-run: the ini is rewritten only if something changed, and modules a previous run
    tagged but this one does not name are removed. Without ``startup`` a copy of the add-in left
    in ``XLSTART`` by an earlier install is removed, which is how a persistent install is
    converted to on demand.
    """
    paths = Paths.discover()
    if not run_checks(paths, startup=args.startup, create_xlstart=args.create_xlstart):
        print("\nprerequisites failed; nothing installed")
        return 1
    modules = args.modules or DEFAULT_MODULES
    if _check_importable(modules) and not args.force:
        print("\nfix the imports above (or pass --force); nothing installed")
        return 1

    print()
    if args.startup:
        if args.dry_run:
            print(f"dry-run: would run xloil install, copying {ADDIN_NAME} to {paths.xlstart}")
        else:
            _xloil("install")
    elif paths.xll.exists():
        if args.dry_run:
            print(f"dry-run: would run xloil remove, deleting {paths.xll}")
        else:
            try:
                _xloil("remove")
            except subprocess.CalledProcessError:
                print(f"FAIL  could not remove {paths.xll}: close Excel and run install again")
                return 1
            print(f"ok    removed {paths.xll}: Excel no longer loads the add-in at startup")

    text = _configured_ini(paths)
    _write_ini(paths, ini.tag(text, modules), dry_run=args.dry_run)
    if args.trust:
        _set_trusted(addin_dir(), dry_run=args.dry_run)

    if args.startup:
        print("\ndone: restart Excel; =QuantToolkit.Utilities.Version() in a cell confirms the functions loaded")
    else:
        print("\ndone: Excel starts without the add-in. Load it for a session with")
        print("      quant_toolkit_xl [workbook.xlsx ...]")
        print(f"      or, from Excel, File > Open {addin_dir() / ADDIN_NAME} before opening the workbook")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    """``uninstall``: remove our modules from the ini and our trusted location, then ``xloil remove``.

    Parameters
    ----------
    args : argparse.Namespace
        Uses ``dry_run``.

    Returns
    -------
    int
        Exit code, always 0. ``LoadModules`` entries the user added by hand are left alone,
        and xlOil leaves the ini file itself in place.
    """
    paths = Paths.discover()
    if paths.ini.exists():
        _write_ini(paths, ini.untag(paths.ini.read_text(encoding="utf-8")), dry_run=args.dry_run)
    _set_trusted(None, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run: would run xloil remove, deleting {paths.xll}")
    else:
        _xloil("remove")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the chosen command.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` if omitted.

    Returns
    -------
    int
        The command's exit code.
    """
    parser = argparse.ArgumentParser(
        prog="quant_toolkit_xl_installer", description="Point xlOil at this env and quant-toolkit's Excel functions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="report on prerequisites and current install state")
    p.add_argument("--create-xlstart", action="store_true", help="create XLSTART if it is missing")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("install", help="point xlOil.ini at this env and tag our modules into it")
    p.add_argument("--startup", action="store_true", help="also copy the add-in into XLSTART (loads with every Excel)")
    p.add_argument("--trust", action="store_true", help="add the add-in folder to Excel's trusted locations")
    p.add_argument("--modules", nargs="+", metavar="MODULE", help=f"modules to load (default: {DEFAULT_MODULES})")
    p.add_argument("--create-xlstart", action="store_true", help="create XLSTART if it is missing")
    p.add_argument("--force", action="store_true", help="tag modules even if they fail to import here")
    p.add_argument("--dry-run", action="store_true", help="show what would change without changing it")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("uninstall", help="untag our modules, drop our trusted location + xloil remove")
    p.add_argument("--dry-run", action="store_true", help="show what would change without changing it")
    p.set_defaults(func=cmd_uninstall)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
