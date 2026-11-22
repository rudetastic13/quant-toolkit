"""``quant_toolkit_xl_installer`` — wire the xlOil add-in and our modules into Excel (prototype).

Commands
--------
check
    Report on the prerequisites: Windows, ``%APPDATA%``, Excel's per-user ``XLSTART`` folder,
    a registered Excel, xlOil importable, Excel not running, and the current install state.
install
    ``check``, then run ``xloil install`` with this interpreter (copies ``xlOil.xll`` into
    ``XLSTART`` and points ``%APPDATA%\\xlOil\\xlOil.ini`` at this Python), then tag our modules
    into ``[xlOil_Python] LoadModules`` (replacing any set a previous run tagged).
uninstall
    Remove our tagged modules from the ini, then ``xloil remove`` (drops the ``.xll`` from
    ``XLSTART``; xlOil leaves the ini in place).

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

from quant_toolkit.xl_installer import ini

DEFAULT_MODULES = ["quant_toolkit.xl.functions"]
ADDIN_NAME = "xlOil.xll"


@dataclass(frozen=True)
class Paths:
    appdata: Path
    xlstart: Path
    ini: Path

    @property
    def xll(self) -> Path:
        return self.xlstart / ADDIN_NAME

    @classmethod
    def discover(cls) -> Paths:
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise SystemExit("FAIL  %APPDATA% is not set; this tool needs a Windows user profile")
        root = Path(appdata)
        return cls(root, root / "Microsoft" / "Excel" / "XLSTART", root / "xlOil" / "xlOil.ini")


# -- checks -------------------------------------------------------------------------------------


def _excel_version() -> str | None:
    """``Excel.Application.16`` style ProgID version from the registry, or None if Excel isn't registered."""
    try:
        import winreg

        return winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, r"Excel.Application\CurVer")
    except OSError:
        return None


def _excel_running() -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq EXCEL.EXE", "/NH"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return False
    return "EXCEL.EXE" in out.upper()


def run_checks(paths: Paths, *, create_xlstart: bool = False) -> bool:
    """Print one line per prerequisite; return True when nothing blocking failed."""
    ok = True

    def report(status: str, msg: str) -> None:
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
    else:
        report("FAIL", f"XLSTART   {paths.xlstart} missing: start Excel once, or pass --create-xlstart")

    version = _excel_version()
    report("ok" if version else "FAIL", f"Excel     {version or 'not registered (desktop Excel not installed?)'}")

    try:
        xloil = importlib.import_module("xloil")
        report("ok", f"xloil     {getattr(xloil, '__version__', '?')} in {sys.prefix}")
    except ImportError:
        report("FAIL", "xloil     not importable: pip install -e apps/excel")

    report("ok", f"python    {sys.version.split()[0]} {'64' if sys.maxsize > 2**32 else '32'}-bit (must match Excel)")

    if _excel_running():
        report("warn", "Excel is running: close it before install/uninstall, changes load on next start")

    report("info", f"addin     {paths.xll} {'present' if paths.xll.exists() else 'not installed'}")
    if paths.ini.exists():
        text = paths.ini.read_text(encoding="utf-8")
        report("info", f"ini       {paths.ini}")
        report("info", f"          LoadModules={ini.load_modules(text)}  ours={ini.tagged_modules(text)}")
    else:
        report("info", f"ini       {paths.ini} not created yet")
    return ok


# -- ini editing --------------------------------------------------------------------------------


def _check_importable(modules: list[str]) -> list[str]:
    bad = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - report every failure, any kind
            print(f"FAIL  {name} does not import in this env: {type(e).__name__}: {e}")
            bad.append(name)
    return bad


def _write_ini(paths: Paths, new: str, *, dry_run: bool) -> None:
    old = paths.ini.read_text(encoding="utf-8")
    if new == old:
        print(f"ok    {paths.ini} already up to date")
        return
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), str(paths.ini), "(new)", lineterm="", n=1)
    print("\n".join(diff))
    if dry_run:
        print("dry-run: ini not written")
        return
    shutil.copy2(paths.ini, paths.ini.with_suffix(".ini.bak"))
    paths.ini.write_text(new, encoding="utf-8")
    print(f"ok    wrote {paths.ini} (backup: {paths.ini.with_suffix('.ini.bak').name})")


def _xloil(*args: str) -> None:
    cmd = [sys.executable, "-m", "xloil.command_line", *args]
    print("run  ", " ".join(cmd))
    subprocess.run(cmd, check=True)


# -- commands -----------------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    return 0 if run_checks(Paths.discover(), create_xlstart=args.create_xlstart) else 1


def cmd_install(args: argparse.Namespace) -> int:
    paths = Paths.discover()
    if not run_checks(paths, create_xlstart=args.create_xlstart):
        print("\nprerequisites failed; nothing installed")
        return 1
    modules = args.modules or DEFAULT_MODULES
    if _check_importable(modules) and not args.force:
        print("\nfix the imports above (or pass --force); nothing installed")
        return 1

    print()
    if args.dry_run:
        print(f"dry-run: would run xloil install, copying {ADDIN_NAME} to {paths.xlstart}")
    else:
        _xloil("install")
    if not paths.ini.exists():  # only possible on a dry run before the first real install
        print(f"dry-run: {paths.ini} doesn't exist yet; would tag {modules} after xloil creates it")
        return 0
    _write_ini(paths, ini.tag(paths.ini.read_text(encoding="utf-8"), modules), dry_run=args.dry_run)
    print("\ndone: restart Excel; =qVersion() in a cell confirms the functions loaded")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    paths = Paths.discover()
    if paths.ini.exists():
        _write_ini(paths, ini.untag(paths.ini.read_text(encoding="utf-8")), dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run: would run xloil remove, deleting {paths.xll}")
    else:
        _xloil("remove")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quant_toolkit_xl_installer", description="Install the xlOil add-in and quant-toolkit's Excel functions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="report on prerequisites and current install state")
    p.add_argument("--create-xlstart", action="store_true", help="create XLSTART if it is missing")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("install", help="xloil install + tag our modules into xlOil.ini")
    p.add_argument("--modules", nargs="+", metavar="MODULE", help=f"modules to load (default: {DEFAULT_MODULES})")
    p.add_argument("--create-xlstart", action="store_true", help="create XLSTART if it is missing")
    p.add_argument("--force", action="store_true", help="tag modules even if they fail to import here")
    p.add_argument("--dry-run", action="store_true", help="show what would change without changing it")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("uninstall", help="untag our modules + xloil remove")
    p.add_argument("--dry-run", action="store_true", help="show what would change without changing it")
    p.set_defaults(func=cmd_uninstall)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
