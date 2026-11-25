"""quant_toolkit.xl_installer — CLI that installs the xlOil add-in and tags our modules into xlOil.ini.

Prototype. Run ``quant_toolkit_xl_installer --help`` (or ``python -m quant_toolkit.xl_installer``).
"""
from __future__ import annotations

from quant_toolkit.xl_installer.cli import main

__all__ = ["main"]
