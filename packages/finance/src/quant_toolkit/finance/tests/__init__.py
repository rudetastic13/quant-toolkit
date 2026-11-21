"""Test suite for ``quant_toolkit.finance`` (mirrors the source tree)."""
from pathlib import Path


def finance_data_dir() -> Path:
    """The finance portion's ``data/`` directory (test fixtures such as ``rates.csv``; repo-only, not shipped).

    Resolved from this file so no test counts parent directories.
    """
    return Path(__file__).resolve().parents[4] / "data"
