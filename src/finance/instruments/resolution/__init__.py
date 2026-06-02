"""Trader-facing instrument resolution: minimal input + convention lookup -> contract."""
from finance.instruments.resolution.resolver import curve_name, resolve_conventions, roll_spot
from finance.instruments.resolution.builders import ResolvedSwap, Swap

__all__ = ["curve_name", "resolve_conventions", "roll_spot", "ResolvedSwap", "Swap"]
