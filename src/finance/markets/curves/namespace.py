
"""CurveNamespace — registered :class:`YieldCurve` objects with version counters.

Lives with the curves it holds (``finance.markets.curves``), so market-data containers like
``MarketContext`` depend only on ``finance.markets`` — not on ``finance.pricing``.

Each curve carries an integer version counter that increments on rebind. Instruments cache
results and compare versions to detect staleness — no observer pattern needed. Multiple
namespaces can coexist (e.g. one per scenario). This is not a singleton.
"""
from __future__ import annotations

from finance.markets.curves.yield_curve import YieldCurve


class CurveNamespace:
    """Key-value store mapping canonical names to convention-aware yield curves."""

    def __init__(self) -> None:
        self._curves: dict[str, YieldCurve] = {}
        self._versions: dict[str, int] = {}

    def bind(self, curve: YieldCurve) -> None:
        """Register a yield curve under its canonical name."""
        name = curve.name
        if name in self._curves:
            raise KeyError(f"'{name}' is already bound. Use rebind() to overwrite.")
        self._curves[name] = curve
        self._versions[name] = 0

    def rebind(self, curve: YieldCurve) -> None:
        """Overwrite the yield curve's binding and increment its version counter."""
        name = curve.name
        self._curves[name] = curve
        self._versions[name] = self._versions.get(name, -1) + 1

    def resolve(self, name: str) -> YieldCurve:
        """Return the curve bound to name. Raises KeyError if absent."""
        try:
            return self._curves[name]
        except KeyError:
            raise KeyError(f"No curve bound under '{name}'") from None

    def version(self, name: str) -> int:
        """Return the current version counter for name. Raises KeyError if absent."""
        try:
            return self._versions[name]
        except KeyError:
            raise KeyError(f"No curve bound under '{name}'") from None

    def snapshot(self) -> dict[str, tuple[YieldCurve, int]]:
        """Return a shallow copy of {name: (curve, version)} for caching / scenario use."""
        return {name: (curve, self._versions[name]) for name, curve in self._curves.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._curves

    def __repr__(self) -> str:
        entries = ", ".join(f"{name}(v{self._versions[name]})" for name in self._curves)
        return f"CurveNamespace({{{entries}}})"


__all__ = ["CurveNamespace"]
