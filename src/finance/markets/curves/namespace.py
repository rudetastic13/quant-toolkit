"""CurveNamespace — registered :class:`YieldCurve` objects by curve key, with version counters.

Lives with the curves it holds (``finance.markets.curves``), so market-data containers like
``MarketContext`` depend only on ``finance.markets`` — not on ``finance.pricing``.

A curve binds under its own key by default; binding the same object under a second name
(``bind(sofr, "USD.STDCSA")``) is how a market says "discount at SOFR": one object, two keys.
Each name carries an integer version counter that increments on rebind.  Multiple namespaces
can coexist (e.g. one per scenario).  This is not a singleton.
"""

from __future__ import annotations

from finance.markets.curves.yield_curve import YieldCurve


class CurveNamespace:
    """Key-value store mapping curve keys to convention-aware yield curves."""

    def __init__(self) -> None:
        self._curves: dict[str, YieldCurve] = {}
        self._versions: dict[str, int] = {}

    @staticmethod
    def _name(curve: YieldCurve, name: str | None) -> str:
        return curve.name if name is None else str(name)

    def bind(self, curve: YieldCurve, name: str | None = None) -> None:
        """Register a yield curve under ``name`` (default: its own key).

        Raises
        ------
        KeyError
            If ``name`` is already bound; use :meth:`rebind` to overwrite.
        """
        name = self._name(curve, name)
        if name in self._curves:
            raise KeyError(f"'{name}' is already bound. Use rebind() to overwrite.")
        self._curves[name] = curve
        self._versions[name] = 0

    def rebind(self, curve: YieldCurve, name: str | None = None) -> None:
        """Overwrite the binding under ``name`` (default: the curve's key) and bump its version."""
        name = self._name(curve, name)
        self._curves[name] = curve
        self._versions[name] = self._versions.get(name, -1) + 1

    def resolve(self, name: str) -> YieldCurve:
        """Return the curve bound to name. Raises KeyError if absent."""
        try:
            return self._curves[name]
        except KeyError:
            raise KeyError(f"No curve bound under '{name}'") from None

    def names_of(self, curve: YieldCurve) -> tuple[str, ...]:
        """Every name the object ``curve`` is bound under (by identity), in binding order."""
        return tuple(name for name, bound in self._curves.items() if bound is curve)

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

    def __iter__(self):
        return iter(self._curves)

    def __len__(self) -> int:
        return len(self._curves)

    def __repr__(self) -> str:
        entries = ", ".join(f"{name}(v{self._versions[name]})" for name in self._curves)
        return f"CurveNamespace({{{entries}}})"
