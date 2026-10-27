"""ConventionRegistry — index families, and the product conventions registered beside them.

Two maps, one key ``(currency, family_name)``:

- **Families** (:class:`IndexFamily`): what a projection key names.  The curve layer
  (``YieldCurve``, the group, the market, ``RateGenerator``) reads only these, so a family can
  be registered, looked up and validated with no instrument layer present.
- **Conventions**: the leg and product bundle (``finance.instruments.conventions``) the
  builders read.  Registered later, beside a family that already exists.  This module never
  imports the bundle type, so ``finance.conventions`` depends on dates only.

Lookups take a :class:`CurveKey` or its text (``"USD.SOFR.1D"``, ``"USD.CME-TERM-SOFR.3M"``):
the key's currency and name select the family, its tenor the :class:`RateIndex`.
"""

from __future__ import annotations

from typing import Any

from common.registry import Registry
from finance.conventions.keys import CurveKey
from finance.conventions.rate_index import IndexFamily, RateIndex

_families: Registry[IndexFamily] = Registry("IndexFamilies")
_conventions: Registry[Any] = Registry("MarketConventions")


class ConventionRegistry:
    """Lookup table: ``(currency, family_name)`` to an :class:`IndexFamily`, and to its product conventions.

    Parameters
    ----------
    families, conventions : Registry or None
        Backing stores; default to the module-level ones the bundled definitions fill.
    """

    def __init__(self, families: Registry[IndexFamily] | None = None, conventions: Registry[Any] | None = None) -> None:
        self._families = families if families is not None else _families
        self._conventions = conventions if conventions is not None else _conventions

    @staticmethod
    def _key(currency: str, family_name: str) -> tuple[str, str]:
        return (currency.strip().upper(), family_name.strip().upper())

    # -- families (the curve layer) ---------------------------------------------------------

    def register_family(self, family: IndexFamily, *, overwrite: bool = False) -> None:
        """Register an index family under its own currency and name."""
        self._families.register(self._key(family.currency, family.name), family, overwrite=overwrite)

    def family(self, key: CurveKey | str) -> IndexFamily:
        """The family ``key`` names (its tenor, if any, is not checked).

        Raises
        ------
        KeyError
            If no family is registered for the key's currency and name.
        """
        parsed = CurveKey.parse(key)
        family = self._families.get(self._key(parsed.currency, parsed.name))
        if family is None:
            raise KeyError(
                f"No index family registered for ({parsed.currency!r}, {parsed.name!r}). "
                "Register one with ConventionRegistry.register_family()."
            )
        return family

    def index(self, key: CurveKey | str) -> RateIndex:
        """The :class:`RateIndex` a projection key names, e.g. ``"USD.SOFR.1D"``.

        Raises
        ------
        KeyError
            If the family is unknown or does not publish the key's tenor.
        ValueError
            If ``key`` is a discount key (no tenor).
        """
        parsed = CurveKey.parse(key)
        if parsed.is_discount:
            raise ValueError(f"{parsed} is a discount key; an index is named by 'CCY.INDEX.TENOR'")
        return self.family(parsed).index(parsed.tenor)

    def validate(self, key: CurveKey | str) -> CurveKey:
        """Parse a key and, for a projection key, check its family publishes the tenor.

        A discount key needs no conventions and passes through.
        """
        parsed = CurveKey.parse(key)
        if parsed.is_projection:
            self.index(parsed)
        return parsed

    def has(self, currency: str, family_name: str) -> bool:
        """Whether a family is registered under ``(currency, family_name)``."""
        return self._families.has(self._key(currency, family_name))

    # -- product conventions (the instrument layer) ---------------------------------------

    def register(self, currency: str, family_name: str, convention: Any, *, overwrite: bool = False) -> None:
        """Register the product conventions of an already-registered family.

        Raises
        ------
        KeyError
            If no family is registered under ``(currency, family_name)``.
        """
        key = self._key(currency, family_name)
        if not self._families.has(key):
            raise KeyError(f"register the index family for {key} before its conventions")
        self._conventions.register(key, convention, overwrite=overwrite)

    def get(self, key: CurveKey | str) -> Any:
        """The product conventions of the family ``key`` names (its tenor, if any, is not checked).

        Raises
        ------
        KeyError
            If no conventions are registered for the family.
        """
        parsed = CurveKey.parse(key)
        result = self._conventions.get(self._key(parsed.currency, parsed.name))
        if result is None:
            raise KeyError(
                f"No market conventions registered for ({parsed.currency!r}, {parsed.name!r}). "
                "Register them with @register_convention or ConventionRegistry.register()."
            )
        return result


def register_convention(currency: str, family_name: str, *, overwrite: bool = False):
    """Decorator registering a family's product conventions in the shared registry.

    Usage::

        @register_convention("USD", "SOFR")
        def _usd_sofr() -> MarketConventions:
            return MarketConventions(family=default_registry.family("USD.SOFR"), ...)

    The decorated callable is invoked immediately and its return value is stored; the family
    must already be registered.
    """

    def decorator(fn):
        default_registry.register(currency, family_name, fn(), overwrite=overwrite)
        return fn

    return decorator


# Module-level default instance backed by the shared registries.
default_registry = ConventionRegistry()
