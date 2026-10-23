"""ConventionRegistry — market conventions per index family, looked up by curve key."""

from __future__ import annotations

from common.registry import Registry
from finance.conventions.keys import CurveKey
from finance.conventions.market_conventions import MarketConventions
from finance.conventions.rate_index import IndexFamily, RateIndex

# Internal registry keyed by (currency, family_name) tuples.
_registry: Registry[MarketConventions] = Registry("ConventionRegistry")


class ConventionRegistry:
    """Lookup table: ``(currency, family_name) -> MarketConventions``.

    Static definitions are registered via ``@register_convention``; runtime overrides with
    :meth:`register`.  Lookups take a :class:`CurveKey` or its text (``"USD.SOFR.1D"``,
    ``"USD.CME-TERM-SOFR.3M"``): the key's currency and name select the family, its tenor the
    :class:`RateIndex`.
    """

    def __init__(self, registry: Registry[MarketConventions] | None = None) -> None:
        self._registry = registry if registry is not None else _registry

    @staticmethod
    def _key(currency: str, family_name: str) -> tuple[str, str]:
        return (currency.strip().upper(), family_name.strip().upper())

    def get(self, key: CurveKey | str) -> MarketConventions:
        """The conventions of the family ``key`` names (its tenor, if any, is not checked).

        Raises
        ------
        KeyError
            If no convention is registered for the family.
        """
        parsed = CurveKey.parse(key)
        result = self._registry.get(self._key(parsed.currency, parsed.name))
        if result is None:
            raise KeyError(
                f"No convention registered for ({parsed.currency!r}, {parsed.name!r}). "
                "Register one with @register_convention or ConventionRegistry.register()."
            )
        return result

    def family(self, key: CurveKey | str) -> IndexFamily:
        return self.get(key).family

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

    def register(
        self,
        currency: str,
        family_name: str,
        convention: MarketConventions,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register or overwrite a family's conventions at runtime."""
        self._registry.register(self._key(currency, family_name), convention, overwrite=overwrite)

    def has(self, currency: str, family_name: str) -> bool:
        return self._registry.has(self._key(currency, family_name))


def register_convention(
    currency: str,
    family_name: str,
    *,
    overwrite: bool = False,
):
    """
    Decorator to statically register a MarketConventions in the module-level registry.

    Usage::

        @register_convention("USD", "SOFR")
        def _usd_sofr() -> MarketConventions:
            return MarketConventions(...)

    The decorated callable is invoked immediately and its return value is stored.
    """

    def decorator(fn):
        convention = fn()
        _registry.register(ConventionRegistry._key(currency, family_name), convention, overwrite=overwrite)
        return fn

    return decorator


# Module-level default instance backed by the shared registry.
default_registry = ConventionRegistry()
