"""ConventionRegistry — market conventions per index family, indices per (family, tenor)."""

from __future__ import annotations

from common.registry import Registry
from finance.conventions.keys import CurveKey
from finance.conventions.market_conventions import MarketConventions
from finance.conventions.rate_index import IndexFamily, RateIndex
from finance.dates import Term

# Internal registry keyed by (currency, family_name) tuples.
_registry: Registry[MarketConventions] = Registry("ConventionRegistry")


def split_index_name(index_name: str, tenor: Term | str | None = None) -> tuple[str, Term | str | None]:
    """``"CME-TERM-SOFR.3M"`` → ``("CME-TERM-SOFR", "3M")``; ``"SOFR"`` → ``("SOFR", tenor)``.

    A tenor embedded in the name and one passed explicitly must not both be given.
    """
    family, dot, embedded = str(index_name).strip().upper().partition(".")
    if dot:
        if tenor is not None:
            raise ValueError(f"tenor given twice: in {index_name!r} and as {tenor!r}")
        return family, embedded
    return family, tenor


class ConventionRegistry:
    """Lookup table: ``(currency, family_name) -> MarketConventions``.

    Static definitions are registered via ``@register_convention``; runtime overrides with
    :meth:`register`.  Index names may carry their tenor (``"CME-TERM-SOFR.3M"``); the
    family part is what is looked up and the tenor part selects the :class:`RateIndex`.
    """

    def __init__(self, registry: Registry[MarketConventions] | None = None) -> None:
        self._registry = registry if registry is not None else _registry

    @staticmethod
    def _key(currency: str, index_name: str) -> tuple[str, str]:
        family, _ = split_index_name(index_name)
        return (currency.upper(), family)

    def get(self, currency: str, index_name: str) -> MarketConventions:
        """The conventions of the family ``index_name`` names (a tenor suffix is ignored).

        Raises
        ------
        KeyError
            If no convention is registered for the family.
        """
        key = self._key(currency, index_name)
        result = self._registry.get(key)
        if result is None:
            raise KeyError(
                f"No convention registered for ({currency!r}, {key[1]!r}). "
                "Register one with @register_convention or ConventionRegistry.register()."
            )
        return result

    def family(self, currency: str, index_name: str) -> IndexFamily:
        return self.get(currency, index_name).family

    def index(self, currency: str, index_name: str, tenor: Term | str | None = None) -> RateIndex:
        """The :class:`RateIndex` for ``currency`` and ``"NAME"`` / ``"NAME.TENOR"`` (+ optional tenor).

        A bare family name is enough for an overnight family; a term family needs the tenor.

        Raises
        ------
        KeyError
            If the family is unknown or does not publish the tenor.
        """
        family_name, tenor = split_index_name(index_name, tenor)
        return self.family(currency, family_name).index(tenor)

    def key(self, currency: str, index_name: str, tenor: Term | str | None = None) -> CurveKey:
        """The projection curve key of :meth:`index`, e.g. ``CurveKey('USD.SOFR.ON')``."""
        return self.index(currency, index_name, tenor).key

    def validate(self, key: CurveKey | str) -> CurveKey:
        """Parse a key and, for a projection key, check its family publishes the tenor.

        A discount key needs no conventions and passes through.
        """
        parsed = CurveKey.parse(key)
        if parsed.is_projection:
            self.family(parsed.currency, parsed.name).index(parsed.tenor)
        return parsed

    def register(
        self,
        currency: str,
        index_name: str,
        convention: MarketConventions,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register or overwrite a family's conventions at runtime."""
        self._registry.register(self._key(currency, index_name), convention, overwrite=overwrite)

    def has(self, currency: str, index_name: str) -> bool:
        return self._registry.has(self._key(currency, index_name))


def register_convention(
    currency: str,
    index_name: str,
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
        key = (currency.upper(), index_name.upper())
        _registry.register(key, convention, overwrite=overwrite)
        return fn

    return decorator


# Module-level default instance backed by the shared registry.
default_registry = ConventionRegistry()
