"""ConventionRegistry — lookup market conventions by (currency, index_name)."""
from __future__ import annotations

from common.registry import Registry
from finance.conventions.market_conventions import MarketConventions

# Internal registry keyed by (currency, index_name) tuples.
_registry: Registry[MarketConventions] = Registry("ConventionRegistry")


class ConventionRegistry:
    """
    Lookup table: (currency: str, index_name: str) -> MarketConventions.

    Static definitions are registered via @register_convention. Runtime
    overrides can be applied with register() for non-standard configurations.
    """

    def __init__(self, registry: Registry[MarketConventions] | None = None) -> None:
        self._registry = registry if registry is not None else _registry

    def get(self, currency: str, index_name: str) -> MarketConventions:
        """
        Return the MarketConventions for (currency, index_name).
        Raises KeyError if no convention is registered.
        """
        key = (currency.upper(), index_name.upper())
        result = self._registry.get(key)
        if result is None:
            raise KeyError(
                f"No convention registered for ({currency!r}, {index_name!r}). "
                "Register one with @register_convention or ConventionRegistry.register()."
            )
        return result

    def register(
        self,
        currency: str,
        index_name: str,
        convention: MarketConventions,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register or overwrite a convention at runtime."""
        key = (currency.upper(), index_name.upper())
        self._registry.register(key, convention, overwrite=overwrite)

    def has(self, currency: str, index_name: str) -> bool:
        return self._registry.has((currency.upper(), index_name.upper()))


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
