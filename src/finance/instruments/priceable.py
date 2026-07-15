"""Priceable — the functor mixin that makes a resolved instrument a standalone calculator.

``instrument(market)`` prices through the exact same compile→reprice machinery as the
portfolio path: the first call compiles the instrument via its registered pricer and caches
the ``PricingProgram``; every subsequent call — including against a *different*
``MarketContext`` — only reprices.  A standalone bump loop therefore costs what the
portfolio path costs per scenario (compile-once/reprice-many is preserved), and there is no
second pricing implementation to drift.

The cached program is keyed to the instrument's state at first call: these are pure-data
dataclasses, so mutate-and-reprice is not supported — build a new instrument instead (the
same contract the portfolio path already implies).

Pricers self-register in ``pricer_registry`` keyed by the resolved type's name (e.g.
``SwapPricer`` registers under ``"Swap"``).  Dispatch imports ``finance.pricing``
lazily inside ``__call__`` so ``instruments`` carries no import-time dependency on the
pricing layer (pricing already imports instruments).
"""
from __future__ import annotations

from typing import Literal, Sequence, get_args

from common.registry import Registry

PricingRequest = Literal["pv", "leg_pvs", "cashflows"]
_VALID_REQUESTS = frozenset(get_args(PricingRequest))

# resolved-instrument type name -> pricer class (SwapPricer registers "Swap", ...)
pricer_registry: Registry = Registry(name="Pricers")


class Priceable:
    """Mixin: ``instrument(market, requests=None) -> PricingResult``.

    ``requests`` is an optional list of :data:`PricingRequest` literals.  ``None`` means
    the default set (``pv`` + ``leg_pvs`` + ``cashflows``).  ``pv``/``leg_pvs`` are always
    populated (the kernel produces them in one pass); ``cashflows`` is skipped unless
    requested, saving the per-flow report assembly in tight loops.
    """

    def __call__(self, market, requests: Sequence[PricingRequest] | None = None):
        if requests is not None:
            unknown = set(requests) - _VALID_REQUESTS
            if unknown:
                valid = ", ".join(sorted(_VALID_REQUESTS))
                raise ValueError(f"unknown pricing request(s) {sorted(unknown)}; valid: {valid}")
        program = self.__dict__.get("_program")
        if program is None:
            program = self._compile()
            self.__dict__["_program"] = program
        with_cashflows = requests is None or "cashflows" in requests
        return program.price(market, with_cashflows=with_cashflows)

    def _compile(self):
        from finance.pricing import pricers  # noqa: F401 — registers the bundled pricers

        pricer_cls = pricer_registry.get(type(self).__name__)
        if pricer_cls is None:
            raise KeyError(
                f"no pricer registered for {type(self).__name__}; register one in "
                "finance.instruments.priceable.pricer_registry"
            )
        return pricer_cls().compile([self])


__all__ = ["Priceable", "PricingRequest", "pricer_registry"]
