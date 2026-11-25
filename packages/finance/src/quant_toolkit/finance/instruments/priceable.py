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
``SwapPricer`` registers under ``"Swap"``).  Dispatch imports ``quant_toolkit.finance.pricing``
lazily inside ``__call__`` so ``instruments`` carries no import-time dependency on the
pricing layer (pricing already imports instruments).
"""
from __future__ import annotations

from typing import Sequence

from quant_toolkit.common.object.serializer import Serializable
from quant_toolkit.common.registry import Registry

# A measure name.  Each program publishes the ones it answers as ``program.measures`` (leg products:
# "pv", "leg_pvs", "cashflows") and rejects the rest.
PricingRequest = str

# resolved-instrument type name -> pricer class (SwapPricer registers "Swap", ...)
pricer_registry: Registry = Registry(name="Pricers")


class Priceable(Serializable):
    """Mixin: ``instrument(market, requests=None) -> PricingResult``; JSON round-trip via ``Serializable``.

    The compiled program is cached on the instance outside its dataclass fields, so it never
    reaches a document.

    ``requests`` is an optional list of measure names, forwarded to the compiled program's
    ``price``.  ``None`` means every measure the program answers.  Requests are lenient, so one
    list serves a mixed book: a measure this instrument does not answer reads ``None`` from
    ``result.get(name)``, and only a name no product answers raises ``ValueError``.  For leg products the measures are ``pv`` + ``leg_pvs`` +
    ``cashflows``: the first two are always populated (one kernel pass), ``cashflows`` is
    skipped unless requested, saving the per-flow report assembly in tight loops.
    """

    def __call__(self, market, requests: Sequence[PricingRequest] | None = None):
        program = self.__dict__.get("_program")
        if program is None:
            program = self._compile()
            self.__dict__["_program"] = program
        return program.price(market, requests)

    def _compile(self):
        from quant_toolkit.finance.pricing import pricers  # noqa: F401 — registers the bundled pricers

        # TODO(pricer-protocol): see quant_toolkit.finance.pricing.pricers.protocol — type the registry as type[Pricer].
        pricer_cls = pricer_registry.get(type(self).__name__)
        if pricer_cls is None:
            raise KeyError(
                f"no pricer registered for {type(self).__name__}; register one in "
                "quant_toolkit.finance.instruments.priceable.pricer_registry"
            )
        return pricer_cls().compile([self])
