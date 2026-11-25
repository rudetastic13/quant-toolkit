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

Pricers self-register in ``pricer_registry`` keyed by the instrument class they price
(``@register_with(pricer_registry, Swap)`` on ``LegPricer``).  :func:`compile_instruments` is
the one lookup everything uses, so no caller names a concrete pricer.  It imports
``quant_toolkit.finance.pricing`` lazily so ``instruments`` carries no import-time dependency
on the pricing layer (pricing already imports instruments).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from quant_toolkit.finance.pricing.pricers.protocol import Pricer, Program

from quant_toolkit.common.object.serializer import Serializable
from quant_toolkit.common.registry import Registry

# A measure name.  Each program publishes the ones it answers as ``program.measures`` (leg products:
# "pv", "leg_pvs", "cashflows") and rejects the rest.
PricingRequest = str

# instrument class -> pricer class (LegPricer registers Swap, Deposit and Fra, ...)
pricer_registry: Registry[type[Pricer]] = Registry(name="Pricers")


def pricer_class(instrument_type: type) -> type[Pricer]:
    """The pricer class registered for ``instrument_type`` or its nearest registered base.

    Raises
    ------
    KeyError
        If no class in the type's MRO has a pricer.
    """
    from quant_toolkit.finance.pricing import pricers  # noqa: F401 — registers the bundled pricers

    for cls in instrument_type.__mro__:
        found = pricer_registry.get(cls)
        if found is not None:
            return found
    raise KeyError(
        f"no pricer registered for {instrument_type.__name__}; register one in "
        "quant_toolkit.finance.instruments.priceable.pricer_registry"
    )


def compile_instruments(instruments: Sequence[Any], *, backend=None) -> Program:
    """Compile instruments that share a pricer into one program.

    Parameters
    ----------
    instruments : sequence
        Non-empty; every element must resolve to the same pricer (swaps, deposits and FRAs
        share one; futures and swaptions each have their own).
    backend : Backend, optional
        ``None`` is the pricer's default.

    Returns
    -------
    Program

    Raises
    ------
    ValueError
        If ``instruments`` is empty or mixes pricers.
    KeyError
        If an instrument type has no registered pricer.
    """
    if not instruments:
        raise ValueError("compile_instruments needs at least one instrument")
    classes = {pricer_class(type(instrument)) for instrument in instruments}
    if len(classes) > 1:
        names = ", ".join(sorted(cls.__name__ for cls in classes))
        raise ValueError(f"instruments resolve to several pricers ({names}); compile each product separately")
    return classes.pop()().compile(list(instruments), backend=backend)


class Priceable(Serializable):
    """Mixin: ``instrument(market, requests=None) -> PricingResult``; JSON round-trip via ``Serializable``.

    The compiled program is cached on the instance outside its dataclass fields, so it never
    reaches a document.  ``instrument.requests()`` lists the measure names it answers.

    ``requests`` is an optional list of measure names, forwarded to the compiled program's
    ``price``.  ``None`` means every measure the program answers.  Requests are lenient, so one
    list serves a mixed book: a measure this instrument does not answer reads ``None`` from
    ``result.get(name)``, and only a name no product answers raises ``ValueError``.  For leg products the measures are ``pv`` + ``leg_pvs`` +
    ``cashflows``: the first two are always populated (one kernel pass), ``cashflows`` is
    skipped unless requested, saving the per-flow report assembly in tight loops.

    Keyword ``settings`` steer measures that need a choice: ``swap(market, ["dv01"],
    curve="USD.STDCSA")`` reports DV01 against that curve instead of the swap's own projection
    curve.  An unrecognised setting is a ``TypeError``.
    """

    def __call__(self, market, requests: Sequence[PricingRequest] | None = None, **settings):
        return self._compiled_program().price(market, requests, **settings)

    def requests(self) -> tuple[str, ...]:
        """Every measure this instrument answers, sorted: what ``instrument(market, requests)`` accepts
        without reading ``None`` back."""
        return tuple(sorted(self._compiled_program().measures))

    def _compiled_program(self):
        """The compiled program, built on first use and cached off the dataclass fields."""
        program = self.__dict__.get("_program")
        if program is None:
            program = compile_instruments([self])
            self.__dict__["_program"] = program
        return program
