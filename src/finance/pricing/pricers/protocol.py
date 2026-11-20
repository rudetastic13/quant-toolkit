"""Pricer and Program protocols — the interface every product pricer satisfies.

A *pricer* is the impure prep step: it takes instruments, builds schedules, resolves curve names and
lowers the lot into a *program*.  A *program* is the compiled, repriceable artifact: ``reprice(market)``
is the cheap repeatable step and must not rebuild anything.  Both are structural (``Protocol``): the
three pricers (``LegPricer``, ``FuturesPricer``, ``SwaptionPricer``) satisfy ``Pricer`` without
inheriting from it, and a Treasury future or option pricer only has to match the shape.

What a pricer pins down
-----------------------
- ``default_backend``: the engine it binds (see ``docs/engine_selection.md``); every ``compile`` /
  ``price`` takes ``backend=None`` meaning that default.
- ``compile(instruments, *, backend=None) -> Program``: compile once.
- ``price(instruments, market, *, backend=None) -> result``: compile and reprice in one call.

What a program pins down
------------------------
- ``reprice(market) -> result``: the product's result type (``PricingResult`` pieces for leg products,
  ``FuturesPricingResult``, ``SwaptionPricingResult``).  Results differ per product on purpose; callers
  that need the fields know the product.

TODO(pricer-protocol): code outside this package still names concrete pricers (``LegPricer()`` in
tests, research scripts, ``quant_toolkit_xl.api``, ``Priceable._compile`` via ``pricer_registry``, the
calibration helpers).  Replace those with ``Pricer`` / ``Program`` annotations and resolve the concrete
class through ``pricer_registry`` throughout the project.  Also make ``pricer_registry`` a
``Registry[type[Pricer]]``, and reconcile ``PricingProgram.reprice`` (returns the raw ``KernelResult``)
with the other programs, whose ``reprice`` returns the product result, so ``Program.reprice`` can be
typed more tightly than ``Any``.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

from finance.pricing.types import Backend

InstrumentT_contra = TypeVar("InstrumentT_contra", contravariant=True)
ProgramT_co = TypeVar("ProgramT_co", covariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)


@runtime_checkable
class Program(Protocol):
    """A compiled book: reprice it against any market without rebuilding."""

    def reprice(self, market: Any) -> Any: ...


@runtime_checkable
class Pricer(Protocol[InstrumentT_contra, ProgramT_co, ResultT_co]):
    """Compiles instruments into a :class:`Program` and prices them against a market."""

    default_backend: Backend

    def compile(
        self, instruments: Sequence[InstrumentT_contra], *, backend: Backend | None = None
    ) -> ProgramT_co: ...

    def price(
        self, instruments: Sequence[InstrumentT_contra], market: Any, *, backend: Backend | None = None
    ) -> ResultT_co: ...
