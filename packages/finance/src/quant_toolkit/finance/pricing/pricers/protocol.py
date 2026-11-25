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
- ``measures``: the request names the program answers (``"pv"``, ``"cashflows"``, ``"greeks"`` ...).
- ``n_positions``: how many instruments it holds; every per-position array has this leading length.
- ``price(market, requests=None) -> result``: the product's result type (``PricingResult`` for leg
  products, ``FuturesPricingResult``, ``SwaptionPricingResult``).  Results differ per product on
  purpose, but each exposes ``value``, the one number per position that risk differentiates.
  ``requests`` names the measures wanted; ``None`` is the base set (the result's own fields).
  On-demand measures (risk ladders, registered in ``measure_registry``) cost an adjoint pass and are
  computed only when named.  Keyword ``settings`` steer the measures that need a choice
  (``curve="USD.STDCSA"`` for ``dv01``); an unknown setting is a ``TypeError``.  Requests are lenient so one
  list can be sent to a mixed book: a measure this product does not answer is skipped and reads back
  as ``None`` from ``result.get(name)``.  Only a name no product answers raises (a typo).
- ``pillar_gradient(market) -> PillarGradient``: ``d value / d z`` over every curve parameter, from
  the analytic adjoint.  This is what ``pillar_risk`` and ``RiskEngine`` consume, so neither names a
  concrete program.

``reprice`` is not part of this protocol: it is the engine-level primitive (``PricingProgram.reprice``
and the engine programs return the raw ``KernelResult``).

Callers outside this package do not name a concrete pricer: ``pricer_registry`` maps an instrument
class to its pricer class, and ``compile_instruments`` (in ``instruments.priceable``) is the lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

from quant_toolkit.common.registry import Registry
from quant_toolkit.finance.pricing.types import Backend

InstrumentT_contra = TypeVar("InstrumentT_contra", contravariant=True)
ProgramT_co = TypeVar("ProgramT_co", covariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)


@dataclass(frozen=True)
class PillarGradient:
    """A program's first-order curve risk at one market.

    Parameters
    ----------
    layout : ParamLayout
        The parameter blocks ``values`` is laid out over.
    values : ndarray of float64, shape (n_positions, n_params)
        ``d value / d z`` per unit of zero rate.
    value : ndarray of float64, shape (n_positions,)
        The differentiated value at the same market.
    """

    layout: Any
    values: np.ndarray
    value: np.ndarray


_known_measures: set[str] = set()


def declare_measures(names: Iterable[str]) -> frozenset[str]:
    """Publish a program's measure names into the shared vocabulary and return them frozen."""
    measures = frozenset(names)
    _known_measures.update(measures)
    return measures


def known_measures() -> frozenset[str]:
    """Every measure some program answers."""
    return frozenset(_known_measures)


# Settings a ``price`` call may pass through to its measures.
KNOWN_SETTINGS = frozenset({"curve"})


@dataclass
class MeasureContext:
    """What one ``price`` call hands each on-demand measure.

    Parameters
    ----------
    cache : dict
        Private to the call, so measures sharing an adjoint pass run it once.
    settings : dict
        The caller's keyword settings (``curve`` ...); a measure reads the ones it understands.
    """

    cache: dict
    settings: dict


# (program class, measure name) -> fn(program, market, context) for measures computed only on request.
measure_registry: Registry[Callable[[Any, Any, MeasureContext], Any]] = Registry("PricingMeasures")


def measure(program_cls: type, name: str) -> Callable:
    """Register an on-demand measure for ``program_cls`` and add ``name`` to its ``measures``."""

    def decorator(fn: Callable[[Any, Any, MeasureContext], Any]) -> Callable[[Any, Any, MeasureContext], Any]:
        measure_registry.register((program_cls, name), fn)
        program_cls.measures = program_cls.measures | declare_measures([name])
        return fn

    return decorator


def on_demand(
    program: Any, market: Any, asked: Iterable[str], cache: dict | None = None, settings: dict | None = None
) -> dict[str, Any]:
    """Evaluate the registered on-demand measures among ``asked``, keyed by request name.

    Parameters
    ----------
    program, market
    asked : iterable of str
    cache : dict, optional
        Seeds the per-call cache with what the pricing pass already produced.
    settings : dict, optional
        The caller's keyword settings.

    Raises
    ------
    TypeError
        If a setting is not one of ``KNOWN_SETTINGS``.
    """
    settings = dict(settings or {})
    unknown = settings.keys() - KNOWN_SETTINGS
    if unknown:
        raise TypeError(f"unknown pricing setting(s) {sorted(unknown)}; known: {', '.join(sorted(KNOWN_SETTINGS))}")
    context = MeasureContext({} if cache is None else cache, settings)
    out: dict[str, Any] = {}
    for name in sorted(asked):
        fn = measure_registry.get((type(program), name))
        if fn is not None:
            out[name] = fn(program, market, context)
    return out


def position_curves(inputs: Any) -> list[str | None]:
    """The curve each instrument's ``dv01`` reports against by default.

    Parameters
    ----------
    inputs : KernelInputs

    Returns
    -------
    list of (str or None), length n_instruments
        The instrument's projection curve; its discount curve when it projects off nothing (a
        fixed-only deposit); ``None`` when its legs project off several curves (a basis swap),
        where only an explicit ``curve=`` is unambiguous.
    """
    leg_of_flow = np.repeat(np.arange(inputs.n_legs), np.diff(np.append(inputs.leg_offsets, inputs.n_flows)))
    instrument = inputs.leg_instrument[leg_of_flow]
    out: list[str | None] = []
    for i in range(inputs.n_instruments):
        mine = instrument == i
        projection = np.unique(inputs.proj_curve[mine])
        projection = projection[projection >= 0]
        ids = projection if projection.size else np.unique(inputs.discount_curve[mine])
        out.append(inputs.curve_names[int(ids[0])] if ids.size == 1 else None)
    return out


def dv01_from(gradient: PillarGradient, inputs: Any, context: MeasureContext, bp: float = 1e-4) -> np.ndarray:
    """Parallel DV01 per position: value change for +``bp`` on every zero pillar of one curve.

    Parameters
    ----------
    gradient : PillarGradient
        The program's first-order curve risk.
    inputs : KernelInputs
        The compiled legs, for each position's default curve (:func:`position_curves`).
    context : MeasureContext
        ``settings["curve"]`` names the curve for every position; without it each position
        reports against its own default.
    bp : float

    Returns
    -------
    ndarray of float64, shape (n_positions,)
        The matching block of the zero ladder, summed and scaled.  ``nan`` for a position with
        no unambiguous default curve.

    Raises
    ------
    KeyError
        If ``settings["curve"]`` is not a curve the program reads.
    """
    n = gradient.values.shape[0]
    chosen = context.settings.get("curve")
    curves = [chosen] * n if chosen is not None else position_curves(inputs)
    out = np.full(n, np.nan)
    for name in {c for c in curves if c is not None}:
        rows = np.array([c == name for c in curves])
        try:
            block = gradient.layout.block_slice(name)
        except KeyError:
            if chosen is not None:
                raise
            continue
        out[rows] = gradient.values[rows, block].sum(axis=1) * bp
    return out


def pillar_measure(program: Any, market: Any, context: MeasureContext) -> Any:
    """``zero_ladder`` for any program: its ``pillar_gradient`` as a labelled :class:`PillarRisk`."""
    from quant_toolkit.finance.pricing.risk.pillar import PillarRisk

    gradient = program.pillar_gradient(market)
    return PillarRisk.from_layout(gradient.layout, gradient.values, gradient.value)


def resolve_requests(
    requests: Iterable[str] | None, measures: frozenset[str], default: Iterable[str]
) -> frozenset[str]:
    """The requested measures this program answers.

    Parameters
    ----------
    requests : iterable of str or None
        ``None`` means ``default``.
    measures : frozenset of str
        What the program answers.
    default : iterable of str
        The base set: what comes out of the pricing pass itself.

    Returns
    -------
    frozenset of str
        ``requests`` restricted to ``measures``; the rest read back as ``None`` from the result.

    Raises
    ------
    ValueError
        If a request is a name no program answers.
    """
    if requests is None:
        return frozenset(default)
    asked = frozenset((requests,) if isinstance(requests, str) else requests)
    unknown = asked - _known_measures
    if unknown:
        raise ValueError(
            f"unknown pricing request(s) {sorted(unknown)}; known: {', '.join(sorted(_known_measures))}"
        )
    return asked & measures


@runtime_checkable
class Program(Protocol):
    """A compiled book: price it against any market without rebuilding."""

    measures: frozenset[str]

    @property
    def n_positions(self) -> int: ...

    def price(self, market: Any, requests: Sequence[str] | None = None, **settings: Any) -> Any: ...

    def pillar_gradient(self, market: Any) -> PillarGradient: ...


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
