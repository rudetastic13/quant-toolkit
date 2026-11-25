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
  (``curve="USD.STDCSA"`` for ``dv01``); an unknown setting is a ``TypeError``.  ``template=`` (a
  ``RiskTemplate``, bound or not) re-expresses the risk measures as risk to the template's quotes:
  the ladders come back as ``QuoteRisk`` (unexplained pillar risk included) and ``dv01`` as the sum
  over the template's quotes on the curve.  Measures that are not risk are unchanged.
  ``gamma_1bp`` / ``gamma_ladder`` are parallel curve gamma, by differencing the exact gradient;
  ``gamma_step=`` sets the bump.  With a template they are in quote space with the template frozen
  as bound; ``rebind_template=True`` rebinds it at each bumped market.  Requests are lenient so one
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

from dataclasses import dataclass, replace
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
KNOWN_SETTINGS = frozenset({"curve", "template", "gamma_step", "rebind_template"})


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

    def bound_template(self, market: Any) -> Any:
        """``settings["template"]`` bound to ``market`` (once per call), or ``None`` without one.

        A ``RiskTemplate`` is bound here; a ``BoundRiskTemplate`` is used as is when it matches
        ``market`` and rebound from its template when the market's curves have moved.
        """
        template = self.settings.get("template")
        if template is None:
            return None
        if "bound_template" not in self.cache:
            bound = template if hasattr(template, "quote_risk") else template.bind(market)
            if not bound.matches(market):
                bound = bound.template.bind(market)
            self.cache["bound_template"] = bound
        return self.cache["bound_template"]

    def gradient(self, program: Any, market: Any) -> PillarGradient:
        """``program.pillar_gradient(market)``, run once per call and shared by every measure that reads it."""
        if "pillar_gradient" not in self.cache:
            self.cache["pillar_gradient"] = program.pillar_gradient(market)
        return self.cache["pillar_gradient"]

    def ladder(self, market: Any, pillar: Any) -> Any:
        """``pillar`` (a ``PillarRisk``) as is, or as ``QuoteRisk`` when a template is set."""
        bound = self.bound_template(market)
        return pillar if bound is None else bound.quote_risk(pillar)


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


def dv01_from(
    gradient: PillarGradient, inputs: Any, market: Any, context: MeasureContext, bp: float = 1e-4
) -> np.ndarray:
    """Parallel DV01 per position: value change for +``bp`` on one curve.

    Without a template that is +``bp`` on every zero pillar of the curve.  With
    ``settings["template"]`` it is +``bp`` on every template quote on the curve.

    Parameters
    ----------
    gradient : PillarGradient
        The program's first-order curve risk.
    inputs : KernelInputs
        The compiled legs, for each position's default curve (:func:`position_curves`).
    market : MarketContext
        What a template is bound to.
    context : MeasureContext
        ``settings["curve"]`` names the curve for every position; without it each position
        reports against its own default.
    bp : float

    Returns
    -------
    ndarray of float64, shape (n_positions,)
        The matching block of the zero ladder (or the matching quotes of the quote ladder),
        summed and scaled.  ``nan`` for a position with no unambiguous default curve, and, under
        a template, for one whose curve has no template quote.

    Raises
    ------
    KeyError
        If ``settings["curve"]`` is not a curve the program reads, or has no template quote.
    """
    n = gradient.values.shape[0]
    chosen = context.settings.get("curve")
    curves = [chosen] * n if chosen is not None else position_curves(inputs)
    bound = context.bound_template(market)
    quote_risk = None
    if bound is not None:
        from quant_toolkit.finance.pricing.risk.pillar import PillarRisk

        quote_risk = bound.quote_risk(PillarRisk.from_layout(gradient.layout, gradient.values, gradient.value)).values
        quote_curves = [instrument.curve for instrument in bound.template.instruments]
    out = np.full(n, np.nan)
    for name in {c for c in curves if c is not None}:
        rows = np.array([c == name for c in curves])
        try:
            block_name = gradient.layout.block_of(name)
            block = gradient.layout.block_slice(name)
        except KeyError:
            if chosen is not None:
                raise
            continue
        if quote_risk is None:
            out[rows] = gradient.values[rows, block].sum(axis=1) * bp
            continue
        columns = [k for k, curve in enumerate(quote_curves) if curve in (name, block_name)]
        if not columns:
            if chosen is not None:
                raise KeyError(f"the template has no quote on '{name}'")
            continue
        out[rows] = quote_risk[np.ix_(rows, columns)].sum(axis=1) * bp
    return out


def _template_columns(bound: Any, layout: Any, name: str) -> list[int]:
    """Template quotes on curve ``name`` (or on the block it reports under)."""
    block = layout.block_of(name)
    return [k for k, instrument in enumerate(bound.template.instruments) if instrument.curve in (name, block)]


def _quote_move(bound: Any, market: Any, columns: list[int], size: float) -> Any:
    """``market`` with every template quote in ``columns`` moved by ``size``."""
    move = bound.transfer[:, columns].sum(axis=1) * size
    out = market
    for curve in bound.curves:
        shift = move[bound.slice(curve)]
        if np.any(shift != 0.0):
            out = out.bump(curve, shift)
    return out


def parallel_gamma(program: Any, inputs: Any, market: Any, context: MeasureContext, bp: float = 1e-4) -> dict:
    """How each position's 01 ladder changes for a +``bp`` parallel move of its curve.

    A central difference of the exact gradient (``program.pillar_gradient``) between two bumped
    markets: two adjoint passes per distinct curve, and no second difference of price.  Computed
    once per ``price`` call and shared by ``gamma_1bp`` and ``gamma_ladder``.

    Without a template the move is +``bp`` on every zero pillar of the curve and the ladder is
    the zero ladder (``H @ 1_c``, Strata's semi-parallel gamma).

    With ``settings["template"]`` the move is +``bp`` on every template quote on the curve and
    the ladder is the quote ladder.  By default the template stays as bound at ``market`` (its
    quote-to-pillar map is frozen), which is the second-order term consistent with a linear
    quote attribution: ``A' H A``.  ``settings["rebind_template"]=True`` rebinds it at each
    bumped market, adding the curvature of the quote-to-pillar map (proportional to the
    position's delta) at the cost of two binds per curve: that is the change the quote DV01
    would actually show after the move.

    Parameters
    ----------
    program : Program
    inputs : KernelInputs
        The compiled legs, for each position's default curve (:func:`position_curves`).
    market : MarketContext
    context : MeasureContext
        ``settings["curve"]`` names the curve for every position; ``settings["gamma_step"]`` is
        the bump (default ``1e-5``, a tenth of a basis point).
    bp : float

    Returns
    -------
    dict
        ``"scalar"`` : ndarray (n_positions,), the ladder summed over the position's own curve
        (its pillars, or its template quotes); ``nan`` where the position has no unambiguous
        curve, or no template quote on it.
        ``"ladder"`` : ``PillarRisk`` or, with a template, ``QuoteRisk``.  Its arrays are already
        per ``bp`` per ``bp``: read ``values``, not ``dv01``.

    Raises
    ------
    KeyError
        If ``settings["curve"]`` is not a curve the program reads, or has no template quote.
    ValueError
        If ``gamma_step`` is not positive.
    """
    if "parallel_gamma" in context.cache:
        return context.cache["parallel_gamma"]
    from quant_toolkit.finance.pricing.risk.pillar import PillarRisk

    step = float(context.settings.get("gamma_step", 1e-5))
    if step <= 0.0:
        raise ValueError("gamma_step must be positive")
    base = context.gradient(program, market)
    bound = context.bound_template(market)
    rebind = bool(context.settings.get("rebind_template", False))
    n = base.values.shape[0]
    chosen = context.settings.get("curve")
    curves = [chosen] * n if chosen is not None else position_curves(inputs)
    scale = bp * bp / (2.0 * step)

    def pillar(gradient: PillarGradient) -> Any:
        return PillarRisk.from_layout(gradient.layout, gradient.values, gradient.value)

    scalar = np.full(n, np.nan)
    if bound is None:
        ladder = np.full(base.values.shape, np.nan)
    else:
        at_base = bound.quote_risk(pillar(base))
        fields = {
            "values": at_base.values, "unexplained": at_base.unexplained, "rebucketed": at_base.rebucketed,
            "pillar_parallel": at_base.pillar_parallel, "template_parallel": at_base.template_parallel,
        }
        fields = {key: np.full(np.shape(value), np.nan) for key, value in fields.items()}
        uncovered = {key: np.full(value.shape, np.nan) for key, value in at_base.uncovered.items()}

    for name in {c for c in curves if c is not None}:
        rows = np.flatnonzero([c == name for c in curves])
        try:
            block = base.layout.block_of(name)
            columns = _template_columns(bound, base.layout, name) if bound is not None else None
        except KeyError:
            if chosen is not None:
                raise
            continue
        if bound is None:
            up = program.pillar_gradient(market.bump(block, step)).values
            down = program.pillar_gradient(market.bump(block, -step)).values
            ladder[rows] = (up[rows] - down[rows]) * scale
            scalar[rows] = ladder[rows][:, base.layout.block_slice(block)].sum(axis=1)
            continue
        if not columns:
            if chosen is not None:
                raise KeyError(f"the template has no quote on '{name}'")
            continue
        market_up, market_down = _quote_move(bound, market, columns, step), _quote_move(bound, market, columns, -step)
        bound_up = bound.template.bind(market_up) if rebind else bound
        bound_down = bound.template.bind(market_down) if rebind else bound
        risk_up = bound_up.quote_risk(pillar(program.pillar_gradient(market_up)))
        risk_down = bound_down.quote_risk(pillar(program.pillar_gradient(market_down)))
        for key, out in fields.items():
            out[rows] = (getattr(risk_up, key)[rows] - getattr(risk_down, key)[rows]) * scale
        for key, out in uncovered.items():
            out[rows] = (risk_up.uncovered[key][rows] - risk_down.uncovered[key][rows]) * scale
        scalar[rows] = fields["values"][rows][:, columns].sum(axis=1)

    if bound is None:
        result = {"scalar": scalar, "ladder": PillarRisk.from_layout(base.layout, ladder, base.value)}
    else:
        result = {"scalar": scalar, "ladder": replace(at_base, uncovered=uncovered, **fields)}
    context.cache["parallel_gamma"] = result
    return result


def gamma_1bp_from(program: Any, inputs: Any, market: Any, context: MeasureContext) -> np.ndarray:
    """Change in ``dv01`` for a +1bp parallel move of the same curve, per position.

    After a parallel move of ``m`` bp, ``dv01`` is about ``dv01 + gamma_1bp * m`` and the value
    change is about ``dv01 * m + 0.5 * gamma_1bp * m**2``.  With a template both the ``dv01`` and
    the move are in the template's quotes.  See :func:`parallel_gamma`.
    """
    return parallel_gamma(program, inputs, market, context)["scalar"]


def gamma_ladder_from(program: Any, inputs: Any, market: Any, context: MeasureContext) -> Any:
    """Change in each pillar's (or template quote's) 01 for a +1bp parallel move of the position's curve.

    A ``PillarRisk``, or a ``QuoteRisk`` with a template.  Its arrays are already per bp per bp,
    so read ``values`` (not ``dv01``); the position's own curve sums to ``gamma_1bp``.  See
    :func:`parallel_gamma`.
    """
    return parallel_gamma(program, inputs, market, context)["ladder"]


def pillar_measure(program: Any, market: Any, context: MeasureContext) -> Any:
    """``zero_ladder`` for any program: its ``pillar_gradient`` as a labelled :class:`PillarRisk`."""
    from quant_toolkit.finance.pricing.risk.pillar import PillarRisk

    gradient = context.gradient(program, market)
    return context.ladder(market, PillarRisk.from_layout(gradient.layout, gradient.values, gradient.value))


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
