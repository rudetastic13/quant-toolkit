"""RiskTemplate — a book's pillar risk re-expressed as risk to the quotes of real instruments.

A template is a *view*: a list of calibration instruments (par swaps, deposits, FRAs, SOFR
futures) on one or more curves, with a rule for how the base curve moves between template nodes
when one template quote moves.  Pricing, fixings and calibration never leave the base curve; a
template changes no PV and triggers no recalibration.  One production curve, many risk views.

The maths (``docs/risk_and_engine_design.md`` §2.2).  With ``G`` the book's pillar risk
``(N, P)``, ``J_T`` the template Jacobian ``(K, P)`` and ``B`` the block-diagonal basis
``(P, K)``: ``M = J_T B``, ``A = B M⁻¹`` (so ``J_T A = I``), quote risk ``R = G A``, unexplained
pillar risk ``U = G − R J_T``, hedge ``h = −R / s`` with ``s_k`` the quote sensitivity of
template instrument ``k`` to its own quote.  A template bump always reprices template
instrument ``k`` to ``q_k + δ`` and the others to their quotes; the basis only decides how pillar
risk between nodes is attributed.  ``U`` is literally the risk left after executing the template
hedge, and is always reported.

Binding is to a market by object identity (``BoundRiskTemplate.matches``): a scenario or a
recalibration rebinds, a reprice on the same market does not.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from finance.conventions import CurveKey
from finance.dates import Date
from finance.pricing.calibration.instruments import (
    DepositHelper,
    FraHelper,
    FuturesHelper,
    QuoteKind,
    SwapHelper,
    deposit_helper,
    fra_helper,
    futures_helper,
    swap_helper,
)
from finance.pricing.engines.layout import ParamLayout
from finance.pricing.risk.basis import BasisRule, LeastSquaresBasis, SchemeBasis, TentBasis
from finance.pricing.risk.jacobian import template_jacobian
from finance.pricing.risk.pillar import PillarRisk, pillar_risk
from finance.pricing.types import Backend

FloatArray = np.ndarray
DateArray = np.ndarray


class TemplateError(ValueError):
    """A template that cannot be bound safely; ``code`` says why and ``node`` names the offender."""

    def __init__(self, code: str, message: str, node: str | None = None) -> None:
        super().__init__(f"{code}: {message}" if node is None else f"{code} at {node}: {message}")
        self.code = code
        self.node = node


@dataclass(frozen=True)
class TemplateLimits:
    """Production safety limits ``bind`` enforces; raising one is a deliberate, explicit act."""

    max_condition: float = 1e8
    warn_condition: float = 1e4
    max_leverage: float = 25.0
    warn_leverage: float = 5.0
    max_abs_residual: float | None = None


@dataclass(frozen=True)
class TemplateDiagnostics:
    """What ``bind`` measured: conditioning, leverage per node, residuals, rank per curve, warnings."""

    condition: float
    leverage: FloatArray
    residuals: FloatArray
    rank: dict[str, int]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RiskStamp:
    """Traceability of a report: valuation date, template name and hash, curves and pillar counts."""

    as_of: Date
    template: str
    digest: str
    curves: tuple[str, ...]
    pillars: tuple[int, ...]


@dataclass(frozen=True)
class QuoteRisk:
    """Risk to template quotes, with everything needed to reconcile it back to pillar risk.

    ``values`` is ``R`` (N, K), derivatives per unit quote; ``unexplained`` is ``U`` (N, P) over
    the template's curves; ``rebucketed`` is the naive tent rebucketing ``G B`` (no Jacobian) as a
    cross-check; ``pillar_parallel`` and ``template_parallel`` are the two DV01s (all pillars +1
    versus all template quotes +1), which differ and are both reported; ``uncovered`` holds the
    pillar blocks of curves no template instrument covers.
    """

    labels: tuple[str, ...]
    kinds: tuple[QuoteKind, ...]
    curves: tuple[str, ...]
    values: FloatArray
    unexplained: FloatArray
    rebucketed: FloatArray
    pillar_parallel: FloatArray
    template_parallel: FloatArray
    uncovered: dict[str, FloatArray]
    diagnostics: TemplateDiagnostics
    stamp: RiskStamp
    positions: tuple[str, ...] | None = None

    @property
    def dv01(self) -> FloatArray:
        """``values`` per +1bp of each quote."""
        return self.values * 1e-4

    @property
    def unexplained_max(self) -> FloatArray:
        """``‖U_i‖_∞`` per position."""
        return np.abs(self.unexplained).max(axis=1) if self.unexplained.shape[1] else np.zeros(self.values.shape[0])

    @property
    def unexplained_sum(self) -> FloatArray:
        """``Σ_j U[i, j]`` per position."""
        return self.unexplained.sum(axis=1)

    def to_pandas(self):
        """``dv01`` per quote with the reconciliation lines as extra columns."""
        import pandas as pd

        frame = pd.DataFrame(self.dv01, columns=self.labels, index=self.positions)
        frame["pillar_parallel_dv01"] = self.pillar_parallel * 1e-4
        frame["template_parallel_dv01"] = self.template_parallel * 1e-4
        frame["unexplained_max_dv01"] = self.unexplained_max * 1e-4
        frame["unexplained_sum_dv01"] = self.unexplained_sum * 1e-4
        return frame


def _tenor_label(kind: str, item) -> str:
    return f"{kind} {item}" if isinstance(item, str) else f"{kind} {' '.join(str(i) for i in item)}"


def _index_of(key: str) -> tuple[str, str]:
    """``'USD.SOFR.ON'`` → ``('USD', 'SOFR')``; ``'USD.CME-TERM-SOFR.1M'`` → ``('USD', 'CME-TERM-SOFR.1M')``."""
    parsed = CurveKey.parse(key)
    if not parsed.is_projection:
        raise TemplateError("MISSING_CURVE", f"template specs name projection curves; '{key}' is a discount key")
    return parsed.currency, parsed.name if parsed.is_overnight else f"{parsed.name}.{parsed.tenor_label}"


@dataclass(frozen=True)
class RiskTemplate:
    """The instruments risk is reported against, a basis rule and the safety limits.

    Parameters
    ----------
    instruments : sequence of CalibrationInstrument
        In node order per curve (strictly increasing pillar dates within a curve).  A quote may
        be ``NaN``: it is then only used for labelling, and no residual is checked.
    basis : BasisRule
        Default :class:`~finance.pricing.risk.basis.TentBasis`.
    limits : TemplateLimits
    labels : tuple of str, optional
        One per instrument; default ``"<curve> <pillar date>"``.
    name : str
        Audit label (``"SOFR par 1M-30Y v3"``).
    spec : mapping, optional
        The JSON-able form this template was built from (see :meth:`from_spec`).
    """

    instruments: tuple
    basis: BasisRule = TentBasis()
    limits: TemplateLimits = TemplateLimits()
    labels: tuple[str, ...] | None = None
    name: str = ""
    spec: Mapping[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(self.instruments))
        if not self.instruments:
            raise TemplateError("MISSING_CURVE", "a template needs at least one instrument")
        if self.labels is not None and len(self.labels) != len(self.instruments):
            raise ValueError(f"{len(self.labels)} labels for {len(self.instruments)} instruments")

    @property
    def curves(self) -> tuple[str, ...]:
        """The curves the instruments quote, first appearance order (keys as written, unresolved)."""
        return tuple(dict.fromkeys(h.curve for h in self.instruments))

    @property
    def kinds(self) -> tuple[QuoteKind, ...]:
        return tuple(h.quote.kind for h in self.instruments)

    def label(self, k: int) -> str:
        if self.labels is not None:
            return self.labels[k]
        h = self.instruments[k]
        return f"{h.curve} {np.datetime_as_string(np.datetime64(h.pillar_date, 'D'))}"

    @classmethod
    def identity(cls, instruments: Sequence, **kw) -> RiskTemplate:
        """The template made of the calibration instruments themselves (square: reproduces ``G J⁻¹``)."""
        return cls(tuple(instruments), **kw)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], as_of: Date, *, registry=None, **kw) -> RiskTemplate:
        """Build from desk configuration.

        ``spec = {"name": ..., "basis": "tent" | "least_squares" | "scheme", "limits": {...},
        "curves": {"USD.SOFR.ON": {"deposit": ["1M", "3M"], "fra": [["6M", "12M"]],
        "future": [["SR3", "U26"]], "swap": ["2Y", "5Y", "10Y"]}}}``.  Instruments are built
        through the calibration factories with ``NaN`` quotes, in the order given per curve.
        """
        basis_name = spec.get("basis", "tent")
        basis = {"tent": TentBasis(), "least_squares": LeastSquaresBasis(), "scheme": SchemeBasis()}[basis_name]
        limits = TemplateLimits(**spec.get("limits", {}))
        instruments, labels = [], []
        factory_kw = {} if registry is None else {"registry": registry}
        for key, groups in spec["curves"].items():
            currency, rate_index = _index_of(key)
            for kind, items in groups.items():
                for item in items:
                    if kind == "deposit":
                        h = deposit_helper(
                            rate=math.nan,
                            tenor=item,
                            as_of=as_of,
                            rate_index=rate_index,
                            currency=currency,
                            **factory_kw,
                        )
                    elif kind == "fra":
                        h = fra_helper(
                            rate=math.nan,
                            start=item[0],
                            end=item[1],
                            as_of=as_of,
                            rate_index=rate_index,
                            currency=currency,
                            **factory_kw,
                        )
                    elif kind == "future":
                        h = futures_helper(
                            price=math.nan,
                            contract=item[0],
                            month=item[1],
                            as_of=as_of,
                            rate_index=rate_index,
                            currency=currency,
                        )
                    elif kind == "swap":
                        h = swap_helper(
                            rate=math.nan,
                            tenor=item,
                            as_of=as_of,
                            rate_index=rate_index,
                            currency=currency,
                            **factory_kw,
                        )
                    else:
                        raise TemplateError("MISSING_CURVE", f"unknown template instrument kind '{kind}'")
                    instruments.append(h)
                    labels.append(f"{key} {_tenor_label(kind, item)}")
        return cls(
            tuple(instruments),
            basis=basis,
            limits=limits,
            labels=tuple(labels),
            name=spec.get("name", ""),
            spec=dict(spec),
            **kw,
        )

    def to_spec(self) -> Mapping[str, Any]:
        if self.spec is None:
            raise ValueError("this template was built from instruments, not from a spec")
        return dict(self.spec)

    @property
    def digest(self) -> str:
        """Hash of the spec when there is one, otherwise of the labels and kinds."""
        payload = (
            self.spec
            if self.spec is not None
            else {
                "labels": [self.label(k) for k in range(len(self.instruments))],
                "kinds": [k.name for k in self.kinds],
            }
        )
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]

    # -- binding ------------------------------------------------------------------------

    def bind(self, market, *, backend: Backend = Backend.Numba) -> BoundRiskTemplate:
        """Compute the Jacobian, the basis, the transfer and the diagnostics against ``market``.

        Raises
        ------
        TemplateError
            With a reason code: ``UNSORTED_NODES``, ``K_GT_P``, ``RANK_DEFICIENT``,
            ``ILL_CONDITIONED``, ``LEVERAGE``, ``RESIDUAL`` or ``MISSING_CURVE``.
        """
        warnings: list[str] = []
        # rule 7: every quoted curve exists with at least one pillar
        for h in self.instruments:
            if h.curve not in market:
                raise TemplateError("MISSING_CURVE", f"'{h.curve}' is not registered in the market")
            if market[h.curve].x.size < 2:
                raise TemplateError("MISSING_CURVE", f"'{h.curve}' has no pillars")
        resolver = ParamLayout.build(market, self.curves)
        block_of = {c: resolver.block_of(c) for c in self.curves}
        template_curves = tuple(dict.fromkeys(block_of[h.curve] for h in self.instruments))

        # rule 1: strictly increasing nodes per curve, in the order given
        rows: dict[str, list[int]] = {c: [] for c in template_curves}
        for k, h in enumerate(self.instruments):
            rows[block_of[h.curve]].append(k)
        origin = market.origin
        node_x: dict[str, FloatArray] = {}
        for c, ks in rows.items():
            x = np.array(
                [(np.datetime64(self.instruments[k].pillar_date, "D") - origin).astype(np.int64) for k in ks],
                dtype=np.float64,
            )
            bad = np.nonzero(np.diff(x) <= 0)[0]
            if bad.size:
                raise TemplateError(
                    "UNSORTED_NODES",
                    "template nodes must be strictly increasing per curve",
                    node=self.label(ks[int(bad[0]) + 1]),
                )
            node_x[c] = x

        # the Jacobian over every block the instruments touch; template curves first
        touched = template_jacobian(self.instruments, market, backend=backend)
        curves = template_curves + tuple(c for c in touched.curves if c not in template_curves)
        pillar_dates = {
            c: (touched.pillar_dates[c] if c in touched.curves else resolver.pillar_dates(c)) for c in curves
        }
        jacobian = touched.aligned(curves, pillar_dates).values
        offsets = np.concatenate([[0], np.cumsum([pillar_dates[c].size for c in curves])]).astype(np.intp)
        cols = {c: slice(int(offsets[j]), int(offsets[j + 1])) for j, c in enumerate(curves)}

        # rules 2 and 3: the basis, block-diagonal, full rank per curve
        K, P = jacobian.shape
        basis = np.zeros((P, K), dtype=np.float64)
        rank: dict[str, int] = {}
        for c in template_curves:
            ks = rows[c]
            base_x = (pillar_dates[c] - origin).astype(np.int64).astype(np.float64)
            if len(ks) > base_x.size:
                raise TemplateError("K_GT_P", f"'{c}' has {len(ks)} template nodes but {base_x.size} base pillars")
            if isinstance(self.basis, TentBasis):
                for k in TentBasis.unsupported_nodes(base_x, node_x[c]):
                    raise TemplateError(
                        "RANK_DEFICIENT",
                        "no base pillar inside the node's support: template finer than the base",
                        node=self.label(ks[k]),
                    )
            block = self.basis.matrix(
                base_x, node_x[c], jacobian_block=jacobian[np.ix_(ks, range(cols[c].start, cols[c].stop))]
            )
            rank[c] = int(np.linalg.matrix_rank(block))
            if rank[c] < len(ks):
                raise TemplateError(
                    "RANK_DEFICIENT", f"basis of '{c}' has rank {rank[c]} for {len(ks)} nodes", node=self.label(ks[-1])
                )
            basis[cols[c], ks] = block

        # rule 4: conditioning of M = J_T B
        response = jacobian @ basis
        condition = float(np.linalg.cond(response))
        if not np.isfinite(condition) or condition > self.limits.max_condition:
            raise TemplateError("ILL_CONDITIONED", f"cond(M) = {condition:.3g} exceeds {self.limits.max_condition:.3g}")
        if condition > self.limits.warn_condition:
            warnings.append(f"cond(M) = {condition:.3g} exceeds the warning level {self.limits.warn_condition:.3g}")
        lu = lu_factor(response.T)

        # rule 5: transfer and leverage
        transfer = basis @ np.linalg.solve(response, np.eye(K))
        leverage = np.abs(transfer).max(axis=0)
        worst = int(np.argmax(leverage))
        if leverage[worst] > self.limits.max_leverage:
            raise TemplateError(
                "LEVERAGE",
                f"a unit move of the node moves a base pillar by {leverage[worst]:.3g}",
                node=self.label(worst),
            )
        for k in np.nonzero(leverage > self.limits.warn_leverage)[0]:
            warnings.append(f"leverage {leverage[k]:.3g} at {self.label(int(k))}")

        # unit sensitivities: each instrument at its own quote, unit size
        unit = self._unit_risk(market, curves, pillar_dates, backend)
        response_unit = unit @ transfer
        sensitivity = np.diag(response_unit).copy()
        off = np.abs(response_unit - np.diag(sensitivity))
        if sensitivity.size > 1 and np.any(off.max(axis=1) > 1e-2 * np.abs(sensitivity)):
            k = int(np.argmax(off.max(axis=1) / np.maximum(np.abs(sensitivity), 1e-300)))
            warnings.append(f"template instrument {self.label(k)} carries risk to other template quotes")

        # rule 6: residuals against the quotes carried
        quotes = np.array([h.quote.value for h in self.instruments], dtype=np.float64)
        residuals = touched.pv - quotes
        if self.limits.max_abs_residual is not None:
            worst = np.nanargmax(np.abs(residuals)) if np.any(~np.isnan(residuals)) else None
            if worst is not None and abs(residuals[worst]) > self.limits.max_abs_residual:
                raise TemplateError(
                    "RESIDUAL", f"implied − quote = {residuals[worst]:.3g}", node=self.label(int(worst))
                )

        anchors = {c: tuple(id(a) for a in market.owned(c)) for c in curves}
        return BoundRiskTemplate(
            template=self,
            curves=curves,
            param_offsets=offsets,
            pillar_dates=pillar_dates,
            node_x=node_x,
            jacobian=jacobian,
            basis=basis,
            transfer=transfer,
            unit_sensitivity=sensitivity,
            unit_risk=unit,
            quotes=quotes,
            implied=touched.pv,
            diagnostics=TemplateDiagnostics(condition, leverage, residuals, rank, tuple(warnings)),
            as_of=market.as_of,
            _lu=lu,
            _anchors=anchors,
        )

    def _unit_risk(self, market, curves, pillar_dates, backend) -> FloatArray:
        """Pillar risk (K, P) of each instrument built at its implied quote in unit size."""
        from finance.pricing.pricers import FuturesPricer, LegPricer

        legs, leg_rows, futures, future_rows = [], [], [], []
        for k, h in enumerate(self.instruments):
            implied = float(h.implied(market))
            if isinstance(h, SwapHelper):
                fixed = [i for i, leg in enumerate(h.swap) if not leg.is_floating]
                swap_legs = list(h.swap)
                for i in fixed:
                    swap_legs[i] = swap_legs[i].replace(coupon_rate=implied)
                legs.append(dataclasses.replace(h.swap, receive_leg=swap_legs[0], pay_leg=swap_legs[1]))
                leg_rows.append(k)
            elif isinstance(h, DepositHelper):
                legs.append(dataclasses.replace(h.deposit, rate=implied, notional=1.0))
                leg_rows.append(k)
            elif isinstance(h, FraHelper):
                legs.append(dataclasses.replace(h.fra, coupon_rate=implied, notional=1.0))
                leg_rows.append(k)
            elif isinstance(h, FuturesHelper):
                futures.append(dataclasses.replace(h.future, contracts=1.0))
                future_rows.append(k)
            else:
                raise NotImplementedError(f"no unit instrument for helper type {type(h).__name__}")
        out = np.zeros((len(self.instruments), int(sum(pillar_dates[c].size for c in curves))), dtype=np.float64)
        if legs:
            risk = pillar_risk(LegPricer().compile(legs), market, backend=backend).aligned(curves, pillar_dates)
            out[leg_rows] = risk.values
        if futures:
            risk = pillar_risk(FuturesPricer().compile(futures), market, backend=backend).aligned(curves, pillar_dates)
            out[future_rows] = risk.values
        return out


@dataclass(frozen=True)
class BoundRiskTemplate:
    """A template bound to one market: ``J_T``, ``B``, ``A``, the unit instruments' risk and diagnostics.

    ``unit_risk`` (K, P) is the pillar risk of each template instrument built at its implied
    quote in unit size; ``unit_sensitivity`` is its diagonal response to its own quote, ``s``.
    """

    template: RiskTemplate
    curves: tuple[str, ...]
    param_offsets: np.ndarray
    pillar_dates: dict[str, DateArray]
    node_x: dict[str, FloatArray]
    jacobian: FloatArray
    basis: FloatArray
    transfer: FloatArray
    unit_sensitivity: FloatArray
    unit_risk: FloatArray
    quotes: FloatArray
    implied: FloatArray
    diagnostics: TemplateDiagnostics
    as_of: Date
    _lu: tuple = field(repr=False, compare=False)
    _anchors: dict[str, tuple[int, ...]] = field(repr=False, compare=False)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.template.label(k) for k in range(len(self.template.instruments)))

    @property
    def stamp(self) -> RiskStamp:
        return RiskStamp(
            self.as_of,
            self.template.name,
            self.template.digest,
            self.curves,
            tuple(int(self.pillar_dates[c].size) for c in self.curves),
        )

    def slice(self, curve: str) -> slice:
        j = self.curves.index(curve)
        return slice(int(self.param_offsets[j]), int(self.param_offsets[j + 1]))

    def matches(self, market) -> bool:
        """Whether ``market`` is the market this template was bound to: same as-of, same curve objects."""
        if market.as_of != self.as_of:
            return False
        try:
            return all(tuple(id(a) for a in market.owned(c)) == self._anchors[c] for c in self.curves)
        except (KeyError, ValueError):
            return False

    def quote_risk(self, pillar: PillarRisk) -> QuoteRisk:
        """``R = G A`` with the reconciliation lines; curves the template does not span are ``uncovered``."""
        uncovered = {c: pillar.block(c) for c in pillar.curves if c not in self.curves}
        G = pillar.aligned(self.curves, self.pillar_dates).values
        X = lu_solve(self._lu, self.basis.T @ G.T)
        R = np.ascontiguousarray(X.T)
        return QuoteRisk(
            labels=self.labels,
            kinds=self.template.kinds,
            curves=self.curves,
            values=R,
            unexplained=G - R @ self.jacobian,
            rebucketed=G @ self.basis,
            pillar_parallel=G.sum(axis=1),
            template_parallel=R.sum(axis=1),
            uncovered=uncovered,
            diagnostics=self.diagnostics,
            stamp=self.stamp,
            positions=pillar.labels,
        )

    def rebucket(self, pillar: PillarRisk) -> FloatArray:
        """The naive tent rebucketing ``G B`` (no Jacobian): Strata's linear rebucketing, a cross-check."""
        return pillar.aligned(self.curves, self.pillar_dates).values @ self.basis

    def hedge(self, risk: QuoteRisk) -> FloatArray:
        """Unit-instrument notionals that flatten ``risk`` on the template: ``−R / s``."""
        return -risk.values / self.unit_sensitivity[None, :]

    def base_move(self, k: int, size: float) -> dict[str, FloatArray]:
        """The base-pillar move (per curve, zero-rate units) of template quote ``k`` moving by ``size``."""
        column = self.transfer[:, k] * size
        return {c: column[self.slice(c)] for c in self.curves}

    def apply(self, market, k: int, size: float):
        """``market`` with template quote ``k`` moved by ``size`` (a frozen scenario copy)."""
        out = market
        for c, move in self.base_move(k, size).items():
            if np.any(move != 0.0):
                out = out.bump(c, move)
        return out
