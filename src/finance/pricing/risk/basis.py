"""Bucket bases: how the base curve moves between template nodes.

A :class:`~finance.pricing.risk.template.RiskTemplate` re-expresses pillar risk as risk to
template quotes.  The template Jacobian fixes what a template quote *is*; the basis fixes how
the base curve moves *between* template nodes when one quote moves, i.e. how pillar risk between
nodes is attributed.  Each rule returns ``B_c`` of shape ``(P_c, K_c)`` for one curve: column
``k`` is the base-pillar move (in zero-rate units) for a unit move of template node ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from common.math.interpolation import Interpolator, Linear
from finance.markets.curves import CurveSpace, ZeroCurve

FloatArray = np.ndarray


@runtime_checkable
class BasisRule(Protocol):
    """``matrix(base_x, node_x, jacobian_block=...) -> (P_c, K_c)``; offsets in days, strictly increasing, ``> 0``."""

    def matrix(self, base_x: FloatArray, node_x: FloatArray, *, jacobian_block: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class TentBasis:
    """Hat functions on the template nodes, flat beyond the first and last node (the default).

    ``B[j, k]`` is the hat of node ``k`` evaluated at base pillar ``j``: linear in days between
    adjacent template nodes, ``1`` on ``[0, x_1]`` for the first node and on ``[x_K, ∞)`` for the
    last.  Local, non-negative, a partition of unity at every base pillar.  This is Strata's
    linear rebucketing rule, implemented with ``np.interp`` so the flat extension and the origin
    behave exactly as stated.
    """

    def matrix(self, base_x: FloatArray, node_x: FloatArray, *, jacobian_block: FloatArray) -> FloatArray:
        base_x = np.asarray(base_x, dtype=np.float64)
        node_x = np.asarray(node_x, dtype=np.float64)
        out = np.empty((base_x.size, node_x.size), dtype=np.float64)
        for k in range(node_x.size):
            hat = np.zeros(node_x.size)
            hat[k] = 1.0
            out[:, k] = np.interp(base_x, node_x, hat)
        return out

    @staticmethod
    def unsupported_nodes(base_x: FloatArray, node_x: FloatArray) -> list[int]:
        """Template nodes with no base pillar strictly inside their support ``(x_{k-1}, x_{k+1})``.

        Such a node's column is zero and the template is rank-deficient: finer than the base.
        """
        base_x = np.asarray(base_x, dtype=np.float64)
        node_x = np.asarray(node_x, dtype=np.float64)
        lower = np.concatenate([[0.0], node_x[:-1]])
        upper = np.concatenate([node_x[1:], [np.inf]])
        return [k for k in range(node_x.size) if not np.any((base_x > lower[k]) & (base_x < upper[k]))]


@dataclass(frozen=True)
class LeastSquaresBasis:
    """``B = J_Tᵀ``: the global basis that minimises the unexplained risk ``‖U‖₂``.

    Reproduces the July risk-map note (``s = g · J⁺``) and ``research/risk_reconciliation.py``;
    kept as a cross-check, not the default.
    """

    def matrix(self, base_x: FloatArray, node_x: FloatArray, *, jacobian_block: FloatArray) -> FloatArray:
        return np.ascontiguousarray(np.asarray(jacobian_block, dtype=np.float64).T)


@dataclass(frozen=True)
class SchemeBasis:
    """A template-shaped ``ZeroCurve`` (origin + template nodes) in a chosen scheme, differentiated at the base pillars.

    ``B = −diag(365 / x_base) · ∂ln DF(x_base)/∂z_template``: smooth bumps, which under a cubic
    overshoot into neighbouring buckets ("curve noise").  Opt-in experiments only.
    """

    space: CurveSpace = CurveSpace.ZeroRate
    interpolator: Interpolator = Linear()

    def matrix(self, base_x: FloatArray, node_x: FloatArray, *, jacobian_block: FloatArray) -> FloatArray:
        base_x = np.asarray(base_x, dtype=np.float64)
        node_x = np.asarray(node_x, dtype=np.float64)
        x = np.concatenate([[0.0], node_x])
        template = ZeroCurve(x, np.exp(-0.03 * x / 365.0), space=self.space, interpolator=self.interpolator)
        _, jac = template.log_df_affine(base_x)
        return -(365.0 / base_x)[:, None] * jac
