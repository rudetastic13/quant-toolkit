"""Weighted sum of discount curves in log-DF space.

``ln DF(x) = Σᵢ wᵢ · ln DFᵢ(x)``, i.e. ``DF = Π DFᵢ^{wᵢ}``.  Zero rates and instantaneous
forwards add with the same weights.  This is the *only* composition that keeps ``ln DF`` affine
in the parameters (the engine contract in :mod:`.discount_curve`), and it covers every
production case: a base plus a spread, a basis between two indices (weights ``(1, -1)``), a
turn-of-year atom added to a smooth curve, or a scenario overlay.  Components are any
:class:`DiscountCurve`, so composites nest.

Components may be labelled by passing a mapping (``{"base": sofr, "spread": s3m}``); labels
address components in ``curve["spread"]``, ``component_slice("spread")`` and
``with_component("spread", ...)`` and name the parameter blocks in risk reports.  Labels never
affect the arithmetic, so they are not part of :attr:`scheme`.

Parameters are the components' parameters concatenated in component order; ``param_offsets``
recovers each block, which is how risk on a composite is reported per component (index risk vs
spread risk vs turn risk) rather than as one undifferentiated ladder.

Extrapolation is each component's own: a query beyond one component's last pillar uses that
component's rule, so a two-node flat-forward spread atom (``ZeroCurve.constant``) continues its
constant spread for ever while a base curve with ``NotAllowed`` still rejects.
"""

from __future__ import annotations

from functools import reduce
from typing import Hashable, Mapping, Sequence

import numpy as np

from finance.markets.curves._curve_impl.discount_curve import DiscountCurve, FloatArray


class CompositeCurve:
    """Weighted sum of discount curves in log-DF space: ``ln DF = Σ wᵢ ln DFᵢ``.

    Parameters
    ----------
    components : sequence of DiscountCurve, or mapping of str to DiscountCurve
        At least one.  Any curve satisfying the protocol, including other composites.  A
        mapping labels the components (insertion order is component order).
    weights : sequence of float, optional
        One per component, finite and non-zero.  Default: all ``1.0``.

    Raises
    ------
    TypeError
        If a component does not satisfy :class:`DiscountCurve`.
    ValueError
        If ``components`` is empty, ``weights`` has the wrong length, a weight is zero or not
        finite, or labels are not unique.

    Notes
    -----
    The parameter vector is the concatenation of the components' parameters in component
    order; :attr:`param_offsets` and :meth:`component_slice` recover the blocks.  ``ln DF`` is
    exactly affine in it because each component's is and the sum is linear.

    Examples
    --------
    A spread curve, a funding basis as the difference of two curves, and a turn added to a
    smooth curve:

    >>> fedfund = CompositeCurve({"base": sofr, "spread": ZeroCurve.constant(-0.0005, sofr.max_x)})  # doctest: +SKIP
    >>> basis = CompositeCurve((fedfund, sofr), weights=(1.0, -1.0))                                  # doctest: +SKIP
    >>> with_turn = CompositeCurve({"smooth": sofr, "turn": turn})                                     # doctest: +SKIP
    >>> fedfund["spread"].zero_rate(x)                                                                 # doctest: +SKIP
    """

    __slots__ = ("_components", "_weights", "_labels", "_x", "_offsets")

    def __init__(
        self,
        components: Sequence[DiscountCurve] | Mapping[str, DiscountCurve],
        weights: Sequence[float] | None = None,
    ) -> None:
        if isinstance(components, Mapping):
            labels: tuple[str, ...] | None = tuple(str(k) for k in components)
            components = tuple(components.values())
            if len(set(labels)) != len(labels):
                raise ValueError(f"component labels must be unique; got {labels}.")
        else:
            labels = None
            components = tuple(components)
        if not components:
            raise ValueError("CompositeCurve needs at least one component.")
        for i, component in enumerate(components):
            if not isinstance(component, DiscountCurve):
                raise TypeError(f"component {i} ({type(component).__name__}) does not satisfy DiscountCurve.")
        if weights is None:
            weights = (1.0,) * len(components)
        weights = tuple(float(w) for w in weights)
        if len(weights) != len(components):
            raise ValueError(f"{len(components)} components but {len(weights)} weights.")
        if not all(np.isfinite(w) and w != 0.0 for w in weights):
            raise ValueError(f"weights must be finite and non-zero; got {weights}.")
        self._init(components, weights, labels)

    def _init(
        self, components: tuple[DiscountCurve, ...], weights: tuple[float, ...], labels: tuple[str, ...] | None
    ) -> None:
        self._components = components
        self._weights = weights
        self._labels = labels
        self._x = reduce(np.union1d, (c.x for c in components))
        self._offsets = np.concatenate([[0], np.cumsum([c.n_params for c in components])]).astype(np.intp)

    @classmethod
    def _from_parts(
        cls, components: tuple[DiscountCurve, ...], weights: tuple[float, ...], labels: tuple[str, ...] | None
    ) -> CompositeCurve:
        """Rebuild without re-validating; the rebinding paths pass parts that are already valid."""
        curve = object.__new__(cls)
        curve._init(components, weights, labels)
        return curve

    # -- state ----------------------------------------------------------------------------

    @property
    def components(self) -> tuple[DiscountCurve, ...]:
        """The component curves, in parameter-block order."""
        return self._components

    @property
    def weights(self) -> tuple[float, ...]:
        """The log-DF weight of each component."""
        return self._weights

    @property
    def labels(self) -> tuple[str, ...] | None:
        """Component labels in component order, or ``None`` when built from a plain sequence."""
        return self._labels

    def _index(self, key: int | str) -> int:
        if isinstance(key, str):
            if self._labels is None:
                raise KeyError(f"this composite has no labels; address components by index (got {key!r}).")
            try:
                return self._labels.index(key)
            except ValueError:
                raise KeyError(f"no component labelled {key!r}; labels are {self._labels}.") from None
        n = len(self._components)
        if not -n <= key < n:
            raise IndexError(f"component index {key} out of range for {n} components.")
        return key % n

    def __getitem__(self, key: int | str) -> DiscountCurve:
        """Component by index or label.

        Parameters
        ----------
        key : int or str

        Returns
        -------
        DiscountCurve

        Raises
        ------
        KeyError
            Unknown label, or a label on an unlabelled composite.
        IndexError
            Index out of range.
        """
        return self._components[self._index(key)]

    def __len__(self) -> int:
        return len(self._components)

    @property
    def x(self) -> FloatArray:
        """Union of the components' pillar offsets, origin-inclusive.

        Returns
        -------
        ndarray of float64
            Sorted, deduplicated, ``x[0] == 0``.
        """
        return self._x

    @property
    def dfs(self) -> FloatArray:
        """Composite discount factors at :attr:`x`."""
        return self.discount_factor(self._x)

    @property
    def max_x(self) -> float:
        """Largest pillar offset of any component."""
        return float(self._x[-1])

    @property
    def scheme(self) -> Hashable:
        """Weights plus each component's pillars and scheme; labels excluded.

        Component pillar positions are part of the identity: the union :attr:`x` alone does not
        say which component owns a pillar, and the affine rows depend on that.  Labels do not
        change the arithmetic, so two composites that differ only in labels share a scheme.
        """
        return (
            "composite",
            self._weights,
            tuple((tuple(c.x.tolist()), c.scheme) for c in self._components),
        )

    @property
    def is_log_linear(self) -> bool:
        """Always ``False``; a composite is not a single log-linear line."""
        return False

    # -- parameters -----------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Sum of the components' parameter counts."""
        return int(self._offsets[-1])

    @property
    def param_offsets(self) -> np.ndarray:
        """Block boundaries of :attr:`params`.

        Returns
        -------
        ndarray of intp, shape (n_components + 1,)
            Block ``i`` is ``params[off[i]:off[i + 1]]``.
        """
        return self._offsets

    def component_slice(self, key: int | str) -> slice:
        """Slice of :attr:`params` (and of any Jacobian's columns) owned by one component.

        Parameters
        ----------
        key : int or str
            Component index or label.

        Returns
        -------
        slice
        """
        i = self._index(key)
        return slice(int(self._offsets[i]), int(self._offsets[i + 1]))

    @property
    def params(self) -> FloatArray:
        """Components' parameters concatenated in component order.

        Returns
        -------
        ndarray of float64, shape (n_params,)
        """
        return np.concatenate([c.params for c in self._components])

    def with_params(self, params: FloatArray) -> CompositeCurve:
        """Rebind every component's parameters from one flat vector.

        Parameters
        ----------
        params : array_like of float64, shape (n_params,)
            Split by :attr:`param_offsets` and handed to each component's ``with_params``.

        Returns
        -------
        CompositeCurve
            Same components' geometry, the same weights and labels.

        Raises
        ------
        ValueError
            If ``params`` has the wrong shape, or a component rejects its block.
        """
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"expected {self.n_params} parameters; got shape {params.shape}.")
        components = tuple(c.with_params(params[self.component_slice(i)]) for i, c in enumerate(self._components))
        return self._from_parts(components, self._weights, self._labels)

    def with_component(self, key: int | str, component: DiscountCurve) -> CompositeCurve:
        """Replace one component with a curve of any geometry; weights and labels are kept.

        Parameters
        ----------
        key : int or str
            Component index or label.
        component : DiscountCurve
            The replacement.

        Returns
        -------
        CompositeCurve
        """
        i = self._index(key)
        components = list(self._components)
        components[i] = component
        if self._labels is None:
            return CompositeCurve(components, self._weights)
        return CompositeCurve(dict(zip(self._labels, components)), self._weights)

    # -- queries --------------------------------------------------------------------------

    def _weighted_sum(self, method: str, xq: FloatArray) -> FloatArray:
        out = np.zeros(np.shape(xq), dtype=np.float64)
        for w, c in zip(self._weights, self._components):
            out += w * getattr(c, method)(xq)
        return out

    def log_discount_factor(self, xq: FloatArray) -> FloatArray:
        """``Σ wᵢ ln DFᵢ(xq)``.

        Parameters
        ----------
        xq : ndarray of float64
            Day offsets from the origin, ``>= 0``.

        Returns
        -------
        ndarray of float64, same shape as ``xq``
        """
        return self._weighted_sum("log_discount_factor", xq)

    def discount_factor(self, xq: FloatArray) -> FloatArray:
        """``Π DFᵢ(xq)^{wᵢ}``; see :meth:`log_discount_factor`."""
        return np.exp(self.log_discount_factor(xq))

    def zero_rate(self, xq: FloatArray) -> FloatArray:
        """``Σ wᵢ zᵢ(xq)``: zero rates add because each is ``-ln DFᵢ / t``."""
        return self._weighted_sum("zero_rate", xq)

    def instantaneous_forward(self, xq: FloatArray) -> FloatArray:
        """``Σ wᵢ fᵢ(xq)``: forwards add because each is ``-d ln DFᵢ / dt``."""
        return self._weighted_sum("instantaneous_forward", xq)

    # -- engine hand-off ------------------------------------------------------------------

    def log_df_affine(self, xq: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Weighted sum of the components' affine maps, Jacobian blocks side by side.

        Exact by construction: no probing or reconstruction check is needed because every
        component already guarantees its own map.

        Parameters
        ----------
        xq : ndarray of float64, shape (n_queries,)

        Returns
        -------
        offset : ndarray of float64, shape (n_queries,)
        jacobian : ndarray of float64, shape (n_queries, n_params)
            Column block ``i`` is ``wᵢ`` times component ``i``'s Jacobian.
        """
        offset = np.zeros(np.shape(xq), dtype=np.float64)
        blocks = []
        for w, c in zip(self._weights, self._components):
            c_offset, c_jac = c.log_df_affine(xq)
            offset += w * c_offset
            blocks.append(w * c_jac)
        return offset, np.hstack(blocks)

    def __repr__(self) -> str:
        if self._labels is None:
            parts = ", ".join(f"{w:+g}*{c!r}" for w, c in zip(self._weights, self._components))
        else:
            parts = ", ".join(f"{lb}={w:+g}*{c!r}" for lb, w, c in zip(self._labels, self._weights, self._components))
        return f"CompositeCurve({parts})"
