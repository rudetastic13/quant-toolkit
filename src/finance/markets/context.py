"""MarketContext — the curves a market knows, by key, plus vols, fixings and conventions.

A market is a ``Mapping[str, DiscountCurve]``: ``market["USD.SOFR.1D"]`` is the mathematical
curve registered under that key and ``market.yield_curve("USD.SOFR.1D")`` the date-aware view
the engines read (origin = ``as_of``, the key's conventions, the key's fixings).  Keys follow
the grammar of :class:`finance.conventions.CurveKey`: ``CCY.INDEX.TENOR`` projects, ``CCY.NAME``
discounts.  The curves live in one :class:`~finance.markets.yield_curve_group.YieldCurveGroup`
per currency; a key is resolved by its currency, so every engine keeps reading ``market[key]``.

Assembly is by ``add`` and reads like finance::

    market = MarketContext(as_of)
    market.add(sofr, funding_names=["USD.STDCSA"])    # projects as USD.SOFR.1D, discounts as USD.STDCSA
    market.add(fedfund)                               # key USD.FEDFUND.1D, built as sofr - 5 * bp

``market["USD.STDCSA"] = sofr`` is the same registration one key at a time (``add`` under that
name alone).  Sharing is by object identity, declared by construction: writing ``sofr - 5 * bp``
already says the Fed Funds curve is built on ``sofr``.  Scenarios propagate through that
structure: ``with_curve`` and ``bump`` return new markets in which the moved object is replaced
wherever it appears, so "bump SOFR" moves Fed Funds and leaves the basis alone.  A registered
key owns the parameters nobody else owns (:meth:`MarketContext.leaf_owner`); ``bump(key, ...)``
moves exactly those.  Ownership is market-wide: a curve in one currency's group may be built on
an atom registered in another's.

Projection and par-rate logic deliberately lives in
:class:`finance.markets.rate_generator.RateGenerator`, not here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from finance.conventions import ConventionRegistry, CurveKey, default_registry
from finance.dates import Date
from finance.markets.curves import DiscountCurve, YieldCurve
from finance.markets.fixings import HistoricalFixings
from finance.markets.yield_curve_group import YieldCurveGroup

FloatArray = np.ndarray


def _components(curve: DiscountCurve) -> tuple[DiscountCurve, ...]:
    """A composite's components; an atom has none."""
    return tuple(getattr(curve, "components", ()))


def _atoms(curve: DiscountCurve, depth: int = 0) -> Iterator[tuple[DiscountCurve, int]]:
    """Every atom under ``curve`` with its depth (composite levels below ``curve``), in parameter order."""
    parts = _components(curve)
    if not parts:
        yield curve, depth
        return
    for part in parts:
        yield from _atoms(part, depth + 1)


def _contains(curve: DiscountCurve, target: DiscountCurve) -> bool:
    """Whether ``target`` appears anywhere under ``curve`` (by identity), ``curve`` itself included."""
    if curve is target:
        return True
    return any(_contains(part, target) for part in _components(curve))


class MarketContext(Mapping[str, DiscountCurve]):
    """Registered curves by key, with the as-of date, conventions, fixings and vols pricers read.

    Parameters
    ----------
    as_of : Date
        Valuation date; the origin of every registered curve.
    curves : Mapping[str, DiscountCurve or YieldCurve], optional
        Initial registrations, applied in order through :meth:`__setitem__`.
    conventions : ConventionRegistry, optional
        Where projection keys look their index family up.  Default: the library registry.
    vols : VolNamespace or None, optional

    Notes
    -----
    A market is mutable while it is being assembled and :meth:`freeze` makes it read-only.
    :meth:`with_curve`, :meth:`bump` and :meth:`with_fixings` return frozen copies; the market
    they were called on is never changed.  ``market[key]`` is the registered ``DiscountCurve``;
    :meth:`yield_curve` is the stored date-aware view, one per key.

    Examples
    --------
    >>> market = MarketContext(Date(2026, 6, 1))                              # doctest: +SKIP
    >>> market.add(sofr, funding_names=["USD.STDCSA"])                        # doctest: +SKIP
    >>> market["USD.FEDFUND.1D"] = market["USD.SOFR.1D"] - 5 * bp             # doctest: +SKIP
    >>> market.leaf_owner()                                                   # doctest: +SKIP
    {<id of sofr>: 'USD.SOFR.1D', <id of the spread>: 'USD.FEDFUND.1D'}
    >>> up = market.bump("USD.SOFR.1D", 1e-4)      # Fed Funds follows, the spread does not  # doctest: +SKIP
    """

    __slots__ = ("_as_of", "_origin", "_groups", "_conventions", "_vols", "_frozen")

    def __init__(
        self,
        as_of: Date | None = None,
        curves: Mapping[str, DiscountCurve | YieldCurve] | None = None,
        *,
        conventions: ConventionRegistry = default_registry,
        vols: Any | None = None,
    ) -> None:
        if as_of is None:
            raise ValueError("MarketContext needs a valuation date: as_of")
        self._as_of = as_of
        self._origin = np.datetime64(as_of.to_numpy(), "D")
        self._groups: dict[str, YieldCurveGroup] = {}
        self._conventions = conventions
        self._vols = vols
        self._frozen = False
        if curves:
            for key, curve in curves.items():
                self[key] = curve

    # -- identity -------------------------------------------------------------------------

    @property
    def as_of(self) -> Date:
        """Valuation date; the origin of every registered curve."""
        return self._as_of

    @property
    def origin(self) -> np.datetime64:
        """:attr:`as_of` as ``datetime64[D]``."""
        return self._origin

    @property
    def conventions(self) -> ConventionRegistry:
        return self._conventions

    @property
    def vols(self) -> Any | None:
        return self._vols

    @property
    def fixings(self) -> Mapping[str, HistoricalFixings]:
        """Historical prints by curve key, read off the registered views (read-only)."""
        return MappingProxyType(
            {name: view.historical_fixings for name, view in self._views().items() if view.historical_fixings is not None}
        )

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def groups(self) -> Mapping[str, YieldCurveGroup]:
        """The per-currency groups, by ISO code, in registration order."""
        return MappingProxyType(self._groups)

    def group(self, currency: str) -> YieldCurveGroup:
        """The group holding ``currency``'s curves.

        Raises
        ------
        KeyError
            If no curve in that currency is registered.
        """
        try:
            return self._groups[str(currency).strip().upper()]
        except KeyError:
            raise KeyError(f"no {currency} curves registered") from None

    @staticmethod
    def _key(key: CurveKey | str) -> str:
        return str(CurveKey.parse(key))

    def _views(self) -> dict[str, YieldCurve]:
        """Every registered view by name: group order, then registration order within a group."""
        out: dict[str, YieldCurve] = {}
        for group in self._groups.values():
            out.update(group)
        return out

    def _view(self, key: CurveKey | str) -> YieldCurve:
        parsed = CurveKey.parse(key)
        group = self._groups.get(parsed.currency)
        if group is None or str(parsed) not in group:
            raise KeyError(f"no curve registered under '{key}'")
        return group[str(parsed)]

    @property
    def _curves(self) -> dict[str, DiscountCurve]:
        return {name: view.zero_curve for name, view in self._views().items()}

    # -- Mapping ----------------------------------------------------------------------------

    def __getitem__(self, key: str) -> DiscountCurve:
        return self._view(key).zero_curve

    def __iter__(self) -> Iterator[str]:
        return iter(self._views())

    def __len__(self) -> int:
        return sum(len(group) for group in self._groups.values())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, (str, CurveKey)):
            return False
        try:
            self._view(key)
        except (KeyError, ValueError):
            return False
        return True

    # -- assembly -------------------------------------------------------------------------

    def _as_view(self, name: str, curve: DiscountCurve | YieldCurve) -> YieldCurve:
        """``curve`` as the view registered under ``name``: a ``YieldCurve`` under another key is
        re-keyed without its fixings (they belong to the index it was built for, not to an alias)."""
        if isinstance(curve, YieldCurve):
            if curve.name == name:
                return curve
            return YieldCurve.from_registry(curve.origin, curve.zero_curve, key=name, registry=self._conventions)
        if isinstance(curve, DiscountCurve):
            return YieldCurve.from_registry(self._origin, curve, key=name, registry=self._conventions)
        raise TypeError(f"'{name}': expected a DiscountCurve or YieldCurve, got {type(curve).__name__}")

    def add(self, yield_curve: YieldCurve, *, funding_names: Iterable[CurveKey | str] = ()) -> None:
        """Register ``yield_curve`` under its key and under each of ``funding_names`` (assembly).

        Parameters
        ----------
        yield_curve : YieldCurve
            Must have this market's origin; its key's family and tenor must be in ``conventions``.
        funding_names : iterable of str or CurveKey, optional
            Discount keys in the same currency the curve also discounts under (``"USD.STDCSA"``).

        Raises
        ------
        TypeError
            If the market is frozen, or ``yield_curve`` is not a ``YieldCurve``.
        ValueError
            If the origin differs, if a key already registered would be redefined in terms of the
            curve it replaces (that is a redefinition, not a scenario: build the new curve from
            the atoms), or if the registration makes a leaf's ownership ambiguous (see
            :meth:`leaf_owner`); the market is left as it was.
        """
        if self._frozen:
            raise TypeError("this MarketContext is frozen; use with_curve() / bump() for scenario copies")
        if not isinstance(yield_curve, YieldCurve):
            raise TypeError(f"expected a YieldCurve, got {type(yield_curve).__name__}")
        if yield_curve.origin != self._origin:
            raise ValueError(
                f"'{yield_curve.name}': YieldCurve origin {yield_curve.origin} is not the market's as_of {self._origin}"
            )
        name = self._conventions.validate(yield_curve.key)
        previous = self._curves.get(str(name))
        if previous is not None and previous is not yield_curve.zero_curve and _contains(yield_curve.zero_curve, previous):
            raise ValueError(
                f"'{name}' would be redefined in terms of the curve it replaces; assign a curve built from "
                "the atoms, or use with_curve() / bump() to move it"
            )
        group = self._groups.get(name.currency)
        if group is None:
            group = self._groups[name.currency] = YieldCurveGroup(name.currency)
        before = dict(group._views)
        group.add(yield_curve, funding_names=funding_names)
        try:
            self.leaf_owner()
        except ValueError:
            group._views = before
            raise

    def __setitem__(self, key: str, curve: DiscountCurve | YieldCurve) -> None:
        """Register ``curve`` under ``key``: :meth:`add` under that one name.

        Parameters
        ----------
        key : str
            ``CCY.INDEX.TENOR`` (its family and tenor must be in ``conventions``) or ``CCY.NAME``.
        curve : DiscountCurve or YieldCurve
            A ``YieldCurve`` must have this market's origin; under its own key its fixings come
            along, under another key they do not.
        """
        if self._frozen:
            raise TypeError("this MarketContext is frozen; use with_curve() / bump() for scenario copies")
        self.add(self._as_view(self._key(key), curve))

    # -- derived views ------------------------------------------------------------------

    def yield_curve(self, key: str) -> YieldCurve:
        """The date-aware view registered under ``market[key]``: origin, the key's conventions and fixings."""
        return self._view(key)

    # -- ownership ----------------------------------------------------------------------

    def leaf_owner(self) -> dict[int, str]:
        """Which registered key owns each atom: ``{id(atom): key}``.

        An atom reached by several keys belongs to the one reaching it by the shortest path
        (its own key when it is registered directly); at equal depth a projection key beats a
        discount key, and among registered objects the first registered wins.

        Raises
        ------
        ValueError
            When two keys of the same kind reach an unregistered atom at the same depth: nobody
            can own it, so register the atom under a key of its own.
        """
        reach: dict[int, list[tuple[int, int, int, str]]] = {}
        atoms: dict[int, DiscountCurve] = {}
        for index, (name, curve) in enumerate(self._curves.items()):
            kind = 0 if CurveKey.parse(name).is_projection else 1
            for atom, depth in _atoms(curve):
                atoms.setdefault(id(atom), atom)
                reach.setdefault(id(atom), []).append((depth, kind, index, name))
        owner: dict[int, str] = {}
        for atom_id, paths in reach.items():
            paths.sort()
            depth, kind, _, name = paths[0]
            rivals = {p[3] for p in paths if p[0] == depth and p[1] == kind} - {name}
            if depth > 0 and rivals:
                raise ValueError(
                    f"{atoms[atom_id]!r} is reached at the same depth by {name!r} and "
                    f"{sorted(rivals)!r}; register it under its own key"
                )
            owner[atom_id] = name
        return owner

    def owned(self, key: str) -> tuple[DiscountCurve, ...]:
        """The atoms ``key`` owns, in the parameter order of ``market[key]``.

        ``bump(key, vector)`` lays ``vector`` out over their parameters in this order.
        """
        name = self._key(key)
        owner = self.leaf_owner()
        out: list[DiscountCurve] = []
        seen: set[int] = set()
        for atom, _depth in _atoms(self[name]):
            if owner[id(atom)] == name and id(atom) not in seen:
                seen.add(id(atom))
                out.append(atom)
        return tuple(out)

    # -- scenarios ----------------------------------------------------------------------

    def _copy(self, views: Mapping[str, YieldCurve]) -> MarketContext:
        """A frozen market holding ``views``, grouped by currency; group settings carry over."""
        out = MarketContext.__new__(MarketContext)
        out._as_of = self._as_of
        out._origin = self._origin
        out._conventions = self._conventions
        out._vols = self._vols
        out._frozen = True
        by_currency: dict[str, dict[str, YieldCurve]] = {ccy: {} for ccy in self._groups}
        for name, view in views.items():
            by_currency.setdefault(view.key.currency, {})[name] = view
        out._groups = {
            ccy: (self._groups[ccy] if ccy in self._groups else YieldCurveGroup(ccy))._rebound(group_views)
            for ccy, group_views in by_currency.items()
        }
        return out

    def _object_owner(self, obj: DiscountCurve) -> str:
        """Among the keys registered to exactly ``obj``: a projection key first, then the first registered."""
        names = [name for name, curve in self._curves.items() if curve is obj]
        for candidate in names:
            if CurveKey.parse(candidate).is_projection:
                return candidate
        return names[0]

    def _replaced(self, old: DiscountCurve, new: DiscountCurve) -> MarketContext:
        """A frozen copy with ``new`` substituted for ``old`` wherever it appears, by identity."""
        views: dict[str, YieldCurve] = {}
        for name, view in self._views().items():
            curve = view.zero_curve.replace(old, new)
            views[name] = view if curve is view.zero_curve else view.with_zero_curve(curve)
        return self._copy(views)

    def with_curve(self, key: str | YieldCurve, curve: DiscountCurve | YieldCurve | None = None) -> MarketContext:
        """A frozen copy in which ``key`` holds ``curve``.

        ``with_curve(yield_curve)`` alone registers it under its own key.

        When ``key`` owns the object it held (a projection key over a discount key, then the
        first registered), that object *moved*: every other key holding it follows, and every
        composite built on it is rebuilt on ``curve``.
        When ``key`` is an alias (``USD.STDCSA`` on the SOFR object) or new, only ``key`` changes.
        Fixings under ``key`` are replaced only when ``curve`` is a ``YieldCurve`` carrying some.

        Raises
        ------
        ValueError
            If ``curve`` is built on the curve ``key`` holds: that is a redefinition, which is an
            assembly step (``market[key] = ...``), not a scenario.
        """
        if curve is None:
            if not isinstance(key, YieldCurve):
                raise TypeError("with_curve(key, curve), or with_curve(yield_curve) to use the curve's own key")
            key, curve = key.name, key
        name = self._key(key)
        view = self._as_view(name, curve)
        if view.origin != self._origin:
            raise ValueError(f"'{name}': YieldCurve origin {view.origin} is not the market's as_of {self._origin}")
        self._conventions.validate(name)
        views = self._views()
        previous = views.get(name)
        if previous is not None and view.historical_fixings is None and previous.historical_fixings is not None:
            view = view.with_historical_fixings(previous.historical_fixings)
        if previous is None or previous.zero_curve is view.zero_curve:
            views[name] = view
            return self._copy(views)
        if _contains(view.zero_curve, previous.zero_curve):
            raise ValueError(
                f"'{name}' would be redefined in terms of the curve it holds; that is an assembly step "
                "(market[key] = ...), not a scenario"
            )
        if self._object_owner(previous.zero_curve) == name:
            out = self._replaced(previous.zero_curve, view.zero_curve)
            moved = out._views()
            moved[name] = view
            return out._copy(moved)
        views[name] = view
        return self._copy(views)

    def with_fixings(self, key: str, fixings: HistoricalFixings | None) -> MarketContext:
        """A frozen copy with ``key``'s historical prints replaced (``None`` removes them)."""
        name = self._key(key)
        views = self._views()
        if name not in views:
            raise KeyError(f"no curve registered under '{key}'")
        views[name] = views[name].with_historical_fixings(fixings)
        return self._copy(views)

    def with_params(self, key: str, params: FloatArray) -> MarketContext:
        """A frozen copy with the parameters ``key`` owns replaced by ``params``.

        Parameters
        ----------
        key : str
        params : array_like of float64
            Laid out over :meth:`owned` in parameter order; this is what a calibrator writes.

        Returns
        -------
        MarketContext
            The re-parameterised atoms are replaced wherever they appear.

        Raises
        ------
        ValueError
            If ``key`` owns no parameters, or ``params`` has the wrong length.
        """
        name = self._key(key)
        atoms = self.owned(name)
        if not atoms:
            raise ValueError(f"'{name}' owns no parameters: every atom it reaches belongs to another key")
        sizes = [atom.n_params for atom in atoms]
        total = int(sum(sizes))
        values = np.asarray(params, dtype=np.float64)
        if values.shape != (total,):
            raise ValueError(f"'{name}' owns {total} parameters; got shape {values.shape}")
        out = self
        start = 0
        for atom, size in zip(atoms, sizes):
            out = out._replaced(atom, atom.with_params(values[start : start + size]))
            start += size
        return out

    def bump(self, key: str, amount: float | FloatArray) -> MarketContext:
        """A frozen copy with the parameters ``key`` owns shifted by ``amount``.

        Parameters
        ----------
        key : str
        amount : float or array_like of float64
            A scalar moves every owned parameter by it; a vector is laid out over
            :meth:`owned` in parameter order (one entry per pillar zero rate, or per spread).

        Returns
        -------
        MarketContext
            The moved atoms are replaced wherever they appear, so every curve built on them
            follows and nothing else changes.

        Raises
        ------
        ValueError
            If ``key`` owns no parameters, or a vector has the wrong length.
        """
        name = self._key(key)
        atoms = self.owned(name)
        if not atoms:
            raise ValueError(f"'{name}' owns no parameters: every atom it reaches belongs to another key")
        current = np.concatenate([atom.params for atom in atoms])
        shift = np.asarray(amount, dtype=np.float64)
        if shift.ndim == 0:
            shift = np.full(current.size, float(shift))
        elif shift.shape != current.shape:
            raise ValueError(f"'{name}' owns {current.size} parameters; got a bump of shape {shift.shape}")
        return self.with_params(name, current + shift)

    def freeze(self) -> MarketContext:
        """Make this market read-only (in place) and return it."""
        if not self._frozen:
            for group in self._groups.values():
                group._freeze()
            self._frozen = True
        return self

    # -- diagnostics ----------------------------------------------------------------------

    def to_mermaid(self) -> str:
        """The registered curves as a Mermaid flowchart.

        Registered keys are boxes; an alias key points (dotted) at the key owning its object;
        a composite points at its components with the component label and weight on the edge;
        unregistered curves are rounded nodes labelled by the component label that reaches them.
        """
        curves = self._curves
        lines = ["flowchart LR"]
        node_of: dict[int, str] = {}
        for index, name in enumerate(curves):
            node_of.setdefault(id(curves[name]), f"K{index}")
        for index, (name, curve) in enumerate(curves.items()):
            lines.append(f'    K{index}["{name}"]')
            if node_of[id(curve)] != f"K{index}":
                lines.append(f"    K{index} -.->|same object| {node_of[id(curve)]}")
        drawn: set[int] = set()
        counter = [0]

        def node(curve: DiscountCurve, label: str) -> str:
            ident = node_of.get(id(curve))
            if ident is None:
                ident = f"L{counter[0]}"
                counter[0] += 1
                node_of[id(curve)] = ident
                lines.append(f'    {ident}(["{label}: {type(curve).__name__}, {curve.n_params} params"])')
            return ident

        def edges(curve: DiscountCurve, parent: str) -> None:
            if id(curve) in drawn:
                return
            drawn.add(id(curve))
            parts = _components(curve)
            labels = getattr(curve, "labels", None) or tuple(str(i) for i in range(len(parts)))
            weights = getattr(curve, "weights", (1.0,) * len(parts))
            for label, weight, part in zip(labels, weights, parts):
                child = node(part, label)
                lines.append(f"    {parent} -->|{label} w={weight:g}| {child}")
                edges(part, child)

        for name, curve in curves.items():
            if node_of[id(curve)] == f"K{list(curves).index(name)}":
                edges(curve, node_of[id(curve)])
        return "\n".join(lines)

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "mutable"
        return f"MarketContext(as_of={self._as_of}, curves={list(self)!r}, {state})"
