"""MarketContext — the curves a market knows, by key, plus vols, fixings and conventions.

A market is a ``Mapping[str, DiscountCurve]``: ``market["USD.SOFR.ON"]`` is the mathematical
curve registered under that key and ``market.yield_curve("USD.SOFR.ON")`` the date-aware view
the engines read (origin = ``as_of``, the key's conventions, the key's fixings).  Keys follow
the grammar of :class:`finance.conventions.CurveKey`: ``CCY.INDEX.TENOR`` projects, ``CCY.NAME``
discounts, and a market registers every key its trades name — discounting at SOFR is the SOFR
object bound under ``USD.STDCSA`` as well.

Assembly is by assignment and reads like finance::

    market = MarketContext(as_of)
    market["USD.SOFR.ON"] = sofr                      # a ZeroCurve, or a YieldCurve built with dates
    market["USD.STDCSA"] = sofr                       # discount at SOFR: one object, two keys
    market["USD.FEDFUND.ON"] = market["USD.SOFR.ON"] - 5 * bp

Sharing is by object identity, declared by construction: writing ``sofr - 5 * bp`` already says
the Fed Funds curve is built on ``sofr``.  Scenarios propagate through that structure:
``with_curve`` and ``bump`` return new markets in which the moved object is replaced wherever it
appears, so "bump SOFR" moves Fed Funds and leaves the basis alone.  A registered key owns the
parameters nobody else owns (:meth:`MarketContext.leaf_owner`); ``bump(key, ...)`` moves exactly
those.

Projection and par-rate logic deliberately lives in
:class:`finance.markets.rate_generator.RateGenerator`, not here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from types import MappingProxyType
from typing import Any

import numpy as np

from finance.conventions import ConventionRegistry, CurveKey, default_registry
from finance.dates import Date
from finance.markets.curves import DiscountCurve, YieldCurve
from finance.markets.fixings import HistoricalFixings

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
    fixings : Mapping[str, HistoricalFixings], optional
        Historical prints by curve key; a ``YieldCurve`` assigned with fixings adds its own.

    Notes
    -----
    A market is mutable while it is being assembled and :meth:`freeze` makes it read-only.
    :meth:`with_curve`, :meth:`bump` and :meth:`with_fixings` return frozen copies; the market
    they were called on is never changed.  ``market[key]`` is the registered ``DiscountCurve``;
    :meth:`yield_curve` is the derived, cached, date-aware view.

    Examples
    --------
    >>> market = MarketContext(Date(2026, 6, 1))                              # doctest: +SKIP
    >>> market["USD.SOFR.ON"] = sofr                                          # doctest: +SKIP
    >>> market["USD.STDCSA"] = sofr                                           # doctest: +SKIP
    >>> market["USD.FEDFUND.ON"] = market["USD.SOFR.ON"] - 5 * bp              # doctest: +SKIP
    >>> market.leaf_owner()                                                   # doctest: +SKIP
    {<id of sofr>: 'USD.SOFR.ON', <id of the spread>: 'USD.FEDFUND.ON'}
    >>> up = market.bump("USD.SOFR.ON", 1e-4)      # Fed Funds follows, the spread does not  # doctest: +SKIP
    """

    __slots__ = ("_as_of", "_origin", "_curves", "_fixings", "_conventions", "_vols", "_yield_cache", "_frozen")

    def __init__(
        self,
        as_of: Date | None = None,
        curves: Mapping[str, DiscountCurve | YieldCurve] | None = None,
        *,
        conventions: ConventionRegistry = default_registry,
        vols: Any | None = None,
        fixings: Mapping[str, HistoricalFixings] | None = None,
    ) -> None:
        if as_of is None:
            raise ValueError("MarketContext needs a valuation date: as_of")
        self._as_of = as_of
        self._origin = np.datetime64(as_of.to_numpy(), "D")
        self._curves: dict[str, DiscountCurve] | MappingProxyType = {}
        self._fixings: dict[str, HistoricalFixings] | MappingProxyType = {}
        self._conventions = conventions
        self._vols = vols
        self._yield_cache: dict[str, tuple[int, int, YieldCurve]] = {}
        self._frozen = False
        if fixings:
            for key, history in fixings.items():
                self._fixings[self._key(key)] = history
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
    def fixings(self) -> MutableMapping[str, HistoricalFixings] | Mapping[str, HistoricalFixings]:
        """Historical prints by curve key; read-only once the market is frozen."""
        return self._fixings

    @property
    def frozen(self) -> bool:
        return self._frozen

    @staticmethod
    def _key(key: CurveKey | str) -> str:
        return str(CurveKey.parse(key))

    # -- Mapping ----------------------------------------------------------------------------

    def __getitem__(self, key: str) -> DiscountCurve:
        curve = self._curves.get(key)
        if curve is None:
            curve = self._curves.get(self._key(key))
            if curve is None:
                raise KeyError(f"no curve registered under '{key}'")
        return curve

    def __iter__(self) -> Iterator[str]:
        return iter(self._curves)

    def __len__(self) -> int:
        return len(self._curves)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, (str, CurveKey)):
            return False
        try:
            return self._key(key) in self._curves
        except ValueError:
            return False

    def __setitem__(self, key: str, curve: DiscountCurve | YieldCurve) -> None:
        """Register ``curve`` under ``key`` (assembly).

        Parameters
        ----------
        key : str
            ``CCY.INDEX.TENOR`` (its family and tenor must be in ``conventions``) or ``CCY.NAME``.
        curve : DiscountCurve or YieldCurve
            A ``YieldCurve`` must have this market's origin; its discount curve is registered and,
            when ``key`` is the curve's own key, its fixings are added under it.

        Raises
        ------
        TypeError
            If the market is frozen, or ``curve`` is neither.
        ValueError
            If ``key`` is already registered and ``curve`` is built on the curve it replaces
            (that is a redefinition, not a scenario: build the new curve from the atoms), or if
            the new registration makes a leaf's ownership ambiguous (see :meth:`leaf_owner`).
        """
        if self._frozen:
            raise TypeError("this MarketContext is frozen; use with_curve() / bump() for scenario copies")
        parsed = CurveKey.parse(key)
        name = str(parsed)
        if parsed.is_projection:  # the family and tenor must be known before anything is stored
            self._conventions.get(parsed.currency, parsed.name).index(parsed.tenor)
        curve, history = self._coerce(name, curve)
        previous = self._curves.get(name)
        if previous is not None and previous is not curve and _contains(curve, previous):
            raise ValueError(
                f"'{name}' would be redefined in terms of the curve it replaces; assign a curve built from "
                "the atoms, or use with_curve() / bump() to move it"
            )
        self._curves[name] = curve
        if history is not None:
            self._fixings[name] = history
        try:
            self.leaf_owner()
        except ValueError:
            if previous is None:
                del self._curves[name]
            else:
                self._curves[name] = previous
            raise
        self._yield_cache.pop(name, None)

    def _coerce(self, name: str, curve: DiscountCurve | YieldCurve) -> tuple[DiscountCurve, HistoricalFixings | None]:
        if isinstance(curve, YieldCurve):
            if curve.origin != self._origin:
                raise ValueError(f"'{name}': YieldCurve origin {curve.origin} is not the market's as_of {self._origin}")
            # fixings belong to the index the curve was built for, not to an alias (``USD.STDCSA``)
            return curve.zero_curve, curve.historical_fixings if curve.name == name else None
        if isinstance(curve, DiscountCurve):
            return curve, None
        raise TypeError(f"'{name}': expected a DiscountCurve or YieldCurve, got {type(curve).__name__}")

    # -- derived views ------------------------------------------------------------------

    def yield_curve(self, key: str) -> YieldCurve:
        """The date-aware view of ``market[key]``: origin, the key's conventions and fixings.

        Cached per ``(key, curve object, fixings object)``; the same object comes back until
        either changes.
        """
        name = key if key in self._curves else self._key(key)
        curve = self[name]
        history = self._fixings.get(name)
        cached = self._yield_cache.get(name)
        if cached is not None and cached[0] == id(curve) and cached[1] == id(history):
            return cached[2]
        view = YieldCurve.from_registry(
            self._origin, curve, key=name, registry=self._conventions, historical_fixings=history
        )
        self._yield_cache[name] = (id(curve), id(history), view)
        return view

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

    def _copy(self, curves: Mapping[str, DiscountCurve], fixings: Mapping[str, HistoricalFixings]) -> MarketContext:
        out = MarketContext.__new__(MarketContext)
        out._as_of = self._as_of
        out._origin = self._origin
        out._conventions = self._conventions
        out._vols = self._vols
        out._curves = MappingProxyType(dict(curves))
        out._fixings = MappingProxyType(dict(fixings))
        out._frozen = True
        out._yield_cache = {
            name: entry
            for name, entry in self._yield_cache.items()
            if name in curves and entry[0] == id(curves[name]) and entry[1] == id(fixings.get(name))
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
        return self._copy({name: curve.replace(old, new) for name, curve in self._curves.items()}, self._fixings)

    def with_curve(self, key: str | YieldCurve, curve: DiscountCurve | YieldCurve | None = None) -> MarketContext:
        """A frozen copy in which ``key`` holds ``curve``.

        ``with_curve(yield_curve)`` alone registers it under its own key.

        When ``key`` owns the object it held (a projection key over a discount key, then the
        first registered), that object *moved*: every other key holding it follows, and every
        composite built on it is rebuilt on ``curve``.
        When ``key`` is an alias (``USD.STDCSA`` on the SOFR object) or new, only ``key`` changes.

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
        curve, history = self._coerce(name, curve)
        fixings = dict(self._fixings)
        if history is not None:
            fixings[name] = history
        previous = self._curves.get(name)
        if previous is None or previous is curve:
            curves = dict(self._curves)
            curves[name] = curve
            return self._copy(curves, fixings)
        if _contains(curve, previous):
            raise ValueError(
                f"'{name}' would be redefined in terms of the curve it holds; that is an assembly step "
                "(market[key] = ...), not a scenario"
            )
        if self._object_owner(previous) == name:
            out = self._replaced(previous, curve)
            return out if history is None else out._copy(out._curves, fixings)
        curves = dict(self._curves)
        curves[name] = curve
        return self._copy(curves, fixings)

    def with_fixings(self, key: str, fixings: HistoricalFixings | None) -> MarketContext:
        """A frozen copy with ``key``'s historical prints replaced (``None`` removes them)."""
        name = self._key(key)
        if name not in self._curves:
            raise KeyError(f"no curve registered under '{key}'")
        history = dict(self._fixings)
        if fixings is None:
            history.pop(name, None)
        else:
            history[name] = fixings
        return self._copy(self._curves, history)

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
        sizes = [atom.n_params for atom in atoms]
        total = int(sum(sizes))
        shift = np.asarray(amount, dtype=np.float64)
        if shift.ndim == 0:
            shift = np.full(total, float(shift))
        elif shift.shape != (total,):
            raise ValueError(f"'{name}' owns {total} parameters; got a bump of shape {shift.shape}")
        out = self
        start = 0
        for atom, size in zip(atoms, sizes):
            out = out._replaced(atom, atom.with_params(atom.params + shift[start : start + size]))
            start += size
        return out

    def freeze(self) -> MarketContext:
        """Make this market read-only (in place) and return it."""
        if not self._frozen:
            self._curves = MappingProxyType(dict(self._curves))
            self._fixings = MappingProxyType(dict(self._fixings))
            self._frozen = True
        return self

    # -- diagnostics ----------------------------------------------------------------------

    def to_mermaid(self) -> str:
        """The registered curves as a Mermaid flowchart.

        Registered keys are boxes; an alias key points (dotted) at the key owning its object;
        a composite points at its components with the component label and weight on the edge;
        unregistered curves are rounded nodes labelled by the component label that reaches them.
        """
        lines = ["flowchart LR"]
        node_of: dict[int, str] = {}
        for index, name in enumerate(self._curves):
            node_of.setdefault(id(self._curves[name]), f"K{index}")
        for index, (name, curve) in enumerate(self._curves.items()):
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

        for name, curve in self._curves.items():
            if node_of[id(curve)] == f"K{list(self._curves).index(name)}":
                edges(curve, node_of[id(curve)])
        return "\n".join(lines)

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "mutable"
        return f"MarketContext(as_of={self._as_of}, curves={list(self._curves)!r}, {state})"
