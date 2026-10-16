"""ParamLayout — the flat parameter vector of a pricing program, one block per owning key.

The engines differentiate PV with respect to one flat vector ``z``.  This module decides what
``z`` is.  A market's curves are built from atoms (``ZeroCurve``) shared by identity: the Fed
Funds curve ``sofr - 5 * bp`` contains the SOFR atom, and so does ``USD.STDCSA``.  Each atom
appears in ``z`` exactly once, in the block of the key that owns it
(:meth:`finance.markets.context.MarketContext.leaf_owner`), so a move of SOFR is one move of
one block whichever curves it reaches, and the adjoints of every path that leads to the atom
land in the same columns.

A block is named after its owning key and holds that key's owned atoms in the parameter order
of the key's curve (:meth:`MarketContext.owned`).  ``ladder("USD.SOFR.ON")`` is then all SOFR
pillar risk however it arrived, and ``ladder("USD.FEDFUND.ON")`` is the basis alone.

For each program curve the layout also records the *column map* from that curve's own
``params`` (components concatenated, see ``CompositeCurve``) to columns of ``z``; a curve's
``log_df_affine`` Jacobian, which is over its own parameters, is scattered through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np

from finance.markets.curves import DiscountCurve

FloatArray = np.ndarray


@dataclass(frozen=True)
class ParamLayout:
    """Where every program curve's parameters live in the flat vector ``z``.

    Parameters
    ----------
    blocks : tuple of str
        Owning keys in block order.
    block_slices : dict[str, slice]
        Columns of ``z`` per block.
    block_atoms : dict[str, tuple[DiscountCurve, ...]]
        The atoms of each block, in column order; their ``params`` concatenated are the block.
    column_maps : dict[str, ndarray of intp]
        For each program curve name, ``column_maps[name][k]`` is the column of ``z`` holding
        the curve's own parameter ``k``.
    curve_blocks : dict[str, str | None]
        The single block a program curve resolves to (its own key when it owns anything,
        otherwise the key owning its atom), or ``None`` when its parameters span several blocks
        and none is its own.
    origin : datetime64[D]
        The market's as-of, so pillar offsets can be turned back into dates.
    signature : Hashable
        The geometry the layout was built on; equal signatures mean the same ``z``.
    """

    blocks: tuple[str, ...]
    block_slices: dict[str, slice]
    block_atoms: dict[str, tuple[DiscountCurve, ...]]
    column_maps: dict[str, np.ndarray]
    curve_blocks: dict[str, str | None]
    origin: np.datetime64
    signature: Hashable = field(repr=False)

    # -- construction -------------------------------------------------------------------

    @classmethod
    def build(cls, market, curve_names: Iterable[str]) -> ParamLayout:
        """Lay out the parameters every curve in ``curve_names`` reaches, by owning key.

        Parameters
        ----------
        market : MarketContext
        curve_names : iterable of str
            The keys a compiled program references (projection and discount roles).

        Returns
        -------
        ParamLayout

        Raises
        ------
        KeyError
            If a name is not registered in ``market``.
        ValueError
            If the market's ownership is ambiguous (see ``MarketContext.leaf_owner``).
        """
        names = tuple(curve_names)
        owner = market.leaf_owner()
        blocks: list[str] = []
        for name in names:
            for leaf in market[name].leaves():
                key = owner[id(leaf.curve)]
                if key not in blocks:
                    blocks.append(key)
        block_atoms = {key: market.owned(key) for key in blocks}
        block_slices: dict[str, slice] = {}
        atom_columns: dict[int, slice] = {}
        start = 0
        for key in blocks:
            for atom in block_atoms[key]:
                atom_columns[id(atom)] = slice(start, start + atom.n_params)
                start += atom.n_params
        # block slices span their atoms (contiguous by construction)
        cursor = 0
        for key in blocks:
            size = sum(atom.n_params for atom in block_atoms[key])
            block_slices[key] = slice(cursor, cursor + size)
            cursor += size

        column_maps: dict[str, np.ndarray] = {}
        curve_blocks: dict[str, str | None] = {}
        for name in names:
            curve = market[name]
            columns = np.empty(curve.n_params, dtype=np.intp)
            reached: list[str] = []
            for leaf in curve.leaves():
                target = atom_columns[id(leaf.curve)]
                columns[leaf.columns] = np.arange(target.start, target.stop)
                key = owner[id(leaf.curve)]
                if key not in reached:
                    reached.append(key)
            column_maps[name] = columns
            if name in block_slices:
                curve_blocks[name] = name
            elif len(reached) == 1:
                curve_blocks[name] = reached[0]
            else:
                curve_blocks[name] = None

        signature = cls._signature(market, names, owner)
        return cls(
            blocks=tuple(blocks),
            block_slices=block_slices,
            block_atoms=block_atoms,
            column_maps=column_maps,
            curve_blocks=curve_blocks,
            origin=np.datetime64(market.origin, "D"),
            signature=signature,
        )

    @staticmethod
    def _signature(market, names: tuple[str, ...], owner: dict[int, str]) -> Hashable:
        """Origin, and per program curve the owner, pillars, scheme, weight and columns of each leaf."""
        parts = []
        for name in names:
            leaves = []
            for leaf in market[name].leaves():
                leaves.append(
                    (
                        owner[id(leaf.curve)],
                        tuple(leaf.curve.x.tolist()),
                        leaf.curve.scheme,
                        leaf.weight,
                        leaf.columns.start,
                        leaf.columns.stop,
                    )
                )
            parts.append((name, tuple(leaves)))
        return (market.origin.astype(np.int64).item(), tuple(parts))

    # -- queries ------------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Length of ``z``."""
        return int(sum(sl.stop - sl.start for sl in self.block_slices.values()))

    def block_of(self, key: str) -> str:
        """The block ``key`` reports under.

        A block's own key, a program curve name resolving to one block (an alias such as
        ``USD.STDCSA`` reports under ``USD.SOFR.ON``), or ``KeyError`` naming the blocks when
        the key spans several and owns none of them.
        """
        if key in self.block_slices:
            return key
        block = self.curve_blocks.get(key)
        if block is None:
            if key in self.curve_blocks:
                raise KeyError(
                    f"'{key}' owns no parameters and spans several blocks; ask for one of {list(self.blocks)}"
                )
            raise KeyError(f"program has no curve named '{key}'; blocks are {list(self.blocks)}")
        return block

    def block_slice(self, key: str) -> slice:
        """Columns of ``z`` for :meth:`block_of` ``key``."""
        return self.block_slices[self.block_of(key)]

    def columns(self, keys: Iterable[str]) -> np.ndarray:
        """Columns of ``z`` for several blocks, in the order given."""
        return np.concatenate(
            [np.arange(self.block_slice(k).start, self.block_slice(k).stop) for k in keys]
            or [np.zeros(0, dtype=np.intp)]
        ).astype(np.intp)

    def params(self, market) -> FloatArray:
        """``z`` read off ``market``: each block's owned atoms' parameters, concatenated.

        The market must be :meth:`compatible`; the caller checks.
        """
        pieces = [atom.params for key in self.blocks for atom in market.owned(key)]
        return np.ascontiguousarray(np.concatenate(pieces) if pieces else np.zeros(0), dtype=np.float64)

    def compatible(self, market, curve_names: Iterable[str]) -> bool:
        """Whether ``market`` has the geometry this layout was built on (same origin, atoms, owners, weights)."""
        try:
            return self._signature(market, tuple(curve_names), market.leaf_owner()) == self.signature
        except (KeyError, ValueError):
            return False

    def pillar_dates(self, key: str) -> np.ndarray:
        """Non-origin pillar dates of :meth:`block_of` ``key``'s atoms, in column order (``datetime64[D]``)."""
        offsets = np.concatenate([atom.x[1:] for atom in self.block_atoms[self.block_of(key)]])
        return self.origin + offsets.astype(np.int64).astype("timedelta64[D]")

    def pillar_years(self, key: str) -> FloatArray:
        """Non-origin pillar offsets of :meth:`block_of` ``key``'s atoms in Act/365 years, column order."""
        return np.concatenate([atom.x[1:] for atom in self.block_atoms[self.block_of(key)]]) / 365.0
