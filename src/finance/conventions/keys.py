"""Curve keys: the one grammar every curve in a market is registered and looked up under.

The slot count carries the kind of curve:

=====================  ==========  ==============================================================
form                   kind        examples
=====================  ==========  ==============================================================
``CCY.INDEX.TENOR``    projection  ``USD.SOFR.ON``, ``USD.FEDFUND.ON``, ``USD.CME-TERM-SOFR.3M``
``CCY.NAME``           discount    ``USD.STDCSA``, ``GBP.USDSTDCSA``, ``USD.UST``
=====================  ==========  ==============================================================

A projection key names what a floating leg resets off; its tenor is always present and ``ON``
marks an overnight index.  A discount key names a funding or collateral regime or an issuer:
nothing resets off it, so it carries no tenor.  Dots separate slots; hyphens are allowed inside
a slot.  This is Strata's discount-map / forward-map split written as keys.

``ON`` is spelled only here: the parser maps it to :data:`OVERNIGHT` (``Term(1, Days)``) and
``str()`` maps it back, so ``Term`` itself is untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from finance.dates import Term
from finance.dates.term.term_type import TermType

OVERNIGHT: Term = Term(1, TermType.Days)
"""The tenor of an overnight index; printed as ``ON`` inside a curve key."""

_ON = "ON"
_SLOT = re.compile(r"^[A-Z0-9][A-Z0-9-]*$")


class CurveKind(Enum):
    """What a curve key names: a projection (forward) curve or a discount curve."""

    Projection = "projection"
    Discount = "discount"


def _tenor_info(tenor: Term | None) -> tuple[int, TermType] | None:
    return None if tenor is None else (int(tenor.term_length), tenor.term_type)


def tenor_label(tenor: Term) -> str:
    """``ON`` for the overnight tenor, otherwise ``"3M"``-style text.

    Parameters
    ----------
    tenor : Term

    Returns
    -------
    str
    """
    if _tenor_info(tenor) == _tenor_info(OVERNIGHT):
        return _ON
    return f"{int(tenor.term_length)}{tenor.term_type.value}"


def parse_tenor(label: str | Term) -> Term:
    """``"ON"`` → :data:`OVERNIGHT`; anything else through ``Term.from_str``.

    Parameters
    ----------
    label : str or Term
        A ``Term`` is returned unchanged.

    Returns
    -------
    Term

    Raises
    ------
    ValueError
        If the label is not a positive tenor.
    """
    if isinstance(label, Term):
        tenor = label
    elif label.strip().upper() == _ON:
        return OVERNIGHT
    else:
        try:
            tenor = Term.from_str(label.strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"'{label}' is not a tenor (expected 'ON', '1M', '3M', ...)") from exc
    if int(tenor.term_length) <= 0:
        raise ValueError(f"a curve tenor must be positive, got {label!r}")
    return tenor


@dataclass(frozen=True, eq=False)
class CurveKey:
    """A curve's registered name, parsed: ``currency``, ``name`` and, for a projection, ``tenor``.

    Parameters
    ----------
    currency : str
        ISO code, upper-cased.
    name : str
        Index family (projection) or funding regime / issuer (discount), upper-cased; letters,
        digits and hyphens.
    tenor : Term or None
        Present for a projection key (:data:`OVERNIGHT` for an overnight index), absent for a
        discount key.

    Notes
    -----
    Equality and hashing are by value, so a key can be a dictionary key even though
    ``Term`` itself cannot.
    """

    currency: str
    name: str
    tenor: Term | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", str(self.currency).strip().upper())
        object.__setattr__(self, "name", str(self.name).strip().upper())
        for slot, value in (("currency", self.currency), ("name", self.name)):
            if not _SLOT.match(value):
                raise ValueError(f"curve key {slot} {value!r} must be letters, digits and hyphens")
        if self.tenor is not None:
            if not isinstance(self.tenor, Term):
                raise TypeError(f"tenor must be a Term or None, got {type(self.tenor).__name__}")
            if int(self.tenor.term_length) <= 0:
                raise ValueError(f"a curve tenor must be positive, got {self.tenor!r}")

    # -- construction -------------------------------------------------------------------

    @classmethod
    def parse(cls, text: str | CurveKey) -> CurveKey:
        """``"USD.SOFR.ON"`` → projection key; ``"USD.STDCSA"`` → discount key.

        Parameters
        ----------
        text : str or CurveKey
            A key is returned unchanged.

        Returns
        -------
        CurveKey

        Raises
        ------
        ValueError
            On anything but two or three dot-separated slots, or a bad tenor.
        """
        if isinstance(text, CurveKey):
            return text
        slots = str(text).strip().upper().split(".")
        if len(slots) == 2:
            return cls(slots[0], slots[1])
        if len(slots) == 3:
            return cls(slots[0], slots[1], parse_tenor(slots[2]))
        raise ValueError(
            f"curve key {text!r} must be 'CCY.INDEX.TENOR' (projection, 'ON' for overnight) or 'CCY.NAME' (discount)"
        )

    @classmethod
    def projection(cls, currency: str, name: str, tenor: Term | str = OVERNIGHT) -> CurveKey:
        """Projection key for an index; ``tenor`` defaults to overnight."""
        return cls(currency, name, parse_tenor(tenor))

    @classmethod
    def discount(cls, currency: str, name: str) -> CurveKey:
        """Discount key for a funding regime or issuer."""
        return cls(currency, name)

    # -- identity -----------------------------------------------------------------------

    @property
    def kind(self) -> CurveKind:
        return CurveKind.Projection if self.tenor is not None else CurveKind.Discount

    @property
    def is_projection(self) -> bool:
        return self.tenor is not None

    @property
    def is_discount(self) -> bool:
        return self.tenor is None

    @property
    def is_overnight(self) -> bool:
        """Whether this is a projection key on an overnight index."""
        return self.tenor is not None and _tenor_info(self.tenor) == _tenor_info(OVERNIGHT)

    @property
    def tenor_label(self) -> str | None:
        """``"ON"``, ``"3M"``, ... or ``None`` for a discount key."""
        return None if self.tenor is None else tenor_label(self.tenor)

    def _value(self) -> tuple:
        return (self.currency, self.name, _tenor_info(self.tenor))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CurveKey):
            return NotImplemented
        return self._value() == other._value()

    def __hash__(self) -> int:
        return hash(self._value())

    def __str__(self) -> str:
        if self.tenor is None:
            return f"{self.currency}.{self.name}"
        return f"{self.currency}.{self.name}.{tenor_label(self.tenor)}"

    def __repr__(self) -> str:
        return f"CurveKey({str(self)!r})"
