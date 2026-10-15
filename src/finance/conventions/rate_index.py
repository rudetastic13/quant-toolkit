"""Rate index definitions — the anchor conventions every product hangs off.

An :class:`IndexFamily` describes an index independently of any product traded on it: its
accrual basis, where fixings come from, how they apply, and the tenors it publishes.  An
overnight family publishes the single tenor :data:`~finance.conventions.keys.OVERNIGHT`; a
term family (CME Term SOFR, an IBOR) lists several.  A :class:`RateIndex` is one family at
one tenor and is what a projection curve key names: ``USD.SOFR.ON``, ``USD.CME-TERM-SOFR.3M``.

Product conventions (``SwapConventions``, ``DepositConventions``, ...) reference the family
for anything index-derived — most importantly the day count, which is the *projection* basis
the curve kernels use to turn discount factors into forward rates.  A float leg's accrual day
count defaults to the index's, so accrual and projection align by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from finance.conventions.keys import OVERNIGHT, CurveKey, parse_tenor, tenor_label
from finance.dates import Term
from finance.dates.enums import DayCountMethod, FixingType


def _same(a: Term, b: Term) -> bool:
    return (int(a.term_length), a.term_type) == (int(b.term_length), b.term_type)


@dataclass(frozen=True, kw_only=True)
class IndexFamily:
    """An index (SOFR, FEDFUND, CME-TERM-SOFR, ...) and its intrinsic conventions.

    Parameters
    ----------
    currency, name : str
        Upper-cased; together with a tenor they form the projection curve key.
    day_count_method : DayCountMethod
        Projection basis.
    fixing_calendar : str
    fixing_type : FixingType
        ``Arrears`` for compounded RFR legs, ``Advance`` for term indices.
    publication_lag : Term or None
        Informational until the kernels consume it.
    tenors : tuple of Term
        The tenors this family publishes; ``(OVERNIGHT,)`` for an overnight index.
    """

    currency: str
    name: str
    day_count_method: DayCountMethod
    fixing_calendar: str
    fixing_type: FixingType = FixingType.Arrears
    publication_lag: Term | None = None
    tenors: tuple[Term, ...] = (OVERNIGHT,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", str(self.currency).strip().upper())
        object.__setattr__(self, "name", str(self.name).strip().upper())
        tenors = tuple(parse_tenor(t) for t in self.tenors)
        if not tenors:
            raise ValueError(f"index family {self.currency}.{self.name} must list at least one tenor")
        labels = [tenor_label(t) for t in tenors]
        if len(set(labels)) != len(labels):
            raise ValueError(f"index family {self.currency}.{self.name} lists a tenor twice: {labels}")
        object.__setattr__(self, "tenors", tenors)

    @property
    def is_overnight(self) -> bool:
        """Whether the family publishes only the overnight tenor."""
        return len(self.tenors) == 1 and _same(self.tenors[0], OVERNIGHT)

    @property
    def tenor_labels(self) -> tuple[str, ...]:
        return tuple(tenor_label(t) for t in self.tenors)

    def __repr__(self) -> str:
        return (
            f"IndexFamily('{self.currency}.{self.name}', tenors={self.tenor_labels!r}, "
            f"day_count={self.day_count_method.name}, fixing_calendar={self.fixing_calendar!r}, "
            f"fixing_type={self.fixing_type.name})"
        )

    def has_tenor(self, tenor: Term | str) -> bool:
        wanted = parse_tenor(tenor)
        return any(_same(wanted, t) for t in self.tenors)

    def index(self, tenor: Term | str | None = None) -> RateIndex:
        """The family at one tenor.

        Parameters
        ----------
        tenor : Term, str or None
            ``None`` is allowed only when the family publishes a single tenor.

        Raises
        ------
        KeyError
            If the family does not publish ``tenor``, or ``tenor`` is omitted for a family
            that publishes several.
        """
        if tenor is None:
            if len(self.tenors) != 1:
                raise KeyError(
                    f"{self.currency}.{self.name} publishes tenors {list(self.tenor_labels)}; "
                    "name one, e.g. "
                    f"'{self.name}.{self.tenor_labels[0]}'"
                )
            return RateIndex(family=self, tenor=self.tenors[0])
        wanted = parse_tenor(tenor)
        for t in self.tenors:
            if _same(wanted, t):
                return RateIndex(family=self, tenor=t)
        raise KeyError(
            f"{self.currency}.{self.name} does not publish a {tenor_label(wanted)} tenor "
            f"(available: {list(self.tenor_labels)})"
        )


@dataclass(frozen=True, kw_only=True)
class RateIndex:
    """One tenor of an :class:`IndexFamily`: what a projection curve key names.

    Parameters
    ----------
    family : IndexFamily
    tenor : Term
        Must be one the family publishes.
    """

    family: IndexFamily
    tenor: Term

    def __post_init__(self) -> None:
        if not self.family.has_tenor(self.tenor):
            raise ValueError(
                f"{self.family.currency}.{self.family.name} does not publish a {tenor_label(self.tenor)} tenor"
            )

    # -- family pass-through ------------------------------------------------------------

    @property
    def currency(self) -> str:
        return self.family.currency

    @property
    def name(self) -> str:
        return self.family.name

    @property
    def day_count_method(self) -> DayCountMethod:
        return self.family.day_count_method

    @property
    def fixing_calendar(self) -> str:
        return self.family.fixing_calendar

    @property
    def fixing_type(self) -> FixingType:
        return self.family.fixing_type

    @property
    def publication_lag(self) -> Term | None:
        return self.family.publication_lag

    # -- identity -----------------------------------------------------------------------

    @property
    def is_overnight(self) -> bool:
        return _same(self.tenor, OVERNIGHT)

    @property
    def key(self) -> CurveKey:
        """The projection curve key, e.g. ``CurveKey('USD.SOFR.ON')``."""
        return CurveKey(self.currency, self.name, self.tenor)

    @property
    def label(self) -> str:
        """The label a floating leg carries in ``rate_index``: the key string, ``'USD.SOFR.ON'``."""
        return str(self.key)

    def __repr__(self) -> str:
        return f"RateIndex({self.label!r})"
