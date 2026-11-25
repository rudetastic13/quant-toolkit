"""Cell value -> library type helpers shared by the worksheet modules."""
from __future__ import annotations

from quant_toolkit.finance.dates import Date, Term


def enum_member(cls, value):
    """An enum member from a cell value.

    Parameters
    ----------
    cls : type[Enum]
    value : Enum, str, int or float
        A member (returned as is), a member name (``"ModifiedFollowing"``), a member value
        (``"MF"``), or a number taken as an integer value (a roll day).

    Returns
    -------
    Enum

    Raises
    ------
    ValueError
        If ``value`` matches neither a name nor a value.
    """
    if isinstance(value, cls):
        return value
    if isinstance(value, str):
        try:
            return cls[value.strip()]
        except KeyError:
            return cls(value.strip())
    return cls(int(value))


def given(value) -> bool:
    """Whether an optional worksheet argument was supplied.

    Parameters
    ----------
    value : object
        What xlOil passed: ``None`` for an omitted argument or an empty cell.

    Returns
    -------
    bool
        False for ``None`` and for a blank string, in which case the caller keeps the convention's value.
    """
    return value is not None and not (isinstance(value, str) and not value.strip())


def term(value) -> Term:
    """A ``Term`` from a cell value.

    Parameters
    ----------
    value : str, int or float
        A term string such as ``"2BD"`` or ``"3M"``; a bare number is taken as calendar days.

    Returns
    -------
    Term
    """
    return Term.from_str(value if isinstance(value, str) else f"{int(value)}D")


def date_or_term(value) -> Date | Term:
    """A dating cell as a ``Date`` or a ``Term``.

    Parameters
    ----------
    value : float or str
        An Excel date serial, or a term string such as ``"2BD"`` or ``"5Y"``.

    Returns
    -------
    Date or Term
        A ``Term`` for a string, a ``Date`` for a number.
    """
    return Term.from_str(value) if isinstance(value, str) else Date.from_excel(value)
