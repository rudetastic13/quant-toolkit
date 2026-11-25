"""``QuantToolkit.Utilities.*`` worksheet functions (xlOil): build Python containers from ranges.

Each returns a handle (``List;;;03be77``, ``Dict;;;5a1c9e``). Elements that are themselves
handles are replaced by their objects, so ``ArrayToList`` over a range of ``ArrayToDict`` cells
is a list of dicts. A date must be tagged with ``TagDate`` first, or it arrives as a number.
"""
from __future__ import annotations

import numpy as np

from quant_toolkit.finance.dates import Date

from quant_toolkit.xl.core import handles
from quant_toolkit.xl.core.register import xl_func


@xl_func("Utilities")
def ArrayToList(Values):  # noqa: N802, N803
    """Pack a range into a Python list. Returns a List handle.

    Parameters
    ----------
    Values : range
        The cells to pack, read row by row. A cell holding a handle becomes that object; a cell
        holding a TagDate value becomes a date.

    Returns
    -------
    str
        A List handle ("List;;;03be77"), accepted wherever a function takes a range of handles.

    Examples
    --------
    ``=QuantToolkit.Utilities.ArrayToList(E10:E17)`` packs eight swap handles into one cell.
    """
    return handles.store("List", handles.flatten(Values), Values)


@xl_func("Utilities")
def ArrayToDict(Keys, Values):  # noqa: N802, N803
    """Pack two ranges into a Python dict. Returns a Dict handle.

    Parameters
    ----------
    Keys : range
        The keys, read row by row; each is used as text.
    Values : range
        The values, the same number of cells as Keys. Handles become their objects and TagDate
        values become dates.

    Returns
    -------
    str
        A Dict handle ("Dict;;;5a1c9e").

    Raises
    ------
    ValueError
        If the two ranges hold a different number of cells.
    """
    keys, values = handles.flatten(Keys), handles.flatten(Values)
    if len(keys) != len(values):
        raise ValueError(f"{len(keys)} keys but {len(values)} values")
    return handles.store("Dict", dict(zip(map(str, keys), values)), Keys, Values)


@xl_func("Utilities")
def TagDate(Dates):  # noqa: N802, N803
    """Tag a date so it stays a date inside a list or dict.

    Parameters
    ----------
    Dates : date or range
        An Excel date, or a range of them.

    Returns
    -------
    str or range of str
        "Date;;;2026-06-01" for each date, in the shape given.

    Notes
    -----
    Excel hands a date over as a plain number, so inside ArrayToList or ArrayToDict it cannot
    be told apart from one. The tag carries the date as text; the container functions turn it
    back into a date.
    """
    if isinstance(Dates, np.ndarray):
        out = np.empty(Dates.shape, dtype=object)
        out.flat = [handles.tag_date(Date.from_excel(v)) for v in Dates.ravel()]
        return out
    return handles.tag_date(Date.from_excel(Dates))


@xl_func("Utilities")
def Version():  # noqa: N802
    """Installed quant-toolkit-xl version; a quick check that the add-in loaded.

    Returns
    -------
    str
        For example "quant-toolkit-xl 0.1.0".
    """
    from importlib.metadata import version

    return f"quant-toolkit-xl {version('quant-toolkit-xl')}"
