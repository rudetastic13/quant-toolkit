"""C++-backed schedule generation via the finance._core.datemath extension.

Epoch contract: the extension speaks days since 1970-01-01 (numpy datetime64[D]
epoch), matching Date.toordinal(). Rich input types are converted once, here at
the boundary: date-like objects (Date, datetime.date) convert through their
year/month/day components with datemath.days_from_ymd, so no Python-side civil
math or ordinal offset is involved (datetime.date.toordinal() is 0001-01-01
based, off by _UNIX_EPOCH_PYORDINAL); raw ints are trusted to be 1970-based by
contract.
"""
import datetime

import numpy as np

from finance._core import datemath, is_available
from finance.dates.date import Date
from finance.dates.enums import Frequency, Roll, Direction

_UNIX_EPOCH_PYORDINAL = 719_163  # datetime.date(1970, 1, 1).toordinal()
_FREQ_INT = {frequency: frequency.int_based_mapping() for frequency in Frequency}
_DT64_D = np.dtype("datetime64[D]")

DateLike = Date | datetime.date | int | np.datetime64


def _to_ordinal(value: DateLike) -> int:
    if isinstance(value, (Date, datetime.date)):
        return datemath.days_from_ymd(value.year, value.month, value.day)
    if isinstance(value, np.datetime64):
        return int(value.view(np.int64))
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"Cannot interpret {type(value).__name__} as a date")


def generate_schedule_cpp(
    start_date: DateLike,
    end_date: DateLike,
    frequency: Frequency,
    first_regular_date: DateLike | None = None,
    last_regular_date: DateLike | None = None,
    roll_convention: Roll = Roll.Empty,
    direction: Direction = Direction.Forward,
) -> np.ndarray:
    """Generate a schedule with the C++ core, mirroring generate_schedule"""
    if not is_available():
        raise RuntimeError("finance._core not built; run bin/build_core.sh")
    start_int = _to_ordinal(start_date)
    end_int = _to_ordinal(end_date)
    first_int = _to_ordinal(first_regular_date) if first_regular_date is not None else start_int
    last_int = _to_ordinal(last_regular_date) if last_regular_date is not None else end_int
    freq_type, freq_value = _FREQ_INT[frequency]
    # Roll/Direction are IntEnums, passed to the binding as plain ints
    res = datemath.generate_schedule(
        start_int, end_int, freq_type, freq_value, first_int, last_int, roll_convention, direction
    )
    return res.view(_DT64_D)
