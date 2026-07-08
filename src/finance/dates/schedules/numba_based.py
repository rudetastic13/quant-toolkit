"""Numba-jitted schedule generation, the @njit twin of the C++ kernel.

The kernel is self-contained (Hinnant civil math re-implemented as @njit
helpers, no lookup-table globals) so it stays structurally comparable to
finance/_core/cpp/civil.hpp. Day counts are days since 1970-01-01.
"""
import numpy as np

from finance.dates.date import Date
from finance.dates.enums import Frequency, Roll, Direction

try:
    from numba import njit
except ImportError:  # numba not installed; the pure-python/vectorized paths still work
    njit = None


def is_available() -> bool:
    return njit is not None


if njit is not None:
    _MARCH_EPOCH = 719_468

    @njit(cache=True)
    def _is_leap(y: int) -> bool:
        return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

    @njit(cache=True)
    def _days_in_month(y: int, m: int) -> int:
        if m == 2:
            return 29 if _is_leap(y) else 28
        if m == 4 or m == 6 or m == 9 or m == 11:
            return 30
        return 31

    @njit(cache=True)
    def _days_from_ymd(y: int, m: int, d: int) -> int:
        yy = y - (1 if m <= 2 else 0)
        era = yy // 400
        yoe = yy - era * 400
        mp = m + (9 if m <= 2 else -3)
        doy = (153 * mp + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        return era * 146_097 + doe - _MARCH_EPOCH

    @njit(cache=True)
    def _ymd_from_days(days: int) -> tuple[int, int, int]:
        z = days + _MARCH_EPOCH
        era = z // 146_097
        doe = z - era * 146_097
        yoe = (doe - doe // 1_460 + doe // 36_524 - doe // 146_096) // 365
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        d = doy - (153 * mp + 2) // 5 + 1
        m = mp + 3 if mp < 10 else mp - 9
        y = yoe + era * 400 + (1 if m <= 2 else 0)
        return y, m, d

    @njit(cache=True)
    def _month_step(month_index: int, delta: int, roll_day: int) -> int:
        mi = month_index + delta
        y = mi // 12
        m = mi % 12 + 1
        if 1 <= roll_day <= 28:
            d = roll_day
        elif roll_day == -1:
            d = _days_in_month(y, m)
        else:
            d = min(roll_day, _days_in_month(y, m))
        return _days_from_ymd(y, m, d)

    @njit(cache=True)
    def _schedule_kernel(
        start: int, end: int, freq_type: int, freq_value: int, first: int, last: int, roll: int, forward: bool
    ) -> np.ndarray:
        # capacity bound mirrors helpers.guess_array_size
        max_dates = end - start + 1
        if freq_type == 1:
            max_dates = (max_dates // 28) // freq_value
        else:
            max_dates = max_dates // freq_value
        out = np.empty(max_dates + 6, dtype=np.int64)

        anchor = first if forward else last
        step = freq_value if forward else -freq_value

        roll_day = roll
        anchor_mi = 0
        if freq_type == 1:
            y, m, d = _ymd_from_days(anchor)
            anchor_mi = y * 12 + m - 1
            if roll_day == 0:
                roll_day = -1 if d == _days_in_month(y, m) else d

        n = 0
        if forward:
            if start != first:
                out[n] = start
                n += 1
            out[n] = first
            n += 1
            if last != first:
                k = 1
                while True:
                    d = anchor + k * step if freq_type == 2 else _month_step(anchor_mi, k * step, roll_day)
                    if d >= last:
                        break
                    out[n] = d
                    n += 1
                    k += 1
                out[n] = last
                n += 1
            if end != last:
                out[n] = end
                n += 1
        else:
            # build descending, then reverse in place
            if end != last:
                out[n] = end
                n += 1
            out[n] = last
            n += 1
            if first != last:
                k = 1
                while True:
                    d = anchor + k * step if freq_type == 2 else _month_step(anchor_mi, k * step, roll_day)
                    if d <= first:
                        break
                    out[n] = d
                    n += 1
                    k += 1
                out[n] = first
                n += 1
            if start != first:
                out[n] = start
                n += 1
            out[:n] = out[:n][::-1].copy()
        return out[:n]


def generate_schedule_numba(
    start_date: Date,
    end_date: Date,
    frequency: Frequency,
    first_regular_date: Date | None = None,
    last_regular_date: Date | None = None,
    roll_convention: Roll = Roll.Empty,
    direction: Direction = Direction.Forward,
) -> np.ndarray:
    """Generate a schedule with the numba kernel, mirroring generate_schedule"""
    if not is_available():
        raise RuntimeError("numba is not installed")
    start_int = start_date.toordinal()
    end_int = end_date.toordinal()
    first_int = first_regular_date.toordinal() if first_regular_date else start_int
    last_int = last_regular_date.toordinal() if last_regular_date else end_int
    if not (start_int <= first_int <= last_int <= end_int):
        raise ValueError("Dates provided are not in correct order")
    freq_type, freq_value = frequency.int_based_mapping()

    if freq_type == -1:
        return np.array([start_int, end_int], dtype="datetime64[D]")
    if freq_type not in (1, 2):
        raise NotImplementedError(f"Frequency type {freq_type} not supported!")

    res = _schedule_kernel(
        start_int,
        end_int,
        freq_type,
        freq_value,
        first_int,
        last_int,
        roll_convention.value,
        direction == Direction.Forward,
    )
    return res.view("datetime64[D]")
