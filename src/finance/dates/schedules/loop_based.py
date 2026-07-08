"""Naive while-loop schedule generation, the readable scalar reference.

Same semantics as vectorized.generate_schedule; the numba and C++ kernels are
structured after this implementation.
"""
import numpy as np

from finance.dates.date import Date
from finance.dates.enums import Frequency, Roll, Direction
from finance.dates.schedules.helpers import days_from_ymd, days_in_month, days_to_month_index, is_eom_days


def generate_schedule_loop(
    start_date: Date,
    end_date: Date,
    frequency: Frequency,
    first_regular_date: Date | None = None,
    last_regular_date: Date | None = None,
    roll_convention: Roll = Roll.Empty,
    direction: Direction = Direction.Forward,
) -> np.ndarray:
    """Generate a schedule with a plain scalar loop, mirroring generate_schedule"""
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

    forward = direction == Direction.Forward
    anchor = first_int if forward else last_int
    step = freq_value if forward else -freq_value

    if freq_type == 1:
        anchor_mi, anchor_day = days_to_month_index(anchor)
        roll_day = roll_convention.value
        if roll_day == 0:
            # infer from the anchor: end-of-month anchors roll to EOM
            roll_day = -1 if is_eom_days(anchor) else anchor_day

        def date_at(k: int) -> int:
            mi = anchor_mi + k * step
            y, m = mi // 12, mi % 12 + 1
            eom = int(days_in_month(y, m))
            d = roll_day if 1 <= roll_day <= 28 else (eom if roll_day == -1 else min(roll_day, eom))
            return days_from_ymd(y, m, d)
    else:

        def date_at(k: int) -> int:
            return anchor + k * step

    dates: list[int] = []
    if forward:
        if start_int != first_int:
            dates.append(start_int)
        dates.append(first_int)
        if last_int != first_int:
            k = 1
            while (d := date_at(k)) < last_int:
                dates.append(d)
                k += 1
            dates.append(last_int)
        if end_int != last_int:
            dates.append(end_int)
    else:
        if end_int != last_int:
            dates.append(end_int)
        dates.append(last_int)
        if first_int != last_int:
            k = 1
            while (d := date_at(k)) > first_int:
                dates.append(d)
                k += 1
            dates.append(first_int)
        if start_int != first_int:
            dates.append(start_int)
        dates.reverse()

    return np.array(dates, dtype="datetime64[D]")
