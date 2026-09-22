"""Use numpy to generate a naive schedule array based on frequency and roll convention"""
import numpy as np
from finance.dates.enums import Frequency, Roll, Direction
from finance.dates.date import Date
from finance.dates.schedules.helpers import (
    guess_array_size,
    _DAYS_IN_MONTH_YEAR,
    _MONTH_TO_DAYS_OFFSET, ymd_from_days
)
from common.array_buffer import ArrayBuffer


_Allocated_Cache = ArrayBuffer(size=5_000_000, dtype=np.int32)

def generate_schedule(
    start_date: Date,
    end_date: Date,
    frequency: Frequency,
    first_regular_date: Date | None = None,
    last_regular_date: Date | None = None,
    roll_convention: Roll = Roll.Empty,
    direction: Direction = Direction.Forward,
) -> np.ndarray:
    """Use numpy to generate a naive schedule array based on frequency and roll convention"""
    start_int = start_date.toordinal()
    end_int = end_date.toordinal()
    first_int = first_regular_date.toordinal() if first_regular_date else start_int
    last_int = last_regular_date.toordinal() if last_regular_date else end_int
    if not (start_int <= first_int <= last_int <= end_int):
        raise ValueError("Dates provided are not in correct order")
    freq_type, freq_value = frequency.int_based_mapping()

    if freq_type == -1:
        return np.array([start_int, end_int], dtype="datetime64[D]")

    # get date bounds
    if direction == Direction.Forward:
        anchor = first_int
        step = freq_value
    else:
        anchor = last_int
        step = -freq_value

    # give a buffer
    arr_size = guess_array_size(start_int, end_int, freq_type, freq_value)
    arr = _Allocated_Cache.get_slice(arr_size + 2)
    arr[:] = np.arange(start=-2, stop=arr_size, dtype=np.int32)
    arr_size = arr.size
    np.multiply(arr, step, out=arr)

    # walk schedule out, simple offsets for day based, month based is more complex
    if freq_type == 2:
        np.add(arr, anchor, out=arr)
    elif freq_type == 1:
        y, m, d = ymd_from_days(anchor)
        roll_day = roll_convention.value
        if roll_day == 0:
            if _DAYS_IN_MONTH_YEAR[y, m] == d:
                roll_day = -1
            else:
                roll_day = d

        # derive month-based index
        np.add(arr, m - 1, out=arr)
        np.add(arr, y * 12, out=arr)

        # derive years from month epoc
        years = _Allocated_Cache.get_slice(arr_size)
        np.floor_divide(arr, 12, out=years)

        # derive months from month epoch
        months = _Allocated_Cache.get_slice(arr_size)
        np.mod(arr, 12, out=months)
        np.add(months, 1, out=months)

        # branching for direct cacls
        d = _Allocated_Cache.get_slice(arr_size)
        if roll_day == -2:
            # IMM third Wednesday: day-of-month from the weekday of the 1st (ordinal 0 is a Thursday, Monday=0)
            d[:] = _MONTH_TO_DAYS_OFFSET[arr]
            np.add(d, 3, out=d)
            np.mod(d, 7, out=d)
            np.subtract(2, d, out=d)
            np.mod(d, 7, out=d)
            np.add(d, 15, out=d)
        elif 1 <= roll_day <= 28:
            d[:] = roll_day
        else:
            if roll_day == -1:
                d[:] = _DAYS_IN_MONTH_YEAR[years, months]
            else:
                d[:] = roll_day
                np.minimum(d, _DAYS_IN_MONTH_YEAR[years, months], out=d)
        # convert back to days
        arr[:] = _MONTH_TO_DAYS_OFFSET[arr]
        np.add(arr, d-1, out=arr) # arr are 1st of month, offset by 1 to get final day
    else:
        raise NotImplementedError(f"Frequency type {freq_type} not supported!")

    # now cleanup array so end points have correct values
    start_idx = 2
    if direction == Direction.Forward:
        arr[start_idx] = first_int
        if start_int != first_int:
            start_idx -= 1
            arr[start_idx] = start_int
        end_idx = np.argmax(arr >= last_int)
        if arr[end_idx] != last_int:
            arr[end_idx] = last_int
        if end_int != last_int:
            end_idx += 1
            arr[end_idx] = end_int
        res = arr[start_idx:end_idx + 1].copy()
    else:
        arr[start_idx] = last_int
        if end_int != last_int:
            start_idx -= 1
            arr[start_idx] = end_int
        end_idx = np.argmax(arr <= first_int)
        if arr[end_idx] != first_int:
            arr[end_idx] = first_int
        if start_int != first_int:
            end_idx += 1
            arr[end_idx] = start_int
        res = arr[start_idx:end_idx + 1][::-1].copy()

    # reset the cache
    _Allocated_Cache.reset()

    # done
    return res.astype("datetime64[D]")
