"""Use numpy to generate a naive schedule array based on frequency and roll convention"""
import numpy as np
from finance.dates.enums import Frequency, Roll, Direction
from finance.dates.date import Date
from finance.dates.schedules.helpers import (
    guess_array_size,
    _DAYS_IN_MONTH_YEAR,
    _MONTH_TO_DAYS_OFFSET, ymd_from_days
)

class _AllocatedArray:
    def __init__(self):
        self.array = np.empty(5_000_000, dtype=np.int32)
        self.current_idx = 0

    def get_slice(self, size: int) -> np.ndarray:
        arr = self.array[self.current_idx:self.current_idx + size]
        self.current_idx += size
        return arr

    def reset(self):
        self.current_idx = 0

_Allocated_Cache = _AllocatedArray()

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
    start_int = start_date.to_ordinal()
    end_int = end_date.to_ordinal()
    first_int = first_regular_date.to_ordinal() if first_regular_date else start_int
    last_int = last_regular_date.to_ordinal() if last_regular_date else end_int
    freq_type, freq_value = frequency.int_based_mapping()

    if freq_type == -1:
        return np.array([start_date.to_numpy(), end_date.to_numpy()], dtype="datetime64[D]")

    # get date bounds
    if direction == Direction.Forward:
        anchor = first_int
        step = freq_value
    else:
        anchor = last_int
        step = -freq_value

    # give a buffer
    arr_size = guess_array_size(start_int, end_int, freq_type, freq_value)
    arr = _Allocated_Cache.get_slice(arr_size)
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
        years = _Allocated_Cache.get_slice(arr_size)
        np.floor_divide(arr, 12, out=years)
        months = _Allocated_Cache.get_slice(arr_size)
        np.mod(arr, 12, out=months)
        np.add(months, 1, out=months)

        # branching for direct cacls
        d = _Allocated_Cache.get_slice(arr_size)
        if 1 <= roll_day <= 28:
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


    arr_size = guess_array_size(anchor, last_int, freq_type, freq_value)

    out = _Allocated_Cache.get_slice(arr_size)

    if freq_type == 2:
        np.multiply(np.arange(start=-2, stop=arr_size, dtype=np.int32), step, out=out)
        np.add(out, anchor, out=out)
    elif freq_type == 1:
        y, m, d = Date.from_ordinal(anchor).to_ymd()
        roll_int = roll_convention.value
        if roll_int == 0:
            if d == _DAYS_IN_MONTH_YEAR[y, m]:
                roll_int = -1
            else:
                roll_int = d
        month_idx_seq = np.arange(start=-2, stop=arr_size, dtype=np.int32)
        np.multiply(month_idx_seq, step, out=month_idx_seq)
        np.add(month_idx_seq, m - 1, out=month_idx_seq)
        np.add(month_idx_seq, y * 12, out=month_idx_seq)

        # get years
        years = out
        np.floor_divide(month_idx_seq, 12, out=years)
        # get months
        months = month_idx_seq
        np.mod(month_idx_seq, 12
)