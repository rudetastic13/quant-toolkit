import numpy as np

from core.dates import Date, NP_EPOCH_ORDINAL
from core.dates.enums import Frequency, Direction, Roll

DAY_MASK = np.array([1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30], dtype=np.uint16)


def scalar_ymd_from_days(day: int) -> tuple[int, int, int]:
    z = day + 719_468
    era = z // 146_097
    doe = z - era * 146_097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146097) // 365
    y = int(yoe) + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    y = y + (m <= 2)
    return y, m, d


def is_eom(y, m, d) -> bool:
    eom = DAY_MASK[m]
    if m == 2 and is_leap_year(y):
        eom = 29
    return d == eom


def is_leap_year(y) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def days_from_ymd(y: np.narray, m: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = np.empty(y.size, dtype=np.int32)
    work = np.empty(y.size, dtype=np.int32)
    work.fill(14)
    np.subtract(work, m, out=work)
    np.floor_divide(work, 12, out=work)
    np.subtract(y, work, out=out)
    np.multiply(out, 146_097, out=out)
    np.floor_divide(out, 400, out=out)
    np.multiply(work, 12, out=work)
    np.add(work, m, out=work)
    np.subtract(work, 3, out=work)
    np.multiply(work, 153, out=work)
    np.add(out, work, out=out)
    np.floor_divide(work, 5, out=work)
    np.add(out, work, out=out)
    np.add(out, d, out=out)
    np.subtract(out, 1, out=out)
    np.subtract(out, 719_468, out=out)
    return out


def days_from_ymd_som(y: np.narray, m: np.ndarray) -> np.ndarray:
    out = np.empty(y.size, dtype=np.int32)
    work = np.empty(y.size, dtype=np.int32)
    work.fill(14)
    np.subtract(work, m, out=work)
    np.floor_divide(work, 12, out=work)
    np.subtract(y, work, out=out)
    np.multiply(out, 146_097, out=out)
    np.floor_divide(out, 400, out=out)
    np.multiply(work, 12, out=work)
    np.add(work, m, out=work)
    np.subtract(work, 3, out=work)
    np.multiply(work, 153, out=work)
    np.add(out, work, out=out)
    np.floor_divide(work, 5, out=work)
    np.add(out, work, out=out)
    np.subtract(out, 719_468, out=out)
    return out


def eoms_from_year_month(year, month) -> np.ndarray:
    days = DAY_MASK[month]
    leap_mask = is_leap_year(year)
    days[days == 28 & leap_mask] = 29
    return days


def roll_from_ymd_to_days(y, m, roll_convention) -> np.ndarray:
    d = eoms_from_year_month(y, m)
    if roll_convention == -1:  # End of month
        np.minimum(d, roll_convention, out=d)
    out = days_from_ymd_som(y, m)
    np.add(out, d, out=out)
    np.subtract(out, 1, out=out)
    return out


def _get_array_size(start_day, end_day, freq_type, freq_value) -> int:
    max_dates = 0
    if freq_type != -1:
        max_dates = end_day - start_day + 1
        if freq_type == 1:
            max_dates = (max_dates // 28) // freq_value
        elif freq_type == 2:
            max_dates = max_dates // freq_value
    return max_dates + 3


def generate_schedule(
    start_date: Date,
    end_date: Date,
    frequency: Frequency,
    first_regular_date: Date | None = None,
    last_regular_date: Date | None = None,
    roll_convention: Roll = Roll.Empty,
    direction: Direction = Direction.Forward,
):
    dt_to_int = lambda dt: dt.to_ordinal() - NP_EPOCH_ORDINAL
    start_int = dt_to_int(start_date)
    end_int = dt_to_int(end_date)
    first_int = dt_to_int(first_regular_date) if first_regular_date else start_int
    last_int = dt_to_int(last_regular_date) if last_regular_date else end_int
    freq_type, freq_value = frequency.to_tuple()

    if freq_type == -1:
        return np.array([start_date.to_numpy(), end_date.to_numpy()], dtype="datetime64[D]")

    # get date bounds
    if direction == Direction.Forward:
        anchor = first_int
        step = freq_value
    else:
        anchor = last_int
        step = -freq_value
    arr_size = _get_array_size(anchor, last_int, freq_type, freq_value)

    if freq_type == 2:
        seq = anchor + step * np.arange(start=-2, stop=arr_size, dtype=np.int32)
    elif freq_type == 1:
        y, m, d = scalar_ymd_from_days(anchor)
        roll_int = roll_convention.value
        if roll_int == 0:
            if is_eom(y, m, d):
                roll_int = -1
            else:
                roll_int = d
        month_idx_seq = np.arange(start=-2, stop=arr_size, dtype=np.int32)
        np.multiply(month_idx_seq, step, out=month_idx_seq)
        np.add(month_idx_seq, m - 1, out=month_idx_seq)
        np.add(month_idx_seq, y * 12, out=month_idx_seq)

        # get years
        years = np.empty(arr_size, dtype=np.int32)
        np.floor_divide(month_idx_seq, 12, out=years)

        # get months
        months = np.empty(arr_size, dtype=np.int32)
        np.mod(month_idx_seq, 12, out=months)
        np.add(months, 1, out=months)

        # get date sequence
        seq = roll_from_ymd_to_days(years, months, roll_int)
    else:
        raise ValueError(f"Unsupported frequency type: {Frequency.Name}")

    # update endpoints
    if direction == Direction.Forward:
        start_idx = 2
        seq[start_idx] = first_int
        if start_int != first_int:
            start_idx -= 1
            seq[start_idx] = start_int
        end_idx = np.argmax(seq >= last_int)
        if seq[end_idx] != last_int:
            seq[end_idx] = last_int
        if end_int != last_int:
            end_idx += 1
            seq[end_idx] = end_int
        return seq[start_idx : end_idx + 1].astype("datetime64[D]")

    # backwards fill logic
    start_idx = 2
    seq[start_idx] = last_int
    if end_int != last_int:
        start_idx -= 1
        seq[start_idx] = end_int
    end_idx = np.argmax(seq <= first_int)
    if seq[end_idx] != first_int:
        seq[end_idx] = first_int
    if start_int != first_int:
        end_idx += 1
        seq[end_idx] = start_int
    return seq[start_idx : end_idx + 1][::-1].astype("datetime64[D]")
