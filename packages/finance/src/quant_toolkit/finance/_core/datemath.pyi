"""Type stubs for the finance._core.datemath C++ extension.

All day counts are days since 1970-01-01 (numpy datetime64[D] epoch).
"""

import numpy as np
from numpy.typing import NDArray

def days_from_ymd(y: int, m: int, d: int) -> int: ...
def ymd_from_days(days: int) -> tuple[int, int, int]: ...
def is_leap_year(y: int) -> bool: ...
def days_in_month(y: int, m: int) -> int: ...
def generate_schedule(
    start: int,
    end: int,
    freq_type: int,
    freq_value: int,
    first_regular: int,
    last_regular: int,
    roll: int,
    direction: int,
) -> NDArray[np.int64]: ...
