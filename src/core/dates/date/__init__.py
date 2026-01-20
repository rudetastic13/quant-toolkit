"""Date composition class"""

import datetime
import numpy as np
import pandas as pd

NP_EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal


class Date:
    """Date class, we apply composition to generate outputs as desired"""

    def __init__(self, year, month, day):
        assert month in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, f"Month provided {month} is invalid"
        boundary = 31
        if month == 2:
            if (year % 4 == 0 and year % 100 != 0) or (year % 400) == 0:
                boundary = 29
            else:
                boundary = 28
        else:
            boundary = 30
        assert day <= boundary, f"Day provided exceeds boundary {boundary!r}!"
        self._date_tuple = (year, month, day)
        self._ordinal_month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    def to_str(self):
        year, month, day = self._date_tuple
        return f"{year}-{month:02d}-{day:02d}"

    def to_int(self):
        year, month, day = self._date_tuple
        return year * 10000 + month * 100 + day

    def to_numpy(self):
        return np.datetime64(self.to_str(), "D")

    def to_pandas(self):
        return pd.Timestamp(self.to_str())

    def to_date(self):
        return datetime.date(*self._date_tuple)

    def to_datetime(self):
        return datetime.datetime(*self._date_tuple)

    def to_ordinal(self):
        year, month, day = self._date_tuple
        month_days = self._ordinal_month_days
        leap = (month > 2) and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        y = year - 1
        days_before_year = 365 * y + y // 4 - y // 100 + y // 400
        days_this_year = month_days[month - 1] + day + (1 if leap else 0)
        return days_before_year + days_this_year
