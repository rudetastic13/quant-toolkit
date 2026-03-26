"""Date composition class"""
from __future__ import annotations
import datetime
from dataclasses import dataclass
from dateutil.parser import parse
import numpy as np
import pandas as pd

# set up some statics
_MARCH_EPOCH = 719_468
_NP_EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal
_MONTHS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
_MONTHS_THIRTY_ONE = {1, 3, 5, 7, 8, 10, 12}
_CIVIL_TO_ISO_WEEKDAY = [6, 0, 1, 2, 3, 4, 5] # [0,6] [Sun, Sat], we shift [0, 6] [Mon, Sun]

@dataclass(slots=True)
class Date:
    """Date class, we apply composition to generate outputs as desired"""
    year: int
    month: int
    day: int

    def __post_init__(self):
        if self.month not in _MONTHS:
            raise ValueError(f"Invalid month {self.month!r}")
        if self.month == 2:
            if (self.year % 4 == 0 and self.year % 100 != 0) or (self.year % 400) == 0:
                boundary = 29
            else:
                boundary = 28
        elif self.month in _MONTHS_THIRTY_ONE:
            boundary = 31
        else:
            boundary = 30
        if not (1 <= self.day <= boundary):
            raise ValueError(f"Invalid day {self.day!r}")
        if not (1 <= self.year <= 9_999):
            raise ValueError(f"Invalid year {self.year!r}")

    #region, to-converters
    def to_ymd(self) -> tuple[int, int, int]:
        return self.year, self.month, self.day

    def to_str(self) -> str:
        """Convert date tuple to iso-format date in YYYY-MM-DD format"""
        year, month, day = self.to_ymd()
        return f"{year}-{month:02d}-{day:02d}"

    def to_int(self) -> int:
        """Convert date tuple to YYYYMMDD integer"""
        year, month, day = self.to_ymd()
        return year * 10000 + month * 100 + day

    def to_numpy(self) -> np.datetime64:
        """Convert date to numpy datetime64[D]"""
        return np.datetime64(self.to_str(), "D")

    def to_pandas(self) -> pd.Timestamp:
        """Convert date to pandas.Timestamp"""
        return pd.Timestamp(self.to_str())

    def to_date(self) -> datetime.date:
        """Convert date to datetime.date"""
        return datetime.date(*self.to_ymd())

    def to_datetime(self) -> datetime.datetime:
        """Convert date to datetime.datetime"""
        return datetime.datetime(*self.to_ymd())

    def toordinal(self) -> int:
        """Epoch is 1970-01-01, numpy style"""
        y, m, d = self.to_ymd()
        yy = y - (1 if m <= 2 else 0)
        era = yy // 400
        yoe = yy - era * 400
        mp = m + (9 if m <=2 else -3)
        doy = (153 * mp + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        return era * 146_097 + doe - _MARCH_EPOCH

    # endregion

    # region, from-converters
    @classmethod
    def from_ymd(cls, dt: tuple[int, int, int]) -> Date:
        return cls(dt[0], dt[1], dt[2])

    @classmethod
    def from_str(cls, dt: str) -> Date:
        parsed = parse(dt)
        return cls(parsed.year, parsed.month, parsed.day)

    @classmethod
    def from_int(cls, dt: int) -> Date:
        dt = str(dt)
        y = int(dt[0:4])
        m = int(dt[4:6])
        d = int(dt[6:8])
        return cls(y, m, d)

    @classmethod
    def from_numpy(cls, dt: np.datetime64) -> Date:
        np_dt = dt.item()
        return cls(np_dt.year, np_dt.month, np_dt.day)

    @classmethod
    def from_pandas(cls, dt: pd.Timestamp) -> Date:
        return cls(dt.year, dt.month, dt.day)

    @classmethod
    def from_date(cls, dt: datetime.date) -> Date:
        return cls(dt.year, dt.month, dt.day)

    @classmethod
    def from_datetime(cls, dt: datetime.datetime) -> Date:
        return cls(dt.year, dt.month, dt.day)

    @classmethod
    def fromordinal(cls, days: int) -> Date:
        """Epoch from 1970-01-01, so day 0 is 1970-01-01"""
        z = days + _MARCH_EPOCH
        era = z // 146_097
        doe = z - era * 146_097
        yoe = (doe - doe // 1_460 + doe // 36_524 - doe // 146_096) // 365
        y = yoe + era * 400
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        d = doy - (153 * mp + 2) // 5 + 1
        m = mp + 3 if mp < 10 else mp - 9
        y = y + (m <= 2)
        return cls(y, m, d)

    # endregion

    # region, common interface methods
    @classmethod
    def today(cls):
        dt = datetime.date.today()
        return cls(dt.year, dt.month, dt.day)

    def weekday(self) -> int:
        """Return the weekday, where [0,6] is [Sun,Sat]"""
        # ordinal 0 is 1970-01-01 which is a Thursday (weekday 3)
        z = self.toordinal()
        return (z + 4) % 7 if z >= -4 else (z + 5) % 7 + 6

    def isoweekday(self) -> int:
        return _CIVIL_TO_ISO_WEEKDAY[self.weekday()]

    def isoformat(self) -> str:
        return self.to_str()
    # endregion

    # region, mutation
    def replace(
        self,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None
    ) -> None:
        y = year or self.year
        m = month or self.month
        d = day or self.day
        val = Date.from_ymd((y, m, d))
        self.year = val.year
        self.day = val.day
        self.month = val.month

    # region, dunder methods
    def _cmp_key(self):
        return self.to_ymd()

    def __eq__(self, other: Date) -> bool:
        if type(other) != Date:
            raise TypeError(f"Compare object is type {other.__class__.__name__}, expected Date")
        return self._cmp_key() == other._cmp_key()

    def __lt__(self, other: Date) -> bool:
        if type(other) != Date:
            raise TypeError(f"Compare object is type {other.__class__.__name__}, expected Date")
        return self.to_int() < other.to_int()

    def __gt__(self, other: Date) -> bool:
        if type(other) != Date:
            raise TypeError(f"Compare object is type {other.__class__.__name__}, expected Date")
        return self.to_int() > other.to_int()

    def __le__(self, other: Date) -> bool:
        return self < other or self == other

    def __ge__(self, other: Date) -> bool:
        return self > other or self == other

    def __hash__(self) -> int:
        return hash(self.to_ymd())

    def __repr__(self) -> str:
        y, m, d = self.to_ymd()
        return f"Date({y:d}, {m:d}, {d:d})"

    def __str__(self) -> str:
        return self.to_str()
    # endregion
