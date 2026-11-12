from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from dateutil.rrule import rrule, rruleset
from .observances import FilterFn, ObservanceFn
from finance.dates import Date

@dataclass
class HolidayRule:
    name: str
    rule: rrule | rruleset
    observance: ObservanceFn = field(default=lambda x: x)
    dt_filter: FilterFn = field(default=lambda _: True)
    # generated holidays per (start_year, end_year); rules are shared across rule sets
    _cache: dict[tuple[int, int], frozenset[Date]] = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def fixed(cls, name: str, *dates: date) -> HolidayRule:
        """Build a rule for one-off closures that follow no recurrence.

        Parameters
        ----------
        name : str
            Rule name.
        *dates : date
            The exact closure dates; no observance or filter is applied.

        Returns
        -------
        HolidayRule
            Rule that generates exactly ``dates``.

        Raises
        ------
        ValueError
            If no dates are given.
        """
        if not dates:
            raise ValueError(f"Fixed rule `{name}` needs at least one date")
        rule = rruleset()
        for d in dates:
            rule.rdate(datetime(d.year, d.month, d.day))
        return cls(name=name, rule=rule)

    def generate(self, start_year: int, end_year: int) -> frozenset[Date]:
        if start_year > end_year:
            raise ValueError(f"start_year must be less or equal to end_year")
        key = (start_year, end_year)
        if key in self._cache:
            return self._cache[key]
        dt_start = datetime(start_year, 1, 1)
        dt_end = datetime(end_year, 12, 31)
        raw_dates: list[datetime] = self.rule.between(dt_start, dt_end, inc=True)

        holidays: set[Date] = set()
        for raw_dt in raw_dates:
            d: date = self.observance(raw_dt.date())
            if not self.dt_filter(d):
                continue
            holidays.add(Date.from_date(d))

        # done
        self._cache[key] = frozenset(holidays)
        return self._cache[key]