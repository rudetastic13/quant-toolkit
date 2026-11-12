from __future__ import annotations
from dataclasses import dataclass, field
from .rule import HolidayRule
from finance.dates import Date

@dataclass
class HolidayRuleSet:
    name: str
    rules: list[HolidayRule] = field(default_factory=list)

    def add(self, rule: HolidayRule) -> None:
        self.rules.append(rule)

    def remove(self, name: str) -> None:
        original_len = len(self.rules)
        self.rules = [r for r in self.rules if r.name != name]
        if len(self.rules) == original_len:
            raise KeyError(f"No rule named `{name}` in rule set `{self.name}`")

    def generate(self, start_year: int, end_year: int) -> frozenset[Date]:

        if start_year > end_year:
            raise ValueError("start_year must be less than or equal to end_year")
        holidays: set[Date] = set()
        if len(self) > 0:
            for rule in self:
                holidays.update(rule.generate(start_year, end_year))
        return frozenset(holidays)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)
