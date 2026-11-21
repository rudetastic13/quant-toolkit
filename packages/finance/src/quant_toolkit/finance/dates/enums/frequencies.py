from enum import IntEnum

_INT_MAPPING = {
    "Once": (-1, -1),
    "Monthly": (1, 1),
    "BiMonthly": (1, 2),
    "Quarterly": (1, 3),
    "SemiAnnually": (1, 6),
    "Annually": (1, 12),
    "TwoYearly": (1, 24),
    "ThreeYearly": (1, 36),
    "FiveYearly": (1, 60),
    "SevenYearly": (1, 84),
    "TenYearly": (1, 120),
    "FifteenYearly": (1, 180),
    "TwentyYearly": (1, 240),
    "ThirtyYearly": (1, 360),
    "Daily": (2, 1),
    "Weekly": (2, 7),
    "BiWeekly": (2, 14),
}

class Frequency(IntEnum):
    Once = 0
    Daily = 1
    Weekly = 7
    BiWeekly = 14
    Monthly = 30
    BiMonthly = 60
    Quarterly = 90
    SemiAnnually = 180
    Annually = 360
    TwoYearly = 720
    ThreeYearly = 1080
    FiveYearly = 1800
    SevenYearly = 2520
    TenYearly = 3600
    FifteenYearly = 5400
    TwentyYearly = 7200
    ThirtyYearly = 10800

    def int_based_mapping(self) -> tuple[int, int]:
        return _INT_MAPPING[self.name]