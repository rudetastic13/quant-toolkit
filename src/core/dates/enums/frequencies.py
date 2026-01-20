from enum import IntEnum


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
        return {
            Frequency.Once: (-1, -1),
            Frequency.Monthly: (1, 1),
            Frequency.BiMonthly: (1, 2),
            Frequency.Quarterly: (1, 3),
            Frequency.SemiAnnually: (1, 6),
            Frequency.Annually: (1, 12),
            Frequency.TwoYearly: (1, 24),
            Frequency.ThreeYearly: (1, 36),
            Frequency.FiveYearly: (1, 60),
            Frequency.SevenYearly: (1, 84),
            Frequency.TenYearly: (1, 120),
            Frequency.FifteenYearly: (1, 180),
            Frequency.TwentyYearly: (1, 240),
            Frequency.ThirtyYearly: (1, 360),
            Frequency.Daily: (2, 1),
            Frequency.Weekly: (2, 7),
            Frequency.BiWeekly: (2, 14),
        }
