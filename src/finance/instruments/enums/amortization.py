from common.containers.enums import SupportedIntEnum

class AmortizationType(SupportedIntEnum):
    """Enumeration for different types of amortization methods."""
    Custom = -1
    Unused = 0
    Bullet = 1 # single payment at end of term
    StraightLine = 2 # balance is evenly split across payment periods
    DailyLinear = 3 # straight line, day-based (Accounting adjustments)
    LevelPay = 4 # total payment constant, increase in principal over time (Mortgage-style)
    NoAmortization = 5 # no principal repayments even if balance exists (IR Swaps)


