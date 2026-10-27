from common.containers.enums import SupportedIntEnum


class AmortizationType(SupportedIntEnum):
    """How a leg's outstanding notional evolves (scheduled prepayment).

    ``Bullet`` keeps the face outstanding to maturity.  ``StraightLine`` repays equal principal
    amounts on each principal date; ``LevelPay`` sizes the principal so that interest plus
    principal is constant per principal period (an annuity), re-solved whenever the coupon rate
    changes.  ``Custom`` means a supplied ``NotionalSchedule`` says what is outstanding when.
    """

    Custom = -1
    Bullet = 0
    StraightLine = 1
    LevelPay = 2

    @property
    def is_generated(self) -> bool:
        """True when the outstanding profile is computed from the leg's amortisation window."""
        return self in (AmortizationType.StraightLine, AmortizationType.LevelPay)
