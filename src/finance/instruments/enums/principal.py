from common.containers.enums import SupportedIntEnum


class PrincipalExchange(SupportedIntEnum):
    """Which principal flows a leg exchanges, on top of its coupons (bullet notional).

    ``Initial`` pays the notional out on the effective date, ``Final`` receives it back on the
    last payment date, ``Both`` does both — a deposit or a bond leg.  An amortising paydown is
    the notional provider's job, not this flag's.
    """

    Unused = 0
    Initial = 1
    Final = 2
    Both = 3

    @property
    def initial(self) -> bool:
        return self in (PrincipalExchange.Initial, PrincipalExchange.Both)

    @property
    def final(self) -> bool:
        return self in (PrincipalExchange.Final, PrincipalExchange.Both)
