from common.containers.enums import SupportedIntEnum


class PrincipalExchange(SupportedIntEnum):
    """Which principal flows a leg exchanges, on top of its coupons (bullet notional).

    ``Initial`` pays the notional out on the effective date, ``Final`` receives it back,
    ``Both`` does both — a deposit or a bond leg.  Under ``Final`` the compiler pays each step
    down in the leg's per-period notional on its payment date, so an amortiser repays as it
    goes and a bullet repays at maturity.
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
