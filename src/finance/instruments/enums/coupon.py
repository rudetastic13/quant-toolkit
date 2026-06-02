from common.containers.enums import SupportedIntEnum

class CouponType(SupportedIntEnum):
    """Enum for different types of coupons."""
    Unused = -2
    Custom = -1
    Zero = 0
    Fixed = 1
    Floating = 2
    ArithmeticAveraged = 3
    GeometricAveraged = 4

    @property
    def is_floating(self) -> bool:
        """True for index-linked coupons (simple float, arithmetic-averaged, compounded)."""
        return self in (CouponType.Floating, CouponType.ArithmeticAveraged, CouponType.GeometricAveraged)

    @property
    def needs_observation_grid(self) -> bool:
        """True for coupons that compound/average daily fixings (Tier-2 grid required)."""
        return self in (CouponType.ArithmeticAveraged, CouponType.GeometricAveraged)
