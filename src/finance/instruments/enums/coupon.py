from common.containers.enums import SupportedIntEnum

class CouponType(SupportedIntEnum):
    """Enum for different types of coupons."""
    Custom = -1
    Zero = 0
    Fixed = 1
    Floating = 2
    ArithmeticAveraged = 3
    GeometricAveraged = 4
