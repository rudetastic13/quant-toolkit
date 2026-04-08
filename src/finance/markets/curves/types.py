from common.containers.enums import SupportedIntEnum

class RateExtrapolator(SupportedIntEnum):
    """
    Enum for rate path extrapolation methods.
    """
    Flat = 1
    NotAllowed = 2

class CurveInterpolator(SupportedIntEnum):
    """
    Enum for curve interpolation methods.
    """
    LogLinearDF = 1
    LogCubicDF = 2
    RateLinear = -1
    RateQuadratic = -2
    RateCubic = -3

