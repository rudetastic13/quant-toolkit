from common.containers.enums import SupportedIntEnum

class RateInterpolator(SupportedIntEnum):
    """
    Enum for rate path interpolation methods.
    """
    Linear = 1
    Flat = 2

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

