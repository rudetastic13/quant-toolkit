"""A base zero curve implementation that takes discount factors as inputs"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PPoly
from finance.markets.curves.types import CurveInterpolator

# declare numpy types
DateArray = NDArray[np.datetime64]   # shape (n,), dtype datetime64[D]
FloatArray = NDArray[np.float64]


# region, common math layers
def _dates_to_floats(dates: DateArray, origin: np.int64) -> FloatArray:
    """Convert an array of datetime64[D] values to float days-since-origin."""
    return (dates.astype(np.int64) - origin).astype(np.float64)

def _compute_zero_rates(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    """
    Compute continuously compounded zero rates r(t) = -ln(DF(t)) / t.

    At t=0 (origin), r(0) is the limit of -ln(DF(t))/t as t->0, which
    equals the instantaneous forward rate.  We estimate it by linear
    extrapolation backward from the first two non-origin pillar rates
    when available, falling back to the rate at the first pillar.
    """
    n = len(x)
    rates = np.zeros(n)
    for i in range(n):
        if x[i] > 0:
            rates[i] = -np.log(df[i]) / x[i]

    # Fix origin rate (x[0] == 0)
    if n >= 3 and x[0] == 0.0 and x[1] > 0 and x[2] > 0:
        r1, r2 = rates[1], rates[2]
        rates[0] = r1 - x[1] * (r2 - r1) / (x[2] - x[1])
    elif n >= 2 and x[0] == 0.0 and x[1] > 0:
        rates[0] = rates[1]

    return rates


def _solve_cubic_second_derivs(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Solve the standard cubic spline tridiagonal system for second
    derivatives M at each knot, with prescribed M[0]=alpha and
    M[-1]=beta as boundary conditions.

    Uses Thomas algorithm (O(n)) rather than dense np.linalg.solve.
    """
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h

    if n <= 2:
        return np.array([alpha, beta][:n])

    # Interior system size = n-2
    m = n - 2
    b_diag = np.zeros(m)
    c_sup = np.zeros(m)
    a_sub = np.zeros(m)
    rhs = np.zeros(m)

    for j in range(m):
        i = j + 1  # index in original system
        b_diag[j] = 2.0 * (h[i - 1] + h[i])
        rhs[j] = 6.0 * (d[i] - d[i - 1])
        if j > 0:
            a_sub[j] = h[i - 1]
        if j < m - 1:
            c_sup[j] = h[i]

    # Subtract known boundary values
    rhs[0] -= h[0] * alpha
    rhs[-1] -= h[-1] * beta

    # Forward elimination (Thomas)
    for j in range(1, m):
        w = a_sub[j] / b_diag[j - 1]
        b_diag[j] -= w * c_sup[j - 1]
        rhs[j] -= w * rhs[j - 1]

    # Back substitution
    M_int = np.zeros(m)
    M_int[-1] = rhs[-1] / b_diag[-1]
    for j in range(m - 2, -1, -1):
        M_int[j] = (rhs[j] - c_sup[j] * M_int[j + 1]) / b_diag[j]

    M = np.empty(n)
    M[0] = alpha
    M[1:-1] = M_int
    M[-1] = beta
    return M


def _build_cubic_ppoly(
    x: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
) -> PPoly:
    """
    Build a scipy PPoly from knots x, values y, and second derivatives M.
    """
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h

    c_arr = np.zeros((4, n - 1))
    for i in range(n - 1):
        hi = h[i]
        c_arr[3, i] = y[i]
        c_arr[2, i] = d[i] - hi * (2.0 * M[i] + M[i + 1]) / 6.0
        c_arr[1, i] = M[i] / 2.0
        c_arr[0, i] = (M[i + 1] - M[i]) / (6.0 * hi)

    return PPoly(c_arr, x)
# endregion


# region, define interpolators
class Interpolator(ABC):

    @abstractmethod
    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None: ...

    @abstractmethod
    def __call__(self, x: FloatArray) -> FloatArray:
        """Return interpolated discount factors for an array of day offsets."""
        ...

class DFLogLinear(Interpolator):
    """
    ln(DF) is piecewise-linear between knots.
    Equivalent to constant instantaneous (overnight) forward rates in each
    interval.  Locally stable: changing one node only affects its two
    adjacent intervals.
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        self._log_df = np.log(df)

    def __call__(self, x: FloatArray) -> FloatArray:
        idx = np.clip(np.searchsorted(self._x, x, side="right") - 1, 0, len(self._x) - 2)
        t = (x - self._x[idx]) / (self._x[idx + 1] - self._x[idx])
        log_df = self._log_df[idx] + t * (self._log_df[idx + 1] - self._log_df[idx])
        return np.exp(log_df)


class DFLogCubic(Interpolator):
    """
    Cubic spline on ln(DF).  Produces smooth, continuous overnight rates.
    Global dependence: every node affects the whole curve.

    Parameters
    ----------
    alpha : float
        Prescribed second derivative of ln(DF) at the left endpoint.
        Default 0.0 gives the *natural* spline boundary condition.
    beta : float
        Prescribed second derivative of ln(DF) at the right endpoint.
        Default 0.0 gives the *natural* spline boundary condition.
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        log_df = np.log(df)
        M = _solve_cubic_second_derivs(x, log_df, alpha, beta)
        self._pp = _build_cubic_ppoly(x, log_df, M)

    def __call__(self, x: FloatArray) -> FloatArray:
        return np.exp(self._pp(x))



class RateLinear(Interpolator):
    """
    Linear interpolation of continuously compounded zero rates.
    Global dependence.  Forward rates are discontinuous at pillar dates
    (this is inherent to rate-linear interpolation, not a bug).
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        self._rates = _compute_zero_rates(x, df)

    def __call__(self, x: FloatArray) -> FloatArray:
        idx = np.clip(np.searchsorted(self._x, x, side="right") - 1, 0, len(self._x) - 2)
        t = (x - self._x[idx]) / (self._x[idx + 1] - self._x[idx])
        r = self._rates[idx] + t * (self._rates[idx + 1] - self._rates[idx])
        return np.exp(-r * x)



class RateQuadratic(Interpolator):
    """
    C1-continuous piecewise-quadratic on continuously compounded zero
    rates, stabilised via blended forward/backward sweeps.

    Parameters
    ----------
    alpha : float
        Prescribed first derivative of r(t) at the left endpoint.
        Default 0.0 (flat rate at the short end).
    beta : float
        Prescribed first derivative of r(t) at the right endpoint.
        Default 0.0 (flat rate at the long end).
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        n = len(x)
        rates = _compute_zero_rates(x, df)
        h = np.diff(x)
        d = np.diff(rates) / h

        # Forward sweep from m[0] = alpha
        m_fwd = np.zeros(n)
        m_fwd[0] = alpha
        for i in range(n - 1):
            m_fwd[i + 1] = 2.0 * d[i] - m_fwd[i]

        # Backward sweep from m[-1] = beta
        m_bwd = np.zeros(n)
        m_bwd[-1] = beta
        for i in range(n - 2, -1, -1):
            m_bwd[i] = 2.0 * d[i] - m_bwd[i + 1]

        if n >= 2:
            w = np.linspace(0.0, 1.0, n)
        else:
            w = np.array([0.5])
        m = (1.0 - w) * m_fwd + w * m_bwd

        m[0] = alpha
        m[-1] = beta

        c_arr = np.zeros((3, n - 1))
        for i in range(n - 1):
            c_arr[2, i] = rates[i]
            c_arr[1, i] = m[i]
            c_arr[0, i] = (d[i] - m[i]) / h[i]

        self._pp = PPoly(c_arr, x)

    def __call__(self, x: FloatArray) -> FloatArray:
        return np.exp(-self._pp(x) * x)

class RateCubic(Interpolator):
    """
    Cubic spline on continuously compounded zero rates.
    Smooth aesthetic curve between knots.  Global dependence.

    Parameters
    ----------
    alpha : float
        Prescribed second derivative of the rate function at the left
        endpoint.  Default 0.0 (natural spline).
    beta : float
        Prescribed second derivative of the rate function at the right
        endpoint.  Default 0.0 (natural spline).
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        rates = _compute_zero_rates(x, df)
        M = _solve_cubic_second_derivs(x, rates, alpha, beta)
        self._pp = _build_cubic_ppoly(x, rates, M)

    def __call__(self, x: FloatArray) -> FloatArray:
        return np.exp(-self._pp(x) * x)

# endregion

# ---------------------------------------------------------------------------
# Registry mapping InterpType -> Interpolator class
# ---------------------------------------------------------------------------

_INTERP_REGISTRY: dict[CurveInterpolator, type[Interpolator]] = {
    CurveInterpolator.LogLinearDF: DFLogLinear,
    CurveInterpolator.LogCubicDF: DFLogCubic,
    CurveInterpolator.RateLinear: RateLinear,
    CurveInterpolator.RateQuadratic: RateQuadratic,
    CurveInterpolator.RateCubic: RateCubic,
}


def build_interpolator(
    interp_type: CurveInterpolator,
    x: np.ndarray,
    df: np.ndarray,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> Interpolator:
    """Factory: construct the correct Interpolator for a given InterpType."""
    cls = _INTERP_REGISTRY[interp_type]
    return cls(x, df, alpha=alpha, beta=beta)



@dataclass
class InterpolatedSegment:
    """
    Wraps an Interpolator together with the raw node data that produced it,
    and knows its own domain [x_min, x_max] in float-day space.
    """
    interp_type: CurveInterpolator
    x: np.ndarray
    df: np.ndarray
    alpha: float
    beta: float
    origin: np.datetime64
    interpolator: Interpolator = field(init=False)

    def __post_init__(self) -> None:
        self.interpolator = build_interpolator(
            self.interp_type, self.x, self.df,
            alpha=self.alpha, beta=self.beta,
        )

    @property
    def x_min(self) -> float:
        return float(self.x[0])

    @property
    def x_max(self) -> float:
        return float(self.x[-1])

    def discount_factor(self, xs: FloatArray) -> FloatArray:
        return self.interpolator(xs)
