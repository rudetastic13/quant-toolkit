"""Reference coupon calculators over a MarketContext.

The slow, explicit twin of the kernel path (``pricing.kernels.compiler.reprice``): each
calculator prices one coupon flavour for one leg directly against a ``MarketContext``,
using the same shared pieces — ``RateGenerator`` for index rates, the tight
``compounded``/``averaged`` reductions, and the single-source shaping algebra (via
``coupons.rates``).  It exists as a per-event cross-check of the compiler lowering, not as
a production path: the engine never calls it.

Calculator signatures line up with ``CouponEvent.calculate``: the event binds its own
dataclass fields (rate parameters) via ``functools.partial``, and the caller supplies the
market plus the date/observation grids from a ``PaymentSchedule``.
"""
import numpy as np

from finance.instruments.enums import CouponType, MarginTreatment
from finance.instruments.resolution.resolver import curve_name
from finance.instruments.schedules.coupon_schedule import CouponSchedule
from finance.coupons.rates import averaged_coupon, compounded_coupon, floating_coupon
from finance.markets.context import MarketContext
from finance.markets.rate_generator import RateGenerator


def _projection_curve(market: MarketContext, rate_index: str) -> str:
    """Resolve an instrument's ``rate_index`` label to a bound curve name.

    Accepts either a curve name directly (``"USD.SOFR"``) or the instrument-level
    ``"CCY INDEX"`` label (``"USD SOFR"``) that builders stamp on legs.
    """
    if rate_index in market.curves:
        return rate_index
    parts = rate_index.split()
    if len(parts) == 2:
        name = curve_name(*parts)
        if name in market.curves:
            return name
    raise KeyError(f"rate_index {rate_index!r} does not resolve to a curve in the market")


def calculate_fixed(coupon_rate: float, out: np.ndarray) -> np.ndarray:
    """Fixed rate: the same rate for every period.

    ``out`` is pre-allocated with one slot per accrual period; returned filled.
    """
    out[:] = coupon_rate
    return out


def calculate_floating(
    rate_index: str,
    spread: float,
    index_floor: float | None,
    cap: float | None,
    floor: float | None,
    *,
    market: MarketContext,
    reset_starts: np.ndarray,
    reset_ends: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Single-fixing float: project each reset window, then shape.

    ``reset_starts``/``reset_ends`` are the per-period projection windows (for the aligned
    reset==payment case these are the accrual windows, matching the kernel).
    """
    idx = RateGenerator(market).simple_rate(
        _projection_curve(market, rate_index),
        reset_starts,
        reset_ends,
    )
    out[:] = floating_coupon(idx, spread=spread or 0.0, index_floor=index_floor, cap=cap, floor=floor)
    return out


def calculate_geometric_average(
    rate_index: str,
    spread: float,
    index_floor: float | None,
    cap: float | None,
    floor: float | None,
    margin_treatment: MarginTreatment,
    *,
    market: MarketContext,
    obs_starts: np.ndarray,
    obs_ends: np.ndarray,
    obs_weights: np.ndarray,
    obs_offsets: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Compounded (geometric-average) coupon over the observation grid.

    ``obs_*`` are the Tier-2 observation-grid arrays from a ``PaymentSchedule`` (fixing
    read windows, accrual weights, per-period ``reduceat`` offsets) — the same arrays the
    compiler lowers, so lookback/lockout shifts flow through identically.
    """
    obs_rate = RateGenerator(market).simple_rate(
        _projection_curve(market, rate_index),
        obs_starts,
        obs_ends,
    )
    out[:] = compounded_coupon(
        obs_rate, obs_weights, obs_offsets,
        spread=spread or 0.0, index_floor=index_floor, cap=cap, floor=floor, margin=margin_treatment,
    )
    return out


def calculate_arithmetic_average(
    rate_index: str,
    spread: float,
    index_floor: float | None,
    cap: float | None,
    floor: float | None,
    margin_treatment: MarginTreatment,
    *,
    market: MarketContext,
    obs_starts: np.ndarray,
    obs_ends: np.ndarray,
    obs_weights: np.ndarray,
    obs_offsets: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Arithmetic-average coupon over the observation grid (see calculate_geometric_average)."""
    obs_rate = RateGenerator(market).simple_rate(
        _projection_curve(market, rate_index),
        obs_starts,
        obs_ends,
    )
    out[:] = averaged_coupon(
        obs_rate, obs_weights, obs_offsets,
        spread=spread or 0.0, index_floor=index_floor, cap=cap, floor=floor, margin=margin_treatment,
    )
    return out


def calculate_custom(
    schedule: CouponSchedule,
    accrual_grid: np.ndarray,
    out: np.ndarray,
    *,
    market: MarketContext,
    reset_starts: np.ndarray | None = None,
    reset_ends: np.ndarray | None = None,
    obs_starts: np.ndarray | None = None,
    obs_ends: np.ndarray | None = None,
    obs_weights: np.ndarray | None = None,
    obs_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Piecewise coupon: dispatch each accrual period to its governing event's calculator.

    The reference twin of the compiler's per-period lowering: ``schedule_from_accrual_grid``
    maps each period to an event, and each event's calculator fills its periods.  Grids are
    sliced per event — including the observation grid, whose per-period segments are
    regrouped with rebased offsets so the segmented reductions stay aligned.
    """
    event_idx = schedule.schedule_from_accrual_grid(accrual_grid)
    n = out.shape[0]
    if obs_offsets is not None:
        obs_bounds = np.append(obs_offsets, obs_weights.shape[0])

    for i, event in enumerate(schedule.events):
        mask = event_idx == i
        if not mask.any():
            continue
        sub_out = np.zeros(int(mask.sum()), dtype=np.float64)
        if event.coupon_type == CouponType.Fixed:
            event.calculate(out=sub_out)
        elif event.coupon_type == CouponType.Floating:
            event.calculate(market=market, reset_starts=reset_starts[mask], reset_ends=reset_ends[mask], out=sub_out)
        else:
            # regroup this event's observation segments and rebase the offsets
            if obs_offsets is None or obs_offsets.shape[0] != n:
                raise ValueError("averaged/compounded events need a per-period observation grid")
            segments = [slice(int(obs_bounds[p]), int(obs_bounds[p + 1])) for p in np.nonzero(mask)[0]]
            sub_lens = np.array([s.stop - s.start for s in segments], dtype=np.intp)
            sub_offsets = np.concatenate([[0], np.cumsum(sub_lens[:-1])])
            take = np.concatenate([np.arange(s.start, s.stop) for s in segments])
            event.calculate(
                market=market, obs_starts=obs_starts[take], obs_ends=obs_ends[take],
                obs_weights=obs_weights[take], obs_offsets=sub_offsets, out=sub_out,
            )
        out[mask] = sub_out
    return out
