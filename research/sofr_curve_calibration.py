"""SOFR curve calibration walkthrough — deposits + par swaps into a ZeroCurve.

Market date: 2026-06-03.  The first two quotes (1D, 2D) are cash deposits; every
other pillar is a par SOFR OIS swap.  The same instrument set is calibrated twice:

  * ``GlobalSolver``  — all 23 nodes at once via scipy least_squares (bump-and-reprice)
  * ``Bootstrapper``  — sequential pillar-by-pillar brentq (valid: LogLinearDF is local)

The script dumps the calibration errors (implied - quote, in bp) for both solvers,
prices a handful of example swaps off the calibrated market, and plots the two
discount-factor curves together with the calibration nodes marked.

Run from the repo root:

    PYTHONPATH=src python research/sofr_curve_calibration.py
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from finance.conventions import default_registry
from finance.dates import Date, Term, TermType, add_term
from finance.instruments.resolution import Deposit, Swap, curve_name
from finance.instruments.resolution.resolver import resolve_conventions
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace
from finance.pricing.calibration import (
    Bootstrapper,
    CalibrationResult,
    CurveCalibrator,
    CurveDefinition,
    DepositHelper,
    GlobalSolver,
    Quote,
    QuoteKind,
    swap_helper,
)
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import Sensitivities
from finance.pricing.risk.autodiff import AutodiffRisk

# ---------------------------------------------------------------------------
# Market inputs — SOFR OIS par quotes (percent) as of 2026-06-03
# ---------------------------------------------------------------------------

AS_OF = Date(2026, 6, 3)
CURVE = curve_name("USD", "SOFR")   # "USD.SOFR"

# (tenor, quote %, helper kind) — 1D/2D are cash deposits, the rest par swaps.
SOFR_QUOTES: list[tuple[str, float, str]] = [
    ("1D",   3.614658, "deposit"),
    ("2D",   3.614658, "deposit"),
    ("1M",   3.614658, "swap"),
    ("3M",   3.649460, "swap"),
    ("6M",   3.712126, "swap"),
    ("9M",   3.788469, "swap"),
    ("12M",  3.869102, "swap"),
    ("15M",  3.902553, "swap"),
    ("18M",  3.922179, "swap"),
    ("21M",  3.934771, "swap"),
    ("24M",  3.944517, "swap"),
    ("36M",  3.932856, "swap"),
    ("48M",  3.927707, "swap"),
    ("60M",  3.938521, "swap"),
    ("84M",  3.992436, "swap"),
    ("120M", 4.088470, "swap"),
    ("144M", 4.156282, "swap"),
    ("180M", 4.241282, "swap"),
    ("240M", 4.311461, "swap"),
    ("300M", 4.308223, "swap"),
    ("360M", 4.269303, "swap"),
    ("40Y",  4.099303, "swap"),
    ("50Y",  3.929303, "swap"),
]


def _short_deposit_helper(rate: float, tenor: str) -> DepositHelper:
    """O/N-style deposit anchored at the market date (no spot lag).

    The stock ``deposit_helper`` factory applies the T+2 spot lag, which rolls both
    the 1D and 2D quotes onto the same Monday pillar (2026-06-08) over the weekend.
    The very short end of a SOFR curve is overnight money, so these two deposits run
    from the market date itself: 1D = as_of -> +1BD, 2D = as_of -> +2BD.
    """
    conv = resolve_conventions("USD", "SOFR", default_registry)
    dep_conv = conv.deposit
    n_days = int(tenor.rstrip("Dd"))
    maturity = Date.from_numpy(
        add_term(
            AS_OF.to_numpy(),
            Term(n_days, TermType.BusinessDays),
            dep_conv.business_day_convention,
            dep_conv.calendar,
        )
    )
    dep = Deposit(
        effective=AS_OF, maturity=maturity, rate=rate,
        day_count_method=conv.deposit_day_count, currency="USD", index_name="SOFR",
    )
    return DepositHelper(deposit=dep, quote=Quote(rate, QuoteKind.SimpleRate), curve=CURVE)


def build_helpers() -> list:
    """One calibration instrument per quote, in quote order."""
    helpers = []
    for tenor, pct, kind in SOFR_QUOTES:
        rate = pct / 100.0
        if kind == "deposit":
            helpers.append(_short_deposit_helper(rate, tenor))
        else:
            helpers.append(swap_helper(rate=rate, tenor=tenor, as_of=AS_OF))
    return helpers


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(solver, *, jacobian: bool = False) -> CalibrationResult:
    base = MarketContext(as_of_date=AS_OF, curves=CurveNamespace())
    calibrator = CurveCalibrator(
        instruments=build_helpers(),
        solver=solver,
        target=CurveDefinition(CURVE, CurveInterpolator.LogLinearDF),
    )
    t0 = time.perf_counter()
    result = calibrator.calibrate(base, jacobian=jacobian)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result
    print(f"\n=== {type(solver).__name__} ===")
    print(f"  converged        : {sr.converged}")
    print(f"  evaluations      : {sr.iterations}")
    print(f"  max|residual|    : {sr.max_abs_residual:.3e}  (rate units)")
    print(f"  wall time        : {elapsed:.2f}s")
    return result


def dump_errors(result: CalibrationResult) -> None:
    """Per-pillar calibration errors: implied vs quote, in basis points."""
    helpers = build_helpers()
    helpers.sort(key=lambda h: h.pillar_date)  # calibrator reports residuals in pillar order
    print(f"  {'tenor':>6} {'pillar':>12} {'quote %':>10} {'implied %':>11} {'error (bp)':>12}")
    for (tenor, pct, _), helper, pillar, resid in zip(
        SOFR_QUOTES, helpers, result.pillar_dates, result.residuals
    ):
        implied = helper.implied(result.market)
        print(
            f"  {tenor:>6} {str(pillar):>12} {pct:>10.6f} {implied * 100:>11.6f} "
            f"{resid * 1e4:>12.3e}"
        )


# ---------------------------------------------------------------------------
# Example swap pricing off the calibrated market
# ---------------------------------------------------------------------------

def price_examples(market: MarketContext) -> None:
    """Price a few vanilla SOFR swaps against the calibrated curve.

    Sign convention: positive notional = receive fixed, negative = pay fixed.
    """
    examples = [
        ("5Y receive-fixed @ par (3.938521%)", dict(notional=100e6, fixed_rate=0.03938521, tenor="5Y")),
        ("10Y pay-fixed @ 4.00%", dict(notional=-100e6, fixed_rate=0.0400, tenor="10Y")),
        ("2Y receive-fixed @ 4.25%", dict(notional=250e6, fixed_rate=0.0425, tenor="2Y")),
        ("7Y receive-fixed @ 3.75% + 10bp float spread", dict(notional=50e6, fixed_rate=0.0375, tenor="7Y", spread=0.0010)),
    ]

    print("\n=== Example swap pricing (calibrated market) ===")
    for label, kwargs in examples:
        swap = Swap.fixed_float_swap(rate_index="SOFR", as_of=AS_OF, **kwargs)
        res = swap(market, requests=["pv", "leg_pvs"])
        pv = float(res.instrument_pv[0])
        recv, pay = (float(x) for x in res.leg_pv)
        print(f"\n  {label}")
        print(f"    PV          : {pv:>18,.2f}")
        print(f"    receive leg : {recv:>18,.2f}")
        print(f"    pay leg     : {pay:>18,.2f}")

    # Model par rates read back off the curve for a few standard tenors.
    print("\n  Model par rates off the calibrated curve:")
    for tenor in ("2Y", "5Y", "10Y", "30Y"):
        helper = swap_helper(rate=0.0, tenor=tenor, as_of=AS_OF)
        print(f"    {tenor:>4} par: {helper.implied(market) * 100:.6f} %")


# ---------------------------------------------------------------------------
# Key-rate durations — bump-and-reprice each zero-rate pillar by 1bp
# ---------------------------------------------------------------------------

KRD_TENORS = ("5Y", "7Y", "10Y")


def _example_swaps(market: MarketContext) -> tuple[list, list[str]]:
    """Receive-fixed $100mm swaps for the KRD tables.

    One per KRD tenor struck at its model par rate (PV ~ 0), plus one off-market 5Y struck
    at 2.8% (deep in the money to a receiver) to contrast an off-par risk profile.
    """
    swaps, labels = [], []
    for tenor in KRD_TENORS:
        par = swap_helper(rate=0.0, tenor=tenor, as_of=AS_OF).implied(market)
        swaps.append(
            Swap.fixed_float_swap(
                notional=100e6, rate_index="SOFR", fixed_rate=par, as_of=AS_OF, tenor=tenor,
            )
        )
        labels.append(f"{tenor} recv-fixed @ {par * 100:.4f}%")

    swaps.append(
        Swap.fixed_float_swap(
            notional=100e6, rate_index="SOFR", fixed_rate=0.038, as_of=AS_OF, tenor="5Y",
        )
    )
    labels.append("5Y recv-fixed @ 3.8000% (off-mkt)")
    return swaps, labels


def _pillar_tenor_map() -> dict[np.datetime64, str]:
    """Map each calibration pillar date -> its quote tenor label (e.g. '5Y')."""
    out: dict[np.datetime64, str] = {}
    for (tenor, _, _), helper in zip(SOFR_QUOTES, build_helpers()):
        pd = helper.pillar_date
        key = pd.to_numpy() if hasattr(pd, "to_numpy") else np.datetime64(pd, "D")
        out[np.datetime64(key, "D")] = tenor
    return out


def key_rate_duration_table(market: MarketContext, save_path: str | None = None) -> "pd.DataFrame":
    """Dump per-pillar KRD (PV change per +1bp) for the example swaps as a markdown table.

    Rows are the curve's zero-rate pillars (labelled by quote tenor); columns are the
    swaps.  Only pillars carrying non-trivial risk are shown, plus a total-DV01 row
    (key-rate additivity: the column sum equals the parallel DV01).
    """
    import pandas as pd

    swaps, labels = _example_swaps(market)
    program = SwapPricer().compile(swaps)
    ladder = Sensitivities(program, market).key_rate_durations(CURVE)

    tenor_of = _pillar_tenor_map()
    row_labels = [
        f"{tenor_of.get(d, '?'):>4}  ({yr:5.2f}y)"
        for d, yr in zip(ladder.pillar_dates, ladder.pillar_years)
    ]
    df = pd.DataFrame(ladder.krd.T, index=row_labels, columns=labels)
    df.index.name = "pillar"

    # Trim to pillars that actually carry risk (far pillars are exactly zero for these swaps).
    keep = df.abs().max(axis=1) > 1.0
    trimmed = df.loc[keep].copy()
    trimmed.loc["Total (DV01)"] = df.sum(axis=0)  # full-ladder sum == parallel DV01

    md = trimmed.to_markdown(floatfmt=",.1f")
    header = (
        "## Key-rate duration — receive-fixed SOFR swaps ($100mm, struck at par)\n\n"
        f"Market date {AS_OF.to_str()}.  Values are PV change in USD per **+1bp** zero-rate "
        "bump at each pillar (central difference).\n\n"
    )
    print("\n=== Key-rate durations (PV change per +1bp, USD) ===")
    print(trimmed.to_string(float_format=lambda v: f"{v:,.1f}"))
    if save_path:
        with open(save_path, "w") as fh:
            fh.write(header + md + "\n")
        print(f"\nSaved KRD table -> {save_path}")
    return trimmed


def par_rate_ladder(
    market: MarketContext, jacobian: np.ndarray, *, backend: str = "numpy"
) -> tuple[np.ndarray, list[str]]:
    """Par-rate (calibration-instrument) KRD: ``(∂V/∂z)·J⁻¹``, per +1bp quote move.

    Buckets each swap's risk onto the **par quotes** that built the curve, via the calibration
    Jacobian ``J = ∂implied/∂z`` (implicit function theorem: ``dz/dq = J⁻¹``).  A par swap then
    shows risk almost entirely to its own par instrument — the profile a trader hedges against.

    Two backends produce the identical number by different routes:

    * ``"numpy"`` — reuse the bump-and-reprice zero ladder (already ``∂V/∂z·bp``) and right-
      multiply by ``J⁻¹``.  Pure numpy/linalg, no JAX, no XLA compile.
    * ``"jax"``   — ``AutodiffRisk.partial_dv01`` reads ``∂V/∂z`` from exact autodiff, then
      applies the same ``J⁻¹``.  Faster once warm, but pays a ~300ms one-off compile per
      process/instance.

    Returns ``(pdv01, labels)`` with ``pdv01`` shaped ``(n_instruments, n_quotes)``.
    """
    swaps, labels = _example_swaps(market)
    program = SwapPricer().compile(swaps)
    if backend == "numpy":
        # ladder.krd is (n_inst, P) already scaled to +1bp; J⁻¹ maps zero-space -> quote-space.
        krd = Sensitivities(program, market).key_rate_durations(CURVE).krd
        pdv01 = krd @ np.linalg.inv(jacobian)
    elif backend == "jax":
        pdv01 = AutodiffRisk(program, market).partial_dv01(CURVE, jacobian)
    else:
        raise ValueError(f"unknown backend {backend!r} (expected 'numpy' or 'jax')")
    return pdv01, labels


def par_rate_key_rate_table(
    market: MarketContext, jacobian: np.ndarray, save_path: str | None = None,
    *, backend: str = "numpy",
) -> "pd.DataFrame":
    """Dump par-rate KRD for the example swaps as markdown (see ``par_rate_ladder``)."""
    import pandas as pd

    pdv01, labels = par_rate_ladder(market, jacobian, backend=backend)

    row_labels = [f"{tenor:>4}" for tenor, _, _ in SOFR_QUOTES]
    df = pd.DataFrame(pdv01.T, index=row_labels, columns=labels)
    df.index.name = "par quote"
    trimmed = df.copy()
    trimmed.loc["Total (DV01)"] = df.sum(axis=0)  # row sums recover the parallel DV01

    md = trimmed.to_markdown(floatfmt=",.1f")
    header = (
        "## Par-rate key-rate duration — receive-fixed SOFR swaps ($100mm, struck at par)\n\n"
        f"Market date {AS_OF.to_str()}.  Values are PV change in USD per **+1bp** move in each "
        f"par calibration quote, via the calibration Jacobian ({backend} backend).\n\n"
    )
    print(f"\n=== Par-rate key-rate durations (PV change per +1bp quote, USD) — {backend} ===")
    print(trimmed.to_string(float_format=lambda v: f"{v:,.1f}"))
    if save_path:
        with open(save_path, "w") as fh:
            fh.write(header + md + "\n")
        print(f"\nSaved par-rate KRD table -> {save_path}")
    return trimmed


def compare_par_backends(market: MarketContext, jacobian: np.ndarray) -> None:
    """Sanity check: the numpy J⁻¹ transform must equal the JAX partial_dv01, and it's faster."""
    numpy_pdv01, _ = par_rate_ladder(market, jacobian, backend="numpy")
    jax_pdv01, _ = par_rate_ladder(market, jacobian, backend="jax")  # warms the JIT
    max_abs = float(np.abs(numpy_pdv01 - jax_pdv01).max())

    t0 = time.perf_counter()
    par_rate_ladder(market, jacobian, backend="numpy")
    t_numpy = time.perf_counter() - t0
    t0 = time.perf_counter()
    par_rate_ladder(market, jacobian, backend="jax")  # fresh instance -> recompiles
    t_jax = time.perf_counter() - t0

    print("\n=== Par-rate backend agreement ===")
    print(f"  max |numpy - jax|   : {max_abs:.3e} USD/bp")
    print(f"  numpy one-shot      : {t_numpy * 1e3:8.2f} ms")
    print(f"  jax one-shot (cold) : {t_jax * 1e3:8.2f} ms  (recompiles per fresh instance)")


# ---------------------------------------------------------------------------
# Plot — both discount-factor curves with the calibration nodes marked
# ---------------------------------------------------------------------------

SERIES_BLUE = "#2a78d6"   # GlobalSolver
SERIES_GREEN = "#008300"  # Bootstrapper
LIGHT_BLUE = "#86b6ef"    # lighter tints for node markers
LIGHT_GREEN = "#66b366"


def plot_curves(
    res_global: CalibrationResult,
    res_boot: CalibrationResult,
    save_path: str | None = None,
) -> plt.Figure:
    origin = np.datetime64(AS_OF.to_str(), "D")
    terminal = res_global.pillar_dates[-1]
    total_days = int((terminal - origin) / np.timedelta64(1, "D"))
    grid = origin + np.arange(0, total_days + 1, 7, dtype="timedelta64[D]")
    t = ((grid - origin) / np.timedelta64(365, "D")).astype(np.float64)

    df_g = res_global.curve.discount_factor(grid)
    df_b = res_boot.curve.discount_factor(grid)

    t_nodes = ((res_global.pillar_dates - origin) / np.timedelta64(365, "D")).astype(np.float64)
    df_nodes = res_global.curve.discount_factor(res_global.pillar_dates)

    fig, (ax_df, ax_diff) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    fig.suptitle(f"USD.SOFR calibrated discount factors — {AS_OF.to_str()}", fontsize=13, fontweight="bold")

    ax_df.plot(t, df_g, color=SERIES_BLUE, lw=2.0, label="GlobalSolver (least_squares)")
    ax_df.plot(t, df_b, color=SERIES_GREEN, lw=2.0, ls="--", label="Bootstrapper (brentq)")
    ax_df.scatter(t_nodes, df_nodes, color="black", s=28, zorder=5, label="Calibration nodes")
    for ty in t_nodes:
        ax_df.axvline(ty, color="grey", lw=0.5, ls=":", alpha=0.35, zorder=1)
    ax_df.set_ylabel("Discount factor  DF(t)")
    ax_df.grid(True, alpha=0.25)
    ax_df.legend(loc="upper right", fontsize=9)
    ax_df.set_title("Both solvers, LogLinearDF interpolation — nodes at instrument pillars", fontsize=10, pad=4)

    diff_bp = (df_g - df_b) * 1e4
    ax_diff.plot(t, diff_bp, color=SERIES_BLUE, lw=1.5)
    ax_diff.axhline(0.0, color="grey", lw=0.8)
    for ty in t_nodes:
        ax_diff.axvline(ty, color="grey", lw=0.5, ls=":", alpha=0.35, zorder=1)
    ax_diff.set_ylabel("DF diff (bp)")
    ax_diff.set_xlabel("Time (years)")
    ax_diff.grid(True, alpha=0.25)
    ax_diff.set_title("GlobalSolver − Bootstrapper", fontsize=9, pad=3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot → {save_path}")
    return fig


def plot_daily_log_df_change(
    res_global: CalibrationResult,
    res_boot: CalibrationResult,
    save_path: str | None = None,
) -> plt.Figure:
    """Daily change in log DF — one panel per solver, nodes marked.

    ``log DF(t+1d) - log DF(t)`` is ``-f_on * tau``, so under LogLinearDF this is the
    piecewise-flat overnight forward profile; the steps break exactly at the nodes.
    Plotted in bp per day.
    """
    origin = np.datetime64(AS_OF.to_str(), "D")
    terminal = res_global.pillar_dates[-1]
    total_days = int((terminal - origin) / np.timedelta64(1, "D"))
    grid = origin + np.arange(0, total_days + 1, dtype="timedelta64[D]")
    t = ((grid[:-1] - origin) / np.timedelta64(365, "D")).astype(np.float64)

    t_nodes = ((res_global.pillar_dates - origin) / np.timedelta64(365, "D")).astype(np.float64)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.15},
    )
    fig.suptitle(
        f"USD.SOFR daily change in log discount factors — {AS_OF.to_str()}",
        fontsize=13, fontweight="bold",
    )

    panels = [
        (axes[0], res_global, SERIES_BLUE, LIGHT_BLUE, "GlobalSolver (least_squares)"),
        (axes[1], res_boot, SERIES_GREEN, LIGHT_GREEN, "Bootstrapper (brentq)"),
    ]
    for ax, res, color, node_color, label in panels:
        log_df = np.log(res.curve.discount_factor(grid))
        d_bp = np.diff(log_df) * 1e4

        for ty in t_nodes:
            ax.axvline(ty, color=node_color, lw=1.0, ls="--", zorder=1)
        ax.plot(t, d_bp, color=color, lw=1.5, label=label, zorder=3)
        ax.plot([], [], color=node_color, lw=1.0, ls="--", label="Calibration nodes")
        ax.set_ylabel("Δ log DF per day (bp)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title(label, fontsize=10, pad=4)

    axes[1].set_xlabel("Time (years)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    res_global = calibrate(GlobalSolver(), jacobian=True)
    dump_errors(res_global)

    res_boot = calibrate(Bootstrapper())
    dump_errors(res_boot)

    node_diff = np.abs(res_global.curve.node_dfs - res_boot.curve.node_dfs).max()
    print(f"\nMax |node DF| difference global vs bootstrap: {node_diff:.3e}")

    price_examples(res_global.market)

    key_rate_duration_table(res_global.market, save_path=None)
    par_rate_key_rate_table(
        res_global.market, res_global.jacobian, save_path=None, backend="numpy"
    )
    compare_par_backends(res_global.market, res_global.jacobian)

    plot_curves(res_global, res_boot, save_path="research/sofr_curve_calibration.png")
    plot_daily_log_df_change(res_global, res_boot, save_path="research/sofr_daily_log_df_change.png")
    plt.show()
