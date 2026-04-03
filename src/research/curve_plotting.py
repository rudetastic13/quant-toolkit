"""
Visualization module for comparing curve interpolation methods.

Produces two figures, each with four panels:
  1. Single-interpolation comparison — all five InterpTypes.
  2. Cutover-interpolation comparison — selected short/long pairs,
     cutover fixed at 2 years from the curve origin.

Panels per figure:
  - Top         : Discount Factor DF(t)
  - 2nd         : Zero Rate r(t)  [%]
  - 3rd         : 1-Day Forward Rates [%]  (grid[:-1] → grid[1:])
  - Bottom      : 1-Month Forward Rates [%]  (grid[:-1] → grid[:-1] + 30 days)

Both forward panels extend 1 year past the terminal node to expose
flat-forward extrapolation behaviour.  A vertical line marks the terminal
node on every forward panel.

Imports all curve machinery from more_interp_curve.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from numpy.typing import NDArray
from finance.dates import Term, TermType
from finance.markets.curves import ZeroCurve, CurveInterpolator

# ---------------------------------------------------------------------------
# Shared node inputs  (np.ndarray[datetime64[D]] + np.ndarray[float64])
# ---------------------------------------------------------------------------

NODE_DATES = np.array([
    "2024-01-01",   # origin — DF must be 1.0
    "2024-07-01",
    "2025-01-01",
    "2026-01-01",   # exactly 2 yr from origin
    "2027-01-01",
    "2028-01-01",
    "2029-01-01",
], dtype="datetime64[D]")

NODE_VALUES = np.array(
    [1.0, 0.9975, 0.9940, 0.9850, 0.9730, 0.9600, 0.9460],
    dtype=np.float64,
)

ORIGIN: np.datetime64   = NODE_DATES[0]
TERMINAL: np.datetime64 = NODE_DATES[-1]
PILLAR_DATES = NODE_DATES[1:]   # exclude origin

CUTOVER_TERM = Term(2, TermType.Years)   # always 2-year cutover

# ---------------------------------------------------------------------------
# Cutover combinations to compare  (short-end method, long-end method)
# ---------------------------------------------------------------------------

CUTOVER_PAIRS: list[tuple[CurveInterpolator, CurveInterpolator]] = [
    (CurveInterpolator.LogLinearDF, CurveInterpolator.LogCubicDF),
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_SINGLE_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
_CUTOVER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ---------------------------------------------------------------------------
# Date grids
# ---------------------------------------------------------------------------

def _date_grid(n: int = 600) -> NDArray[np.datetime64]:
    """Dense grid of datetime64[D] from day 1 to the terminal node."""
    total_days = int((TERMINAL - ORIGIN) / np.timedelta64(1, "D"))
    offsets = np.round(np.linspace(1, total_days, n)).astype(int)
    return ORIGIN + offsets.astype("timedelta64[D]")


def _fwd_date_grid() -> NDArray[np.datetime64]:
    """
    Daily grid from the curve origin to 1 year past the terminal node.
    Used for both forward rate panels; the extra year shows flat-forward
    extrapolation behaviour.
    """
    end = TERMINAL + np.timedelta64(365, "D")
    total_days = int((end - ORIGIN) / np.timedelta64(1, "D"))
    return ORIGIN + np.arange(0, total_days + 1, dtype="timedelta64[D]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_years(dates: NDArray[np.datetime64]) -> NDArray[np.float64]:
    return ((dates - ORIGIN) / np.timedelta64(365, "D")).astype(np.float64)


def _eval_curve(
    curve: ZeroCurve,
    dates: NDArray[np.datetime64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (discount_factors, zero_rates_pct) for a smooth date grid."""
    return curve.discount_factor(dates), curve.rate(dates) * 100.0


def _eval_fwd(
    curve: ZeroCurve,
    fwd_grid: NDArray[np.datetime64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Return (t_start_years, fwd_1d_pct, fwd_1m_pct).

    1-day:   starts = fwd_grid[:-1],  ends = fwd_grid[1:]       (each step = 1 day)
    1-month: starts = fwd_grid[:-1],  ends = starts + 30 days
    """
    starts  = fwd_grid[:-1]
    ends_1d = fwd_grid[1:]
    ends_1m = starts + np.timedelta64(30, "D")

    fwd_1d = curve.forward_rate(starts, ends_1d) * 100.0
    fwd_1m = curve.forward_rate(starts, ends_1m) * 100.0
    return _to_years(starts), fwd_1d, fwd_1m


def _pillar_values(
    curve: ZeroCurve,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Pillar times-in-years, DF values, and zero-rate-pct values."""
    t_yr = _to_years(PILLAR_DATES)
    return t_yr, curve.discount_factor(PILLAR_DATES), curve.rate(PILLAR_DATES) * 100.0


def _label(itype: CurveInterpolator) -> str:
    return itype.name


def _cutover_label(short: CurveInterpolator, long: CurveInterpolator) -> str:
    return f"{_label(short)}  →  {_label(long)}"


def _mark_terminal(ax: plt.Axes, terminal_t: float) -> None:
    """Draw a vertical line at the terminal node on a forward rate panel."""
    ax.axvline(terminal_t, color="dimgrey", lw=1.2, ls="--", alpha=0.85, zorder=3)
    ylim = ax.get_ylim()
    ax.text(
        terminal_t + 0.05, ylim[0] + (ylim[1] - ylim[0]) * 0.97,
        "terminal node\n(extrapolation →)",
        color="dimgrey", fontsize=7, va="top", rotation=90,
    )


def _mark_cutover(ax: plt.Axes, cutover_t: float) -> None:
    """Draw a vertical line at the 2-year cutover on an axes."""
    ax.axvline(cutover_t, color="grey", lw=1.2, ls=":", alpha=0.7, zorder=3)
    ylim = ax.get_ylim()
    ax.text(
        cutover_t + 0.05, ylim[0] + (ylim[1] - ylim[0]) * 0.03,
        "2yr cutover",
        color="grey", fontsize=7, va="bottom", rotation=90,
    )


# ---------------------------------------------------------------------------
# Figure 1: Single-interpolation comparison
# ---------------------------------------------------------------------------

def plot_single_interpolations(save_path: str | None = None) -> plt.Figure:
    """
    Four-panel figure comparing all five single-interpolation methods:
      - Panel 1 : Discount Factor DF(t)
      - Panel 2 : Zero Rate r(t) [%]
      - Panel 3 : 1-Day Forward Rate [%]   — grid[:-1] → grid[1:]
      - Panel 4 : 1-Month Forward Rate [%] — grid[:-1] → grid[:-1] + 30 days
    Panels 3 & 4 extend 1 year past the terminal node; a vertical line
    marks where extrapolation begins.
    """
    dates      = _date_grid()
    fwd_grid   = _fwd_date_grid()
    t_years    = _to_years(dates)
    terminal_t = float(_to_years(np.array([TERMINAL]))[0])

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 16), sharex=False,
        gridspec_kw={"hspace": 0.38},
    )
    ax_df, ax_rate, ax_1d, ax_1m = axes
    fig.suptitle(
        "Single-Interpolation Comparison — All Five Methods\n"
        "(Darbyshire §6.3.1)",
        fontsize=13, fontweight="bold",
    )

    ref_curve = ZeroCurve(NODE_DATES, NODE_VALUES, CurveInterpolator.LogLinearDF)
    t_pil, df_pil, r_pil = _pillar_values(ref_curve)

    for idx, itype in enumerate(CurveInterpolator):
        curve = ZeroCurve(NODE_DATES, NODE_VALUES, itype)
        dfs, rates = _eval_curve(curve, dates)
        t_mid, fwd_1d, fwd_1m = _eval_fwd(curve, fwd_grid)
        lbl   = _label(itype)
        color = _SINGLE_COLORS[idx]

        ax_df.plot(t_years, dfs,    color=color, lw=1.8, label=lbl)
        ax_rate.plot(t_years, rates, color=color, lw=1.8, label=lbl)
        ax_1d.plot(t_mid, fwd_1d,   color=color, lw=1.4, label=lbl)
        ax_1m.plot(t_mid, fwd_1m,   color=color, lw=1.4, label=lbl)

    ax_df.scatter(t_pil, df_pil, color="black", zorder=6, s=45, label="Pillar nodes")
    ax_rate.scatter(t_pil, r_pil, color="black", zorder=6, s=45)

    ax_df.set_ylabel("Discount Factor  DF(t)", fontsize=11)
    ax_df.set_xlabel("Time (years)", fontsize=10)
    ax_df.legend(fontsize=9, loc="upper right")
    ax_df.grid(True, alpha=0.3)
    ax_df.set_title("Discount Factor", fontsize=10, pad=4)

    ax_rate.set_ylabel("Zero Rate  r(t)  [%]", fontsize=11)
    ax_rate.set_xlabel("Time (years)", fontsize=10)
    ax_rate.legend(fontsize=9, loc="lower right")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_title("Continuously Compounded Zero Rate", fontsize=10, pad=4)

    ax_1d.set_ylabel("1-Day Forward Rate  [%]", fontsize=11)
    ax_1d.set_xlabel("Start Date (years from origin)", fontsize=10)
    ax_1d.legend(fontsize=9, loc="upper right")
    ax_1d.grid(True, alpha=0.3)
    ax_1d.set_title("1-Day Forward Rate", fontsize=10, pad=4)
    _mark_terminal(ax_1d, terminal_t)

    ax_1m.set_ylabel("1-Month Forward Rate  [%]", fontsize=11)
    ax_1m.set_xlabel("Start Date (years from origin)", fontsize=10)
    ax_1m.legend(fontsize=9, loc="upper right")
    ax_1m.grid(True, alpha=0.3)
    ax_1m.set_title("1-Month Forward Rate", fontsize=10, pad=4)
    _mark_terminal(ax_1m, terminal_t)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Cutover-interpolation comparison (cutover always at 2 years)
# ---------------------------------------------------------------------------

def plot_cutover_interpolations(save_path: str | None = None) -> plt.Figure:
    """
    Four-panel figure comparing selected short/long cutover combinations.
    Cutover is fixed at 2 years from the curve origin.

      - Panel 1 : Discount Factor DF(t)
      - Panel 2 : Zero Rate r(t) [%]
      - Panel 3 : 1-Day Forward Rate [%]
      - Panel 4 : 1-Month Forward Rate [%]

    Dotted vertical line marks the 2-year cutover on all panels.
    Dashed vertical line marks the terminal node on the forward panels.
    """
    dates      = _date_grid()
    fwd_grid   = _fwd_date_grid()
    t_years    = _to_years(dates)
    cutover_t  = 2.0
    terminal_t = float(_to_years(np.array([TERMINAL]))[0])

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 16), sharex=False,
        gridspec_kw={"hspace": 0.38},
    )
    ax_df, ax_rate, ax_1d, ax_1m = axes
    fig.suptitle(
        "Cutover-Interpolation Comparison — 2-Year Cutover\n"
        "(short-end method  →  long-end method)",
        fontsize=13, fontweight="bold",
    )

    for idx, (short, long) in enumerate(CUTOVER_PAIRS):
        curve = ZeroCurve(
            NODE_DATES,
            NODE_VALUES,
            interpolation=short,
            interpolation_long=long,
            interpolation_cutover=CUTOVER_TERM,
        )
        dfs, rates = _eval_curve(curve, dates)
        t_mid, fwd_1d, fwd_1m = _eval_fwd(curve, fwd_grid)
        lbl   = _cutover_label(short, long)
        color = _CUTOVER_COLORS[idx]

        ax_df.plot(t_years, dfs,    color=color, lw=1.8, label=lbl)
        ax_rate.plot(t_years, rates, color=color, lw=1.8, label=lbl)
        ax_1d.plot(t_mid, fwd_1d,   color=color, lw=1.4, label=lbl)
        ax_1m.plot(t_mid, fwd_1m,   color=color, lw=1.4, label=lbl)

    ref_curve = ZeroCurve(NODE_DATES, NODE_VALUES, CurveInterpolator.LogLinearDF)
    t_pil, df_pil, r_pil = _pillar_values(ref_curve)
    ax_df.scatter(t_pil, df_pil, color="black", zorder=6, s=45, label="Pillar nodes")
    ax_rate.scatter(t_pil, r_pil, color="black", zorder=6, s=45)

    # Cutover marker on all four panels
    for ax in axes:
        _mark_cutover(ax, cutover_t)

    # Terminal node marker on the two forward panels only
    _mark_terminal(ax_1d, terminal_t)
    _mark_terminal(ax_1m, terminal_t)

    cutover_line = mlines.Line2D([], [], color="grey", ls=":", lw=1.2, label="2yr cutover")

    ax_df.set_ylabel("Discount Factor  DF(t)", fontsize=11)
    ax_df.set_xlabel("Time (years)", fontsize=10)
    ax_df.grid(True, alpha=0.3)
    ax_df.set_title("Discount Factor", fontsize=10, pad=4)
    handles_df, labels_df = ax_df.get_legend_handles_labels()
    ax_df.legend(handles_df + [cutover_line], labels_df + ["2yr cutover"],
                 fontsize=8, loc="upper right")

    ax_rate.set_ylabel("Zero Rate  r(t)  [%]", fontsize=11)
    ax_rate.set_xlabel("Time (years)", fontsize=10)
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_title("Continuously Compounded Zero Rate", fontsize=10, pad=4)
    ax_rate.legend(fontsize=8, loc="lower right")

    ax_1d.set_ylabel("1-Day Forward Rate  [%]", fontsize=11)
    ax_1d.set_xlabel("Start Date (years from origin)", fontsize=10)
    ax_1d.legend(fontsize=8, loc="upper right")
    ax_1d.grid(True, alpha=0.3)
    ax_1d.set_title("1-Day Forward Rate", fontsize=10, pad=4)

    ax_1m.set_ylabel("1-Month Forward Rate  [%]", fontsize=11)
    ax_1m.set_xlabel("Start Date (years from origin)", fontsize=10)
    ax_1m.legend(fontsize=8, loc="upper right")
    ax_1m.grid(True, alpha=0.3)
    ax_1m.set_title("1-Month Forward Rate", fontsize=10, pad=4)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig1 = plot_single_interpolations(
        save_path="single_interp_comparison.png"
    )
    fig2 = plot_cutover_interpolations(
        save_path="cutover_interp_comparison.png"
    )
    plt.show()