"""Figures and snippet outputs for ``docs/curves_line1d.md``.

Run from the repo root (``PYTHONPATH=src python research/curve_doc_figures.py``); writes
PNGs to ``docs/img/curves/`` and prints the values quoted in the doc's snippets.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.math.interpolation import Cubic, Flat, Linear, Mixed, Quadratic  # noqa: E402
from common.math.line import Extrapolation, Line1d  # noqa: E402
from finance.dates import Date, Term  # noqa: E402
from finance.markets import HistoricalFixings, RateGenerator  # noqa: E402
from finance.markets.context import MarketContext  # noqa: E402
from finance.markets.curves import CurveInterpolator, CurveNamespace, CurveSpace, YieldCurve, ZeroCurve  # noqa: E402

OUT = "docs/img/curves"

# validated categorical palette, fixed slot order (dataviz skill reference instance)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "font.family": "sans-serif",
        "font.size": 10,
    }
)


def nodes(ax, x, y):
    ax.scatter(x, y, s=42, color=INK, zorder=5, edgecolors=SURFACE, linewidths=2)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/{name}.png")


# ---------------------------------------------------------------------------------------
# 1. Line1d: the four schemes on the same nodes
# ---------------------------------------------------------------------------------------
X = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
Y = np.array([1.0, 0.5, 2.0, 1.5, 3.0])
XQ = np.linspace(0.0, 6.0, 601)

fig, (ax_v, ax_d) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"hspace": 0.3})
for color, scheme in zip(SERIES, (Flat(), Linear(), Cubic(), Quadratic())):
    line = Line1d(X, Y, scheme)
    label = type(scheme).__name__
    ax_v.plot(XQ, line(XQ), color=color, label=label)
    ax_d.plot(XQ, line.derivative(XQ), color=color, label=label)
nodes(ax_v, X, Y)
ax_v.set_title("Line1d(x, y, interpolator)(xq) — same five nodes, four schemes")
ax_v.set_ylabel("y")
ax_v.legend(loc="upper left", ncol=4)
ax_d.set_title("line.derivative(xq)")
ax_d.set_ylabel("dy/dx")
ax_d.set_xlabel("x")
save(fig, "line1d_interpolators")

# ---------------------------------------------------------------------------------------
# 2. Line1d: extrapolation per side
# ---------------------------------------------------------------------------------------
XQ_WIDE = np.linspace(-1.5, 7.5, 901)
fig, ax = plt.subplots(figsize=(9, 3.8))
for color, (left, right, label) in zip(
    SERIES,
    (
        (Extrapolation.Flat, Extrapolation.Flat, "left=Flat, right=Flat"),
        (Extrapolation.Linear, Extrapolation.Linear, "left=Linear, right=Linear"),
        (Extrapolation.Flat, Extrapolation.Linear, "left=Flat, right=Linear"),
    ),
):
    line = Line1d(X, Y, Cubic(), left=left, right=right)
    ax.plot(XQ_WIDE, line(XQ_WIDE), color=color, label=label)
nodes(ax, X, Y)
for xe in (X[0], X[-1]):
    ax.axvline(xe, color=AXIS, linewidth=0.8)
ax.set_title("Cubic interior, extrapolation chosen per side (NotAllowed raises instead)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper left")
save(fig, "line1d_extrapolation")

# ---------------------------------------------------------------------------------------
# 3. ZeroCurve: the same discount factors under the five named schemes
# ---------------------------------------------------------------------------------------
ORIGIN = np.datetime64("2026-06-01", "D")
NODE_DATES = np.array(
    [
        "2026-06-01",
        "2026-09-01",
        "2026-12-01",
        "2027-06-01",
        "2028-06-01",
        "2029-06-01",
        "2031-06-01",
        "2033-06-01",
        "2036-06-01",
    ],
    dtype="datetime64[D]",
)
X_NODES = (NODE_DATES.view(np.int64) - ORIGIN.astype(np.int64)).astype(np.float64)
Z_NODES = np.array([0.0, 0.0425, 0.0410, 0.0385, 0.0365, 0.0360, 0.0372, 0.0385, 0.0398])
DFS = np.exp(-Z_NODES * X_NODES / 365.0)
DFS[0] = 1.0
XQ_C = np.arange(0.0, X_NODES[-1] + 2 * 365.0, 2.0)
YEARS = XQ_C / 365.0

fig, (ax_z, ax_f) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True, gridspec_kw={"hspace": 0.3})
for color, member in zip(SERIES, CurveInterpolator):
    space, interpolator = member.resolve()
    zc = ZeroCurve(X_NODES, DFS, space=space, interpolator=interpolator)
    ax_z.plot(YEARS, zc.zero_rate(XQ_C) * 100, color=color, label=member.name)
    ax_f.plot(YEARS, zc.instantaneous_forward(XQ_C) * 100, color=color, label=member.name)
nodes(ax_z, X_NODES[1:] / 365.0, Z_NODES[1:] * 100)
for ax in (ax_z, ax_f):
    ax.axvline(X_NODES[-1] / 365.0, color=AXIS, linewidth=0.8)
ax_z.set_title("ZeroCurve.zero_rate(xq) — same pillar DFs, CurveInterpolator.<name>.resolve()")
ax_z.set_ylabel("zero rate, % (Act/365 cc)")
ax_z.legend(loc="lower right", ncol=3)
ax_f.set_title("ZeroCurve.instantaneous_forward(xq) — flat-forward past the last pillar")
ax_f.set_ylabel("instantaneous forward, %")
ax_f.set_xlabel("years from origin")
save(fig, "zero_curve_schemes")


# ---------------------------------------------------------------------------------------
# 4. YieldCurve: pseudo cash rates (3M and 1Y simple, index day count) by scheme
# ---------------------------------------------------------------------------------------
def market_for(yc: YieldCurve) -> MarketContext:
    ns = CurveNamespace()
    ns.bind(yc)
    return MarketContext(as_of_date=Date.from_numpy(ORIGIN), curves=ns)


STARTS = ORIGIN + np.arange(0, int(X_NODES[-1]) - 365, 3).astype("timedelta64[D]")
fig, (ax_3m, ax_1y) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True, gridspec_kw={"hspace": 0.3})
for color, member in zip(
    SERIES,
    (
        CurveInterpolator.LogLinearDF,
        CurveInterpolator.LogCubicDF,
        CurveInterpolator.RateLinear,
        CurveInterpolator.RateCubic,
    ),
):
    space, interpolator = member.resolve()
    yc = YieldCurve.build(NODE_DATES, DFS, currency="USD", index_name="SOFR", space=space, interpolator=interpolator)
    rates = RateGenerator(market_for(yc))
    t = (STARTS.view(np.int64) - ORIGIN.astype(np.int64)) / 365.0
    ax_3m.plot(
        t, rates.simple_rate("USD.SOFR", STARTS, STARTS + np.timedelta64(91, "D")) * 100, color=color, label=member.name
    )
    ax_1y.plot(
        t,
        rates.simple_rate("USD.SOFR", STARTS, STARTS + np.timedelta64(365, "D")) * 100,
        color=color,
        label=member.name,
    )
ax_3m.set_title("RateGenerator.simple_rate — 3M forward cash rate by start date (index day count)")
ax_3m.set_ylabel("3M simple rate, %")
ax_3m.legend(loc="lower right", ncol=2)
ax_1y.set_title("1Y forward cash rate by start date")
ax_1y.set_ylabel("1Y simple rate, %")
ax_1y.set_xlabel("start date, years from origin")
save(fig, "yield_curve_cash_rates")

# ---------------------------------------------------------------------------------------
# 5. Mixed: log-linear front end, natural cubic beyond 2Y
# ---------------------------------------------------------------------------------------
base = YieldCurve.build(NODE_DATES, DFS, currency="USD", index_name="SOFR")
switch = base.node_index(Term.from_str("2Y"))
mixed_zc = ZeroCurve(base.zero_curve.x, base.zero_curve.dfs, interpolator=Mixed(Linear(), Cubic(), switch_node=switch))
mixed = base.with_zero_curve(mixed_zc)
cubic = base.with_zero_curve(ZeroCurve(base.zero_curve.x, base.zero_curve.dfs, interpolator=Cubic()))

fig, (ax_f, ax_c) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True, gridspec_kw={"hspace": 0.3})
t_q = XQ_C / 365.0
t_s = (STARTS.view(np.int64) - ORIGIN.astype(np.int64)) / 365.0
for color, (yc, label) in zip(
    SERIES,
    ((base, "Linear (log-DF)"), (cubic, "Cubic (log-DF)"), (mixed, f"Mixed(Linear, Cubic, switch_node={switch})")),
):
    ax_f.plot(t_q, yc.zero_curve.instantaneous_forward(XQ_C) * 100, color=color, label=label)
    ax_c.plot(
        t_s,
        RateGenerator(market_for(yc)).simple_rate("USD.SOFR", STARTS, STARTS + np.timedelta64(91, "D")) * 100,
        color=color,
        label=label,
    )
for ax in (ax_f, ax_c):
    ax.axvline(base.zero_curve.x[switch] / 365.0, color=AXIS, linewidth=0.8)
    ax.axvline(X_NODES[-1] / 365.0, color=AXIS, linewidth=0.8)
ax_f.set_title("instantaneous forward — Mixed switches scheme at the 2Y pillar (C0 join)")
ax_f.set_ylabel("instantaneous forward, %")
ax_f.legend(loc="lower right")
ax_c.set_title("3M forward cash rate")
ax_c.set_ylabel("3M simple rate, %")
ax_c.set_xlabel("years from origin")
save(fig, "yield_curve_mixed")

# ---------------------------------------------------------------------------------------
# Snippet outputs quoted in the doc
# ---------------------------------------------------------------------------------------
np.set_printoptions(precision=6, suppress=True)
print("\n--- Line1d basics")
line = Line1d(X, Y, Linear())
print("line(np.array([0.5, 1.75, 6.0])) ->", line(np.array([0.5, 1.75, 6.0])))
print("line.derivative(np.array([0.5, 1.75])) ->", line.derivative(np.array([0.5, 1.75])))
print("line.coefficients ->\n", line.coefficients)
flat = Line1d(np.array([0.0, 3.0]), np.array([5.30, 5.32]), Flat(), left=Extrapolation.Flat, right=Extrapolation.Flat)
print("flat(np.array([0., 1., 2., 3.])) ->", flat(np.array([0.0, 1.0, 2.0, 3.0])))
try:
    line(np.array([6.5, 9.0]))
except ValueError as exc:
    print("NotAllowed ->", exc)
ext = Line1d(X, Y, Linear(), left=Extrapolation.Flat, right=Extrapolation.Linear)
print("ext(np.array([-2., 8.])) ->", ext(np.array([-2.0, 8.0])))
cubic_line = Line1d(X, Y, Cubic(bc_type=((2, 0.5), (1, 0.0))))
print("cubic_line.coefficients.shape ->", cubic_line.coefficients.shape)
print("with_y ->", line.with_y(Y * 2)(np.array([1.75])))

print("\n--- ZeroCurve")
zc = ZeroCurve(X_NODES, DFS)
print("zc ->", zc)
xq = np.array([0.0, 182.0, 365.0, 3650.0, 4000.0])
print("zc.discount_factor(xq) ->", zc.discount_factor(xq))
print("zc.zero_rate(xq) ->", zc.zero_rate(xq))
print("zc.instantaneous_forward(xq) ->", zc.instantaneous_forward(xq))
print("zc.node_zero_rates ->", zc.node_zero_rates)
print("zc.is_log_linear ->", zc.is_log_linear, "| line.coefficients.shape ->", zc.line.coefficients.shape)
rc = ZeroCurve(X_NODES, DFS, space=CurveSpace.ZeroRate, interpolator=Cubic())
print("rc ->", rc)
print("rc.zero_rate(xq) ->", rc.zero_rate(xq))
print("CurveInterpolator.RateCubic.resolve() ->", CurveInterpolator.RateCubic.resolve())

print("\n--- YieldCurve")
yc = YieldCurve.build(NODE_DATES, DFS, currency="USD", index_name="SOFR")
print("yc ->", yc)
dates = np.array(["2026-06-01", "2027-06-01", "2040-06-01"], dtype="datetime64[D]")
print("yc.to_x(dates) ->", yc.to_x(dates))
print("yc.discount_factor(dates) ->", yc.discount_factor(dates))
print("yc.zero_rate(dates) ->", yc.zero_rate(dates))
print("yc.node_dates[:3] ->", yc.node_dates[:3])
print("yc.node_index(Term.from_str('2Y')) ->", yc.node_index(Term.from_str("2Y")))
print("yc.node_index(np.datetime64('2031-06-01')) ->", yc.node_index(np.datetime64("2031-06-01")))
history = HistoricalFixings(
    dates=np.array(["2026-05-27", "2026-05-29"], dtype="datetime64[D]"), values=np.array([0.0428, 0.0431])
)
seasoned = yc.with_historical_fixings(history)
rates = RateGenerator(market_for(seasoned))
starts = np.array(["2026-05-28", "2026-05-30", "2026-06-01"], dtype="datetime64[D]")
print(
    "simple_rate over history + projection ->", rates.simple_rate("USD.SOFR", starts, starts + np.timedelta64(1, "D"))
)
print("mixed ->", mixed.zero_curve)
print(
    "mixed forward at 2Y - 1d / + 1d ->",
    mixed.zero_curve.instantaneous_forward(np.array([X_NODES[switch] - 1.0, X_NODES[switch] + 1.0])),
)
