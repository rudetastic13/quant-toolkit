# python
import numpy as np
import matplotlib.pyplot as plt

from src.core.dates.date import Date
from src.core.markets.curves.types import (
    ValueType,
    CurveInterpolator,
    CurveExtrapolator,
)
from src.core.markets.curves._curve_impl.zero_curve import ZeroCurve


def _make_example_inputs(curve_date: Date):
    # node offsets in days (roughly quarterly-ish out to 10Y)
    offsets = np.array([0, 30, 90, 180, 365, 2 * 365, 3 * 365, 5 * 365, 7 * 365, 10 * 365], dtype=np.int64)

    # create a "reasonable" continuously-compounded zero curve in decimals
    rng = np.random.default_rng(7)
    base = 0.020
    slope = 0.012  # long-end around 3.2%
    t = offsets.astype(np.float64) / 365
    zero_rates = base + slope * (1.0 - np.exp(-t / 3.0))  # smooth upward shape
    zero_rates += rng.normal(0.0, 0.0008, size=zero_rates.shape)  # small noise
    zero_rates = np.clip(zero_rates, 0.0001, None)

    # consistent discount factors and log DFs for those nodes
    log_dfs = zero_rates * t  # log_df = r * t
    dfs = np.exp(-log_dfs)

    # node dates as datetime64[D] for querying
    curve_date_np = curve_date.to_numpy().astype("datetime64[D]")
    node_dates = curve_date_np + offsets.astype("timedelta64[D]")

    return offsets, node_dates, dfs.astype(np.float64), log_dfs.astype(np.float64), zero_rates.astype(np.float64)


def main():
    curve_date = Date(2026, 1, 21)
    offsets, node_dates, dfs, log_dfs, zero_rates = _make_example_inputs(curve_date)

    curves = [
        ("From DF", ZeroCurve(curve_date, offsets, dfs, ValueType.DiscountFactor, CurveInterpolator.Linear, CurveExtrapolator.FlatForward)),
        ("From log DF", ZeroCurve(curve_date, offsets, log_dfs, ValueType.LogDiscountFactor, CurveInterpolator.Linear, CurveExtrapolator.FlatForward)),
        ("From zero rates", ZeroCurve(curve_date, offsets, zero_rates, ValueType.ZeroRate, CurveInterpolator.Linear, CurveExtrapolator.FlatForward)),
    ]

    curve_date_np = curve_date.to_numpy().astype("datetime64[D]")

    # query dates (monthly grid out to 10Y) \-\- ignore day 0
    query_offsets = np.arange(0, 10 * 365 + 1, 10, dtype=np.int64)
    query_offsets = query_offsets[query_offsets != 0]
    query_dates = curve_date_np + query_offsets.astype("timedelta64[D]")

    plt.figure(figsize=(10, 5))
    for label, c in curves:
        zr = c.zero_rate(query_dates)
        plt.plot(query_offsets / 365, 100.0 * zr, label=label, linewidth=2)

    # overlay node points \-\- ignore day 0 node
    node_mask = offsets != 0
    node_zr = curves[0][1].zero_rate(node_dates[node_mask])
    plt.scatter(
        offsets[node_mask] / 365,
        100.0 * node_zr,
        color="black",
        s=25,
        zorder=5,
        label="Nodes",
    )

    plt.title("ZeroCurve zero rates (continuous compounding)")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Zero rate (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()