"""Seasoned SOFR cashflows: locked history, stubbed index risk, and expired payments."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date, Term, TermType
from finance.instruments.resolution import Swap
from finance.markets import HistoricalFixings
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, YieldCurve, ZeroCurve
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import NumbaRisk, bumped_curve
from finance.pricing.types import Backend, RateKind

pytest.importorskip("numba")

CURVE = "USD.SOFR"
ORIGIN = np.datetime64("2026-06-01", "D")


def _sofr_history() -> HistoricalFixings:
    """Load the repository's New York Fed-style SOFR history fixture."""
    path = Path(__file__).parents[4] / "data" / "rates.csv"
    observations: list[tuple[np.datetime64, float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Rate Type"] == "SOFR":
                date = datetime.strptime(row["Effective Date"], "%m/%d/%Y").date()
                observations.append((np.datetime64(date, "D"), float(row["Rate (%)"]) / 100.0))
    observations.sort(key=lambda item: item[0])
    return HistoricalFixings(
        dates=np.array([item[0] for item in observations], dtype="datetime64[D]"),
        values=np.array([item[1] for item in observations], dtype=np.float64),
    )


def _market() -> MarketContext:
    dates = np.array([ORIGIN, "2027-06-01", "2028-06-01", "2030-06-01"], dtype="datetime64[D]")
    years = (dates.astype(np.int64) - ORIGIN.astype(np.int64)) / 365.0
    zero = ZeroCurve(dates, np.exp(-0.04 * years), CurveInterpolator.LogLinearDF)
    yield_curve = YieldCurve.from_registry(
        zero,
        currency="USD",
        index_name="SOFR",
        historical_fixings=_sofr_history(),
    )
    curves = CurveNamespace()
    curves.bind(yield_curve)
    return MarketContext(as_of_date=Date.from_numpy(ORIGIN), curves=curves)


def _seasoned_book() -> list[Swap]:
    # Final accrual ends before the curve date but a long payment delay leaves it unpaid:
    # its floating coupon is pure history and must have zero index risk.
    pure_history = Swap.fixed_float_swap(
        notional=25e6,
        rate_index="SOFR",
        fixed_rate=0.04,
        tenor="2Y",
        as_of=Date(2024, 5, 27),
        payment_delay=Term(60, TermType.BusinessDays),
    )
    # The middle annual coupon straddles the curve origin, so only its remaining daily
    # observations carry projection risk. Its first annual payment is already expired.
    straddling = Swap.fixed_float_swap(
        notional=25e6,
        rate_index="SOFR",
        fixed_rate=0.04,
        tenor="3Y",
        as_of=Date(2024, 9, 16),
    )
    return [pure_history, straddling]


@pytest.mark.risk
class TestSeasonedSofrRisk(UnitTest):
    COVERAGE = [
        "finance.markets.fixings",
        "finance.markets.rate_generator",
        "finance.pricing.engines.numba.program",
        "finance.pricing.risk.adjoint",
    ]

    def setUp(self):
        self.market = _market()
        self.program = SwapPricer().compile(_seasoned_book(), backend=Backend.Numba)
        self.inputs = self.program.inputs
        self.pricing = self.program.price(self.market)
        self.risk = NumbaRisk(self.program, self.market)

    def _observation_classification(self) -> tuple[np.ndarray, np.ndarray]:
        pure_history = np.zeros(self.inputs.n_flows, dtype=np.bool_)
        straddling = np.zeros(self.inputs.n_flows, dtype=np.bool_)
        observation_count = self.inputs.obs_starts.size
        ends = np.append(self.inputs.obs_offsets[1:], observation_count)
        for period, flow in enumerate(self.inputs.obs_flow):
            starts = self.inputs.obs_starts[self.inputs.obs_offsets[period] : ends[period]]
            pure_history[flow] = bool(np.all(starts < ORIGIN))
            straddling[flow] = bool(np.any(starts < ORIGIN) and np.any(starts >= ORIGIN))
        return pure_history, straddling

    def test_expired_payments_have_zero_df_pv_and_risk(self):
        expired = self.inputs.pay_dates < ORIGIN
        self.assertTrue(expired.any())
        np.testing.assert_array_equal(self.pricing.cashflows.df[expired], 0.0)
        np.testing.assert_array_equal(self.pricing.cashflows.flow_pv[expired], 0.0)
        np.testing.assert_array_equal(self.risk.cashflow_index_delta(CURVE)[expired], 0.0)
        np.testing.assert_array_equal(self.risk.cashflow_funding_delta(CURVE)[expired], 0.0)

    def test_unpaid_pure_history_coupon_has_no_index_risk(self):
        pure_history, _ = self._observation_classification()
        unpaid = self.inputs.pay_dates >= ORIGIN
        floating = self.inputs.rate_kind == int(RateKind.Compounded)
        target = pure_history & unpaid & floating
        self.assertTrue(target.any())
        np.testing.assert_array_equal(self.risk.cashflow_index_delta(CURVE)[target], 0.0)
        self.assertGreater(np.abs(self.risk.cashflow_funding_delta(CURVE)[target]).sum(), 0.0)

    def test_straddling_coupon_has_stubbed_index_risk_matching_reprice(self):
        _, straddling = self._observation_classification()
        target = straddling & (self.inputs.pay_dates >= ORIGIN)
        self.assertTrue(target.any())
        index_delta = self.risk.cashflow_index_delta(CURVE)[target]
        analytic = self.risk.cashflow_zero_delta(CURVE)[target]
        self.assertGreater(np.abs(index_delta).sum(), 0.0)

        curve = self.market.zero_curve(CURVE)
        yield_curve = self.market.yield_curve(CURVE)
        finite_difference = np.zeros_like(analytic)
        for column, pillar in enumerate(range(1, curve.node_dates.size)):
            up = self.market.with_curve(
                yield_curve.with_zero_curve(bumped_curve(curve, 1e-4, pillar=pillar))
            )
            down = self.market.with_curve(
                yield_curve.with_zero_curve(bumped_curve(curve, -1e-4, pillar=pillar))
            )
            up_flow = self.program.price(up).cashflows.flow_pv[target]
            down_flow = self.program.price(down).cashflows.flow_pv[target]
            finite_difference[:, column] = (up_flow - down_flow) / 2.0
        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-4, atol=1e-4)

    def test_numba_and_numpy_seasoned_prices_match(self):
        numpy = self.program.reprice(self.market, backend=Backend.Numpy)
        numba = self.program.reprice(self.market, backend=Backend.Numba)
        np.testing.assert_allclose(numba.instrument_pv, numpy.instrument_pv, rtol=1e-11, atol=1e-6)
        np.testing.assert_allclose(numba.flow_pv, numpy.flow_pv, rtol=1e-11, atol=1e-7)
        np.testing.assert_allclose(numba.df, numpy.df, rtol=1e-13, atol=1e-13)

    def test_jax_and_numba_agree_on_seasoned_index_risk(self):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        from finance.pricing.risk.autodiff import AutodiffRisk

        pure_history, straddling = self._observation_classification()
        active = self.inputs.pay_dates >= ORIGIN
        jax_index = AutodiffRisk(self.program, self.market).cashflow_index_delta(CURVE)
        numba_index = self.risk.cashflow_index_delta(CURVE)
        np.testing.assert_array_equal(jax_index[pure_history & active], 0.0)
        self.assertGreater(np.abs(jax_index[straddling & active]).sum(), 0.0)
        np.testing.assert_allclose(jax_index, numba_index, rtol=2e-9, atol=1e-6)
