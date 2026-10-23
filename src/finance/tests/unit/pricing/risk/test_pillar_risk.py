"""R1: PillarRisk adapters agree with today's doors; template_jacobian agrees with the calibrator and with bumps."""

import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import SofrFuture, Swap, Swaption, SwaptionModel
from finance.markets.context import MarketContext
from finance.markets.curves import YieldCurve, bp
from finance.markets.vols import FlatVolSurface, VolNamespace, VolUnits
from finance.pricing.calibration import deposit_helper, fra_helper, futures_helper, swap_helper
from finance.pricing.pricers import FuturesPricer, SwapPricer, SwaptionPricer
from finance.pricing.risk import NumbaRisk, calibration_jacobian, pillar_risk, template_jacobian

SOFR = "USD.SOFR.1D"
STDCSA = "USD.STDCSA"
FEDFUND = "USD.FEDFUND.1D"

pytest.importorskip("numba")


def _market(as_of: Date) -> MarketContext:
    origin = as_of.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 91, 182, 365, 730, 1825, 3650)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.041, 0.040, 0.039, 0.038, 0.040, 0.043])
    sofr = YieldCurve.build(dates, np.exp(-z * t), key=SOFR)
    vols = VolNamespace()
    vols.bind(SOFR, FlatVolSurface(0.01, VolUnits.Normal))
    market = MarketContext(as_of, vols=vols)
    market[SOFR] = sofr
    market[STDCSA] = sofr
    market[FEDFUND] = market[SOFR] - 5 * bp
    return market.freeze()


def _fd(helpers, market, key, h=1e-6):
    """Central differences of every helper's implied quote over the block ``key``."""
    n = sum(a.n_params for a in market.owned(key))
    out = np.empty((len(helpers), n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        up, dn = market.bump(key, e), market.bump(key, -e)
        out[:, j] = [(hp.implied(up) - hp.implied(dn)) / (2 * h) for hp in helpers]
    return out


@pytest.mark.risk
class TestPillarRiskAdapters(UnitTest):
    COVERAGE = ["finance.pricing.risk.pillar", "finance.pricing.pricers.futures", "finance.pricing.pricers.swaption"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.book = [
            Swap.fixed_float_swap(notional=100e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=self.as_of),
            Swap.fixed_float_swap(notional=-50e6, rate_index="USD.FEDFUND.1D", fixed_rate=0.04, tenor="3Y", as_of=self.as_of),
        ]

    def test_leg_program_matches_numba_risk_per_block(self):
        program = SwapPricer().compile(self.book)
        risk = pillar_risk(program, self.market, labels=("5Y SOFR", "3Y FF"))
        door = NumbaRisk(program, self.market)
        self.assertEqual(risk.curves, (SOFR, FEDFUND))
        self.assertEqual(risk.labels, ("5Y SOFR", "3Y FF"))
        np.testing.assert_array_equal(risk.block(SOFR), door.unit_zero_jacobian(SOFR))
        np.testing.assert_array_equal(risk.block(FEDFUND), door.unit_zero_jacobian(FEDFUND))
        np.testing.assert_allclose(risk.pv, program.reprice(self.market).instrument_pv, rtol=1e-10)
        np.testing.assert_array_equal(risk.pillar_dates[SOFR], self.market.yield_curve(SOFR).node_dates[1:])
        np.testing.assert_allclose(risk.dv01, risk.values * 1e-4)
        self.assertEqual(risk.block(FEDFUND).shape, (2, 1))
        self.assertAlmostEqual(risk.block(FEDFUND)[0, 0], 0.0)  # the SOFR swap carries no basis risk

    def test_futures_program_matches_the_door_and_handles_an_expired_contract(self):
        from finance.instruments.enums import CouponType

        expired = SofrFuture(  # settled a year ago; the factory refuses these, a book can still hold one
            contract="SR3",
            ref_start=Date(2025, 3, 19),
            ref_end=Date(2025, 6, 18),
            coupon_type=CouponType.GeometricAveraged,
            entry_price=96.0,
        )
        strip = [
            expired,
            SofrFuture.from_contract_month(contract="SR3", month="U26", as_of=self.as_of, entry_price=96.0),
            SofrFuture.from_contract_month(contract="SR3", month="Z26", as_of=self.as_of),
        ]
        program = FuturesPricer().compile(strip)
        risk = pillar_risk(program, self.market)
        self.assertEqual(risk.curves, (SOFR,))
        np.testing.assert_array_equal(risk.block(SOFR), program.risk(self.market, SOFR).pnl_gradient)
        self.assertTrue(np.isfinite(risk.values).all())
        np.testing.assert_array_equal(risk.values[0], 0.0)  # expired: no risk
        self.assertGreater(np.abs(risk.values[1]).sum(), 0.0)
        pnl = program.reprice(self.market).pnl
        np.testing.assert_array_equal(risk.pv[:2], pnl[:2])
        self.assertEqual(risk.pv[2], 0.0)  # no entry price

    def test_swaption_program_matches_the_door(self):
        option = Swaption.european(
            notional=1e6,
            strike=0.04,
            expiry="1Y",
            swap_tenor="5Y",
            as_of=self.as_of,
            payer=True,
            model=SwaptionModel.Bachelier,
        )
        program = SwaptionPricer().compile([option])
        risk = pillar_risk(program, self.market)
        np.testing.assert_array_equal(risk.block(SOFR), program.risk(self.market, SOFR).curve_gradient)
        np.testing.assert_array_equal(risk.pv, program.reprice(self.market).pv)

    def test_aligned_reorders_zero_fills_and_checks_pillars(self):
        risk = pillar_risk(SwapPricer().compile(self.book[:1]), self.market)
        dates = {
            FEDFUND: self.market.yield_curve(FEDFUND).node_dates[-1:],
            SOFR: risk.pillar_dates[SOFR],
            "USD.UST": np.array(["2027-06-01"], dtype="datetime64[D]"),
        }
        moved = risk.aligned((FEDFUND, "USD.UST", SOFR), dates)
        self.assertEqual(moved.curves, (FEDFUND, "USD.UST", SOFR))
        np.testing.assert_array_equal(moved.block(SOFR), risk.block(SOFR))
        np.testing.assert_array_equal(moved.block("USD.UST"), 0.0)
        with self.assertRaisesRegex(ValueError, "pillar dates"):
            risk.aligned((SOFR,), {SOFR: risk.pillar_dates[SOFR][:-1]})
        with self.assertRaises(KeyError):
            risk.block("USD.UST")


@pytest.mark.risk
class TestTemplateJacobian(UnitTest):
    COVERAGE = ["finance.pricing.risk.jacobian"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.sofr_helpers = [
            deposit_helper(rate=0.0, tenor="3M", as_of=self.as_of),
            fra_helper(rate=0.0, start="3M", end="6M", as_of=self.as_of),
            futures_helper(price=96.0, contract="SR3", month="U26", as_of=self.as_of),
            swap_helper(rate=0.0, tenor="2Y", as_of=self.as_of),
            swap_helper(rate=0.0, tenor="5Y", as_of=self.as_of),
        ]
        self.ff_helpers = [swap_helper(rate=0.0, tenor=t, as_of=self.as_of, rate_index="USD.FEDFUND.1D") for t in ("2Y", "5Y")]

    def test_rows_follow_helper_order_and_carry_the_implied_quote(self):
        jac = template_jacobian(self.sofr_helpers, self.market)
        self.assertEqual(jac.curves, (SOFR,))
        self.assertEqual(jac.n_positions, 5)
        np.testing.assert_allclose(jac.pv, [h.implied(self.market) for h in self.sofr_helpers], rtol=1e-12)
        self.assertTrue(jac.labels[0].startswith("USD.SOFR.1D 2026-09"))

    def test_single_curve_blocks_match_finite_differences(self):
        jac = template_jacobian(self.sofr_helpers, self.market)
        np.testing.assert_allclose(jac.block(SOFR), _fd(self.sofr_helpers, self.market, SOFR), rtol=1e-6, atol=1e-8)

    def test_calibration_jacobian_is_the_target_block(self):
        jac = template_jacobian(self.sofr_helpers, self.market)
        np.testing.assert_array_equal(calibration_jacobian(self.sofr_helpers, SOFR, self.market), jac.block(SOFR))
        np.testing.assert_array_equal(calibration_jacobian(self.sofr_helpers, STDCSA, self.market), jac.block(SOFR))

    def test_multi_curve_blocks_match_finite_differences(self):
        helpers = self.sofr_helpers + self.ff_helpers
        jac = template_jacobian(helpers, self.market)
        self.assertEqual(jac.curves, (SOFR, FEDFUND))
        np.testing.assert_allclose(jac.block(SOFR), _fd(helpers, self.market, SOFR), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(jac.block(FEDFUND), _fd(helpers, self.market, FEDFUND), rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(jac.block(FEDFUND)[:5], 0.0)  # SOFR quotes do not see the basis
        self.assertTrue((np.abs(jac.block(FEDFUND)[5:]) > 0.5).all())  # a Fed Funds par rate moves ~1:1 with its basis

    def test_requested_curve_order_and_zero_columns(self):
        jac = template_jacobian(self.ff_helpers, self.market, curves=(FEDFUND, SOFR))
        self.assertEqual(jac.curves, (FEDFUND, SOFR))
        self.assertEqual(jac.n_params, self.market[SOFR].n_params + 1)
        with self.assertRaisesRegex(ValueError, "not in the template"):
            template_jacobian(self.ff_helpers, self.market, curves=(FEDFUND,))
        with self.assertRaises(ValueError):
            template_jacobian([], self.market)
