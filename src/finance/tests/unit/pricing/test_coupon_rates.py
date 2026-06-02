"""Coupon shaping layer: the full coupon matrix over the tight kernels.

Covers: simple float (IBOR), averaged, compounded, averaged+index_floor,
compounded margin inclusive vs exclusive, caps & floors — including the subtle
cap/floor-on-final-period-coupon (NOT daily-running) convention.
"""
import numpy as np

from common.testing import UnitTest
from finance.instruments.enums import MarginTreatment
from finance.coupons.rates import fixed_coupon, floating_coupon, compounded_coupon, averaged_coupon

INCL = MarginTreatment.Inclusive
EXCL = MarginTreatment.Exclusive


def _one_period_grid(obs_rate, w=1.0 / 360.0):
    obs_rate = np.asarray(obs_rate, dtype=np.float64)
    obs_w = np.full(obs_rate.shape, w)
    offsets = np.array([0])
    return obs_rate, obs_w, offsets


# -- reference implementations (slow, explicit) -----------------------------
def _ref_compounded_one(obs_rate, obs_w, spread, index_floor, cap, floor, margin):
    r = obs_rate.copy()
    if index_floor is not None:
        r = np.maximum(r, index_floor)
    if spread and margin == INCL:
        r = r + spread
    rate = (np.prod(1.0 + r * obs_w) - 1.0) / obs_w.sum()
    if spread and margin == EXCL:
        rate += spread
    if floor is not None:
        rate = max(rate, floor)
    if cap is not None:
        rate = min(rate, cap)
    return rate


class TestSimpleFloatIBOR(UnitTest):
    COVERAGE = ["finance.coupons.rates"]

    def test_index_plus_spread(self):
        idx = np.array([0.03, 0.05])
        np.testing.assert_allclose(floating_coupon(idx, spread=0.0025), [0.0325, 0.0525], atol=1e-15)

    def test_order_index_floor_spread_floor_cap(self):
        idx = np.array([0.03, 0.05, -0.01])
        got = floating_coupon(idx, spread=0.002, index_floor=0.0, cap=0.051, floor=0.0)
        # -0.01 -> index_floor 0 -> +0.002 = 0.002 ; 0.05 -> 0.052 -> cap 0.051
        np.testing.assert_allclose(got, [0.032, 0.051, 0.002], atol=1e-15)

    def test_fixed(self):
        np.testing.assert_array_equal(fixed_coupon(0.045, 4), np.full(4, 0.045))


class TestAveragedCoupon(UnitTest):
    COVERAGE = ["finance.coupons.rates"]

    def test_plain_average(self):
        obs_rate, obs_w, offsets = _one_period_grid([0.02, 0.04, 0.06])
        # equal weights -> simple mean
        np.testing.assert_allclose(averaged_coupon(obs_rate, obs_w, offsets)[0], 0.04, atol=1e-15)

    def test_with_index_floor(self):
        # index_floor lifts the negative fixings before averaging
        obs_rate, obs_w, offsets = _one_period_grid([-0.01, 0.04, 0.06])
        got = averaged_coupon(obs_rate, obs_w, offsets, index_floor=0.0)[0]
        self.assertAlmostEqual(got, (0.0 + 0.04 + 0.06) / 3.0, places=15)

    def test_inclusive_spread_equals_plain_plus_spread(self):
        # averaging is linear, so inclusive spread == average + spread
        obs_rate, obs_w, offsets = _one_period_grid([0.03, 0.05])
        incl = averaged_coupon(obs_rate, obs_w, offsets, spread=0.01, margin=INCL)[0]
        excl = averaged_coupon(obs_rate, obs_w, offsets, spread=0.01, margin=EXCL)[0]
        self.assertAlmostEqual(incl, 0.04 + 0.01, places=15)
        self.assertAlmostEqual(incl, excl, places=15)  # linear: order doesn't matter for average


class TestCompoundedCoupon(UnitTest):
    COVERAGE = ["finance.coupons.rates"]

    def test_plain_matches_reference(self):
        rng = np.random.default_rng(3)
        obs_rate, obs_w, offsets = _one_period_grid(0.04 + 0.01 * rng.standard_normal(50))
        got = compounded_coupon(obs_rate, obs_w, offsets)[0]
        exp = _ref_compounded_one(obs_rate, obs_w, 0.0, None, None, None, INCL)
        self.assertAlmostEqual(got, exp, places=14)

    def test_margin_inclusive_vs_exclusive_differ(self):
        # compounding is non-linear, so spread-inclusive != spread-exclusive
        obs_rate, obs_w, offsets = _one_period_grid([0.04] * 60)
        incl = compounded_coupon(obs_rate, obs_w, offsets, spread=0.01, margin=INCL)[0]
        excl = compounded_coupon(obs_rate, obs_w, offsets, spread=0.01, margin=EXCL)[0]
        self.assertNotAlmostEqual(incl, excl, places=8)
        # inclusive compounds the spread, so it should be slightly larger
        self.assertGreater(incl, excl)

    def test_inclusive_matches_reference(self):
        rng = np.random.default_rng(9)
        obs_rate, obs_w, offsets = _one_period_grid(0.03 + 0.02 * rng.standard_normal(40))
        got = compounded_coupon(obs_rate, obs_w, offsets, spread=0.0025, index_floor=0.0, margin=INCL)[0]
        exp = _ref_compounded_one(obs_rate, obs_w, 0.0025, 0.0, None, None, INCL)
        self.assertAlmostEqual(got, exp, places=14)


class TestCapsAndFloors(UnitTest):
    COVERAGE = ["finance.coupons.rates"]

    def test_period_floor_binds(self):
        obs_rate, obs_w, offsets = _one_period_grid([0.001] * 30)  # ~0.1% compounded
        got = compounded_coupon(obs_rate, obs_w, offsets, floor=0.02)[0]
        self.assertAlmostEqual(got, 0.02, places=15)

    def test_period_cap_binds(self):
        obs_rate, obs_w, offsets = _one_period_grid([0.10] * 30)
        got = compounded_coupon(obs_rate, obs_w, offsets, cap=0.05)[0]
        self.assertAlmostEqual(got, 0.05, places=15)

    def test_cap_is_on_FINAL_period_coupon_not_daily(self):
        """The documented convention: cap/floor act on the realized period coupon.

        Construct a period whose running compounded rate spikes ABOVE the cap early
        (high fixings) but ends BELOW the cap (negative fixings later). Final-period
        treatment must NOT cap it; a daily-running cap would have. This pins the
        convention so a future refactor can't silently switch to daily treatment.
        """
        # 10 days at 12%, then 50 days at -2% -> final compounded well under the 5% cap,
        # even though the partial after the first 10 days is ~0.33% of period... build it
        # so the *annualized running* rate exceeds 5% mid-period.
        obs = np.concatenate([np.full(10, 0.12), np.full(50, -0.02)])
        obs_rate, obs_w, offsets = _one_period_grid(obs)

        uncapped = compounded_coupon(obs_rate, obs_w, offsets)[0]
        capped = compounded_coupon(obs_rate, obs_w, offsets, cap=0.05)[0]

        # running rate after the high stretch (annualized) breaches the cap...
        hi = obs_rate[:10]
        hw = obs_w[:10]
        running = (np.prod(1.0 + hi * hw) - 1.0) / hw.sum()
        self.assertGreater(running, 0.05)  # would have been capped on a daily basis
        # ...but the final coupon is below the cap, so final-period treatment leaves it alone
        self.assertLess(uncapped, 0.05)
        self.assertAlmostEqual(capped, uncapped, places=15)
