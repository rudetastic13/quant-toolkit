"""Tests for the shared coupon-shaping module (sentinel-encoded, xp-parameterized).

``test_coupon_rates.py`` covers the same algebra through the scalar/None convenience API in
``coupons.rates``; this file pins the columnar sentinel contract both engines consume, and
numpy/JAX parity on the shared code path.
"""
import numpy as np
import pytest

from common.testing import UnitTest
from finance.pricing.kernels.inputs import NO_CAP, NO_FLOOR
from finance.pricing.kernels.shaping import prep_obs, shape_float, shape_period


class TestShapeFloat(UnitTest):
    COVERAGE = ["finance.pricing.kernels.shaping"]

    def test_order_index_floor_spread_floor_cap(self):
        idx = np.array([0.03, 0.05, -0.01])
        got = shape_float(
            idx,
            index_floor=np.full(3, 0.0),
            spread=np.full(3, 0.002),
            floor=np.full(3, 0.0),
            cap=np.full(3, 0.051),
        )
        # -0.01 -> index_floor 0 -> +0.002 = 0.002 ; 0.05 -> 0.052 -> cap 0.051
        np.testing.assert_allclose(got, [0.032, 0.051, 0.002], atol=1e-15)

    def test_sentinels_are_no_ops(self):
        idx = np.array([-0.5, 0.0, 0.5])
        got = shape_float(idx, np.full(3, NO_FLOOR), np.zeros(3), np.full(3, NO_FLOOR), np.full(3, NO_CAP))
        np.testing.assert_array_equal(got, idx)

    def test_input_not_mutated(self):
        idx = np.array([0.03, 0.05])
        shape_float(idx, np.full(2, 0.04), np.full(2, 0.01), np.full(2, NO_FLOOR), np.full(2, NO_CAP))
        np.testing.assert_array_equal(idx, [0.03, 0.05])


class TestPrepAndShapePeriod(UnitTest):
    COVERAGE = ["finance.pricing.kernels.shaping"]

    def test_prep_obs_inclusive_mask_gates_spread(self):
        obs = np.array([0.03, 0.03])
        got = prep_obs(obs, np.full(2, NO_FLOOR), np.full(2, 0.01), np.array([True, False]))
        np.testing.assert_allclose(got, [0.04, 0.03], atol=1e-15)

    def test_prep_obs_index_floor_before_spread(self):
        obs = np.array([-0.02])
        got = prep_obs(obs, np.array([0.0]), np.array([0.005]), np.array([True]))
        np.testing.assert_allclose(got, [0.005], atol=1e-15)

    def test_shape_period_exclusive_mask_gates_spread(self):
        rate = np.array([0.04, 0.04])
        got = shape_period(rate, np.full(2, 0.01), np.array([True, False]), np.full(2, NO_FLOOR), np.full(2, NO_CAP))
        np.testing.assert_allclose(got, [0.05, 0.04], atol=1e-15)

    def test_shape_period_floor_cap_after_spread(self):
        rate = np.array([0.001, 0.10])
        got = shape_period(rate, np.zeros(2), np.zeros(2, dtype=bool), np.full(2, 0.02), np.full(2, 0.05))
        np.testing.assert_allclose(got, [0.02, 0.05], atol=1e-15)


class TestNumpyJaxParity(UnitTest):
    COVERAGE = ["finance.pricing.kernels.shaping"]

    def test_shaping_matches_across_namespaces(self):
        jnp = pytest.importorskip("jax.numpy")
        import jax

        jax.config.update("jax_enable_x64", True)

        rng = np.random.default_rng(7)
        idx = 0.03 + 0.02 * rng.standard_normal(64)
        index_floor = np.where(rng.random(64) < 0.5, 0.0, NO_FLOOR)
        spread = np.full(64, 0.0025)
        floor = np.where(rng.random(64) < 0.5, 0.01, NO_FLOOR)
        cap = np.where(rng.random(64) < 0.5, 0.045, NO_CAP)
        incl = rng.random(64) < 0.5

        np_float = shape_float(idx, index_floor, spread, floor, cap)
        jx_float = shape_float(jnp.asarray(idx), jnp.asarray(index_floor), jnp.asarray(spread),
                               jnp.asarray(floor), jnp.asarray(cap), xp=jnp)
        np.testing.assert_allclose(np.asarray(jx_float), np_float, atol=1e-15)

        np_obs = prep_obs(idx, index_floor, spread, incl)
        jx_obs = prep_obs(jnp.asarray(idx), jnp.asarray(index_floor), jnp.asarray(spread),
                          jnp.asarray(incl), xp=jnp)
        np.testing.assert_allclose(np.asarray(jx_obs), np_obs, atol=1e-15)

        np_per = shape_period(idx, spread, ~incl, floor, cap)
        jx_per = shape_period(jnp.asarray(idx), jnp.asarray(spread), jnp.asarray(~incl),
                              jnp.asarray(floor), jnp.asarray(cap), xp=jnp)
        np.testing.assert_allclose(np.asarray(jx_per), np_per, atol=1e-15)

    def test_shaping_is_jittable(self):
        jnp = pytest.importorskip("jax.numpy")
        import jax

        jax.config.update("jax_enable_x64", True)

        @jax.jit
        def f(idx):
            return shape_float(idx, jnp.full(4, 0.0), jnp.full(4, 0.001),
                               jnp.full(4, NO_FLOOR), jnp.full(4, 0.05), xp=jnp)

        got = f(jnp.array([0.03, -0.01, 0.06, 0.02]))
        np.testing.assert_allclose(np.asarray(got), [0.031, 0.001, 0.05, 0.021], atol=1e-15)
