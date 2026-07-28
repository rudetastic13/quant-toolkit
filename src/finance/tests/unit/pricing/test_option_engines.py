"""Black/Bachelier values and greeks across NumPy and Numba."""
import numpy as np

from common.testing import UnitTest
from finance.pricing.engines.numba import bachelier_greeks as nb_bachelier
from finance.pricing.engines.numba import black_greeks as nb_black
from finance.pricing.engines.numpy.option import bachelier_greeks, black_greeks


class TestOptionKernels(UnitTest):
    COVERAGE = ["finance.pricing.engines.numpy.option", "finance.pricing.engines.numba.kernels"]

    def test_numba_matches_numpy_for_calls_and_puts(self):
        forward = np.array([0.02, 0.04, 0.06])
        strike = np.array([0.03, 0.04, 0.05])
        expiry = np.array([0.25, 1.0, 5.0])
        for is_call in (False, True):
            for ref_fn, nb_fn, vol in (
                (black_greeks, nb_black, np.array([0.2, 0.3, 0.4])),
                (bachelier_greeks, nb_bachelier, np.array([0.005, 0.01, 0.015])),
            ):
                expected = ref_fn(forward, strike, expiry, vol, is_call)
                actual = nb_fn(forward, strike, expiry, vol, is_call)
                for expected_array, actual_array in zip(expected.__dict__.values(), actual.__dict__.values()):
                    np.testing.assert_allclose(actual_array, expected_array, rtol=1e-13, atol=1e-14)

    def test_black_delta_and_vega_match_finite_difference(self):
        f, k, t, vol = np.array([0.04]), np.array([0.045]), np.array([2.0]), np.array([0.3])
        result = black_greeks(f, k, t, vol)
        h = 1e-6
        delta = (black_greeks(f + h, k, t, vol).value - black_greeks(f - h, k, t, vol).value) / (2 * h)
        vega = (black_greeks(f, k, t, vol + h).value - black_greeks(f, k, t, vol - h).value) / (2 * h)
        np.testing.assert_allclose(result.delta, delta, rtol=1e-8)
        np.testing.assert_allclose(result.vega, vega, rtol=1e-8)

    def test_zero_expiry_returns_intrinsic_without_singular_greeks(self):
        result = bachelier_greeks(np.array([0.05]), np.array([0.04]), np.array([0.0]), np.array([0.01]))
        np.testing.assert_allclose(result.value, [0.01], atol=1e-16)
        np.testing.assert_array_equal(result.gamma, [0.0])
