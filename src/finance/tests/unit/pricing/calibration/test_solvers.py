"""Solver unit tests — pure root-finders, no market data.

Uses a lower-triangular linear system so residual ``i`` depends only on ``x[0..i]`` — i.e. a
*local* system the sequential Bootstrapper is valid on, letting us check it agrees with the
GlobalSolver and recovers the exact root.
"""
import numpy as np
import pytest

from common.testing import UnitTest
from finance.pricing.calibration.solvers import Bootstrapper, GlobalSolver


@pytest.mark.calibration
class TestSolvers(UnitTest):
    COVERAGE = ["finance.pricing.calibration.solvers"]

    def setUp(self):
        # Lower-triangular (local) system L x = b.
        self.L = np.array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [1.0, 3.0, 0.0, 0.0],
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.3, 1.0, 4.0],
            ]
        )
        self.b = np.array([0.04, 0.10, 0.09, 0.20])
        self.x_true = np.linalg.solve(self.L, self.b)

    def _resid(self):
        def resid(x):
            return self.L @ np.asarray(x, dtype=np.float64) - self.b

        return resid

    def test_global_solves(self):
        r = GlobalSolver().solve(self._resid(), np.zeros(4))
        self.assertTrue(r.converged)
        np.testing.assert_allclose(r.x, self.x_true, atol=1e-10)

    def test_bootstrap_solves_local_system(self):
        r = Bootstrapper().solve(self._resid(), np.zeros(4))
        self.assertTrue(r.converged)
        np.testing.assert_allclose(r.x, self.x_true, atol=1e-10)

    def test_global_and_bootstrap_agree(self):
        rg = GlobalSolver().solve(self._resid(), np.zeros(4))
        rb = Bootstrapper().solve(self._resid(), np.zeros(4))
        np.testing.assert_allclose(rg.x, rb.x, atol=1e-9)

    def test_bootstrap_expands_bracket(self):
        # Root at 0.5 sits outside the initial ±0.05 bracket → forces expansion.
        def resid(x):
            return np.array([np.asarray(x, dtype=np.float64)[0] - 0.5])

        r = Bootstrapper().solve(resid, np.zeros(1))
        self.assertTrue(r.converged)
        np.testing.assert_allclose(r.x, [0.5], atol=1e-10)

    def test_bootstrap_bad_bracket_raises(self):
        # Strictly positive residual: no sign change, cannot bracket a root.
        def resid(x):
            return np.asarray(x, dtype=np.float64) ** 2 + 1.0

        with self.assertRaises(RuntimeError):
            Bootstrapper(max_bracket_expansions=3).solve(resid, np.zeros(1))
