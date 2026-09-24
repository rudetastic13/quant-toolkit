"""Line1d-based ZeroCurve reproduces the legacy (pre-Line1d) implementation.

The expects were generated from the previous ``ZeroCurve`` for every named scheme and for
the legacy dual-segment ("cutover") curve, on a grid that runs two years past the last
pillar.  The interior must match to the printed precision.  Beyond the last pillar the
legacy code estimated the terminal forward with a one-day finite difference while the new
code uses the exact end derivative, so the extrapolated region of the non-linear schemes is
compared to a looser tolerance (the one-day chord is off by ~f'(T)/2 per day, worst for the
blended-sweep quadratic).
"""

import numpy as np
import pytest

from common.math.interpolation import Cubic, Linear, Mixed
from common.testing import UnitTest
from common.testing.expects_loader import ModuleLoader
from finance.markets.curves import CurveInterpolator, CurveSpace, ZeroCurve

EXPECTS = ModuleLoader("finance.tests.unit.markets.expects.zero_curve_legacy").load()
NODES = EXPECTS["nodes"]
X = NODES["x"].to_numpy()
DFS = NODES["df"].to_numpy()


def _check(curve: ZeroCurve, name: str, *, extrapolated_rtol: float) -> None:
    table = EXPECTS[name]
    xq = table["x"].to_numpy()
    interior = xq <= X[-1]
    got = {
        "df": curve.discount_factor(xq),
        "log_df": curve.log_discount_factor(xq),
        "zero_rate": curve.zero_rate(xq),
    }
    for column, values in got.items():
        expected = table[column].to_numpy()
        np.testing.assert_allclose(
            values[interior], expected[interior], rtol=1e-10, atol=1e-11, err_msg=f"{name}.{column} interior"
        )
        np.testing.assert_allclose(
            values[~interior],
            expected[~interior],
            rtol=extrapolated_rtol,
            atol=1e-11,
            err_msg=f"{name}.{column} extrapolated",
        )


@pytest.mark.regression
class TestZeroCurveMatchesLegacy(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve", "common.math.interpolation", "common.math.line"]

    def test_named_schemes(self):
        for member in CurveInterpolator:
            space, interpolator = member.resolve()
            with self.subTest(scheme=member.name):
                exact = interpolator == Linear() and space is CurveSpace.LogDF
                _check(
                    ZeroCurve(X, DFS, space=space, interpolator=interpolator),
                    member.name,
                    extrapolated_rtol=1e-10 if exact else 1e-4,
                )

    def test_mixed_reproduces_legacy_cutover_curve(self):
        curve = ZeroCurve(X, DFS, interpolator=Mixed(Linear(), Cubic(), switch_node=3))
        _check(curve, "Mixed_LogLinear_LogCubic_at_node3", extrapolated_rtol=1e-4)
