"""Property tests for the interpolation schemes (skipped when hypothesis is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from common.math.interpolation import Cubic, Flat, Linear, Quadratic  # noqa: E402
from common.math.line import Extrapolation, Line1d  # noqa: E402
from common.testing import UnitTest  # noqa: E402

_nodes = st.lists(st.floats(0.1, 10.0), min_size=2, max_size=12).map(lambda gaps: np.cumsum(np.asarray(gaps)))
_values = st.lists(st.floats(-10.0, 10.0), min_size=2, max_size=12)


@pytest.mark.hypothesis
class TestSchemesRecoverNodes(UnitTest):
    COVERAGE = ["common.math.interpolation", "common.math.line"]

    @settings(max_examples=100, deadline=None)
    @given(x=_nodes, y=_values)
    def test_every_scheme_recovers_its_nodes(self, x, y):
        n = min(x.size, len(y))
        x, y = x[:n], np.asarray(y[:n])
        for interpolator in (Flat(), Linear(), Cubic(), Quadratic()):
            line = Line1d(x, y, interpolator, left=Extrapolation.Flat, right=Extrapolation.Flat)
            np.testing.assert_allclose(line(x), y, rtol=1e-9, atol=1e-9)
