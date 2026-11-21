"""DCF kernel: portfolio-wide PV via reduceat must match hand-computed PVs."""
import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.pricing.engines.numpy.dcf import dcf


class TestDcfKernel(UnitTest):

    def test_single_annuity_pv(self):
        # 3 flows of 1.0 each, receive (+1), discounted.
        cash = np.array([1.0, 1.0, 1.0])
        df = np.array([0.99, 0.97, 0.94])
        sign = np.array([1.0, 1.0, 1.0])
        row_offsets = np.array([0])
        pv = dcf(cash, df, sign, row_offsets)
        self.assertAlmostEqual(pv[0], 0.99 + 0.97 + 0.94, places=12)

    def test_two_instruments_segmented(self):
        # inst A: 2 flows; inst B: 3 flows. One reduceat returns both PVs.
        cash = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
        df = np.array([0.99, 0.98, 0.99, 0.97, 0.95])
        sign = np.array([1.0, 1.0, -1.0, -1.0, -1.0])  # B is a pay leg
        row_offsets = np.array([0, 2])
        pv = dcf(cash, df, sign, row_offsets)
        self.assertAlmostEqual(pv[0], 0.99 + 0.98, places=12)
        self.assertAlmostEqual(pv[1], -(2 * 0.99 + 2 * 0.97 + 2 * 0.95), places=12)
