"""Tight rate kernels: bare segmented reductions must match a per-period loop.

These test the kernels in isolation — no spread/caps/margin (that's the shaping layer,
see test_coupon_rates.py). The kernels take only (obs_rate, obs_w, offsets).
"""
import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.pricing.engines.numpy.rates import compounded, averaged


def _ragged_grid(seed: int, n_periods: int = 40):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 90, size=n_periods)
    m = int(counts.sum())
    obs_rate = 0.04 + 0.01 * rng.standard_normal(m)
    obs_w = np.where(rng.random(m) < 0.3, 3.0 / 360.0, 1.0 / 360.0)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)
    return obs_rate, obs_w, offsets, counts


def _ref_compounded(obs_rate, obs_w, counts):
    out, i = [], 0
    for c in counts:
        rr, ww = obs_rate[i : i + c], obs_w[i : i + c]
        out.append((np.prod(1.0 + rr * ww) - 1.0) / ww.sum())
        i += c
    return np.array(out)


def _ref_averaged(obs_rate, obs_w, counts):
    out, i = [], 0
    for c in counts:
        rr, ww = obs_rate[i : i + c], obs_w[i : i + c]
        out.append(np.sum(rr * ww) / np.sum(ww))
        i += c
    return np.array(out)


class TestCompoundedKernel(UnitTest):

    def test_matches_reference(self):
        obs_rate, obs_w, offsets, counts = _ragged_grid(1)
        np.testing.assert_allclose(compounded(obs_rate, obs_w, offsets), _ref_compounded(obs_rate, obs_w, counts), atol=1e-12)

    def test_telescopes_to_df_ratio(self):
        rng = np.random.default_rng(7)
        w = np.full(30, 1.0 / 360.0)
        dfs = np.cumprod(np.concatenate([[1.0], 1.0 / (1.0 + (0.04 + 0.005 * rng.standard_normal(30)) * w)]))
        r = (dfs[:-1] / dfs[1:] - 1.0) / w
        rate = compounded(r, w, np.array([0]))[0]
        self.assertAlmostEqual(rate, (dfs[0] / dfs[-1] - 1.0) / w.sum(), places=12)

    def test_rejects_non_increasing_offsets(self):
        with self.assertRaises(ValueError):
            compounded(np.array([0.04, 0.04]), np.array([1.0, 1.0]), np.array([0, 0]))

    def test_rejects_nonzero_first_offset(self):
        with self.assertRaises(ValueError):
            compounded(np.array([0.04, 0.04]), np.array([1.0, 1.0]), np.array([1]))


class TestAveragedKernel(UnitTest):

    def test_matches_reference(self):
        obs_rate, obs_w, offsets, counts = _ragged_grid(11)
        np.testing.assert_allclose(averaged(obs_rate, obs_w, offsets), _ref_averaged(obs_rate, obs_w, counts), atol=1e-12)
