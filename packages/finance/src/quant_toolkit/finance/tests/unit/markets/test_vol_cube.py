"""Vol surfaces and the cube: interpolation, the model pairing, and the parameter gradient."""
from dataclasses import dataclass

import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.markets.vols import FlatVolSurface, GridVolSurface, ModelVolCube, VolNamespace, VolUnits

EXPIRIES = np.array([1.0, 2.0, 5.0])
TENORS = np.array([2.0, 10.0])
VOLS = np.array([[0.010, 0.012], [0.011, 0.013], [0.009, 0.0095]])


@dataclass(frozen=True)
class _SkewModel:
    """Toy smile: ``vol = level + skew * (strike - forward) + curve * expiry``; params are (level, skew, curve)."""

    def vol(self, params, expiry, strike, forward):
        return params[:, 0] + params[:, 1] * (strike - forward) + params[:, 2] * expiry

    def d_params(self, params, expiry, strike, forward):
        return np.stack([np.ones_like(expiry), strike - forward, expiry], axis=1)


def _bumped_gradient(cube, *query, h=1e-6):
    base = cube.params
    columns = []
    for i in range(base.size):
        shift = np.zeros_like(base)
        shift[i] = h
        columns.append((cube.with_params(base + shift).vol(*query) - cube.with_params(base - shift).vol(*query)) / (2 * h))
    return np.stack(columns, axis=1)


class TestGridVolSurface(UnitTest):

    def setUp(self):
        self.cube = ModelVolCube(GridVolSurface(EXPIRIES, TENORS, VOLS), VolUnits.Normal)

    def test_nodes_are_reproduced(self):
        expiry, tenor = np.meshgrid(EXPIRIES, TENORS, indexing="ij")
        np.testing.assert_allclose(self.cube.vol(expiry, tenor, 0.04, 0.04), VOLS.ravel(), rtol=1e-15)

    def test_interior_point_is_bilinear(self):
        # halfway between the 1Y and 2Y expiries, a quarter of the way from the 2Y to the 10Y tenor
        expected = 0.5 * (0.75 * 0.010 + 0.25 * 0.012) + 0.5 * (0.75 * 0.011 + 0.25 * 0.013)
        np.testing.assert_allclose(self.cube.vol(1.5, 4.0, 0.04, 0.04), [expected], rtol=1e-15)

    def test_extrapolation_is_flat(self):
        np.testing.assert_allclose(self.cube.vol([0.1, 30.0], [1.0, 30.0], 0.04, 0.04), [VOLS[0, 0], VOLS[-1, -1]])

    def test_weights_sum_to_one(self):
        _, weight = self.cube.surface.node_weights(np.array([0.1, 1.5, 3.0, 9.0]), np.array([1.0, 4.0, 10.0, 12.0]))
        np.testing.assert_allclose(weight.sum(axis=1), 1.0, rtol=1e-15)

    def test_gradient_matches_a_parameter_bump(self):
        query = (np.array([0.5, 1.5, 3.0, 5.0, 8.0]), np.array([1.0, 4.0, 10.0, 7.0, 20.0]), 0.04, 0.04)
        np.testing.assert_allclose(self.cube.vol_gradient(*query), _bumped_gradient(self.cube, *query), atol=1e-9)

    def test_a_single_node_grid_is_flat(self):
        cube = ModelVolCube(GridVolSurface(np.array([1.0]), np.array([5.0]), np.array([[0.01]])), VolUnits.Normal)
        np.testing.assert_allclose(cube.vol([0.5, 3.0], [2.0, 9.0], 0.04, 0.04), [0.01, 0.01])
        np.testing.assert_allclose(cube.vol_gradient([0.5, 3.0], [2.0, 9.0], 0.04, 0.04), [[1.0], [1.0]])

    def test_with_params_round_trips_the_grid(self):
        moved = self.cube.with_params(self.cube.params + 0.001)
        np.testing.assert_allclose(moved.surface.values, VOLS + 0.001)
        np.testing.assert_allclose(self.cube.surface.values, VOLS)

    def test_bad_grids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            GridVolSurface(EXPIRIES, TENORS, VOLS.T)
        with self.assertRaisesRegex(ValueError, "increasing"):
            GridVolSurface(EXPIRIES[::-1], TENORS, VOLS)


class TestFlatVolSurface(UnitTest):

    def test_one_vol_everywhere(self):
        cube = ModelVolCube(FlatVolSurface(0.01), VolUnits.Normal)
        np.testing.assert_allclose(cube.vol([0.5, 3.0, 20.0], 5.0, 0.04, 0.04), [0.01, 0.01, 0.01])
        np.testing.assert_allclose(cube.vol_gradient([0.5, 3.0], 5.0, 0.04, 0.04), [[1.0], [1.0]])
        np.testing.assert_allclose(cube.with_params(np.array([0.02])).vol(1.0, 5.0, 0.04, 0.04), [0.02])


class TestModelPairing(UnitTest):

    def test_a_flat_surface_feeds_the_model_its_parameters(self):
        cube = ModelVolCube(FlatVolSurface(np.array([0.01, -0.2, 0.001])), VolUnits.Normal, _SkewModel())
        vol = cube.vol(2.0, 5.0, np.array([0.03, 0.04, 0.05]), 0.04)
        np.testing.assert_allclose(vol, 0.01 - 0.2 * np.array([-0.01, 0.0, 0.01]) + 0.002, rtol=1e-14)

    def test_gradient_chains_the_model_through_the_grid(self):
        values = np.stack([VOLS, np.full_like(VOLS, -0.2) + VOLS, 0.1 * VOLS], axis=2)
        cube = ModelVolCube(GridVolSurface(EXPIRIES, TENORS, values), VolUnits.Normal, _SkewModel())
        query = (np.array([0.5, 1.5, 3.0, 8.0]), np.array([1.0, 4.0, 10.0, 20.0]), np.array([0.03, 0.04, 0.05, 0.045]), 0.04)
        self.assertEqual(cube.vol_gradient(*query).shape, (4, 18))
        np.testing.assert_allclose(cube.vol_gradient(*query), _bumped_gradient(cube, *query), atol=1e-9)

    def test_no_model_needs_one_vol_per_node(self):
        with self.assertRaisesRegex(ValueError, "one vol per node"):
            ModelVolCube(FlatVolSurface(np.array([0.01, -0.2])), VolUnits.Normal)


class TestVolNamespace(UnitTest):

    def test_bind_rebind_and_version(self):
        vols = VolNamespace()
        cube = ModelVolCube(FlatVolSurface(0.01), VolUnits.Normal)
        vols.bind("USD.SOFR.1D", cube)
        self.assertIn("USD.SOFR.1D", vols)
        self.assertIs(vols.resolve("USD.SOFR.1D"), cube)
        with self.assertRaises(KeyError):
            vols.bind("USD.SOFR.1D", cube)
        vols.rebind("USD.SOFR.1D", cube.with_params(np.array([0.02])))
        self.assertEqual(vols.version("USD.SOFR.1D"), 1)
        with self.assertRaises(KeyError):
            vols.resolve("missing")
        with self.assertRaises(KeyError):
            vols.version("missing")
