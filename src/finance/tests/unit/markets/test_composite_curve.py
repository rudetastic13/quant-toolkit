"""DiscountCurve protocol and CompositeCurve: weighted sums in log-DF space that are exactly affine
in their parameters, carry labelled component blocks, and price through the Numba engine like an atom."""

import numpy as np
import pytest

from common.math.interpolation import Cubic, Linear
from common.testing import UnitTest
from finance.markets.curves import CompositeCurve, CurveSpace, DiscountCurve, RateExtrapolator, ZeroCurve

COVERAGE = [
    "finance.markets.curves._curve_impl.discount_curve",
    "finance.markets.curves._curve_impl.composite_curve",
    "finance.markets.curves._curve_impl.zero_curve",
]

BASE_X = np.array([0.0, 90.0, 365.0, 730.0, 1825.0, 3650.0])
BASE_Z = np.array([0.0, 0.041, 0.040, 0.039, 0.041, 0.043])
SPREAD_X = np.array([0.0, 365.0, 1825.0, 7300.0])
SPREAD_Z = np.array([0.0, 0.0010, 0.0015, 0.0020])
# pillars of one or both, interior points, beyond the base (3650) and beyond the spread (7300)
XQ = np.array([0.0, 30.0, 365.0, 500.0, 1825.0, 3000.0, 5000.0, 9000.0])


def _zero(x, z, **kwargs) -> ZeroCurve:
    dfs = np.exp(-z * x / 365.0)
    dfs[0] = 1.0
    return ZeroCurve(x, dfs, **kwargs)


def _base(**kwargs) -> ZeroCurve:
    return _zero(BASE_X, BASE_Z, **kwargs)


def _spread(**kwargs) -> ZeroCurve:
    return _zero(SPREAD_X, SPREAD_Z, **kwargs)


def _flat_spread(base: DiscountCurve, rate: float) -> CompositeCurve:
    return CompositeCurve({"base": base, "spread": ZeroCurve.constant(rate, base.max_x)})


class TestDiscountCurveProtocol(UnitTest):
    COVERAGE = COVERAGE

    def test_every_implementation_satisfies_the_protocol(self):
        base, spread = _base(), _spread()
        for curve in (base, CompositeCurve((base, spread)), _flat_spread(base, 0.001)):
            with self.subTest(curve=repr(curve)[:20]):
                self.assertIsInstance(curve, DiscountCurve)

    def test_zero_curve_params_are_the_node_zero_rates(self):
        c = _base()
        self.assertEqual(c.n_params, BASE_X.size - 1)
        np.testing.assert_allclose(c.params, BASE_Z[1:], rtol=1e-14)
        np.testing.assert_array_equal(c.params, c.node_zero_rates)

    def test_zero_curve_with_params_matches_with_dfs(self):
        c = _base(interpolator=Cubic())
        z = BASE_Z[1:] + 0.005
        via_params = c.with_params(z)
        via_dfs = c.with_dfs(np.concatenate([[1.0], np.exp(-z * BASE_X[1:] / 365.0)]))
        np.testing.assert_allclose(via_params.discount_factor(XQ), via_dfs.discount_factor(XQ), rtol=1e-15)
        self.assertEqual(via_params.scheme, c.scheme)

    def test_zero_curve_with_params_validates_shape(self):
        with self.assertRaises(ValueError):
            _base().with_params(np.zeros(3))

    def test_zero_curve_scheme_is_space_and_interpolator(self):
        self.assertEqual(_base().scheme, (CurveSpace.LogDF, Linear()))
        self.assertNotEqual(_base().scheme, _base(interpolator=Cubic()).scheme)
        self.assertNotEqual(_base().scheme, _base(space=CurveSpace.ZeroRate).scheme)


class TestZeroCurveConstant(UnitTest):
    COVERAGE = COVERAGE

    def test_constant_rate_everywhere(self):
        c = ZeroCurve.constant(0.00085, 3650.0)
        self.assertEqual(c.n_params, 1)
        self.assertAlmostEqual(c.params[0], 0.00085, places=15)
        self.assertEqual(c.max_x, 3650.0)
        positive = XQ[XQ > 0]
        np.testing.assert_allclose(c.zero_rate(positive), 0.00085, rtol=1e-12)
        np.testing.assert_allclose(c.instantaneous_forward(XQ), 0.00085, rtol=1e-12)
        np.testing.assert_allclose(c.discount_factor(XQ), np.exp(-0.00085 * XQ / 365.0), rtol=1e-14)

    def test_negative_rate_and_short_horizon(self):
        c = ZeroCurve.constant(-0.0005, 365.0)
        np.testing.assert_allclose(c.zero_rate(np.array([100.0, 9000.0])), -0.0005, rtol=1e-12)
        with self.assertRaises(ValueError):
            ZeroCurve.constant(0.001, 0.0)
        with self.assertRaises(ValueError):
            ZeroCurve.constant(0.001, 365.0, extrapolation=RateExtrapolator.NotAllowed).discount_factor(
                np.array([400.0])
            )


class TestCompositeCurve(UnitTest):
    COVERAGE = COVERAGE

    def setUp(self):
        self.base = _base()
        self.spread = _spread(space=CurveSpace.ZeroRate, interpolator=Cubic())
        self.weights = (1.0, -0.5)
        self.composite = CompositeCurve((self.base, self.spread), self.weights)

    def test_log_df_is_the_weighted_sum(self):
        expected = 1.0 * self.base.log_discount_factor(XQ) - 0.5 * self.spread.log_discount_factor(XQ)
        np.testing.assert_allclose(self.composite.log_discount_factor(XQ), expected, rtol=1e-15)
        np.testing.assert_allclose(self.composite.discount_factor(XQ), np.exp(expected), rtol=1e-15)

    def test_zero_rates_and_forwards_add(self):
        np.testing.assert_allclose(
            self.composite.zero_rate(XQ), self.base.zero_rate(XQ) - 0.5 * self.spread.zero_rate(XQ), rtol=1e-14
        )
        np.testing.assert_allclose(
            self.composite.instantaneous_forward(XQ),
            self.base.instantaneous_forward(XQ) - 0.5 * self.spread.instantaneous_forward(XQ),
            rtol=1e-14,
        )

    def test_each_component_extrapolates_by_its_own_rule(self):
        beyond_base = np.array([5000.0])  # past the base's last pillar, inside the spread's
        f_base = self.base.instantaneous_forward(np.array([3650.0]))[0]  # flat forward continues
        expected = f_base - 0.5 * self.spread.instantaneous_forward(beyond_base)[0]
        self.assertAlmostEqual(self.composite.instantaneous_forward(beyond_base)[0], expected, places=14)
        strict = CompositeCurve((_base(extrapolation=RateExtrapolator.NotAllowed), self.spread), self.weights)
        with self.assertRaises(ValueError):
            strict.discount_factor(beyond_base)
        with self.assertRaises(ValueError):
            self.composite.discount_factor(np.array([-1.0]))

    def test_geometry(self):
        np.testing.assert_array_equal(self.composite.x, np.union1d(BASE_X, SPREAD_X))
        self.assertEqual(self.composite.max_x, 7300.0)
        np.testing.assert_allclose(self.composite.dfs, self.composite.discount_factor(self.composite.x))
        self.assertEqual(self.composite.dfs[0], 1.0)
        self.assertEqual(self.composite.n_params, (BASE_X.size - 1) + (SPREAD_X.size - 1))
        np.testing.assert_array_equal(self.composite.param_offsets, [0, 5, 8])
        self.assertEqual(self.composite.component_slice(1), slice(5, 8))
        self.assertEqual(self.composite.component_slice(-1), slice(5, 8))
        np.testing.assert_allclose(self.composite.params, np.concatenate([BASE_Z[1:], SPREAD_Z[1:]]), rtol=1e-13)
        self.assertFalse(self.composite.is_log_linear)
        self.assertIs(self.composite.components[0], self.base)
        self.assertIs(self.composite[0], self.base)
        self.assertIs(self.composite[-1], self.spread)
        self.assertEqual(len(self.composite), 2)
        self.assertEqual(self.composite.weights, self.weights)
        self.assertIsNone(self.composite.labels)

    def test_default_weights_are_one(self):
        c = CompositeCurve((self.base, self.spread))
        self.assertEqual(c.weights, (1.0, 1.0))
        np.testing.assert_allclose(
            c.log_discount_factor(XQ), self.base.log_discount_factor(XQ) + self.spread.log_discount_factor(XQ)
        )

    def test_affine_map_is_exact_and_blockwise(self):
        offset, jac = self.composite.log_df_affine(XQ)
        self.assertEqual(jac.shape, (XQ.size, self.composite.n_params))
        np.testing.assert_allclose(
            offset + jac @ self.composite.params, self.composite.log_discount_factor(XQ), atol=1e-13
        )
        b_off, b_jac = self.base.log_df_affine(XQ)
        s_off, s_jac = self.spread.log_df_affine(XQ)
        np.testing.assert_allclose(offset, b_off - 0.5 * s_off, atol=1e-15)
        np.testing.assert_allclose(jac[:, :5], b_jac, atol=1e-15)
        np.testing.assert_allclose(jac[:, 5:], -0.5 * s_jac, atol=1e-15)

    def test_affine_jacobian_matches_central_differences(self):
        h = 1e-6
        _, jac = self.composite.log_df_affine(XQ)
        z0 = self.composite.params
        for k in range(z0.size):
            e = np.zeros(z0.size)
            e[k] = h
            up = self.composite.with_params(z0 + e).log_discount_factor(XQ)
            dn = self.composite.with_params(z0 - e).log_discount_factor(XQ)
            np.testing.assert_allclose(jac[:, k], (up - dn) / (2.0 * h), atol=1e-7)

    def test_with_params_moves_only_the_addressed_block(self):
        z = self.composite.params.copy()
        z[6] += 0.01  # spread's second pillar
        moved = self.composite.with_params(z)
        self.assertIsInstance(moved, CompositeCurve)
        np.testing.assert_allclose(moved.params, z, rtol=1e-13)
        np.testing.assert_allclose(moved.components[0].discount_factor(XQ), self.base.discount_factor(XQ))
        self.assertEqual(moved.scheme, self.composite.scheme)
        self.assertEqual(moved.weights, self.weights)
        with self.assertRaises(ValueError):
            self.composite.with_params(z[:-1])

    def test_with_component_replaces_and_revalidates(self):
        new_spread = _spread(interpolator=Cubic())
        swapped = self.composite.with_component(1, new_spread)
        self.assertIs(swapped.components[1], new_spread)
        self.assertEqual(swapped.weights, self.weights)
        self.assertNotEqual(swapped.scheme, self.composite.scheme)
        with self.assertRaises(TypeError):
            self.composite.with_component(1, "not a curve")
        with self.assertRaises(IndexError):
            self.composite.with_component(2, new_spread)
        with self.assertRaises(KeyError):
            self.composite.with_component("spread", new_spread)  # unlabelled composite

    def test_nesting_equals_flat_with_product_weights(self):
        turn = _zero(np.array([0.0, 400.0]), np.array([0.0, 0.002]))
        nested = CompositeCurve((self.composite, turn), (2.0, 1.0))
        flat = CompositeCurve((self.base, self.spread, turn), (2.0, -1.0, 1.0))
        np.testing.assert_allclose(nested.log_discount_factor(XQ), flat.log_discount_factor(XQ), rtol=1e-14)
        self.assertEqual(nested.n_params, flat.n_params)
        n_off, n_jac = nested.log_df_affine(XQ)
        f_off, f_jac = flat.log_df_affine(XQ)
        np.testing.assert_allclose(n_off, f_off, atol=1e-15)
        np.testing.assert_allclose(n_jac, f_jac, atol=1e-15)

    def test_scheme_distinguishes_pillar_ownership_but_not_labels(self):
        # Same union of pillars, same schemes, different partition between the components.
        a = CompositeCurve(
            (
                _zero(np.array([0.0, 365.0]), np.array([0.0, 0.04])),
                _zero(np.array([0.0, 730.0]), np.array([0.0, 0.001])),
            )
        )
        b = CompositeCurve(
            (
                _zero(np.array([0.0, 730.0]), np.array([0.0, 0.04])),
                _zero(np.array([0.0, 365.0]), np.array([0.0, 0.001])),
            )
        )
        np.testing.assert_array_equal(a.x, b.x)
        self.assertNotEqual(a.scheme, b.scheme)
        self.assertEqual(a.scheme, a.with_params(a.params + 0.01).scheme)
        labelled = CompositeCurve(dict(zip(("p", "q"), a.components)))
        self.assertEqual(labelled.scheme, a.scheme)
        hash(a.scheme)

    def test_validation(self):
        with self.assertRaises(ValueError):
            CompositeCurve(())
        with self.assertRaises(ValueError):
            CompositeCurve((self.base, self.spread), (1.0,))
        for bad in (0.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                CompositeCurve((self.base, self.spread), (1.0, bad))
        with self.assertRaises(TypeError):
            CompositeCurve((self.base, object()))

    def test_repr_names_weights_and_components(self):
        text = repr(self.composite)
        self.assertIn("CompositeCurve(", text)
        self.assertIn("-0.5*ZeroCurve", text)


class TestCompositeLabels(UnitTest):
    """Labelled composites: a spread curve is ``{"base": ..., "spread": ...}``."""

    COVERAGE = COVERAGE

    def setUp(self):
        self.base = _base(interpolator=Cubic())
        self.spread = _spread()
        self.curve = CompositeCurve({"base": self.base, "spread": self.spread})

    def test_discount_factor_is_the_product(self):
        np.testing.assert_allclose(
            self.curve.discount_factor(XQ), self.base.discount_factor(XQ) * self.spread.discount_factor(XQ), rtol=1e-15
        )
        np.testing.assert_allclose(self.curve["spread"].zero_rate(XQ), self.spread.zero_rate(XQ))
        np.testing.assert_allclose(
            self.curve.zero_rate(XQ) - self.curve["base"].zero_rate(XQ), self.spread.zero_rate(XQ), atol=1e-15
        )
        np.testing.assert_allclose(
            self.curve.instantaneous_forward(XQ),
            self.base.instantaneous_forward(XQ) + self.spread.instantaneous_forward(XQ),
        )

    def test_labels_address_components_and_blocks(self):
        self.assertEqual(self.curve.labels, ("base", "spread"))
        self.assertIs(self.curve["base"], self.base)
        self.assertIs(self.curve["spread"], self.spread)
        self.assertEqual(self.curve.n_params, 8)
        self.assertEqual(self.curve.component_slice("base"), slice(0, 5))
        self.assertEqual(self.curve.component_slice("spread"), slice(5, 8))
        self.assertEqual(self.curve.component_slice(1), slice(5, 8))
        np.testing.assert_array_equal(self.curve.params[self.curve.component_slice("base")], self.base.params)
        np.testing.assert_array_equal(self.curve.params[self.curve.component_slice("spread")], self.spread.params)
        with self.assertRaises(KeyError):
            self.curve["basis"]
        self.assertIn("spread=+1*ZeroCurve", repr(self.curve))

    def test_duplicate_labels_rejected(self):
        with self.assertRaises(ValueError):  # keys 1 and "1" both stringify to "1"
            CompositeCurve({1: self.base, "1": self.spread})

    def test_with_params_and_with_component_keep_labels(self):
        z = self.curve.params + 0.001
        moved = self.curve.with_params(z)
        self.assertEqual(moved.labels, ("base", "spread"))
        np.testing.assert_allclose(moved.params, z, rtol=1e-13)
        other_base = _base(space=CurveSpace.ZeroRate)
        rebased = self.curve.with_component("base", other_base)
        self.assertIs(rebased["base"], other_base)
        self.assertIs(rebased["spread"], self.spread)
        self.assertEqual(rebased.labels, ("base", "spread"))
        self.assertIs(self.curve.with_component(1, other_base)["spread"], other_base)

    def test_affine_map_blocks_follow_labels(self):
        offset, jac = self.curve.log_df_affine(XQ)
        self.assertEqual(jac.shape, (XQ.size, 8))
        np.testing.assert_allclose(offset + jac @ self.curve.params, self.curve.log_discount_factor(XQ), atol=1e-13)
        np.testing.assert_allclose(jac[:, self.curve.component_slice("base")], self.base.log_df_affine(XQ)[1])

    def test_flat_spread_is_constant_everywhere(self):
        s = 0.00085
        flat = _flat_spread(self.base, s)
        self.assertEqual(flat.n_params, self.base.n_params + 1)
        self.assertAlmostEqual(flat.params[-1], s, places=15)
        positive = XQ[XQ > 0]
        np.testing.assert_allclose(flat["spread"].zero_rate(positive), s, rtol=1e-12)
        np.testing.assert_allclose(flat.zero_rate(positive), self.base.zero_rate(positive) + s, rtol=1e-12)
        np.testing.assert_allclose(flat.instantaneous_forward(XQ), self.base.instantaneous_forward(XQ) + s, rtol=1e-12)
        np.testing.assert_allclose(
            flat.discount_factor(XQ), self.base.discount_factor(XQ) * np.exp(-s * XQ / 365.0), rtol=1e-14
        )


@pytest.mark.risk
class TestCompositeThroughNumbaEngine(UnitTest):
    """A labelled spread composite prices and risks through the Numba engine exactly like the equivalent atom."""

    COVERAGE = COVERAGE + ["finance.pricing.engines.numba.program"]

    def setUp(self):
        pytest.importorskip("numba")
        from finance.dates import Date
        from finance.instruments.resolution import Swap
        from finance.markets.context import MarketContext
        from finance.markets.curves import CurveNamespace, YieldCurve
        from finance.pricing.pricers import SwapPricer

        self.as_of = Date(2026, 6, 1)
        origin = self.as_of.to_numpy()
        self.spread = 0.0025
        self.base = _base()
        self.spread_curve = _flat_spread(self.base, self.spread)
        # Equivalent atom: log-linear DF reproduces a constant spread exactly (``-s·x/365`` is linear in x).
        self.atom = self.base.with_params(self.base.params + self.spread)

        def market(curve):
            ns = CurveNamespace()
            ns.bind(YieldCurve.from_registry(origin, curve, currency="USD", index_name="SOFR"))
            return MarketContext(self.as_of, ns)

        self.market_spread = market(self.spread_curve)
        self.market_atom = market(self.atom)
        self.market_fn = market
        book = [
            Swap.fixed_float_swap(notional=1e6, rate_index="SOFR", fixed_rate=0.04, tenor="5Y", as_of=self.as_of),
            Swap.fixed_float_swap(notional=2e6, rate_index="SOFR", fixed_rate=0.042, tenor="7Y", as_of=self.as_of),
        ]
        self.program = SwapPricer().compile(book)

    def _grad(self, market):
        from finance.pricing.types import Backend

        result = self.program.prepare(market, backend=Backend.Numba).value_and_grad(market)
        return result, result.instrument_gradient("USD.SOFR", role="total")

    def test_pv_matches_the_equivalent_atom(self):
        from finance.pricing.types import Backend

        pv_spread = self.program.reprice(self.market_spread, backend=Backend.Numba).instrument_pv
        pv_atom = self.program.reprice(self.market_atom, backend=Backend.Numba).instrument_pv
        pv_numpy = self.program.reprice(self.market_spread).instrument_pv
        np.testing.assert_allclose(pv_spread, pv_atom, rtol=1e-12)
        np.testing.assert_allclose(pv_spread, pv_numpy, rtol=1e-11)

    def test_adjoint_splits_into_base_and_spread_blocks(self):
        result, g_spread = self._grad(self.market_spread)
        _, g_atom = self._grad(self.market_atom)
        self.assertEqual(g_spread.shape, (2, self.base.n_params + 1))
        self.assertEqual(result.curve_slice("USD.SOFR"), slice(0, self.base.n_params + 1))
        # Base block equals the atom's pillar risk; the spread parameter carries the parallel risk.
        np.testing.assert_allclose(g_spread[:, self.spread_curve.component_slice("base")], g_atom, rtol=1e-11)
        np.testing.assert_allclose(
            g_spread[:, self.spread_curve.component_slice("spread")][:, 0], g_atom.sum(axis=1), rtol=1e-10
        )

    def test_spread_gradient_matches_finite_difference(self):
        _, g = self._grad(self.market_spread)
        h = 1e-6

        def pv_at(s):
            return self.program.reprice(self.market_fn(_flat_spread(self.base, s))).instrument_pv

        fd = (pv_at(self.spread + h) - pv_at(self.spread - h)) / (2.0 * h)
        np.testing.assert_allclose(g[:, -1], fd, rtol=1e-6)

    def test_prepared_program_is_reused_for_a_spread_move_and_rejects_a_geometry_change(self):
        from finance.pricing.types import Backend

        prepared = self.program.prepare(self.market_spread, backend=Backend.Numba)
        moved = self.market_fn(_flat_spread(self.base, self.spread + 0.001))
        self.assertTrue(prepared.compatible(moved))
        self.assertAlmostEqual(prepared.params_from_market(moved)[-1], self.spread + 0.001, places=15)
        regeared = self.market_fn(CompositeCurve({"base": self.base, "spread": _spread()}))
        self.assertFalse(prepared.compatible(regeared))
        with self.assertRaises(ValueError):
            prepared.params_from_market(regeared)

    def test_bump_helper_refuses_a_composite(self):
        from finance.pricing.risk.sensitivities import bumped_curve

        with self.assertRaises(TypeError):
            bumped_curve(self.spread_curve, 1e-4)
