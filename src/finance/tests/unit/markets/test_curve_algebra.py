"""Curve algebra: operators build the same composites as the constructors; leaves() and replace()
expose and propagate through the sharing structure."""

import numpy as np

from common.testing import UnitTest
from finance.markets.curves import CompositeCurve, Leaf, ZeroCurve, bp, pct

COVERAGE = [
    "finance.markets.curves._curve_impl.algebra",
    "finance.markets.curves._curve_impl.composite_curve",
    "finance.markets.curves._curve_impl.zero_curve",
    "finance.markets.curves.units",
]

X = np.array([0.0, 90.0, 365.0, 730.0, 1825.0, 3650.0])
Z = np.array([0.0, 0.041, 0.040, 0.039, 0.041, 0.043])
XQ = np.array([30.0, 500.0, 1825.0, 5000.0])


def _zero(x=X, z=Z, **kwargs) -> ZeroCurve:
    dfs = np.exp(-z * x / 365.0)
    dfs[0] = 1.0
    return ZeroCurve(x, dfs, **kwargs)


class TestUnits(UnitTest):
    COVERAGE = COVERAGE

    def test_values(self):
        self.assertEqual(bp, 1e-4)
        self.assertEqual(pct, 1e-2)
        self.assertAlmostEqual(5 * bp, 0.0005)


class TestCurveOperators(UnitTest):
    COVERAGE = COVERAGE

    def setUp(self):
        self.sofr = _zero()
        self.s3m = _zero(np.array([0.0, 365.0, 3650.0]), np.array([0.0, 0.0008, 0.0012]))

    def test_add_curves_is_a_labelled_spread(self):
        c = self.sofr + self.s3m
        self.assertIsInstance(c, CompositeCurve)
        self.assertEqual(c.labels, ("base", "spread"))
        self.assertIs(c["base"], self.sofr)
        self.assertIs(c["spread"], self.s3m)
        explicit = CompositeCurve({"base": self.sofr, "spread": self.s3m})
        np.testing.assert_allclose(c.log_discount_factor(XQ), explicit.log_discount_factor(XQ), rtol=1e-15)
        self.assertEqual(c.scheme, explicit.scheme)

    def test_add_scalar_is_a_flat_spread(self):
        fedfund = self.sofr - 5 * bp
        self.assertEqual(fedfund.labels, ("base", "spread"))
        self.assertEqual(fedfund.n_params, self.sofr.n_params + 1)
        self.assertAlmostEqual(fedfund.params[-1], -0.0005, places=15)
        self.assertEqual(fedfund["spread"].max_x, self.sofr.max_x)
        np.testing.assert_allclose(fedfund.zero_rate(XQ), self.sofr.zero_rate(XQ) - 0.0005, rtol=1e-12)
        plus = self.sofr + 5 * bp
        np.testing.assert_allclose(plus["spread"].zero_rate(XQ), 0.0005, rtol=1e-12)
        np.testing.assert_allclose((5 * bp + self.sofr).zero_rate(XQ), plus.zero_rate(XQ), rtol=1e-15)
        explicit = CompositeCurve({"base": self.sofr, "spread": ZeroCurve.constant(0.0005, self.sofr.max_x)})
        self.assertEqual(plus.scheme, explicit.scheme)

    def test_scalar_minus_curve(self):
        c = 0.0005 - self.sofr
        self.assertEqual(c.weights, (1.0, -1.0))
        np.testing.assert_allclose(c.zero_rate(XQ), 0.0005 - self.sofr.zero_rate(XQ), rtol=1e-12)

    def test_subtract_curves_is_an_unlabelled_basis(self):
        fedfund = self.sofr - 5 * bp
        basis = fedfund - self.sofr
        self.assertIsNone(basis.labels)
        self.assertEqual(basis.weights, (1.0, -1.0))
        self.assertIs(basis.components[0], fedfund)
        np.testing.assert_allclose(basis.zero_rate(XQ), -0.0005, atol=1e-15)

    def test_scale_and_negate(self):
        half = 0.5 * self.sofr
        self.assertEqual(half.weights, (0.5,))
        np.testing.assert_allclose(half.log_discount_factor(XQ), 0.5 * self.sofr.log_discount_factor(XQ), rtol=1e-15)
        np.testing.assert_allclose((self.sofr * 0.5).log_discount_factor(XQ), half.log_discount_factor(XQ), rtol=1e-15)
        neg = -self.sofr
        self.assertEqual(neg.weights, (-1.0,))
        np.testing.assert_allclose(neg.zero_rate(XQ), -self.sofr.zero_rate(XQ), rtol=1e-15)

    def test_composites_support_the_same_operators(self):
        fedfund = self.sofr - 5 * bp
        chained = fedfund + self.s3m
        self.assertIs(chained["base"], fedfund)
        self.assertEqual(chained.n_params, self.sofr.n_params + 1 + self.s3m.n_params)
        twice = 2 * chained
        np.testing.assert_allclose(twice.log_discount_factor(XQ), 2 * chained.log_discount_factor(XQ), rtol=1e-15)

    def test_forbidden_operations(self):
        with self.assertRaises(TypeError):
            self.sofr * self.s3m
        with self.assertRaises(TypeError):
            self.sofr / 2.0
        with self.assertRaises(TypeError):
            self.sofr + "spread"
        with self.assertRaises(TypeError):
            self.sofr + True
        with self.assertRaises(ValueError):
            0.0 * self.sofr  # a zero weight has no parameters to own

    def test_operators_define_but_do_not_move(self):
        defined = self.sofr + 10 * bp
        moved = self.sofr.with_params(self.sofr.params + 10 * bp)
        np.testing.assert_allclose(defined.discount_factor(XQ), moved.discount_factor(XQ), rtol=1e-12)
        self.assertEqual(defined.n_params, moved.n_params + 1)
        self.assertNotEqual(defined.scheme, moved.scheme)


class TestLeavesAndReplace(UnitTest):
    COVERAGE = COVERAGE

    def setUp(self):
        self.sofr = _zero()
        self.s1m = ZeroCurve.constant(0.0008, 3650.0)
        self.b6m1m = ZeroCurve.constant(0.0004, 3650.0)
        self.term1m = self.sofr + self.s1m
        self.term6m = self.term1m + self.b6m1m  # three deep: (sofr + s1m) + b6m1m

    def test_atom_leaf(self):
        (leaf,) = self.sofr.leaves()
        self.assertIsInstance(leaf, Leaf)
        self.assertIs(leaf.curve, self.sofr)
        self.assertEqual(leaf.weight, 1.0)
        self.assertEqual(leaf.columns, slice(0, 5))

    def test_leaves_of_a_chain_accumulate_weights_and_shift_columns(self):
        leaves = (2.0 * self.term6m).leaves()
        self.assertEqual([lf.curve for lf in leaves], [self.sofr, self.s1m, self.b6m1m])
        self.assertEqual([lf.weight for lf in leaves], [2.0, 2.0, 2.0])
        self.assertEqual([lf.columns for lf in leaves], [slice(0, 5), slice(5, 6), slice(6, 7)])
        basis = self.term6m - self.sofr
        self.assertEqual([lf.weight for lf in basis.leaves()], [1.0, 1.0, 1.0, -1.0])
        self.assertIs(basis.leaves()[0].curve, basis.leaves()[-1].curve)  # sofr reached twice
        self.assertEqual(basis.leaves()[-1].columns, slice(7, 12))

    def test_leaf_columns_index_the_affine_jacobian(self):
        _, jac = self.term6m.log_df_affine(XQ)
        for leaf in self.term6m.leaves():
            _, leaf_jac = leaf.curve.log_df_affine(XQ)
            np.testing.assert_allclose(jac[:, leaf.columns], leaf.weight * leaf_jac, atol=1e-15)

    def test_replace_propagates_through_nesting(self):
        shifted = self.sofr.with_params(self.sofr.params + 10 * bp)
        new6m = self.term6m.replace(self.sofr, shifted)
        self.assertIsNot(new6m, self.term6m)
        self.assertIs(new6m["base"]["base"], shifted)
        self.assertIs(new6m["base"]["spread"], self.s1m)  # untouched components stay shared
        self.assertIs(new6m["spread"], self.b6m1m)
        self.assertEqual(new6m.labels, ("base", "spread"))
        np.testing.assert_allclose(new6m.zero_rate(XQ), self.term6m.zero_rate(XQ) + 10 * bp, rtol=1e-10)
        self.assertEqual(new6m.scheme, self.term6m.scheme)

    def test_replace_is_identity_when_absent_and_by_identity_not_value(self):
        other = _zero()  # equal values, different object
        self.assertIs(self.term6m.replace(other, self.s1m), self.term6m)
        self.assertIs(self.sofr.replace(other, self.s1m), self.sofr)
        self.assertIs(self.sofr.replace(self.sofr, other), other)
        self.assertIs(self.term6m.replace(self.term6m, other), other)

    def test_replace_a_middle_node(self):
        new1m = self.sofr + ZeroCurve.constant(0.0010, 3650.0)
        new6m = self.term6m.replace(self.term1m, new1m)
        self.assertIs(new6m["base"], new1m)
        np.testing.assert_allclose(new6m.zero_rate(XQ), self.term6m.zero_rate(XQ) + 0.0002, rtol=1e-10)
