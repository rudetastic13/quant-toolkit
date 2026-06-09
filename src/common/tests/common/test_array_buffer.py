from __future__ import annotations

import numpy as np

from common.array_buffer import (
    ArrayBuffer,
    ArrayInitializationStrategy,
    BufferOverflowError,
    ExpandableArrayBuffer,
)
from common.testing import UnitTest


class TestArrayInitializationStrategy(UnitTest):
    COVERAGE = ["common.array_buffer"]

    def test_zeros_strategy(self):
        buf = ArrayBuffer(size=5, strategy=ArrayInitializationStrategy.Zeros)
        np.testing.assert_array_equal(buf.array, np.zeros(5))

    def test_ones_strategy(self):
        buf = ArrayBuffer(size=4, strategy=ArrayInitializationStrategy.Ones)
        np.testing.assert_array_equal(buf.array, np.ones(4))

    def test_empty_strategy_correct_shape(self):
        buf = ArrayBuffer(size=6, strategy=ArrayInitializationStrategy.Empty)
        self.assertEqual(buf.array.shape, (6,))

    def test_dtype_is_applied(self):
        buf = ArrayBuffer(size=3, dtype=np.float32, strategy=ArrayInitializationStrategy.Zeros)
        self.assertEqual(buf.array.dtype, np.float32)


class TestArrayBuffer(UnitTest):
    COVERAGE = ["common.array_buffer"]

    def _buf(self, size: int = 10) -> ArrayBuffer:
        return ArrayBuffer(size=size, strategy=ArrayInitializationStrategy.Zeros)

    def test_get_slice_returns_correct_view(self):
        buf = self._buf(10)
        s = buf.get_slice(3)
        self.assertEqual(len(s), 3)

    def test_get_slice_advances_cursor(self):
        buf = self._buf(10)
        buf.get_slice(3)
        self.assertEqual(buf.current_idx, 3)
        buf.get_slice(2)
        self.assertEqual(buf.current_idx, 5)

    def test_get_slice_overflow_raises(self):
        buf = self._buf(5)
        buf.get_slice(4)
        with self.assertRaises(BufferOverflowError):
            buf.get_slice(2)

    def test_reset_resets_cursor(self):
        buf = self._buf(10)
        buf.get_slice(7)
        buf.reset()
        self.assertEqual(buf.current_idx, 0)

    def test_write_via_slice(self):
        buf = ArrayBuffer(size=5, dtype=np.float64, strategy=ArrayInitializationStrategy.Zeros)
        s = buf.get_slice(3)
        s[:] = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(buf.array[:3], [1.0, 2.0, 3.0])


class TestExpandableArrayBuffer(UnitTest):
    COVERAGE = ["common.array_buffer"]

    def test_initial_size(self):
        buf = ExpandableArrayBuffer(size=4, strategy=ArrayInitializationStrategy.Zeros)
        self.assertEqual(buf.size, 4)

    def test_get_slice_within_capacity(self):
        buf = ExpandableArrayBuffer(size=10, strategy=ArrayInitializationStrategy.Zeros)
        s = buf.get_slice(5)
        self.assertEqual(len(s), 5)

    def test_auto_expands_when_overflows(self):
        buf = ExpandableArrayBuffer(size=4, strategy=ArrayInitializationStrategy.Zeros)
        buf.get_slice(4)
        # Next request exceeds current size — should expand without raising
        s = buf.get_slice(4)
        self.assertEqual(len(s), 4)
        self.assertGreater(buf.size, 4)

    def test_data_preserved_after_expansion(self):
        buf = ExpandableArrayBuffer(size=4, dtype=np.float64, strategy=ArrayInitializationStrategy.Zeros)
        s1 = buf.get_slice(4)
        s1[:] = [1.0, 2.0, 3.0, 4.0]
        buf.get_slice(4)  # triggers expand
        np.testing.assert_array_equal(buf.array[:4], [1.0, 2.0, 3.0, 4.0])

    def test_invalid_expansion_factor_raises(self):
        with self.assertRaises(ValueError):
            ExpandableArrayBuffer(size=10, expansion_factor=0.0, strategy=ArrayInitializationStrategy.Zeros)
        with self.assertRaises(ValueError):
            ExpandableArrayBuffer(size=10, expansion_factor=1.5, strategy=ArrayInitializationStrategy.Zeros)

    def test_valid_expansion_factor_boundary(self):
        # 1.0 is a valid upper boundary
        buf = ExpandableArrayBuffer(size=4, expansion_factor=1.0, strategy=ArrayInitializationStrategy.Zeros)
        self.assertAlmostEqual(buf.expansion_factor, 1.0)

    def test_reset_resets_cursor(self):
        buf = ExpandableArrayBuffer(size=10, strategy=ArrayInitializationStrategy.Zeros)
        buf.get_slice(5)
        buf.reset()
        self.assertEqual(buf.current_idx, 0)
