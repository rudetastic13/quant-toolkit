"""Backend registry coverage for NumPy, Numba, option models, and the Rust seam."""
import numpy as np

from common.testing import UnitTest
from finance.pricing.engines import has_engine
from finance.pricing.engines.numpy.option import bachelier, black
from finance.pricing.types import Backend, KERNEL_DCF, KERNEL_BACHELIER, RATE_COMPOUNDED


class TestBackendSeams(UnitTest):
    COVERAGE = ["finance.pricing.engines"]

    def test_numpy_kernels_registered(self):
        self.assertTrue(has_engine(KERNEL_DCF, Backend.Numpy))
        self.assertTrue(has_engine(RATE_COMPOUNDED, Backend.Numpy))

    def test_numba_kernels_registered_and_rust_absent(self):
        for kernel in (KERNEL_DCF, RATE_COMPOUNDED):
            self.assertTrue(has_engine(kernel, Backend.Numba))
            self.assertFalse(has_engine(kernel, Backend.Rust))

    def test_option_kernels_registered(self):
        self.assertTrue(has_engine(KERNEL_BACHELIER, Backend.Numpy))
        self.assertTrue(has_engine(KERNEL_BACHELIER, Backend.Numba))

    def test_option_kernels_return_finite_values(self):
        a = np.array([0.04])
        for fn in (bachelier, black):
            self.assertTrue(np.isfinite(fn(a, a, a, a)).all())

    def test_seam_packages_import(self):
        import finance.pricing.engines.numba  # noqa: F401
        import finance.pricing.engines.rust  # noqa: F401
