"""Backend seams: numba/rust and option kernels are reserved keys, not yet implemented.

Pins the swappable-backend contract — the registry knows the numpy kernels and cleanly
rejects unimplemented backends, so adding numba/rust later is purely additive.
"""
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

    def test_numba_rust_backends_absent(self):
        for kernel in (KERNEL_DCF, RATE_COMPOUNDED):
            self.assertFalse(has_engine(kernel, Backend.Numba))
            self.assertFalse(has_engine(kernel, Backend.Rust))

    def test_option_kernels_reserved(self):
        self.assertFalse(has_engine(KERNEL_BACHELIER, Backend.Numpy))

    def test_option_stubs_raise(self):
        a = np.array([0.04])
        for fn in (bachelier, black):
            with self.assertRaises(NotImplementedError):
                fn(a, a, a, a)

    def test_seam_packages_import(self):
        import finance.pricing.engines.numba  # noqa: F401
        import finance.pricing.engines.rust  # noqa: F401
