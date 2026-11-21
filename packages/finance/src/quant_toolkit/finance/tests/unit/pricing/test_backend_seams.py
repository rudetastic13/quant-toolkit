"""Engine seams: the program registry (Backend -> engine) and the narrow option-kernel registry."""
import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.pricing.engines import has_kernel, has_program, kernel, program_class
from quant_toolkit.finance.pricing.engines.numpy.option import bachelier, bachelier_greeks, black
from quant_toolkit.finance.pricing.engines.numpy.program import NumpyProgram
from quant_toolkit.finance.pricing.types import Backend, KERNEL_BACHELIER, KERNEL_BLACK


class TestProgramRegistry(UnitTest):

    def test_numpy_is_registered_and_resolves(self):
        self.assertTrue(has_program(Backend.Numpy))
        self.assertIs(program_class(Backend.Numpy), NumpyProgram)

    def test_numba_is_registered_when_installed(self):
        from quant_toolkit.finance.pricing.engines.numba import NumbaProgram

        self.assertIs(program_class(Backend.Numba), NumbaProgram)

    def test_unregistered_backend_raises(self):
        self.assertFalse(has_program(Backend.Jax))
        with self.assertRaises(NotImplementedError):
            program_class(Backend.Jax)


class TestOptionKernelRegistry(UnitTest):

    def test_option_kernels_registered_for_both_backends(self):
        for kernel_id in (KERNEL_BLACK, KERNEL_BACHELIER):
            for backend in (Backend.Numpy, Backend.Numba):
                self.assertTrue(has_kernel(kernel_id, backend))

    def test_dispatch_returns_greeks(self):
        f, k, t, v = (np.array([x]) for x in (0.04, 0.04, 1.0, 0.01))
        for backend in (Backend.Numpy, Backend.Numba):
            greeks = kernel((KERNEL_BACHELIER, backend), f, k, t, v)
            np.testing.assert_allclose(greeks.value, bachelier_greeks(f, k, t, v).value, rtol=1e-13)

    def test_value_wrappers_finite(self):
        a = np.array([0.04])
        for fn in (bachelier, black):
            self.assertTrue(np.isfinite(fn(a, a, a, a)).all())

    def test_rate_and_dcf_reductions_are_not_registered(self):
        self.assertFalse(has_kernel("dcf", Backend.Numpy))
        self.assertFalse(has_kernel("rate_compounded", Backend.Numba))
