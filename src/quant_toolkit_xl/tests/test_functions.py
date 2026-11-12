"""Tests for the q* UDF layer — delegation and the #QERR error surface."""
import numpy as np

from common.testing import UnitTest
from quant_toolkit_xl import cache, functions as f


def _curve_dfs():
    pillars = ["2026-06-01", "2027-06-01", "2031-06-01"]
    dfs = np.exp(-0.04 * np.array([0.0, 1.0, 5.0]))
    return pillars, dfs


class TestQFunctions(UnitTest):
    COVERAGE = ["quant_toolkit_xl.functions"]

    def setUp(self):
        cache.clear()

    def test_qbuildcurve_returns_handle(self):
        pillars, dfs = _curve_dfs()
        self.assertTrue(f.qBuildCurve(pillars, dfs).startswith("Curve::"))

    def test_qprice_end_to_end(self):
        pillars, dfs = _curve_dfs()
        as_of = "2026-06-01"
        ch = f.qBuildCurve(pillars, dfs)
        sh = f.qSwap(100e6, "USD.SOFR.1D", 0.041, "5Y", as_of)
        prog = f.qCompileProgram([sh])
        mkt = f.qMarket(as_of, "USD", "USD.SOFR.1D", ch)
        pv = f.qPrice(prog, mkt)
        self.assertIsInstance(pv, np.ndarray)
        self.assertTrue(np.isfinite(pv).all())

    def test_error_becomes_qerr_string(self):
        # bad interpolation name -> readable cell string, not an exception
        pillars, dfs = _curve_dfs()
        out = f.qBuildCurve(pillars, dfs, "NotAnInterpolator")
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("#QERR:"))

    def test_wrong_handle_kind_becomes_qerr_string(self):
        sh = f.qSwap(100e6, "USD.SOFR.1D", 0.041, "5Y", "2026-06-01")
        out = f.qDiscountFactor(sh, ["2026-06-01"])  # swap handle where curve expected
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("#QERR:"))


# VBA reserved words. xlwings copies Python parameter names verbatim into the generated VBA
# wrapper module, and one reserved word there is a syntax error that breaks every q* function
# in the workbook (``currency`` did exactly that: ``Currency`` is a VBA data type).
_VBA_RESERVED = frozenset(
    """
    and as boolean byref byte byval call case class const currency date decimal declare dim do double each
    else elseif empty end enum erase error event exit false for friend function get goto if implements in
    integer is let like long loop me mod new next not nothing null object on option optional or paramarray
    preserve private property public redim rem resume return select set single static step stop string sub
    then to true type typeof until variant wend while with xor
    """.split()
)


class TestUdfSignaturesAreVbaSafe(UnitTest):
    COVERAGE = ["quant_toolkit_xl.functions"]

    def test_no_parameter_is_a_vba_reserved_word(self):
        import inspect

        udfs = [getattr(f, n) for n in dir(f) if n.startswith("q") and callable(getattr(f, n))]
        self.assertTrue(udfs)
        for udf in udfs:
            for name in inspect.signature(udf).parameters:
                with self.subTest(udf=udf.__name__, parameter=name):
                    self.assertNotIn(name.lower(), _VBA_RESERVED)
