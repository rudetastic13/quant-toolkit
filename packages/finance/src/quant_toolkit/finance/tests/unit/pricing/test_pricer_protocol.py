"""Every product pricer and program satisfies the Pricer / Program protocols."""
import inspect

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.pricing.pricers import FuturesPricer, LegPricer, Pricer, PricingProgram, Program, SwaptionPricer
from quant_toolkit.finance.pricing.pricers.futures import FuturesProgram
from quant_toolkit.finance.pricing.pricers.swaption import SwaptionProgram
from quant_toolkit.finance.pricing.types import Backend

PRICERS = (LegPricer, FuturesPricer, SwaptionPricer)
PROGRAMS = (PricingProgram, FuturesProgram, SwaptionProgram)


class TestPricerProtocol(UnitTest):

    def test_pricers_satisfy_protocol(self):
        for cls in PRICERS:
            with self.subTest(pricer=cls.__name__):
                self.assertIsInstance(cls(), Pricer)

    def test_programs_satisfy_protocol(self):
        for cls in PROGRAMS:
            with self.subTest(program=cls.__name__):
                self.assertTrue(callable(getattr(cls, "reprice", None)))
                self.assertTrue(issubclass(cls, Program))

    def test_default_backend_is_a_backend(self):
        for cls in PRICERS:
            with self.subTest(pricer=cls.__name__):
                self.assertIsInstance(cls.default_backend, Backend)

    def test_backend_is_keyword_only_with_none_default(self):
        for cls in PRICERS:
            for name in ("compile", "price"):
                with self.subTest(pricer=cls.__name__, method=name):
                    param = inspect.signature(getattr(cls, name)).parameters["backend"]
                    self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
                    self.assertIsNone(param.default)

    def test_non_pricer_is_rejected(self):
        self.assertNotIsInstance(object(), Pricer)
