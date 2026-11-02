"""Every product round-trips through the Serializable mixin and prices identically after reload."""

import json

import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Deposit, Fra, SofrFuture, Swap, Swaption
from finance.markets.context import MarketContext
from finance.markets.curves import YieldCurve
from finance.markets.vols import FlatVolSurface, VolNamespace, VolUnits

AS_OF = Date(2026, 6, 1)


def _market() -> MarketContext:
    dates = np.array(["2026-06-01", "2028-06-01", "2031-06-01", "2041-06-01"], dtype="datetime64[D]")
    t = (dates - dates[0]).astype(np.int64) / 365.0
    sofr = YieldCurve.build(dates, np.exp(-0.04 * t), key="USD.SOFR.1D")
    vols = VolNamespace()
    vols.bind("USD.SOFR.1D", FlatVolSurface(level=0.20, units=VolUnits.Lognormal))
    return MarketContext(AS_OF, {"USD.SOFR.1D": sofr, "USD.STDCSA": sofr}, vols=vols)


class TestInstrumentSerialization(UnitTest):
    COVERAGE = [
        "finance.instruments.priceable",
        "finance.instruments.resolution.builders",
        "finance.instruments.resolution.futures",
        "finance.instruments.resolution.swaption",
    ]

    def setUp(self):
        self.market = _market()
        self.swap = Swap.fixed_float_swap(notional=1e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=AS_OF)
        self.instruments = {
            "swap": self.swap,
            "deposit": Deposit.spot_deposit(notional=1e6, rate=0.05, tenor="3M", as_of=AS_OF),
            "fra": Fra.forward_starting(notional=1e6, rate=0.045, start="3M", end="6M", as_of=AS_OF),
            "future": SofrFuture.from_contract_month(contract="SR3", month="U26", as_of=AS_OF),
            "swaption": Swaption.european(notional=1e6, strike=0.04, expiry="1Y", swap_tenor="5Y", as_of=AS_OF),
        }

    def test_every_product_round_trips_as_an_equal_object(self):
        for name, instrument in self.instruments.items():
            with self.subTest(name):
                text = instrument.to_json()
                back = type(instrument).from_json(text)
                self.assertEqual(back, instrument)
                self.assertEqual(json.loads(text)["__type__"], f"{type(instrument).__module__}.{type(instrument).__qualname__}")

    def test_reloaded_instruments_price_the_same(self):
        from finance.pricing.pricers import FuturesPricer

        for name in ("swap", "deposit", "fra"):
            with self.subTest(name):
                instrument = self.instruments[name]
                back = type(instrument).from_json(instrument.to_json())
                self.assertEqual(back(self.market).pv, instrument(self.market).pv)
        future = self.instruments["future"]
        back = SofrFuture.from_json(future.to_json())
        np.testing.assert_array_equal(
            FuturesPricer().price([back], self.market).model_price, FuturesPricer().price([future], self.market).model_price
        )

    def test_the_compiled_program_cache_stays_out_of_the_document(self):
        self.swap(self.market)  # populates the cache
        self.assertIn("_program", self.swap.__dict__)
        doc = self.swap.serialize()
        self.assertNotIn("_program", doc)
        self.assertEqual(set(doc) - {"__type__"}, {"receive_leg", "pay_leg", "currency", "index_name", "funding_id"})

    def test_a_derived_coupon_schedule_is_rebuilt_on_load_not_stored(self):
        doc = self.swap.serialize()
        self.assertNotIn("coupon_schedule", doc["receive_leg"])   # flat leg: schedule derives from the flat fields
        self.assertNotIn("notional_schedule", doc["receive_leg"])
        back = Swap.deserialize(doc)
        self.assertEqual(back.receive_leg.coupon_schedule, self.swap.receive_leg.coupon_schedule)
