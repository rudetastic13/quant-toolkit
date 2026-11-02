"""Frozen documents must load forever.

One file per ``(stable name, version)`` ever shipped lives in ``expects/serialized``.  Files are
never edited: a version bump adds ``Name.v<N>.json`` and keeps the old ones, and every file must
still load with the current code and equal the object built from the same inputs today.
"""

import json
from pathlib import Path

from common.object import Serializable
from common.testing import UnitTest
from finance.dates import BDC, Date, DayCountMethod, Frequency
from finance.instruments import AmortizationType, CommonInstrument
from finance.instruments.enums import PrincipalExchange
from finance.instruments.resolution import Deposit, Fra, SofrFuture, Swap, Swaption
from finance.instruments.schedules import CompoundedCouponEvent, CouponSchedule, FixedCouponEvent

FIXTURES = Path(__file__).parent / "expects" / "serialized"
AS_OF = Date(2026, 6, 1)


def _today() -> dict[str, Serializable]:
    """The same objects the fixtures were frozen from, built with today's code."""
    return {
        "Swap": Swap.fixed_float_swap(notional=1e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=AS_OF),
        "Deposit": Deposit.spot_deposit(notional=1e6, rate=0.05, tenor="3M", as_of=AS_OF),
        "Fra": Fra.forward_starting(notional=1e6, rate=0.045, start="3M", end="6M", as_of=AS_OF),
        "SofrFuture": SofrFuture.from_contract_month(contract="SR3", month="U26", as_of=AS_OF),
        "Swaption": Swaption.european(notional=1e6, strike=0.04, expiry="1Y", swap_tenor="5Y", as_of=AS_OF),
        "CommonInstrument": CommonInstrument(
            effective=AS_OF, maturity=Date(2031, 6, 1), currency="USD", notional=1e6, payment_frequency=Frequency.Quarterly,
            day_count_method=DayCountMethod.Actual360, business_day_convention=BDC.ModifiedFollowing, rate_index="USD.SOFR.1D",
            reset_frequency=Frequency.Daily, principal_exchange=PrincipalExchange.Final,
            amortization=AmortizationType.StraightLine, amort_start=Date(2027, 6, 1), principal_frequency=Frequency.Annually,
            coupon_schedule=CouponSchedule(events=[
                FixedCouponEvent(start_date=AS_OF, coupon_rate=0.03),
                CompoundedCouponEvent(start_date=Date(2028, 6, 1), rate_index="USD.SOFR.1D", spread=0.001),
            ]),
        ),
    }


class TestSerializedFixtures(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_every_frozen_document_loads_and_equals_todays_object(self):
        today = _today()
        files = sorted(FIXTURES.glob("*.v*.json"))
        self.assertTrue(files, f"no fixtures under {FIXTURES}")
        seen = set()
        for path in files:
            name, version = path.stem.split(".v")
            seen.add(name)
            with self.subTest(file=path.name):
                doc = json.loads(path.read_text())
                self.assertEqual(doc["__type__"], name)
                self.assertEqual(doc["__version__"], int(version))
                loaded = Serializable.deserialize(doc)        # resolved by stable name, upgraded if older than the class
                self.assertEqual(loaded, today[name])
                self.assertEqual(loaded.serialize()["__version__"], type(loaded).__serial_version__)
        self.assertEqual(seen, set(today), "every product has at least one frozen document")

    def test_every_registered_finance_type_has_a_fixture(self):
        from common.object.serializer import _types

        finance_types = {name for name, cls in _types.all().items() if cls.__module__.startswith("finance.")}
        frozen = {p.stem.split(".v")[0] for p in FIXTURES.glob("*.v*.json")}
        events = {"FixedCouponEvent", "FloatingCouponEvent", "AveragedCouponEvent", "CompoundedCouponEvent",
                  "CouponSchedule", "NotionalEvent", "NotionalSchedule"}  # frozen inside the CommonInstrument document
        self.assertEqual(finance_types - events, frozen)
