"""Pinned quote risk of a fixed book on a production-shaped SOFR curve and a par-swap template.

A change in the adjoint, the layout, the Jacobian, the basis or the transfer shows up here as
a number moving.  Refresh deliberately with::

    REGRESSION_UPDATE_EXPECTS=1 pytest src/finance/tests/regression/test_risk_template.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from common.testing import UnitTest
from common.testing.expects_loader import ModuleLoader
from finance.dates import Date
from finance.instruments.resolution import SofrFuture, Swap
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator
from finance.pricing.calibration import (
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    deposit_helper,
    futures_helper,
    swap_helper,
)
from finance.pricing.pricers import FuturesPricer, SwapPricer
from finance.pricing.risk import pillar_risk
from finance.pricing.risk.template import RiskTemplate

EXPECTS_MODULE = "finance.tests.regression.expects.risk_template"
UPDATE = os.environ.get("REGRESSION_UPDATE_EXPECTS") == "1"
AS_OF = Date(2026, 6, 1)
SOFR = "USD.SOFR.1D"

DEPOSITS = {"3M": 0.0431, "6M": 0.0428}
FUTURES = {"Z26": 95.78, "H27": 95.94, "M27": 96.07, "U27": 96.16, "Z27": 96.21, "H28": 96.24, "M28": 96.25}
SWAPS = {"3Y": 0.0385, "4Y": 0.0383, "5Y": 0.0384, "7Y": 0.0390, "10Y": 0.0398, "20Y": 0.0412, "30Y": 0.0415}
TEMPLATE = ("3M", "6M", "9M", "12M", "15M", "18M", "21M", "24M", "3Y", "4Y", "5Y", "7Y", "10Y", "20Y", "30Y")


def production_market() -> MarketContext:
    helpers = (
        [deposit_helper(rate=r, tenor=t, as_of=AS_OF) for t, r in DEPOSITS.items()]
        + [futures_helper(price=p, contract="SR3", month=m, as_of=AS_OF) for m, p in FUTURES.items()]
        + [swap_helper(rate=r, tenor=t, as_of=AS_OF) for t, r in SWAPS.items()]
    )
    definition = CurveDefinition(SOFR, CurveInterpolator.LogLinearDF, discount_keys=("USD.STDCSA",))
    return CurveCalibrator(helpers, GlobalSolver(), definition).calibrate(MarketContext(AS_OF)).market


@pytest.mark.regression
class TestRiskTemplateRegression(UnitTest):
    COVERAGE = ["finance.pricing.risk.template", "finance.pricing.risk.jacobian", "finance.pricing.risk.pillar"]

    def setUp(self):
        pytest.importorskip("numba")
        self.market = production_market()
        self.template = RiskTemplate(
            [swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in TEMPLATE], labels=TEMPLATE, name="SOFR par"
        )
        self.bound = self.template.bind(self.market)
        swaps = SwapPricer().compile(
            [
                Swap.fixed_float_swap(notional=100e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=AS_OF),
                Swap.fixed_float_swap(notional=-60e6, rate_index="USD.SOFR.1D", fixed_rate=0.041, tenor="10Y", as_of=AS_OF),
                Swap.fixed_float_swap(notional=30e6, rate_index="USD.SOFR.1D", fixed_rate=0.043, tenor="18M", as_of=AS_OF),
            ]
        )
        futures = FuturesPricer().compile(
            [
                SofrFuture.from_contract_month(contract="SR3", month=m, as_of=AS_OF, contracts=50)
                for m in ("Z26", "H27", "M27", "U27")
            ]
        )
        self.swap_risk = self.bound.quote_risk(
            pillar_risk(swaps, self.market, labels=("5Y recv", "10Y pay", "18M recv"))
        )
        self.future_risk = self.bound.quote_risk(pillar_risk(futures, self.market, labels=("Z26", "H27", "M27", "U27")))

    def _tables(self) -> dict[str, pd.DataFrame]:
        def quote_table(risk):
            frame = pd.DataFrame(risk.dv01, columns=list(risk.labels), index=list(risk.positions)).reset_index(
                names="position"
            )
            frame["pillar_parallel"] = risk.pillar_parallel * 1e-4
            frame["template_parallel"] = risk.template_parallel * 1e-4
            frame["unexplained_max"] = risk.unexplained_max * 1e-4
            return frame

        diagnostics = pd.DataFrame(
            {
                "node": list(self.bound.labels),
                "leverage": self.bound.diagnostics.leverage,
                "unit_sensitivity": self.bound.unit_sensitivity,
                "implied": self.bound.implied,
            }
        )
        diagnostics["condition"] = self.bound.diagnostics.condition
        return {"swaps": quote_table(self.swap_risk), "futures": quote_table(self.future_risk), "template": diagnostics}

    def test_quote_risk_matches_expects(self):
        tables = self._tables()
        loader = ModuleLoader(EXPECTS_MODULE)
        if UPDATE:
            loader.update("risk_template", tables)
            self.skipTest("expects refreshed")
        expects = loader.load()
        for name, got in tables.items():
            with self.subTest(table=name):
                self.assertIn(name, expects, "no stored expects; run with REGRESSION_UPDATE_EXPECTS=1")
                want = expects[name]
                self.assertEqual(list(got.columns), list(want.columns))
                for column in got.columns:
                    if got[column].dtype.kind == "f":
                        np.testing.assert_allclose(
                            got[column].to_numpy(), want[column].to_numpy().astype(float), rtol=1e-8, atol=1e-6
                        )
                    else:
                        self.assertEqual(list(got[column]), list(want[column]))
