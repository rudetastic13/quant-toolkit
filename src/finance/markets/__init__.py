"""Define the market"""
import numpy as np
from common.containers.enums import SupportedIntEnum
from finance.dates import Date, np_date_based_utils_module, Term
from dataclasses import dataclass, field
from .curves import ZeroCurve, CurveInterpolator
from .paths import SinglePath, FlatPath, InvertedPath

_np_dt_utils = np_date_based_utils_module()

class MarketType(SupportedIntEnum):
    Curve = 1
    SinglePath = 2


@dataclass
class Market:
    as_of_date: Date
    market_type: MarketType
    curves: dict | None= field(default_factory=dict)
    paths: dict | None= field(default_factory=dict)

    def __post_init__(self):
        self._market_rates = self.curves if self.market_type == MarketType.Curve else self.paths

    def get_rates(self, rate_index: str, dates: np.ndarray) -> np.ndarray:
        if self.market_type == MarketType.Curve:
            return self._get_rates_curves(rate_index, dates)
        elif self.market_type == MarketType.SinglePath:
            return self._get_rates_paths(rate_index, dates)
        else:
            raise ValueError(f"Unsupported market type {self.market_type}")

    def _get_rates_paths(self, rate_index: str, dates: np.ndarray) -> np.ndarray:
        path = self.paths[rate_index]
        return path.get_values(dates)

    def _get_rates_curves(self, rate_index: str, dates: np.ndarray) -> np.ndarray:
        ccy, rate_index, tenor = rate_index.split()
        my_curve = None
        # ``self.curves`` is a dict keyed by name; match on the curve's currency.
        for curve in self.curves.values():
            if getattr(curve, "currency", None) == ccy:
                my_curve = curve
        if my_curve is None:
            raise IndexError(f"Requested rate_index {rate_index} is not defined in curve set")

        sub_curve = my_curve.get_curve(rate_index)
        term = Term.from_str(tenor)
        end_dates = dates + term

        # done
        return sub_curve.get_rates(dates, end_dates)



