"""Leg and product conventions: the ``MarketConventions`` bundle the builders read.

Registered beside the index families in ``finance.conventions``; importing this package
registers the bundled USD definitions.
"""

from finance.instruments.conventions.market_conventions import (
    MATCH_TENOR,
    DepositConventions,
    FixedLegConventions,
    FloatLegConventions,
    FraConventions,
    MarketConventions,
    SwapConventions,
    frequency_for_tenor,
)

# Import for side effect: registers the bundled product conventions (USD SOFR, Fed Funds,
# term SOFR, UST-OTR, XXX-FTP) beside their families on package import.
from finance.instruments.conventions import definitions as _definitions  # noqa: F401

__all__ = [
    "MATCH_TENOR",
    "frequency_for_tenor",
    "FixedLegConventions",
    "FloatLegConventions",
    "SwapConventions",
    "DepositConventions",
    "FraConventions",
    "MarketConventions",
]
