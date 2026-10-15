"""Conventions: curve keys, index families and the per-family market convention registry."""

from finance.conventions.keys import OVERNIGHT, CurveKey, CurveKind, parse_tenor, tenor_label
from finance.conventions.rate_index import IndexFamily, RateIndex
from finance.conventions.market_conventions import (
    MATCH_TENOR,
    DepositConventions,
    FixedLegConventions,
    FloatLegConventions,
    FraConventions,
    MarketConventions,
    SwapConventions,
    frequency_for_tenor,
)
from finance.conventions.convention_registry import (
    ConventionRegistry,
    register_convention,
    default_registry,
    split_index_name,
)

# Import for side effect: registers the bundled MarketConventions definitions
# (e.g. USD SOFR) into the shared registry on package import.
from finance.conventions import definitions as _definitions  # noqa: F401

__all__ = [
    "CurveKey",
    "CurveKind",
    "OVERNIGHT",
    "parse_tenor",
    "tenor_label",
    "IndexFamily",
    "RateIndex",
    "MATCH_TENOR",
    "frequency_for_tenor",
    "FixedLegConventions",
    "FloatLegConventions",
    "SwapConventions",
    "DepositConventions",
    "FraConventions",
    "MarketConventions",
    "ConventionRegistry",
    "register_convention",
    "default_registry",
    "split_index_name",
]
