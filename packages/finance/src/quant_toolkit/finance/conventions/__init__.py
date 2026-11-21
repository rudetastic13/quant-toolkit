"""Conventions the curve layer reads: curve keys, index families and the family registry.

Leg and product conventions (the ``MarketConventions`` bundle the builders read) live in
``quant_toolkit.finance.instruments.conventions`` and are registered beside the families here, so this
package depends on ``quant_toolkit.finance.dates`` only.
"""

from quant_toolkit.finance.conventions.keys import OVERNIGHT, CurveKey, parse_tenor, tenor_label
from quant_toolkit.finance.conventions.rate_index import IndexFamily, RateIndex
from quant_toolkit.finance.conventions.convention_registry import (
    ConventionRegistry,
    register_convention,
    default_registry,
)

# Import for side effect: registers the bundled index families (USD SOFR, Fed Funds, term SOFR,
# UST-OTR, XXX-FTP) into the shared registry on package import.
from quant_toolkit.finance.conventions import definitions as _definitions  # noqa: F401

__all__ = [
    "CurveKey",
    "OVERNIGHT",
    "parse_tenor",
    "tenor_label",
    "IndexFamily",
    "RateIndex",
    "ConventionRegistry",
    "register_convention",
    "default_registry",
]
