from finance.dates.enums import FixingType

DEFINED = {
    "SOFR": {
        "long_name": "Secured Overnight Financing Rate",
        "rate_type": "Compound",
        "rate_source": "Federal Reserve Bank of New York",
        "term_structure": "SOFR OIS",
        "lag": 0,
        "fixing_type": FixingType.Arrears,
    }
}
