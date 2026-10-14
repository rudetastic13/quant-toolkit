"""Join legs end to end: piecewise-regular dating without a custom payment schedule.

A leg's dating is generated from its flat fields — one frequency, one calendar, one day count.
A product whose dating changes at a date (monthly for two years, then quarterly) is several
legs with the same sign, each generated regularly, joined where one matures and the next
starts.  ``LegPricer`` already rolls every leg up into the instrument PV, so nothing new is
needed to price it; this helper only enforces that the pieces actually abut.
"""
from __future__ import annotations

from finance.instruments.common_instrument import CommonInstrument


def join_legs(*legs: CommonInstrument, sign: float = 1.0) -> tuple[tuple[CommonInstrument, float], ...]:
    """Signed ``(leg, sign)`` pairs for legs that run back to back.

    Each leg must mature on the next one's effective date and share its currency.  Fixing
    continuity across a join is the caller's responsibility: a join must sit on a reset
    boundary, otherwise the second leg's first fixing is built from its own reset cadence
    and may not be the fixing actually in force.
    """
    if not legs:
        raise ValueError("join_legs needs at least one leg")
    for i, (first, second) in enumerate(zip(legs, legs[1:])):
        if first.maturity != second.effective:
            raise ValueError(
                f"leg {i} matures {first.maturity} but leg {i + 1} starts {second.effective}; joined legs must abut"
            )
        if first.currency != second.currency:
            raise ValueError(f"leg {i} is {first.currency} but leg {i + 1} is {second.currency}")
    return tuple((leg, float(sign)) for leg in legs)


__all__ = ["join_legs"]
