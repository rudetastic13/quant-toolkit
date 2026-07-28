"""European swaption contracts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from finance.dates import Date, Term
from finance.instruments.resolution.builders import Swap


class SwaptionModel(IntEnum):
    Black = 1
    Bachelier = 2


@dataclass(frozen=True, kw_only=True)
class Swaption:
    """European option on a forward-starting fixed/float swap."""

    underlying: Swap
    expiry: Date
    payer: bool = True
    model: SwaptionModel = SwaptionModel.Black
    vol_name: str | None = None

    @classmethod
    def european(
        cls,
        *,
        notional: float,
        strike: float,
        expiry: str,
        swap_tenor: str,
        as_of: Date,
        payer: bool = True,
        model: SwaptionModel = SwaptionModel.Black,
        rate_index: str = "SOFR",
        currency: str = "USD",
        funding_id: str = "STDCSA",
        vol_name: str | None = None,
    ) -> "Swaption":
        expiry_date = as_of + Term.from_str(expiry)
        signed_notional = -abs(float(notional)) if payer else abs(float(notional))
        underlying = Swap.fixed_float_swap(
            notional=signed_notional,
            rate_index=rate_index,
            fixed_rate=float(strike),
            tenor=swap_tenor,
            as_of=expiry_date,
            currency=currency,
            funding_id=funding_id,
        )
        return cls(
            underlying=underlying,
            expiry=expiry_date,
            payer=payer,
            model=SwaptionModel(model),
            vol_name=vol_name,
        )


__all__ = ["Swaption", "SwaptionModel"]
