"""European swaption contracts."""
from __future__ import annotations

from dataclasses import dataclass

from common.object.serializer import serializable, upgrade
from finance.dates import Date, Term
from finance.instruments.common_instrument import CommonInstrument
from finance.instruments.enums import CouponType
from finance.instruments.priceable import Priceable
from finance.instruments.resolution.builders import Swap


@serializable("Swaption", version=2)
@dataclass(frozen=True, kw_only=True)
class Swaption(Priceable):
    """European option to enter ``underlying``, a forward-starting fixed/float swap.

    The contract is the swap plus when it can be entered and which side of the option is
    held.  Everything else is read off the underlying: ``strike`` is its fixed rate and
    ``payer`` is whether it pays fixed.  The pricing model is not contract data; it follows
    the quoting units of the vol cube the option resolves (``vol_name``, default the cube
    named after the underlying's index).

    Parameters
    ----------
    underlying : Swap
        One flat fixed leg against one floating leg, starting on or after ``expiry``.
    expiry : Date
    long : bool, optional
        ``True`` holds the option, ``False`` has written it.
    vol_name : str or None, optional

    Raises
    ------
    ValueError
        If the underlying is not one flat fixed leg against one floating leg, or starts
        before ``expiry``.
    """

    underlying: Swap
    expiry: Date
    long: bool = True
    vol_name: str | None = None

    def __post_init__(self):
        fixed = [leg for leg in self.underlying if not leg.is_floating]
        if len(fixed) != 1 or fixed[0].coupon_type is not CouponType.Fixed:
            raise ValueError(
                "a swaption's underlying is one flat fixed leg (coupon_type Fixed) against one floating leg; "
                "a strike is not defined otherwise"
            )
        if fixed[0].effective < self.expiry:
            raise ValueError(f"underlying starts {fixed[0].effective}, before expiry {self.expiry}")

    @property
    def fixed_leg(self) -> CommonInstrument:
        return next(leg for leg in self.underlying if not leg.is_floating)

    @property
    def strike(self) -> float:
        """The underlying's fixed rate."""
        return self.fixed_leg.coupon_rate

    @property
    def payer(self) -> bool:
        """The option is on paying fixed (a call on the swap rate)."""
        return next(sign for leg, sign in self.underlying.legs if not leg.is_floating) < 0

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
        long: bool = True,
        rate_index: str = "USD.SOFR.1D",
        currency: str = "USD",
        funding_id: str = "STDCSA",
        vol_name: str | None = None,
    ) -> "Swaption":
        """Build a swaption on a standard spot-from-expiry swap.

        ``notional`` is the size (its sign is ignored); ``payer`` and ``long`` say which option
        and which side of it.
        """
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
        return cls(underlying=underlying, expiry=expiry_date, long=long, vol_name=vol_name)


@upgrade("Swaption", from_version=1)
def _swaption_1_to_2(doc: dict) -> dict:
    # v2: ``payer`` is read off the underlying and ``model`` off the vol cube's units
    doc.pop("payer", None)
    doc.pop("model", None)
    return doc
