"""European swaption contracts."""
from __future__ import annotations

from dataclasses import dataclass

from quant_toolkit.common.object.serializer import serializable, upgrade
from quant_toolkit.finance.dates import Date, Term
from quant_toolkit.finance.instruments.priceable import Priceable
from quant_toolkit.finance.instruments.resolution.builders import Swap


@serializable("Swaption", version=2)
@dataclass(frozen=True, kw_only=True)
class Swaption(Priceable):
    """European option on a forward-starting fixed/float swap.

    Contract terms only.  The option model is not a contract term: it follows the quoting
    convention of the vol cube the swaption resolves (``vol_name``, or the cube named after
    its rate index), so a Normal cube prices Bachelier and a Lognormal cube prices Black.
    ``swaption(market, requests)`` prices standalone through ``SwaptionPricer``.

    Parameters
    ----------
    underlying : Swap
        One fixed leg against one floating leg.  Its direction is the option's: paying fixed
        makes a payer, receiving fixed a receiver (see :attr:`payer`).  The fixed coupon is
        the strike.
    expiry : Date
    position : float
        Number of options held: positive is bought, negative is sold.  Scales PV, greeks and
        risk.
    vol_name : str or None
        The vol cube to resolve; ``None`` is the cube named after the rate index.

    Raises
    ------
    ValueError
        If the underlying is not one fixed leg against one floating leg.
    """

    underlying: Swap
    expiry: Date
    position: float = 1.0
    vol_name: str | None = None

    def __post_init__(self):
        floating = [leg.is_floating for leg in self.underlying]
        if sorted(floating) != [False, True]:
            raise ValueError("a swaption's underlying must be one fixed leg against one floating leg")

    @property
    def payer(self) -> bool:
        """True for the right to pay fixed: the underlying's pay leg is its fixed leg."""
        return not self.underlying.pay_leg.is_floating

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
        position: float = 1.0,
        rate_index: str = "USD.SOFR.1D",
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
        return cls(underlying=underlying, expiry=expiry_date, position=float(position), vol_name=vol_name)


@upgrade("Swaption", from_version=1)
def _swaption_1_to_2(doc: dict) -> dict:
    """v2 drops ``model`` (it follows the resolved vol cube's units) and ``payer`` (it is the
    underlying's direction; every v1 document was built with the two in agreement)."""
    doc.pop("model", None)
    doc.pop("payer", None)
    return doc
