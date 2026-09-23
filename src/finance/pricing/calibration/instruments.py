"""Calibration instruments — pricers that each solve for a single quoted number.

A calibration instrument carries one market quote plus a way to compute the model value of
that *same* measure off a market.  The residual (``implied - quote``, in rate units) is what
the solver drives to zero.  Crucially the measure lives on the instrument, not the
calibrator: a deposit quotes a simple money-market rate, a swap quotes a par rate, and a
future bond helper would quote a yield — each slots into the same calibrator unchanged.

Deposits/FRAs are closed-form (one ``RateGenerator.simple_rate`` call); swaps go through the
pricing kernel (compile once, reprice per solver iteration) and read the par rate off the
leg PVs by *classifying legs*, never by positional index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np

from finance.instruments.resolution import (
    Deposit,
    Fra,
    SofrFuture,
    Swap,
    curve_name,
)
from finance.markets.context import MarketContext
from finance.markets.rate_generator import RateGenerator
from finance.conventions import ConventionRegistry, default_registry
from finance.pricing.pricers.base import PricingProgram
from finance.pricing.pricers.swap import SwapPricer
from finance.pricing.pricers.futures import FuturesPricer


class QuoteKind(IntEnum):
    """The measure a quote is expressed in — owned by the instrument, not the calibrator."""

    ParRate = 1      # swaps
    SimpleRate = 2   # deposits / FRAs (money-market simple rate)
    Yield = 3        # bonds — designed seam, no helper yet
    FuturesRate = 4  # (100 - quoted price) / 100


@dataclass(frozen=True)
class Quote:
    """A market quote: a value in the units of its ``kind``."""

    value: float
    kind: QuoteKind


@runtime_checkable
class CalibrationInstrument(Protocol):
    """A single-quote pricer.  ``curve`` is the curve name it pins; ``pillar_date`` is the
    node it anchors; ``implied`` is the model value of the quoted measure."""

    quote: Quote
    curve: str

    @property
    def pillar_date(self) -> np.datetime64: ...

    def implied(self, market: MarketContext) -> float: ...

    def residual(self, market: MarketContext) -> float: ...


def _as_date_array(d) -> np.ndarray:
    return np.array([d.to_numpy()], dtype="datetime64[D]")


@dataclass
class DepositHelper:
    """A cash deposit whose quote is the simple money-market rate over its accrual window."""

    deposit: Deposit
    quote: Quote
    curve: str

    @property
    def pillar_date(self) -> np.datetime64:
        return np.datetime64(self.deposit.maturity.to_numpy(), "D")

    def implied(self, market: MarketContext) -> float:
        starts = _as_date_array(self.deposit.effective)
        ends = _as_date_array(self.deposit.maturity)
        return float(
            RateGenerator(market).simple_rate(
                self.curve,
                starts,
                ends,
                self.deposit.day_count_method,
            )[0]
        )

    def residual(self, market: MarketContext) -> float:
        return self.implied(market) - self.quote.value


@dataclass
class FraHelper:
    """A forward rate agreement whose quote is the simple forward rate over [start, end]."""

    fra: Fra
    quote: Quote
    curve: str

    @property
    def pillar_date(self) -> np.datetime64:
        return np.datetime64(self.fra.maturity.to_numpy(), "D")

    def implied(self, market: MarketContext) -> float:
        starts = _as_date_array(self.fra.effective)
        ends = _as_date_array(self.fra.maturity)
        return float(
            RateGenerator(market).simple_rate(
                self.curve,
                starts,
                ends,
                self.fra.day_count_method,
            )[0]
        )

    def residual(self, market: MarketContext) -> float:
        return self.implied(market) - self.quote.value


@dataclass
class SwapHelper:
    """A par swap whose quote is the par fixed rate.

    Compiles its swap once (``notional=1``, ``fixed_rate=1`` so the fixed leg PV *is* the
    unit-rate annuity).  ``implied`` classifies the compiled legs as fixed vs floating by
    iterating the instrument — generalising past two-leg vanillas — and returns
    ``-(Σ floating leg PV) / (Σ fixed leg PV)``.  Both legs carry their pay/receive sign, so
    this is the par rate regardless of receive- vs pay-fixed orientation.
    """

    swap: Swap
    quote: Quote
    curve: str
    _program: PricingProgram = field(init=False, repr=False)
    _fixed_idx: tuple[int, ...] = field(init=False, repr=False)
    _float_idx: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._program = SwapPricer().compile([self.swap])
        fixed: list[int] = []
        floating: list[int] = []
        for i, leg in enumerate(self.swap):
            (floating if leg.coupon_type.is_floating else fixed).append(i)
        self._fixed_idx = tuple(fixed)
        self._float_idx = tuple(floating)

    @property
    def pillar_date(self) -> np.datetime64:
        ki = self._program.inputs
        last = ki.pay_dates.max()
        if ki.obs_ends.size:
            last = max(last, ki.obs_ends.max())
        return np.datetime64(last, "D")

    def implied(self, market: MarketContext) -> float:
        return RateGenerator(market).par_rate_from_program(
            self._program,
            fixed_legs=self._fixed_idx,
            floating_legs=self._float_idx,
        )

    def residual(self, market: MarketContext) -> float:
        return self.implied(market) - self.quote.value


@dataclass
class FuturesHelper:
    """SOFR future helper targeting the convexity-adjusted curve forward."""

    future: SofrFuture
    quote: Quote
    curve: str
    convexity: float | object = 0.0
    _program: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._program = FuturesPricer().compile([self.future])

    @property
    def pillar_date(self) -> np.datetime64:
        return np.datetime64(self.future.ref_end.to_numpy(), "D")

    def _convexity(self, market: MarketContext) -> float:
        if callable(self.convexity):
            return float(self.convexity(market))
        return float(self.convexity)

    def implied(self, market: MarketContext) -> float:
        # Read the unadjusted curve period rate directly; FuturesProgram.reprice adds the
        # contract's display convexity, while calibration consumes CA on the quote side.
        return float(self._program.program.reprice(market).rate[0])

    def residual(self, market: MarketContext) -> float:
        return self.implied(market) - (self.quote.value - self._convexity(market))

    def numba_gradient(self, market: MarketContext) -> np.ndarray:
        return self._program.risk(market, self.curve).rate_gradient[0]


# -- trader-facing factories (minimal input; conventions do the work) -----------------------


def deposit_helper(
    *, rate: float, tenor: str, as_of, rate_index: str = "SOFR", currency: str = "USD",
    funding_id: str = "STDCSA", registry: ConventionRegistry = default_registry,
) -> DepositHelper:
    dep = Deposit.spot_deposit(
        rate=rate, tenor=tenor, as_of=as_of, rate_index=rate_index,
        currency=currency, funding_id=funding_id, registry=registry,
    )
    return DepositHelper(
        deposit=dep, quote=Quote(float(rate), QuoteKind.SimpleRate),
        curve=curve_name(currency, rate_index),
    )


def fra_helper(
    *, rate: float, start: str, end: str, as_of, rate_index: str = "SOFR", currency: str = "USD",
    funding_id: str = "STDCSA", registry: ConventionRegistry = default_registry,
) -> FraHelper:
    fra = Fra.forward_starting(
        rate=rate, start=start, end=end, as_of=as_of, rate_index=rate_index,
        currency=currency, funding_id=funding_id, registry=registry,
    )
    return FraHelper(
        fra=fra, quote=Quote(float(rate), QuoteKind.SimpleRate),
        curve=curve_name(currency, rate_index),
    )


def swap_helper(
    *, rate: float, tenor: str, as_of, rate_index: str = "SOFR", currency: str = "USD",
    funding_id: str = "STDCSA", registry: ConventionRegistry = default_registry,
) -> SwapHelper:
    swap = Swap.fixed_float_swap(
        notional=1.0, rate_index=rate_index, fixed_rate=1.0, tenor=tenor, as_of=as_of,
        currency=currency, funding_id=funding_id, registry=registry,
    )
    return SwapHelper(
        swap=swap, quote=Quote(float(rate), QuoteKind.ParRate),
        curve=curve_name(currency, rate_index),
    )


def futures_helper(
    *,
    price: float,
    contract: str,
    month: str | tuple[int, int],
    as_of,
    convexity: float | object = 0.0,
    rate_index: str = "SOFR",
    currency: str = "USD",
    funding_id: str = "STDCSA",
) -> FuturesHelper:
    future = SofrFuture.from_contract_month(
        contract=contract,
        month=month,
        as_of=as_of,
        currency=currency,
        rate_index=rate_index,
        funding_id=funding_id,
    )
    return FuturesHelper(
        future=future,
        quote=Quote((100.0 - float(price)) / 100.0, QuoteKind.FuturesRate),
        curve=curve_name(currency, rate_index),
        convexity=convexity,
    )


__all__ = [
    "QuoteKind",
    "Quote",
    "CalibrationInstrument",
    "DepositHelper",
    "FraHelper",
    "SwapHelper",
    "FuturesHelper",
    "deposit_helper",
    "fra_helper",
    "swap_helper",
    "futures_helper",
]
