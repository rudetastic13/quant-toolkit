"""Defining a custom coupon schedule"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from functools import partial, cached_property
from typing import Sequence, ClassVar, Callable, Protocol, Self
import numpy as np
from finance.dates import Date
from finance.instruments.enums import CouponType, MarginTreatment
from .base_schedule import BaseEvent, BaseSchedule


class InstrumentLike(Protocol):
    """Protocol for instruments that can be used to define coupon events."""
    effective_date: Date
    coupon_type: CouponType
    coupon_rate: float
    index_floor: float
    cap: float
    floor: float
    rate_index: str
    margin_treatment: MarginTreatment
    schedules: dict[str, BaseSchedule]

@dataclass
class CouponEvent(BaseEvent):
    """Defines a coupon payment event, anchored to the accrual grid."""
    coupon_type: ClassVar[CouponType] = CouponType.Unused

    @property
    def params(self) -> dict:
        return {key: val for key, val in asdict(self).items() if key not in {"start_date", "coupon_type"}}

    @cached_property
    def _calculator(self) -> Callable:
        return NotImplementedError

    @property
    def calculate(self) -> Callable:
        func = partial(self._calculator, **self.params)
        func.__name__ = self._calculator.__name__
        return func

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> CouponEvent:
        if data["coupon_type"] == CouponType.Fixed:
            return FixedCouponEvent(**data)
        elif data["coupon_type"] == CouponType.Floating:
            return FloatingCouponEvent(**data)
        elif data["coupon_type"] == CouponType.ArithmeticAveraged:
            return AveragedCouponEvent(**data)
        elif data["coupon_type"] == CouponType.GeometricAveraged:
            return CompoundedCouponEvent(**data)
        return NotImplementedError


@dataclass
class FixedCouponEvent(CouponEvent):
    """Defines a fixed coupon payment event."""
    coupon_type: ClassVar[CouponType] = CouponType.Fixed
    coupon_rate: float = field(init=True, default=0)

    @cached_property
    def _calculator(self) -> Callable:
        from finance.coupons.calculators import calculate_fixed
        return calculate_fixed


@dataclass
class FloatingCouponEvent(CouponEvent):
    """Defines a fixed coupon payment event."""
    coupon_type: ClassVar[CouponType] = CouponType.Floating
    rate_index: str = field(init=True, default="undefined")
    spread: float = field(init=True, default=0)
    index_floor: float = field(init=True, default=0)
    cap: float = field(init=True, default=0)
    floor: float = field(init=True, default=0)

    @cached_property
    def _calculator(self) -> Callable:
        from finance.coupons.calculators import calculate_floating
        return calculate_floating


@dataclass
class AveragedCouponEvent(FloatingCouponEvent):
    """Defines a fixed coupon payment event."""
    coupon_type: ClassVar[CouponType] = CouponType.ArithmeticAveraged
    margin_treatment: MarginTreatment = field(init=True, default=MarginTreatment.Inclusive)

    @cached_property
    def _calculator(self) -> Callable:
        from finance.coupons.calculators import calculate_arithmetic_average
        return calculate_arithmetic_average

@dataclass
class CompoundedCouponEvent(FloatingCouponEvent):
    """Defines a fixed coupon payment event."""
    coupon_type: ClassVar[CouponType] = CouponType.GeometricAveraged
    margin_treatment: MarginTreatment = field(init=True, default=MarginTreatment.Inclusive)

    @cached_property
    def _calculator(self) -> Callable:
        from finance.coupons.calculators import calculate_geometric_average
        return calculate_geometric_average

@dataclass
class CouponSchedule(BaseSchedule):
    """Defines a schedule of coupon payment events, anchored to the accrual grid."""
    events: list[CouponEvent] = field(default_factory=list)

    def add_event(self, event: CouponEvent) -> None:
        """Add a coupon event to the schedule, maintaining sorted order."""
        self.events.append(event)
        self.events.sort()

    def to_list(self):
        """Convert the schedule of coupon events to a list of dictionaries for serialization."""
        return [event.to_dict() for event in self.events]

    @staticmethod
    def from_list(data: Sequence[dict]) -> CouponSchedule:
        """Create a CouponSchedule from a list of event data dictionaries."""
        events = [CouponEvent.from_dict(event_data) for event_data in data]
        return CouponSchedule(events=events)

    @classmethod
    def schedule_from_instrument(cls, instrument: InstrumentLike) -> Self:
        if instrument.schedules and "coupon" in instrument.schedules:
            return instrument.schedules["coupon"]
        else:
            return cls(events=[CouponEvent.from_dict(instrument.__dict__)])


    def schedule_from_accrual_grid(self, accrual_grid: np.ndarray) -> np.ndarray:
        """Define the schedule of coupon events based on the accrual grid, returns array of event indices for each accrual date."""
        schedule_dates = np.array([event.start_date.to_numpy() for event in self.events], dtype="datetime64[D]")
        res = np.searchsorted(schedule_dates, accrual_grid, side="left") - 1
        np.clip(res, 0, accrual_grid.shape[0], out=res)
        return res
