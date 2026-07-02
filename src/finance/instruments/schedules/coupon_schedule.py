"""Defining a custom coupon schedule"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from functools import partial, cached_property
from typing import Sequence, ClassVar, Callable, Protocol, Self
import numpy as np
from finance.dates import Date
from finance.instruments.enums import CouponType, MarginTreatment
from .base_schedule import BaseEvent, BaseSchedule


class InstrumentLike(Protocol):
    """Protocol for instruments that can be used to define coupon events."""
    effective: Date
    coupon_type: CouponType
    coupon_rate: float
    index_floor: float
    cap: float
    floor: float
    rate_index: str
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
        raise NotImplementedError("CouponEvent subclasses bind their reference calculator")

    @property
    def calculate(self) -> Callable:
        func = partial(self._calculator, **self.params)
        func.__name__ = self._calculator.__name__
        return func

    def to_dict(self) -> dict:
        # ``coupon_type`` is a ClassVar, so asdict() omits it — add it back explicitly;
        # from_dict needs it to dispatch to the right event class.
        return {"coupon_type": self.coupon_type, **asdict(self)}

    @staticmethod
    def from_dict(data: dict) -> CouponEvent:
        """Build the concrete event for ``data['coupon_type']``.

        Extra keys are ignored (so a ``CommonInstrument.__dict__`` can be passed through),
        and a ``start_date`` serialized by ``asdict`` (a plain year/month/day dict) is
        rebuilt into a ``Date``.
        """
        try:
            event_cls = _EVENT_TYPES[data["coupon_type"]]
        except KeyError:
            raise ValueError(f"unknown coupon_type {data.get('coupon_type')!r}") from None
        init_fields = {f.name for f in fields(event_cls) if f.init}
        kwargs = {key: val for key, val in data.items() if key in init_fields}
        if isinstance(kwargs.get("start_date"), dict):
            kwargs["start_date"] = Date(**kwargs["start_date"])
        return event_cls(**kwargs)


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
    """Defines a floating coupon payment event.

    Bounds use ``None`` for "no bound" so a genuine 0% level (e.g. a SOFR coupon floored at
    0%) is distinguishable from "unfloored" — ``0.0`` means a real 0% floor, ``None`` means
    none.  ``spread`` keeps a ``0.0`` default because 0 spread is a normal additive value
    with no sentinel ambiguity.
    """
    coupon_type: ClassVar[CouponType] = CouponType.Floating
    rate_index: str = field(init=True, default="undefined")
    spread: float = field(init=True, default=0)
    index_floor: float | None = field(init=True, default=None)
    cap: float | None = field(init=True, default=None)
    floor: float | None = field(init=True, default=None)

    @cached_property
    def _calculator(self) -> Callable:
        from finance.coupons.calculators import calculate_floating
        return calculate_floating


@dataclass
class AveragedCouponEvent(FloatingCouponEvent):
    """Defines an arithmetically-averaged floating coupon payment event."""
    coupon_type: ClassVar[CouponType] = CouponType.ArithmeticAveraged
    margin_treatment: MarginTreatment = field(init=True, default=MarginTreatment.Inclusive)

    @cached_property
    def _calculator(self) -> Callable:
        from finance.coupons.calculators import calculate_arithmetic_average
        return calculate_arithmetic_average

@dataclass
class CompoundedCouponEvent(FloatingCouponEvent):
    """Defines a compounded (geometrically-averaged) floating coupon payment event."""
    coupon_type: ClassVar[CouponType] = CouponType.GeometricAveraged
    margin_treatment: MarginTreatment = field(init=True, default=MarginTreatment.Inclusive)

    @cached_property
    def _calculator(self) -> Callable:
        from finance.coupons.calculators import calculate_geometric_average
        return calculate_geometric_average

# from_dict dispatch: CouponType -> concrete event class. GeometricAveraged maps to
# CompoundedCouponEvent (daily-compounded in arrears, e.g. SOFR OIS legs).
_EVENT_TYPES: dict[CouponType, type[CouponEvent]] = {
    CouponType.Fixed: FixedCouponEvent,
    CouponType.Floating: FloatingCouponEvent,
    CouponType.ArithmeticAveraged: AveragedCouponEvent,
    CouponType.GeometricAveraged: CompoundedCouponEvent,
}


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
        # Single-event schedule anchored at the leg's effective date. None values are
        # dropped so event-class defaults apply (e.g. spread=0.0 when the leg carries None).
        data = {key: val for key, val in vars(instrument).items() if val is not None}
        data["start_date"] = instrument.effective
        return cls(events=[CouponEvent.from_dict(data)])


    def schedule_from_accrual_grid(self, accrual_grid: np.ndarray) -> np.ndarray:
        """Map each accrual date to the index of the event governing it.

        An event governs every accrual period whose start is on/after the event's anchor
        date and before the next event's. ``side="right"`` is required so a period that
        starts exactly on an event boundary picks up that event (not the prior one).
        """
        schedule_dates = np.array([event.start_date.to_numpy() for event in self.events], dtype="datetime64[D]")
        res = np.searchsorted(schedule_dates, accrual_grid, side="right") - 1
        np.clip(res, 0, len(self.events) - 1, out=res)
        return res
