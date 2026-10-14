"""Coupon schedules: the source of truth for what a leg pays.

A :class:`CouponSchedule` is a sorted list of :class:`CouponEvent` segments, each anchored at
the accrual period it starts governing.  A vanilla leg has one event; a step-up or a
fixed→float switch has several.  ``CommonInstrument`` derives a single-event schedule from
its flat coupon fields when none is supplied, and the compiler lowers *only* the schedule —
the flat fields never reach the pricer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Sequence, ClassVar, Self
import numpy as np
from finance.dates import Date
from finance.instruments.enums import CouponType, MarginTreatment
from .base_schedule import BaseEvent, BaseSchedule


@dataclass
class CouponEvent(BaseEvent):
    """Defines a coupon payment event, anchored to the accrual grid."""
    coupon_type: ClassVar[CouponType] = CouponType.Unused

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


@dataclass
class AveragedCouponEvent(FloatingCouponEvent):
    """Defines an arithmetically-averaged floating coupon payment event."""
    coupon_type: ClassVar[CouponType] = CouponType.ArithmeticAveraged
    margin_treatment: MarginTreatment = field(init=True, default=MarginTreatment.Inclusive)


@dataclass
class CompoundedCouponEvent(FloatingCouponEvent):
    """Defines a compounded (geometrically-averaged) floating coupon payment event."""
    coupon_type: ClassVar[CouponType] = CouponType.GeometricAveraged
    margin_treatment: MarginTreatment = field(init=True, default=MarginTreatment.Inclusive)


# from_dict dispatch: CouponType -> concrete event class.  GeometricAveraged maps to
# CompoundedCouponEvent (daily-compounded in arrears, e.g. SOFR OIS legs); Zero is a fixed
# event at the leg's (default 0) coupon rate — principal-only legs set PrincipalExchange.
_EVENT_TYPES: dict[CouponType, type[CouponEvent]] = {
    CouponType.Zero: FixedCouponEvent,
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

    # -- what the schedule pays ---------------------------------------------------------

    @property
    def coupon_types(self) -> tuple[CouponType, ...]:
        return tuple(event.coupon_type for event in self.events)

    @property
    def is_floating(self) -> bool:
        """Any segment is index-linked (the leg needs a projection curve)."""
        return any(event.coupon_type.is_floating for event in self.events)

    @property
    def needs_observation_grid(self) -> bool:
        """Any segment compounds or averages daily fixings (the leg needs the Tier-2 grid)."""
        return any(event.coupon_type.needs_observation_grid for event in self.events)

    # -- (de)serialization --------------------------------------------------------------

    def to_list(self):
        """Convert the schedule of coupon events to a list of dictionaries for serialization."""
        return [event.to_dict() for event in self.events]

    @staticmethod
    def from_list(data: Sequence[dict]) -> CouponSchedule:
        """Create a CouponSchedule from a list of event data dictionaries."""
        events = [CouponEvent.from_dict(event_data) for event_data in data]
        return CouponSchedule(events=events)

    @classmethod
    def schedule_from_instrument(cls, instrument) -> Self:
        """The single-event schedule a leg's flat coupon fields describe, anchored at its effective date.

        ``None`` values are dropped so event-class defaults apply (e.g. spread=0.0 when the
        leg carries None).
        """
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
