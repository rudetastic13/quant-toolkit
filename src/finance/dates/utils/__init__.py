"""Common utilities we use for date stuff"""
from importlib import import_module
from typing import Any
from types import ModuleType
import numpy as np
from finance.dates import Date, Calendar, BDC
from finance.dates.types import DateType, BoolType


def date_based_utils_module() -> ModuleType:
    return import_module("finance.dates.utils.impl.dt_based")

def np_date_based_utils_module() -> ModuleType:
    return import_module("finance.dates.utils.impl.np_based")

_date_based_utils_module = date_based_utils_module
_np_date_based_utils_module = np_date_based_utils_module

type_map: dict[Any, ModuleType] = {
    Date: _date_based_utils_module,
    np.datetime64: _np_date_based_utils_module,
    np.ndarray: _np_date_based_utils_module,
}

def is_good_bd(dt: DateType, calendar: str | Calendar) -> BoolType:
    return type_map[type(dt)].is_good_bd(dt, calendar)

def adjust_date(dt: DateType, bdc: str | BDC, calendar: str | Calendar) -> DateType:
    return type_map[type(dt)].adjust_date(dt, bdc, calendar)

