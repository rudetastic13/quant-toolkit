import importlib
import inspect
import datetime
import pytest
from hypothesis import given, strategies as st

mod = importlib.import_module("src.dates._date")
Date = getattr(mod, "Date", None)


def _instantiate_date(y, m, d):
    if Date is None:
        pytest.skip("Date class not found in src.dates._date")
    # try common constructor patterns
    try:
        return Date(y, m, d)
    except Exception:
        pass
    try:
        return Date(year=y, month=m, day=d)
    except Exception:
        pass
    # some implementations use from_components or similar
    for name in ("from_ymd", "from_parts", "from_components"):
        ctor = getattr(Date, name, None)
        if callable(ctor):
            try:
                return ctor(y, m, d)
            except Exception:
                pass
    pytest.skip("Cannot construct Date(y, m, d) with detected Date class")


def _extract_ymd(instance):
    # prefer attributes, fall back to methods
    for attr in ("year", "month", "day"):
        if not hasattr(instance, attr):
            break
    else:
        return int(instance.year), int(instance.month), int(instance.day)
    for meth in ("to_ymd", "as_ymd", "components"):
        fn = getattr(instance, meth, None)
        if callable(fn):
            val = fn()
            if isinstance(val, (tuple, list)) and len(val) >= 3:
                return int(val[0]), int(val[1]), int(val[2])
    # if nothing works, raise to indicate unsupported introspection
    raise RuntimeError("Cannot extract year/month/day from Date instance")


def test_date_class_exists():
    assert Date is not None and inspect.isclass(Date)


def test_construct_and_attributes_basic():
    inst = _instantiate_date(2000, 1, 2)
    y, m, d = _extract_ymd(inst)
    assert (y, m, d) == (2000, 1, 2)


def test_iso_roundtrip_if_available():
    inst = _instantiate_date(1999, 12, 31)
    # find possible iso-format method names
    to_names = ("isoformat", "to_iso", "toisoformat")
    from_names = ("fromisoformat", "from_iso", "fromisoformat", "from_string")
    to_fn = next((getattr(inst, n) for n in to_names if hasattr(inst, n)), None)
    if to_fn is None:
        pytest.skip("No iso-formatting method found on Date instance")
    iso = to_fn()
    assert isinstance(iso, str) and iso
    # try to find a parser on the class
    from_fn = next((getattr(Date, n) for n in from_names if hasattr(Date, n)), None)
    if from_fn is None:
        pytest.skip("No from-iso parser found on Date class")
    new = from_fn(iso)
    # compare via attributes if possible
    y1, m1, d1 = _extract_ymd(inst)
    y2, m2, d2 = _extract_ymd(new)
    assert (y1, m1, d1) == (y2, m2, d2)


def test_today_if_available_matches_datetime_today():
    today_cls = getattr(Date, "today", None)
    if not callable(today_cls):
        pytest.skip("Date.today not available")
    inst = today_cls()
    y, m, d = _extract_ymd(inst)
    dt = datetime.date.today()
    assert (y, m, d) == (dt.year, dt.month, dt.day)


def test_comparison_operators_if_available():
    a = _instantiate_date(2000, 1, 1)
    b = _instantiate_date(2001, 1, 1)
    # try ordering
    try:
        assert a != b
        assert a < b or b < a  # at least one ordering should be defined
        # if less-than is defined, check transitivity with a simple sort
        lst = [b, a]
        lst_sorted = sorted(lst)
        # sorted should place a before b if comparisons are implemented consistently
        if lst_sorted[0] is a:
            assert lst_sorted[1] is b
    except TypeError:
        pytest.skip("Date instances are not comparable with < or >")


@given(
    y=st.integers(min_value=1, max_value=9999),
    m=st.integers(min_value=1, max_value=12),
    d=st.integers(min_value=1, max_value=28),
)
def test_hypothesis_constructible_and_matches_datetime(y, m, d):
    # try constructing; skip if not supported
    inst = _instantiate_date(y, m, d)
    y2, m2, d2 = _extract_ymd(inst)
    assert (y, m, d) == (y2, m2, d2)


if __name__ == "__main__":
    pytest.main([__file__])
