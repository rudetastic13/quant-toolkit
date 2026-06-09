from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum, auto
from typing import Union

import numpy as np

from common.object import CommonObject, Serializable, register_type
from common.object.serializer import deserialize_value, serialize_value
from common.testing import UnitTest


class Color(IntEnum):
    Red = auto()
    Green = auto()
    Blue = auto()


class Status(StrEnum):
    Active = "active"
    Inactive = "inactive"


@dataclass(kw_only=True)
class SimpleObj(CommonObject):
    name: str
    value: float
    count: int = 0
    flag: bool = True


@dataclass(kw_only=True)
class ObjWithEnum(CommonObject):
    color: Color
    status: Status = Status.Active


@dataclass(kw_only=True)
class ObjWithOptional(CommonObject):
    name: str
    label: str | None = None
    score: float | None = None


@dataclass(kw_only=True)
class Inner(CommonObject):
    x: int


@dataclass(kw_only=True)
class Outer(CommonObject):
    inner: Inner
    items: list[Inner] = field(default_factory=list)


@dataclass(kw_only=True)
class ObjWithCollections(CommonObject):
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, float] = field(default_factory=dict)


class TestSerializePrimitives(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_round_trip_simple(self):
        obj = SimpleObj(name="test", value=3.14, count=5, flag=False)
        data = obj.serialize()
        restored = SimpleObj.deserialize(data)
        self.assertEqual(restored.name, "test")
        self.assertAlmostEqual(restored.value, 3.14)
        self.assertEqual(restored.count, 5)
        self.assertFalse(restored.flag)

    def test_json_round_trip(self):
        obj = SimpleObj(name="hello", value=1.0)
        json_str = obj.to_json()
        restored = SimpleObj.from_json(json_str)
        self.assertEqual(restored.name, "hello")
        self.assertAlmostEqual(restored.value, 1.0)
        self.assertEqual(restored.count, 0)
        self.assertTrue(restored.flag)

    def test_serialize_contains_type_key(self):
        obj = SimpleObj(name="a", value=0.0)
        data = obj.serialize()
        self.assertIn("__type__", data)


class TestSerializeEnums(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_int_enum_round_trip(self):
        obj = ObjWithEnum(color=Color.Green)
        data = obj.serialize()
        self.assertEqual(data["color"], "Green")
        restored = ObjWithEnum.deserialize(data)
        self.assertEqual(restored.color, Color.Green)

    def test_str_enum_round_trip(self):
        obj = ObjWithEnum(color=Color.Red, status=Status.Inactive)
        restored = ObjWithEnum.deserialize(obj.serialize())
        self.assertEqual(restored.status, Status.Inactive)


class TestSerializeOptional(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_none_fields(self):
        obj = ObjWithOptional(name="test")
        data = obj.serialize()
        self.assertIsNone(data["label"])
        restored = ObjWithOptional.deserialize(data)
        self.assertIsNone(restored.label)
        self.assertIsNone(restored.score)

    def test_present_optional_fields(self):
        obj = ObjWithOptional(name="test", label="ok", score=9.5)
        restored = ObjWithOptional.deserialize(obj.serialize())
        self.assertEqual(restored.label, "ok")
        self.assertAlmostEqual(restored.score, 9.5)


class TestSerializeNested(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_nested_object(self):
        obj = Outer(inner=Inner(x=42), items=[Inner(x=1), Inner(x=2)])
        data = obj.serialize()
        self.assertIsInstance(data["inner"], dict)
        self.assertEqual(len(data["items"]), 2)

        restored = Outer.deserialize(data)
        self.assertEqual(restored.inner.x, 42)
        self.assertEqual(len(restored.items), 2)
        self.assertEqual(restored.items[0].x, 1)
        self.assertEqual(restored.items[1].x, 2)

    def test_nested_json_round_trip(self):
        obj = Outer(inner=Inner(x=10))
        restored = Outer.from_json(obj.to_json())
        self.assertEqual(restored.inner.x, 10)


class TestSerializeCollections(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_list_and_dict(self):
        obj = ObjWithCollections(tags=["a", "b"], metadata={"x": 1.0, "y": 2.0})
        restored = ObjWithCollections.deserialize(obj.serialize())
        self.assertEqual(restored.tags, ["a", "b"])
        self.assertAlmostEqual(restored.metadata["x"], 1.0)

    def test_empty_collections(self):
        obj = ObjWithCollections()
        restored = ObjWithCollections.deserialize(obj.serialize())
        self.assertEqual(restored.tags, [])
        self.assertEqual(restored.metadata, {})


class TestSerializeNumpy(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_numpy_scalars(self):
        self.assertEqual(serialize_value(np.int64(5)), 5)
        self.assertIsInstance(serialize_value(np.int64(5)), int)
        self.assertAlmostEqual(serialize_value(np.float64(3.14)), 3.14)
        self.assertIsInstance(serialize_value(np.float64(3.14)), float)

    def test_numpy_array_envelope_shape(self):
        """serialize_value on an ndarray produces a dtype-aware envelope dict."""
        arr = np.array([1.0, 2.0, 3.0])
        env = serialize_value(arr)
        self.assertTrue(env["__ndarray__"])
        self.assertEqual(env["dtype"], str(arr.dtype))
        self.assertEqual(env["shape"], [3])
        self.assertEqual(env["data"], [1.0, 2.0, 3.0])

    def test_numpy_array_round_trip_float64(self):
        arr = np.array([1.0, 2.0, 3.0])
        restored = deserialize_value(serialize_value(arr), None)
        self.assertEqual(restored.dtype, arr.dtype)
        np.testing.assert_array_equal(restored, arr)

    def test_numpy_array_round_trip_float32(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        restored = deserialize_value(serialize_value(arr), None)
        self.assertEqual(restored.dtype, np.float32)
        np.testing.assert_array_equal(restored, arr)

    def test_numpy_array_round_trip_int8(self):
        arr = np.array([1, -3, 127], dtype=np.int8)
        restored = deserialize_value(serialize_value(arr), None)
        self.assertEqual(restored.dtype, np.int8)
        np.testing.assert_array_equal(restored, arr)

    def test_numpy_array_round_trip_bool(self):
        arr = np.array([True, False, True])
        restored = deserialize_value(serialize_value(arr), None)
        self.assertEqual(restored.dtype, np.bool_)
        np.testing.assert_array_equal(restored, arr)

    def test_numpy_array_round_trip_2d(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        restored = deserialize_value(serialize_value(arr), None)
        self.assertEqual(restored.dtype, np.float32)
        self.assertEqual(restored.shape, (2, 2))
        np.testing.assert_array_equal(restored, arr)

    def test_numpy_array_round_trip_datetime64_day(self):
        arr = np.array(["2025-01-01", "2025-06-01"], dtype="datetime64[D]")
        restored = deserialize_value(serialize_value(arr), None)
        self.assertEqual(restored.dtype, arr.dtype)
        np.testing.assert_array_equal(restored, arr)

    def test_numpy_array_round_trip_datetime64_ns(self):
        arr = np.array(["2025-01-01T00:00:00", "2025-06-01T12:30:00"], dtype="datetime64[ns]")
        restored = deserialize_value(serialize_value(arr), None)
        self.assertEqual(restored.dtype, arr.dtype)
        np.testing.assert_array_equal(restored, arr)

    def test_numpy_datetime64_scalar(self):
        """Scalar np.datetime64 still serializes to a plain string (unchanged)."""
        dt = np.datetime64("2025-04-03")
        result = serialize_value(dt)
        self.assertEqual(result, "2025-04-03")


class MyCustom:
    def __init__(self, val: int):
        self.val = val


register_type(
    MyCustom,
    serializer=lambda obj: {"custom_val": obj.val},
    deserializer=lambda data: MyCustom(data["custom_val"]),
)


@dataclass(kw_only=True)
class ObjWithCustom(CommonObject):
    custom: MyCustom


class TestRegisterType(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_custom_type_round_trip(self):
        obj = ObjWithCustom(custom=MyCustom(99))
        data = obj.serialize()
        self.assertEqual(data["custom"], {"custom_val": 99})

        restored = ObjWithCustom.deserialize(data)
        self.assertEqual(restored.custom.val, 99)


class TestPolymorphicDeserialization(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_deserialize_via_base_class(self):
        obj = SimpleObj(name="poly", value=1.0)
        data = obj.serialize()
        # Deserialize via Serializable base — should resolve to SimpleObj
        restored = Serializable.deserialize(data)
        self.assertIsInstance(restored, SimpleObj)
        self.assertEqual(restored.name, "poly")

    def test_missing_fields_use_defaults(self):
        data = {
            "__type__": f"{SimpleObj.__module__}.{SimpleObj.__qualname__}",
            "name": "minimal",
            "value": 0.0,
        }
        restored = SimpleObj.deserialize(data)
        self.assertEqual(restored.count, 0)
        self.assertTrue(restored.flag)


# ── Marker-driven deserialization (untyped containers) ─────────────────────────


@dataclass(kw_only=True)
class ObjWithUntypedDict(CommonObject):
    payload: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class ObjWithArray(CommonObject):
    values: np.ndarray = field(default_factory=lambda: np.array([]))


class TestMarkerDrivenDeserialization(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_ndarray_inside_untyped_dict_round_trips(self):
        """An ndarray nested inside an untyped dict field survives a full JSON round-trip."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        obj = ObjWithUntypedDict(payload={"my_array": arr})
        restored = ObjWithUntypedDict.from_json(obj.to_json())
        result = restored.payload["my_array"]
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, arr)

    def test_serializable_inside_untyped_dict_round_trips(self):
        """A Serializable nested inside an untyped dict field is reconstructed to its concrete type."""
        inner = SimpleObj(name="nested", value=7.0)
        obj = ObjWithUntypedDict(payload={"item": inner.serialize()})
        restored = ObjWithUntypedDict.from_json(obj.to_json())
        item = restored.payload["item"]
        self.assertIsInstance(item, SimpleObj)
        self.assertEqual(item.name, "nested")

    def test_polymorphic_from_json_with_nested_array(self):
        """Polymorphic from_json via Serializable base works when the object carries an ndarray."""
        arr = np.array([10, 20, 30], dtype=np.int16)
        obj = ObjWithArray(values=arr)
        json_str = obj.to_json()
        restored = Serializable.from_json(json_str)
        self.assertIsInstance(restored, ObjWithArray)
        self.assertEqual(restored.values.dtype, np.int16)
        np.testing.assert_array_equal(restored.values, arr)

    def test_plain_list_with_no_type_hint_recurses(self):
        """deserialize_value with type_hint=None on a plain list recurses into elements."""
        arr = np.array([7.0, 8.0], dtype=np.float64)
        serialized = [serialize_value(arr)]
        result = deserialize_value(serialized, None)
        self.assertIsInstance(result[0], np.ndarray)
        np.testing.assert_array_equal(result[0], arr)

    def test_plain_dict_with_no_type_hint_recurses(self):
        """deserialize_value with type_hint=None on a plain dict (no markers) recurses values."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        raw = {"nested": serialize_value(arr)}
        result = deserialize_value(raw, None)
        self.assertIsInstance(result["nested"], np.ndarray)
        np.testing.assert_array_equal(result["nested"], arr)

    def test_scalar_with_no_type_hint_passes_through(self):
        """deserialize_value with type_hint=None on a primitive returns it unchanged."""
        self.assertEqual(deserialize_value(42, None), 42)
        self.assertEqual(deserialize_value("hello", None), "hello")


# ── Additional edge-case coverage ─────────────────────────────────────────────


@dataclass(kw_only=True)
class ObjWithSet(CommonObject):
    tags: set[str] = field(default_factory=set)


@dataclass(kw_only=True)
class ObjWithTuple(CommonObject):
    pair: tuple[int, str] = field(default_factory=lambda: (0, ""))


@dataclass(kw_only=True)
class ObjWithBareList(CommonObject):
    items: list = field(default_factory=list)


@dataclass(kw_only=True)
class ObjWithNonInitField(CommonObject):
    name: str = ""
    computed: str = field(init=False, default="computed")


class TestEdgeCases(UnitTest):
    COVERAGE = ["common.object.serializer"]

    def test_set_serialization(self):
        """Sets are serialized as lists."""
        result = serialize_value({"a", "b"})
        self.assertIsInstance(result, list)
        self.assertCountEqual(result, ["a", "b"])

    def test_set_round_trip(self):
        obj = ObjWithSet(tags={"x", "y", "z"})
        restored = ObjWithSet.deserialize(obj.serialize())
        self.assertEqual(restored.tags, {"x", "y", "z"})

    def test_tuple_round_trip(self):
        obj = ObjWithTuple(pair=(5, "hello"))
        restored = ObjWithTuple.deserialize(obj.serialize())
        self.assertEqual(restored.pair, (5, "hello"))

    def test_bare_list_hint_recurses_marker_driven(self):
        """A field typed as plain list still round-trips ndarray envelopes inside it."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        envelope = serialize_value(arr)
        obj = ObjWithBareList(items=[envelope])
        data = obj.serialize()
        restored = ObjWithBareList.deserialize(data)
        self.assertIsInstance(restored.items[0], np.ndarray)
        np.testing.assert_array_equal(restored.items[0], arr)

    def test_non_init_field_not_serialized(self):
        """Fields with init=False are excluded from serialization."""
        obj = ObjWithNonInitField(name="test")
        data = obj.serialize()
        self.assertNotIn("computed", data)
        restored = ObjWithNonInitField.deserialize(data)
        self.assertEqual(restored.computed, "computed")

    def test_union_value_none_path(self):
        """When a Union/Optional field has value=None, None is returned."""
        obj = ObjWithOptional(name="x", label=None, score=None)
        restored = ObjWithOptional.deserialize(obj.serialize())
        self.assertIsNone(restored.label)
        self.assertIsNone(restored.score)

    def test_union_fallthrough_returns_value(self):
        """When all non-None Union branches raise, the raw value passes through."""
        # "not_a_number" → int("not_a_number") → ValueError
        #                → float("not_a_number") → ValueError → fallthrough
        result = deserialize_value("not_a_number", Union[int, float])
        self.assertEqual(result, "not_a_number")

    def test_deserialize_unknown_type_key_falls_back_to_cls(self):
        """If __type__ is missing from registry, deserialize falls back to the called class."""
        data = {"__type__": "nonexistent.module.Foo", "name": "x", "value": 1.0}
        # SimpleObj.deserialize should ignore the bad __type__ and use SimpleObj
        restored = SimpleObj.deserialize(data)
        self.assertEqual(restored.name, "x")

    def test_final_fallthrough_returns_value_unchanged(self):
        """Values whose type hint is not handled are returned unchanged."""

        class Opaque:
            pass

        opaque = Opaque()
        result = deserialize_value(opaque, Opaque)
        self.assertIs(result, opaque)

    def test_hint_driven_serializable_without_type_key(self):
        """Deserializing a raw dict (no __type__) into a known Serializable hint works."""
        # A hand-crafted dict for Inner (no __type__ stamp) under a field typed as Inner
        raw = {"x": 99}
        result = deserialize_value(raw, Inner)
        self.assertIsInstance(result, Inner)
        self.assertEqual(result.x, 99)

    def test_bare_typing_tuple_no_args(self):
        """Tuple with no type params (typing.Tuple) returns tuple(value)."""
        from typing import Tuple  # noqa: UP006 — intentional bare Tuple for coverage

        result = deserialize_value([1, 2, 3], Tuple)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1, 2, 3))
