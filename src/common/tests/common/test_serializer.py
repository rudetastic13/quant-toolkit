from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum, auto

import numpy as np

from common.object import CommonObject, Serializable, register_type
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
        from common.object.serializer import serialize_value

        self.assertEqual(serialize_value(np.int64(5)), 5)
        self.assertIsInstance(serialize_value(np.int64(5)), int)
        self.assertAlmostEqual(serialize_value(np.float64(3.14)), 3.14)
        self.assertIsInstance(serialize_value(np.float64(3.14)), float)

    def test_numpy_array(self):
        from common.object.serializer import serialize_value

        arr = np.array([1.0, 2.0, 3.0])
        result = serialize_value(arr)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_numpy_datetime64(self):
        from common.object.serializer import serialize_value

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
