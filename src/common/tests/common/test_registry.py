from __future__ import annotations

from common.registry import Registry, RegistryError, create_factory, get_and_execute, register_with
from common.testing import UnitTest


class TestRegistry(UnitTest):
    COVERAGE = ["common.registry"]

    def _make(self, name: str = "Test") -> Registry:
        return Registry(name=name)

    def test_register_and_get(self):
        reg = self._make()
        reg.register("key", "value")
        self.assertEqual(reg.get("key"), "value")

    def test_contains(self):
        reg = self._make()
        reg.register("k", 1)
        self.assertIn("k", reg)
        self.assertNotIn("missing", reg)

    def test_has(self):
        reg = self._make()
        reg.register("k", 1)
        self.assertTrue(reg.has("k"))
        self.assertFalse(reg.has("nope"))

    def test_all_returns_copy(self):
        reg = self._make()
        reg.register("a", 1)
        reg.register("b", 2)
        all_items = reg.all()
        self.assertEqual(all_items, {"a": 1, "b": 2})
        # Mutating the copy does not affect the registry
        all_items["c"] = 3
        self.assertNotIn("c", reg)

    def test_register_duplicate_raises(self):
        reg = self._make()
        reg.register("dup", 1)
        with self.assertRaises(ValueError):
            reg.register("dup", 2)

    def test_register_overwrite_allowed(self):
        reg = self._make()
        reg.register("k", 1)
        reg.register("k", 99, overwrite=True)
        self.assertEqual(reg.get("k"), 99)

    def test_get_missing_returns_none(self):
        reg = self._make()
        self.assertIsNone(reg.get("nope"))

    def test_register_none_keys_uses_name(self):
        """When keys=None, the object's __name__ attribute is used as the key."""

        def my_func():
            return 42

        reg: Registry = self._make()
        reg.register(None, my_func)
        self.assertIn("my_func", reg)


class TestGetAndExecute(UnitTest):
    COVERAGE = ["common.registry"]

    def test_calls_callable(self):
        reg: Registry = Registry("funcs")
        reg.register("add", lambda x, y: x + y)
        result = get_and_execute(reg, "add", 2, 3)
        self.assertEqual(result, 5)

    def test_returns_non_callable_directly(self):
        reg: Registry = Registry("data")
        reg.register("pi", 3.14159)
        result = get_and_execute(reg, "pi")
        self.assertAlmostEqual(result, 3.14159)

    def test_missing_key_raises_key_error(self):
        reg: Registry = Registry("empty")
        with self.assertRaises(KeyError):
            get_and_execute(reg, "ghost")

    def test_callable_error_raises_registry_error(self):
        reg: Registry = Registry("err")

        def explode():
            raise RuntimeError("boom")

        reg.register("bad", explode)
        with self.assertRaises(RegistryError) as ctx:
            get_and_execute(reg, "bad")
        self.assertEqual(ctx.exception.key, "bad")
        self.assertIsInstance(ctx.exception.original_exception, RuntimeError)


class TestCreateFactory(UnitTest):
    COVERAGE = ["common.registry"]

    def test_factory_dispatches_by_key(self):
        reg: Registry = Registry("ops")
        reg.register("double", lambda x: x * 2)
        reg.register("negate", lambda x: -x)
        factory = create_factory(reg)
        self.assertEqual(factory("double", 5), 10)
        self.assertEqual(factory("negate", 7), -7)

    def test_factory_merges_default_kwargs(self):
        reg: Registry = Registry("greet")
        reg.register("hello", lambda name, greeting="hi": f"{greeting}, {name}")
        factory = create_factory(reg, greeting="hey")
        self.assertEqual(factory("hello", "world"), "hey, world")

    def test_factory_name_is_set(self):
        reg: Registry = Registry("My Registry")
        factory = create_factory(reg)
        self.assertEqual(factory.__name__, "my_registry_factory")


class TestRegisterWith(UnitTest):
    COVERAGE = ["common.registry"]

    def test_decorator_registers_object(self):
        reg: Registry = Registry("deco")

        @register_with(reg, keys="my_key")
        def my_fn():
            return "decorated"

        self.assertIn("my_key", reg)
        self.assertEqual(reg.get("my_key")(), "decorated")

    def test_decorator_returns_original(self):
        reg: Registry = Registry("deco2")

        @register_with(reg, keys="fn2")
        def fn2():
            return 99

        self.assertEqual(fn2(), 99)
