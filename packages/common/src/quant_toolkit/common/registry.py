"""Generic registry for stroring and retrieving objects"""
import inspect
from collections.abc import Callable
from typing import Hashable


class Registry[T]:
    """Key -> object store.  A key is any hashable; a tuple is one composite key, not several keys."""

    def __init__(self, name: str = "Registry"):
        self.name = name
        self._registry: dict[Hashable, T] = {}

    def __contains__(self, item: Hashable) -> bool:
        return item in self._registry

    def register(self, keys: Hashable | None, obj: T, *, overwrite: bool = False) -> None:
        if keys is None:
            keys = [obj.__name__]
        else:
            keys = [keys]

        if not overwrite and any(key in self._registry for key in keys):
            raise ValueError("One or more keys already exist in the registry, set overwrite=True to overwrite existing keys")

        self._registry.update({key: obj for key in keys})

    def get(self, key: Hashable) -> T | None:
        return self._registry.get(key)

    def has(self, key: Hashable) -> bool:
        return key in self._registry

    def all(self) -> dict[Hashable, T]:
        return self._registry.copy()

class RegistryError(Exception):

    def __init__(self, key: Hashable, obj_type: str, original_exception: Exception):
        super().__init__(f"Failed to {obj_type} registered with key '{key}': {str(original_exception)}")
        self.key = key
        self.obj_type = obj_type
        self.original_exception = original_exception


def get_and_execute[R](
    registry: Registry[Callable[..., R]],
    key: Hashable,
    *args,
    **kwargs
) -> R:
    obj = registry.get(key)
    if obj is None:
        raise KeyError(f"Key '{key}' not found in registry '{registry.name}'")
    try:
        if callable(obj):
            return obj(*args, **kwargs)
        else:
            return obj
    except Exception as e:
        obj_dtype = "instantiate" if inspect.isclass(obj) else "execute"
        raise RegistryError(key, obj_dtype, e) from e

def create_factory[R](
    registry: Registry[Callable[..., R]],
    *default_args,
    **default_kwargs,
) -> Callable[..., R]:
    def _factory(key: Hashable, *args, **kwargs) -> R:
        all_args = (*default_args, *args)
        all_kwargs = {**default_kwargs, **kwargs}
        return get_and_execute(registry, key, *all_args, **all_kwargs)
    _factory.__name__ = f"{registry.name.lower().replace(' ', '_')}_factory"
    _factory.__doc__ = f"Factory function under registry '{registry.name}'"
    return _factory

def register_with[T](
    registry: Registry[T],
    keys: Hashable | None = None,
    **kwargs,
    ) -> Callable[[T], T]:
    def decorator(obj: T) -> T:
        registry.register(keys, obj, **kwargs)
        return obj
    return decorator