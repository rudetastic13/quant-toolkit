"""Generic registry for stroring and retrieving objects"""
from collections.abc import Callable
from typing import (
    TypeVar,
    Generic,
    Hashable,
    Sequence,
    TypeVar,
    Union,
    Optional,
    Any
)

T = TypeVar('T')
R = TypeVar('R')

class Registry(Generic[T]):
    def __init__(self, name: str = "Registry"):
        self.name = name
        self._registry: dict[str, T] = {}

    def __contains__(self, item):
        return item in self._registry

    def register(self, keys: Hashable | Sequence[Hashable], obj: T, *, overwrite: bool = False) -> None:
        if keys is None:
            keys = [obj.__name__]
        else:
            keys = [keys]

        if not overwrite and any(key in self._registry for key in keys):
            raise ValueError("One or more keys already exist in the registry, set overwrite=True to overwrite existing keys")

        self._registry.update({key: obj for key in keys})

    def get(self, key: T) -> Optional[T]:
        return self._registry.get(key)

    def has(self, key: T) -> bool:
        return key in self._registry

    def all(self) -> dict[Hashable, T]:
        return self._registry.copy()

def register_with(
    registry: Registry[T],
    keys: Hashable | Sequence[Hashable] | None = None,
    **kwargs,
    ) -> Callable[[T], T]:
    def decorator(obj: T) -> T:
        registry.register(keys, obj, **kwargs)
        return obj
    return decorator