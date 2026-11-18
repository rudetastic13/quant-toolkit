"""A versioned bind/resolve store, so option pricers resolve a cube by name much as linear
pricers resolve a curve from ``MarketContext``."""
from __future__ import annotations

from finance.markets.vols.cube import ModelVolCube


class VolNamespace:
    """Name -> ModelVolCube store with a version counter.

    Not a singleton; one per scenario is fine.
    """

    def __init__(self) -> None:
        self._surfaces: dict[str, ModelVolCube] = {}
        self._versions: dict[str, int] = {}

    def bind(self, name: str, cube: ModelVolCube) -> None:
        if name in self._surfaces:
            raise KeyError(f"'{name}' is already bound. Use rebind() to overwrite.")
        self._surfaces[name] = cube
        self._versions[name] = 0

    def rebind(self, name: str, cube: ModelVolCube) -> None:
        self._surfaces[name] = cube
        self._versions[name] = self._versions.get(name, -1) + 1

    def resolve(self, name: str) -> ModelVolCube:
        try:
            return self._surfaces[name]
        except KeyError:
            raise KeyError(f"No vol cube bound under '{name}'") from None

    def version(self, name: str) -> int:
        try:
            return self._versions[name]
        except KeyError:
            raise KeyError(f"No vol cube bound under '{name}'") from None

    def __contains__(self, name: str) -> bool:
        return name in self._surfaces
