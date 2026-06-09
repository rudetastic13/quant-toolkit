"""Generic reusable containers for Enums"""
from __future__ import annotations
from enum import Enum, IntEnum, EnumMeta

class SupportedIntEnum(IntEnum):

    def is_supported(self) -> bool:
        return self.value >= 0


class CaseInsensitiveEnumMeta(EnumMeta):

    def __new__(meta, cls, base, classdict):
        if len(classdict.keys()) != len(set(key.upper() for key in classdict.keys())):
            seen = set()
            bad_vals = []
            for key, val in classdict.items():
                if key.upper() not in seen:
                    seen.add(key.upper())
                else:
                    bad_vals.append({"member": key, "value": val})
            msg = f"Duplicate members detectedin {cls!s}, ignoring casing : {bad_vals}"
            raise RuntimeError(msg)
        return super().__new__(meta, cls, base, classdict)

    def __call__(cls, value, *args, **kwargs):
        # where value is string, check if match to any member value and find first match
        if isinstance(value, str):
            for item in cls._member_map_.values():
                if isinstance(item.value, str) and item.value.lower() == value.lower():
                    return super().__call__(item.value, *args, **kwargs)
        # fall back to parent class call
        return super().__call__(value, *args, **kwargs)

    def __getitem__(cls, item):
        # find membership via member map first
        if item in cls._member_map_:
            return super().__getitem__(item)

        # now check case matching if that fails
        for key in cls._member_map_:
            if key.lower() == item.lower():
                return super().__getitem__(key)

        # fallback to parent method
        return super().__getitem__(item)

class CaseInsensitiveEnum(Enum, metaclass=CaseInsensitiveEnumMeta):

    @classmethod
    def parse(cls, value: str) -> CaseInsensitiveEnum:
        if cls.__members__:
            try:
                return cls[value]
            except KeyError:
                return cls(value)
        raise IndexError(f"No members defined for Enum {cls.__name__}")

