"""Tagged edits to ``xlOil.ini``: add/remove our modules in ``[xlOil_Python] LoadModules``.

The ini is TOML but full of comments users care about, so (like ``xloil install`` itself) we
edit it as text: only the ``LoadModules`` line is rewritten, and a marker comment directly
above it records exactly which modules we added. ``untag`` reads that marker back, so it
removes only our entries and never one the user added by hand. Every result is re-parsed with
``tomllib`` before it is returned, so a bad edit raises instead of being written.
"""
from __future__ import annotations

import re
import tomllib

MARKER = "# [quant_toolkit_xl_installer] added to LoadModules:"
SECTION = "xlOil_Python"

_SECTION_RE = re.compile(rf"^[ \t]*\[{SECTION}\][ \t]*$", re.M)
_NEXT_TABLE_RE = re.compile(r"^[ \t]*\[", re.M)
# module names never contain "]", so [^\]]* also covers a list split over several lines
_LOAD_RE = re.compile(r"^(?P<indent>[ \t]*)LoadModules[ \t]*=[ \t]*(?P<value>\[[^\]]*\])", re.M)
_MARKER_RE = re.compile(rf"^[ \t]*{re.escape(MARKER)}[ \t]*(?P<value>\[[^\]]*\])[ \t]*\r?\n", re.M)


def _section(text: str) -> tuple[int, int]:
    m = _SECTION_RE.search(text)
    if m is None:
        raise ValueError(f"no [{SECTION}] table in the ini; is this an xlOil.ini?")
    nxt = _NEXT_TABLE_RE.search(text, m.end())
    return m.end(), nxt.start() if nxt else len(text)


def _parse_list(value: str) -> list[str]:
    return list(tomllib.loads(f"v = {value}")["v"])


def _render_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{s}"' for s in items) + "]"


def load_modules(text: str) -> list[str]:
    """The ``[xlOil_Python] LoadModules`` list (empty if the key is absent)."""
    return list(tomllib.loads(text).get(SECTION, {}).get("LoadModules", []))


def tagged_modules(text: str) -> list[str]:
    """Modules our marker says we added (empty if never tagged)."""
    start, end = _section(text)
    m = _MARKER_RE.search(text[start:end])
    return _parse_list(m["value"]) if m else []


def tag(text: str, modules: list[str]) -> str:
    """Return ``text`` with our managed set in ``LoadModules`` replaced by ``modules``.

    Idempotent. Modules a previous run tagged but ``modules`` no longer names are dropped (so
    a renamed module doesn't linger and fail to import). A module the user listed themselves
    is never claimed by the marker, so ``untag`` leaves it alone.
    """
    prior = tagged_modules(text)
    start, end = _section(text)
    section = _MARKER_RE.sub("", text[start:end], count=1)

    load = _LOAD_RE.search(section)
    current = _parse_list(load["value"]) if load else []
    user_owned = [m for m in current if m not in prior]
    ours = [m for m in dict.fromkeys(modules) if m not in user_owned]
    merged = list(dict.fromkeys(user_owned + ours))

    indent = load["indent"] if load else ""
    block = f"{indent}{MARKER} {_render_list(ours)}\n{indent}LoadModules={_render_list(merged)}"
    if load:
        section = section[: load.start()] + block + section[load.end() :]
    else:
        section = "\n" + block + section

    out = text[:start] + section + text[end:]
    tomllib.loads(out)  # never hand back an unparsable ini
    return out


def untag(text: str) -> str:
    """Return ``text`` with our marker and the modules it records removed from ``LoadModules``."""
    ours = tagged_modules(text)
    if not ours:
        return text
    start, end = _section(text)
    section = _MARKER_RE.sub("", text[start:end], count=1)
    load = _LOAD_RE.search(section)
    if load:
        kept = [m for m in _parse_list(load["value"]) if m not in ours]
        section = section[: load.start()] + f"{load['indent']}LoadModules={_render_list(kept)}" + section[load.end() :]
    out = text[:start] + section + text[end:]
    tomllib.loads(out)
    return out
