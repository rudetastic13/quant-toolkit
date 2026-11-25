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
    """Where the body of the ``[xlOil_Python]`` table sits in the ini text.

    Parameters
    ----------
    text : str
        The whole ini.

    Returns
    -------
    start, end : int
        Offsets of the table body: from just after its header to the next table header, or
        to the end of the text.

    Raises
    ------
    ValueError
        If the table is missing.
    """
    m = _SECTION_RE.search(text)
    if m is None:
        raise ValueError(f"no [{SECTION}] table in the ini; is this an xlOil.ini?")
    nxt = _NEXT_TABLE_RE.search(text, m.end())
    return m.end(), nxt.start() if nxt else len(text)


def _parse_list(value: str) -> list[str]:
    """A TOML list literal as a Python list.

    Parameters
    ----------
    value : str
        For example ``'["a.b", "c.d"]'``; may span several lines.

    Returns
    -------
    list of str
    """
    return list(tomllib.loads(f"v = {value}")["v"])


def _render_list(items: list[str]) -> str:
    """A list of names as a one-line TOML list literal.

    Parameters
    ----------
    items : list of str

    Returns
    -------
    str
        For example ``'["a.b", "c.d"]'``.
    """
    return "[" + ", ".join(f'"{s}"' for s in items) + "]"


def load_modules(text: str) -> list[str]:
    """The modules xlOil is currently told to load.

    Parameters
    ----------
    text : str
        The whole ini.

    Returns
    -------
    list of str
        ``[xlOil_Python] LoadModules``; empty if the key is absent.
    """
    return list(tomllib.loads(text).get(SECTION, {}).get("LoadModules", []))


def tagged_modules(text: str) -> list[str]:
    """The modules a previous run of the installer added.

    Parameters
    ----------
    text : str
        The whole ini.

    Returns
    -------
    list of str
        The list recorded in the marker comment above ``LoadModules``; empty if the ini was
        never tagged.

    Raises
    ------
    ValueError
        If the ini has no ``[xlOil_Python]`` table.
    """
    start, end = _section(text)
    m = _MARKER_RE.search(text[start:end])
    return _parse_list(m["value"]) if m else []


def tag(text: str, modules: list[str]) -> str:
    """The ini with our managed set in ``LoadModules`` replaced by ``modules``.

    Parameters
    ----------
    text : str
        The whole ini.
    modules : list of str
        The modules the add-in should load; duplicates are dropped, order kept.

    Returns
    -------
    str
        The new ini text. Only the ``LoadModules`` line and the marker comment above it
        change; every other line, comments included, is preserved.

    Raises
    ------
    ValueError
        If the ini has no ``[xlOil_Python]`` table.
    tomllib.TOMLDecodeError
        If the result would not parse; nothing is returned in that case.

    Notes
    -----
    Idempotent. Modules a previous run tagged but ``modules`` no longer names are dropped,
    so a renamed module doesn't linger and fail to import. A module the user listed
    themselves is never claimed by the marker, so :func:`untag` leaves it alone.
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
    """The ini with our marker and the modules it records removed from ``LoadModules``.

    Parameters
    ----------
    text : str
        The whole ini.

    Returns
    -------
    str
        The new ini text; ``text`` unchanged if it was never tagged. Entries the user added
        by hand stay.

    Raises
    ------
    tomllib.TOMLDecodeError
        If the result would not parse.
    """
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
