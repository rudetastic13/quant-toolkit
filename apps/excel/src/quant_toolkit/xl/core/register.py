"""``xl_func``: the one place worksheet functions are registered with xlOil.

The Excel name comes from the Python function name, so ``@xl_func("Dates")`` on ``AddTerm``
registers ``QuantToolkit.Dates.AddTerm`` in the ``QuantToolkit.Dates`` category. Argument names
are shown to the user as written, hence CamelCase functions and arguments in the worksheet modules.

The docstring is the Excel help. Worksheet functions are documented in numpydoc like the rest of
the library, and ``xl_func`` splits that for the function wizard: the text before the first section
becomes the function description and each ``Parameters`` entry becomes that argument's help. The
wizard shows at most 255 characters of either, so both are cut to fit.
"""
from __future__ import annotations

import functools
import inspect
import re

import numpy as np
import xloil as xlo

PREFIX = "QuantToolkit"
WIZARD_LIMIT = 255  # Excel's function wizard shows no more than this per description

_SECTION = re.compile(r"^(\w[\w ]*)\n-{3,}$", re.MULTILINE)
_ENTRY = re.compile(r"^(\w+)\s*:.*$", re.MULTILINE)


def _fit(text: str) -> str:
    """Collapse whitespace and cut to the wizard limit, ending in an ellipsis if anything was dropped."""
    text = " ".join(text.split())
    return text if len(text) <= WIZARD_LIMIT else text[: WIZARD_LIMIT - 3].rstrip() + "..."


def split_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a numpydoc docstring into the wizard description and per-argument help.

    Parameters
    ----------
    doc : str or None
        The function's docstring.

    Returns
    -------
    summary : str
        Everything before the first section header, on one line, at most 255 characters.
    arguments : dict[str, str]
        Argument name to its ``Parameters`` description, each on one line, at most 255
        characters. Arguments the docstring does not describe are absent.
    """
    doc = inspect.cleandoc(doc or "")
    sections = list(_SECTION.finditer(doc))
    summary = _fit(doc[: sections[0].start()] if sections else doc)
    arguments: dict[str, str] = {}
    for i, section in enumerate(sections):
        if section.group(1) != "Parameters":
            continue
        body = doc[section.end() : sections[i + 1].start() if i + 1 < len(sections) else len(doc)]
        entries = list(_ENTRY.finditer(body))
        for j, entry in enumerate(entries):
            text = body[entry.end() : entries[j + 1].start() if j + 1 < len(entries) else len(body)]
            arguments[entry.group(1)] = _fit(text)
    return summary, arguments


def elementwise(fn):
    """Let a scalar function take ranges: broadcast the arguments and call it per element.

    Parameters
    ----------
    fn : callable
        A function of scalar cell values returning one cell value.

    Returns
    -------
    callable
        ``fn`` unchanged for scalar arguments. If any argument is a range (a 2-D array), the
        arguments are broadcast against each other and the result is an object array of that
        shape, which Excel spills.
    """

    @functools.wraps(fn)
    def wrapper(*args):
        """Call ``fn`` once for scalars, or once per broadcast element for ranges."""
        if not any(isinstance(a, np.ndarray) for a in args):
            return fn(*args)
        grid = np.broadcast(*args)
        out = np.empty(grid.shape, dtype=object)
        out.flat = [fn(*(v.item() if isinstance(v, np.generic) else v for v in vals)) for vals in grid]
        return out

    return wrapper


def xl_func(group: str, *, broadcast: bool = False):
    """Register the decorated function as ``QuantToolkit.<group>.<function name>``.

    Parameters
    ----------
    group : str
        The part after ``QuantToolkit.``: ``"Dates"``, ``"IRProducts"``. Also the category the
        function is listed under in Excel's function wizard.
    broadcast : bool
        Wrap a scalar function with :func:`elementwise`, so ranges are broadcast against each
        other and the result spills.

    Returns
    -------
    callable
        A decorator. The function it returns is still callable from Python with the same
        arguments, which is how the worksheet layer is exercised outside Excel.

    Notes
    -----
    The wizard description and the per-argument help come from the docstring through
    :func:`split_docstring`. A parameter with a default is optional in Excel; an omitted
    argument and an empty cell both arrive as ``None``.
    """
    category = f"{PREFIX}.{group}"

    def decorate(fn):
        """Register ``fn`` with xlOil under the group's name and return the registered function."""
        summary, arguments = split_docstring(fn.__doc__)
        target = elementwise(fn) if broadcast else fn
        return xlo.func(target, name=f"{category}.{fn.__name__}", group=category, help=summary, args=arguments)

    return decorate
