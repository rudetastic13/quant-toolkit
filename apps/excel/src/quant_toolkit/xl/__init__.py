"""quant_toolkit.xl — the Excel add-in (xlOil): worksheet functions over quant_toolkit.

Each module is a set of ``@xl_func`` functions that call the library directly, registered in
Excel as ``QuantToolkit.<Group>.<Name>`` with CamelCase arguments:

- ``dates``      — ``QuantToolkit.Dates.*``: business-day and term arithmetic, IMM dates, schedules.
- ``products``   — ``QuantToolkit.IRProducts.*``: instruments, returned as handles.
- ``solver``     — ``QuantToolkit.Solver.*``: calibration quotes and ``Calibrate``.
- ``curves``     — ``QuantToolkit.Curves.*``: discount factors and rates off a calibration.
- ``pricing``    — ``QuantToolkit.Pricing.*``: a compiled portfolio and the measures requested of it.
- ``risk``       — ``QuantToolkit.Risk.*``: risk templates and the risk report.
- ``utilities``  — ``QuantToolkit.Utilities.*``: lists and dicts from ranges, date tags.

``core`` holds what they share: the ``xl_func`` decorator, the handle store behind the handle
strings (``Swap;;;8f2a1c``), cell-value parsing and the compiled portfolio. The modules
import ``xloil`` (Windows-only), so nothing is imported here: xlOil loads each module by name
from ``LoadModules`` in ``xlOil.ini``.

``MODULES`` is the list of modules the add-in loads. Adding a module there is what promotes it:
``quant_toolkit_xl_installer install`` writes that list into ``xlOil.ini``. The last entry,
``core.ready``, registers nothing: it marks the session as loaded for the ``quant_toolkit_xl`` launcher.

The registered names and argument orders are a contract: saved workbooks bind to them.
"""

# The modules xlOil loads, in load order. The installer reads this; nothing else needs editing to add one.
# "core.ready" stays last: it tells the launcher that everything before it is registered.
MODULES = ("dates", "products", "utilities", "solver", "curves", "pricing", "risk", "core.ready")


def module_names() -> list[str]:
    """The fully qualified module names xlOil loads.

    Returns
    -------
    list of str
        ``quant_toolkit.xl.dates`` and so on, one per entry of ``MODULES``, in that order.
    """
    return [f"{__name__}.{name}" for name in MODULES]
