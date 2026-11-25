"""Helpers shared by the worksheet modules; nothing here registers a worksheet function.

- ``register`` — ``xl_func``, the decorator every worksheet function is registered through.
- ``handles``  — the store behind the handle strings (``Swap;;;8f2a1c``).
- ``convert``  — cell value to library type (enums, terms, dates).
- ``book``     — the compiled portfolio behind a ``Portfolio`` handle and its measures.
"""
