"""Numba kernel backend — SEAM (no kernels implemented yet).

To add a numba implementation, register it under the SAME kernel id with ``Backend.Numba``
and the SAME ``KernelInputs``/``KernelResult`` contract as the numpy kernel, e.g.::

    from numba import njit
    from common.registry import register_with
    from finance.pricing.engines import engine_registry
    from finance.pricing.types import RATE_COMPOUNDED, Backend

    @register_with(engine_registry, (RATE_COMPOUNDED, Backend.Numba))
    def compounded_numba(obs_rate, obs_w, offsets): ...

Nothing in the pricer changes — selection is a one-line key swap at the call site. Because
the boundary is contiguous numpy arrays, the kernel body is a drop-in replacement.
"""
