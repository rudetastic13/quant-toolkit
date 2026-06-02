"""Rust (PyO3/maturin) kernel backend — SEAM (no kernels implemented yet).

Same contract as the numpy/numba backends: register the compiled entry point under the same
kernel id with ``Backend.Rust``, consuming the contiguous arrays of ``KernelInputs`` and
returning a ``KernelResult``. The columnar struct-of-arrays form is exactly what a Rust
kernel wants (zero-copy slices), so no marshalling layer is required beyond the numpy buffer
protocol.
"""
