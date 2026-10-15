# Engine op graph — DESIGN NOTE (not started)

> Companion to [pricing_architecture.md](pricing_architecture.md), [numba_engine.md](numba_engine.md),
> [engine_selection.md](engine_selection.md) and [risk_map_and_dual_ad.md](risk_map_and_dual_ad.md).
> This records a decision and a shape so the work can start cold later. Nothing here is implemented.
>
> **2026-10-13:** the implementation plan (representation, extended op vocabulary with reverse
> rules, row maps, the `risk()` entry, migration gates) is §4 of
> [risk_and_engine_design.md](risk_and_engine_design.md). "The math-upgrades doc" referred to
> below is not in the repository; its item 1 became the `RiskTemplate` (§2 there) and item 2 is
> this note.

## Why

Three findings from the October 2026 review, all with the same root cause:

1. **Three fused monoliths.** `kernels/compiler.reprice` (numpy), `engines/numba/program._execute`
   and `engines/jax/program` each re-implement the full control flow — fixed / float / observation
   grid / shaping / discount / reduce — and are held together only by parity tests.
   `kernels/shaping.py` is the one shared piece.
2. **Six risk front doors.** `Sensitivities`, `NumbaRisk`, `AutodiffRisk`, `RiskEngine`,
   `SwaptionProgram.risk`, `FuturesProgram.risk`. `RiskEngine` refuses futures.
3. **The swaption lives outside the IR.** `SwaptionProgram` compiles a unit-coupon swap, reads
   forward and annuity off leg PVs, and applies Black/Bachelier in Python. The option formula is
   not an op the engine knows; vol risk and gamma are bolted on per product.

Plus two concrete defects in the Numba adjoint that fall out of the same shape:

- Dense per-flow gradient rows `index_grad`/`funding_grad` of shape `(F, Q)`
  (`engines/numba/program.py`). 10k swaps × 40 flows × 30 pillars × 8 bytes × 2 ≈ 200 MB per
  call. Only `SwaptionProgram.risk` and `FuturesProgram.risk` consume the per-cashflow rows;
  everything else reduces to per-instrument immediately.
- The adjoint is fused into the primal behind a `calculate_adjoint` flag, so there is no
  reverse pass to reuse for a different scalar (a weighted book total, a par rate, an option
  value).

Decision taken (see the review thread): **not a tape, not JAX at runtime.** The graph is static
after compile, so a runtime tape records nothing we don't already know, and XLA's per-shape
compile and dispatch cost are the wrong trade for a low-latency book. Hand-written VJPs over a
small fixed op vocabulary — the Strata `PointSensitivities` shape — with JAX kept as the
test-time oracle it already is.

## Shape

### Ops

An **op** is one registered kernel with a primal and a reverse rule, per backend:

```
op = (kernel_id, Backend) -> (primal(inputs, residuals_out), vjp(cotangent, residuals, inputs_bar))
```

`engine_registry` already keys `(kernel_id, Backend)` → primal. The change is that the entry
becomes a pair. The vocabulary is small and stable:

| op | primal | consumes | produces |
|---|---|---|---|
| `df` | `exp(const + coef·z)` over the static query table | `z` | `DF[q]` |
| `forward` | `(DF[s]/DF[e] − 1)/τ` | `DF` | `rate_obs` or `rate_flow` |
| `shape_float` | `shaping.shape_float` | `rate_flow`, bounds | `rate_flow` |
| `prep_obs` / `shape_period` | `shaping.*` | `rate_obs` | `rate_obs` / `rate_flow` |
| `compounded` / `averaged` | segmented `reduceat` | `rate_obs`, `w`, `offsets` | `rate_flow` |
| `dcf` | `N · r · τ · DF · sign`, segment sum | `rate_flow`, `DF` | `flow_pv`, `leg_pv` |
| `black` / `bachelier` | closed form + greeks | `forward`, `annuity`, `vol`, `K`, `T` | `pv` |
| `reduce` | `bincount` / weighted sum | any per-item array | per-instrument / scalar |

A **program** is an ordered list of op instances over named columnar slots (the existing
`KernelInputs` columns plus the query table from `NumbaProgram`). The compiler emits it once.

### Reverse pass

One sweep: seed the cotangent of the requested scalar (book PV, one instrument, a par rate, an
option value), walk the op list backwards, each `vjp` scatters into its inputs' cotangent
buffers. The curve-parameter cotangent accumulates **by query id** — the sparse
`(query, param, coef)` rows the Numba program already holds — never as a dense `(F, Q)` block.
Per-instrument and per-cashflow views are reductions of the same buffer, materialised on
request.

Buffers are allocated once per program and zeroed per sweep. If a parallel sweep is ever
wanted: per-worker buffers merged in a fixed order, so results are bitwise reproducible.

### What this buys

- **One risk entry.** `risk(program, market, scalar=...)` → gradient; ladders, index/funding
  split, trade-by-quote and futures/swaption risk are all seeds + reductions of one sweep.
  `RiskEngine`, `NumbaRisk`, `SwaptionProgram.risk`, `FuturesProgram.risk` collapse into it;
  `Sensitivities` (bump) stays as the universal cross-check.
- **Swaption inside the IR.** `black`/`bachelier` become ops over `(forward, annuity, vol)`
  slots; vol risk is a cotangent on the vol slot, curve risk flows through `forward` and
  `annuity` exactly as today's chained adjoint does, and gamma stops being product-specific.
- **Backends stop being monoliths.** numpy / Numba / JAX implement the *same* op list; parity
  tests become per-op instead of per-engine. The numpy `reprice` becomes "run the op list with
  the numpy entries", and ceases to be a hand-maintained twin.
- **Items 1, 2 and 4 of the math-upgrades doc land here.** Item 1 (calibration adjoint: cached
  `lu_factor` of `Jᵀ`, no `pinv`/`inv`, condition number reported) is linear algebra on top and
  can be done first, independently. Item 2 *is* this note. Item 4 (quote-space curvature) is
  central differences of item-1 gradients at `q ± εv` — a few sweeps per factor direction —
  with forward-over-reverse deferred until a real use case needs hundreds of directions.

### Reporting edges

A cashflow engine (compile + gather + schedule metadata) is a *consumer* of the program, not a
pricer concern. The compiler already lowers a per-flow `rate_kind` from each period's
governing `CouponEvent`; the report can carry the per-period `CouponType`, spread, bounds and
the leg-level `CouponType` (`Custom` when a schedule was supplied) alongside `flow_pv` with no
new lowering — only capturing what the compiler computes. Keep it in `pricing/results.py`
territory, off the instrument.

## Order

1. Item 1 of the math doc (adjoint linear algebra, cached factorization, diagnostics) — small,
   independent, do first.
2. Op registry: `(kernel_id, Backend)` → `(primal, vjp)`; JAX parity test per op.
3. Split `_execute` into the op list; reverse pass with query-id accumulation; delete the dense
   `(F, Q)` rows.
4. Single `risk(...)` entry; retire the six doors.
5. Swaption as ops; vol slot; retire `SwaptionProgram.risk`.
6. numpy and JAX engines as runners of the same op list.

## Non-goals

- A runtime tape or operator-overloading AAD.
- JAX as a production path (oracle only; `docs/risk_map_and_dual_ad.md`).
- Changing `KernelInputs` columns or the instrument layer — the op graph consumes the current
  IR.
