# Swap performance study

A timing study of the **compile-once / reprice-many** design (see
[`docs/pricing_architecture.md`](../../docs/pricing_architecture.md) §3–4) over a
1,000-swap population. It demonstrates the central claim: pricing a population is *one*
vectorized array pass, and a full key-rate ladder is just `2·P` of those passes over the
same compiled program — never a per-instrument or per-scenario rebuild.

## Run

```bash
PYTHONPATH=src python research/swap_performance/benchmark.py
```

## What it does

1. **Calibrate** a SOFR curve from a money-market + swap quote strip (1M…30Y).
2. **Generate** 1,000 randomized receive/pay-fixed vanilla SOFR swaps (seeded), tenors 1Y–30Y.
3. **Time** the three hot operations:
   - `compile` — `ResolvedSwap`s → one columnar `PricingProgram` (paid once).
   - `price` — a single `reprice(market)` over the whole population.
   - `KRD` — `Sensitivities.key_rate_durations`: bump-and-reprice across every pillar.

## Sample run

`as_of=2026-06-01, n=1000, seed=20260601`, Python 3.14, single-threaded numpy:

```
population: 1000 swaps  ->  29,838 flows (accrual periods) compiled

compile (once)                  308.06 ms      308.1 us/swap
price (1 reprice)               230.73 ms      230.7 us/swap
reprice (warm)                  169.85 ms      169.9 us/swap
KRD ladder (2*P reprices)      3341.99 ms     3342.0 us/swap

KRD did 22 repricings (11 pillars x 2) over the same program => 151.91 ms/reprice
```

## Reading it

- **~30k flows** (accrual periods) across the population compile into one struct-of-arrays;
  every subsequent reprice touches those same arrays.
- A **warm reprice is ~0.17 ms/swap** for the full population — the per-swap cost is amortised
  across vectorized kernel calls grouped by `rate_kind`, so it doesn't scale with a Python
  loop over instruments.
- The **KRD ladder scales linearly in repricings**: 11 pillars × 2 (central difference) = 22
  repricings, each ≈ one warm reprice. No rebuild per pillar — that's the inversion of the
  legacy rebuild-per-pillar KRD the architecture replaces.

## Optimization history

Both the compile and reprice hot paths were dominated by **sorting data that was already
sorted / bounded-integer dates** — replaced with O(M + span) bucket factorization (boolean
mask + `flatnonzero`), no JIT required. PVs are unchanged to the cent.

| stage | before | after | speedup |
|---|---|---|---|
| compile | ~700 ms | ~308 ms | ~2.3× |
| reprice (warm) | ~491 ms | ~170 ms | ~2.9× |
| KRD ladder | ~10.8 s | ~3.3 s | ~3.3× |

- **reprice** — `_project_dedup` deduped the fixing-date curve lookup via `np.unique`
  (argsort over ~millions of dates, 54% of reprice). Now a bounded-integer bucket map.
- **compile** — `build_observation_grid` built the daily fixing calendar via `union1d`
  (sort-based union of already-sorted business-day arrays, 42% of compile). Now a day-offset
  boolean mask read back sorted.

These are wall-clock numbers from the pure-numpy backend; they're the baseline a future
numba/rust kernel (same `KernelInputs` contract) would be measured against. Remaining
levers: JIT the per-leg date arithmetic (`generate_schedule`, `add_business_days`) and
vectorize the per-flow `_lower_event` lowering.
