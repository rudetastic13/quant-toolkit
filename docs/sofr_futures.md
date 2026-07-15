# SOFR futures — instrument layer, contract months, and convexity-adjusted calibration

> Companion to [pricing_architecture.md](pricing_architecture.md) (the contract/market/engine
> layering and the calibration layer), [engine_selection.md](engine_selection.md) (futures
> price through the linear-rates engine), and [numba_engine.md](numba_engine.md).

> **Status: design spec — not yet implemented.** None of the symbols specified below
> (`SofrFuture`, `ResolvedSofrFuture`, `FuturesHelper`, `QuoteKind.FuturesRate`, `Roll.IMM`,
> `imm_date` / `next_quarterly_imm`) exist in `src/` yet; the work items are G1–G6 (§6).
> The rate-kind mapping it builds on (SR3 → `GeometricAveraged`, SR1 → `ArithmeticAveraged`)
> is current.

SOFR futures are the liquid front-to-mid of the USD curve (out to ~2–3y), sitting between
deposits (very short) and swaps (2y+). This document specifies how they enter the **instrument
layer**, how **contract months** resolve to reference periods, and how a futures **calibration
helper** optionally carries a **convexity adjustment** that is absorbed into the calibrated
curve.

The pleasant part: the two CME contracts map *exactly* onto rate kinds the kernel already has.

| Contract | Months | Reference period | Settlement rate | Maps to |
|---|---|---|---|---|
| **SR3** (3-Month SOFR) | Quarterly IMM (H, M, U, Z) | 3rd-Wed → 3rd-Wed (~3m) | daily-**compounded** SOFR, Act/360, annualized | `RateKind.Compounded` |
| **SR1** (1-Month SOFR) | Serial — all 12 months | 1st → last calendar day | arithmetic **average** of daily SOFR | `RateKind.Averaged` |

Both quote as **price = 100 − rate**. So a futures "rate" is `(100 − price)/100`, over its
reference period — i.e. the *same compounded/averaged measure* the OIS obs-grid kernel already
computes. No new rate math; only a new contract wrapper, contract-month dating, and the
convexity step at calibration.

---

## 1. Instrument layer

A SOFR future is a pure-data contract in the resolution layer, alongside
`Deposit`/`Fra`/`Swap` — no curve, no pricing methods.

```python
@dataclass
class ResolvedSofrFuture:
    """One SOFR futures contract — pure data (no curve)."""
    contract: str          # e.g. "SR3", "SR1"
    ref_start: Date        # reference-period start (IMM 3rd-Wed for SR3; month 1st for SR1)
    ref_end: Date          # reference-period end   (next IMM for SR3; month-end for SR1)
    rate_kind: RateKind    # Compounded (SR3) | Averaged (SR1)
    day_count_method: DayCountMethod   # Actual360
    currency: str = "USD"
    index_name: str = "SOFR"
    funding_id: str = "STDCSA"
    point_value: float = 25.0          # $/bp per contract (SR3 = $25; SR1 per its 1m spec)
```

The trader-facing **builder** takes a contract code + month and resolves the reference period
from the contract-month conventions (§2):

```python
def SofrFuture(*, contract: str, month: str | tuple[int, int], as_of: Date,
               currency="USD", rate_index="SOFR", funding_id="STDCSA") -> ResolvedSofrFuture:
    ...   # "SR3", "Z26"  or  ("SR3", (2026, 12))  -> Dec-2026 IMM reference period
```

Lowering to `KernelInputs` reuses the **observation grid** unchanged: the reference period is a
single coupon period whose daily fixings are built by
`build_payment_schedule(..., build_observations=True)`, with `coupon_type` →
`GeometricAveraged` (SR3) or `ArithmeticAveraged` (SR1). The compiler already emits the obs
columns; the rate kernels already compound/average them. So a futures contract is, to the
engine, one ragged compounded/averaged coupon.

---

## 2. Contract months

Contract-month handling is the only genuinely new dating logic.

### 2.1 Month codes

Standard futures month letters: `F G H J K M N Q U V X Z` (Jan…Dec). Quarterly IMM uses
`H M U Z` (Mar/Jun/Sep/Dec). A contract label is `<code><yy>` (e.g. `Z26` = Dec 2026) or an
explicit `(year, month)`.

### 2.2 IMM dates (SR3)

The reference period of an SR3 runs **IMM-to-IMM**: from the 3rd Wednesday of the contract
month to the 3rd Wednesday of the next quarterly month. Needed date utility (new):

```python
def imm_date(year: int, month: int) -> Date:      # 3rd Wednesday of (year, month)
def next_quarterly_imm(d: Date) -> Date:          # roll to the following H/M/U/Z IMM
```

`ref_start = imm_date(Y, M)`, `ref_end = next_quarterly_imm(ref_start)`. Consecutive SR3
periods are **contiguous** (Mar→Jun, Jun→Sep, …), which is what lets a futures strip bootstrap
pillar-by-pillar (§4) exactly like an FRA strip. Reuse / extend the existing `Roll` machinery
(an `Roll.IMM` variant) rather than a parallel calendar.

### 2.3 Serial months (SR1)

SR1 references the **calendar month**: `ref_start = first_business_day(month)`,
`ref_end = last_business_day(month)`, average over the daily fixings within. SR1 periods are
monthly and can overlap an SR3 quarter — see §4.3 on strip composition.

---

## 3. Pricing (mark-to-market)

A futures contract prices through the **linear-rates engine** (`Numba`/`Numpy` per
[engine_selection.md](engine_selection.md)). The model futures rate is the compounded (SR3) or
averaged (SR1) SOFR over the reference period, read off the curve — *plus the convexity
adjustment* (§5), since the curve carries *forwards*:

```
rate_fut(curve) = period_rate(curve, ref_start, ref_end)  +  CA
price_model     = 100 · (1 − rate_fut(curve))
P&L per contract = (price_model − entry_price) / 0.01  ·  point_value
```

`period_rate` is exactly the obs-grid compounded/averaged rate (it telescopes to
`DF(ref_start)/DF(ref_end)` for the spread-free SR3 case). DV01 per contract is the
`point_value` ($25/bp for SR3). Risk flows through the same analytic first-order path as any
linear-rates flow (see [numba_engine.md §3](numba_engine.md#3-feature-f2--analytic-first-order-sensitivities-linear-rates)).

---

## 4. Calibration helper

A `FuturesHelper` implements the existing `CalibrationInstrument` protocol — one quote, one
pillar, an `implied(market)` measure — so it drops into `CurveCalibrator` unchanged.

### 4.1 Measure and residual

```python
@dataclass
class FuturesHelper:
    future: ResolvedSofrFuture
    quote: Quote                      # QuoteKind.FuturesRate, value = (100 - price)/100
    curve: str
    convexity: float | ConvexityModel = 0.0   # optional; see §5

    @property
    def pillar_date(self) -> np.datetime64:    # the node this contract anchors
        return np.datetime64(self.future.ref_end.to_numpy(), "D")   # next IMM / month-end

    def implied(self, market) -> float:
        # compounded (SR3) / averaged (SR1) SOFR over [ref_start, ref_end] off the curve
        return period_rate(market, self.curve, self.future)

    def residual(self, market) -> float:
        ca = self._convexity(market)           # float, or model(market) -> float
        # curve forward must equal the convexity-ADJUSTED futures rate
        return self.implied(market) - (self.quote.value - ca)
```

The residual matches the **curve forward** to `futures_rate − CA`. Equivalently the helper
targets the *forward* implied by the futures, not the futures rate itself — see §5. The quote
units are rate (consistent with every other helper → a well-conditioned Jacobian).

A new `QuoteKind.FuturesRate` makes the measure explicit; the convexity-adjusted forward could
also be expressed as a `SimpleRate`, but a distinct kind documents the contract.

### 4.2 Pillar placement and bootstrap

`pillar_date = ref_end` (the next IMM date / month-end), so each future pins the curve node at
the end of its reference window — given the start node from earlier instruments. Because SR3
periods are contiguous, an SR3 strip bootstraps sequentially under `Bootstrapper` (each future
solves one new IMM node) just like the FRA strip; under `GlobalSolver` it works for any
interpolation. The IMM dates become genuine curve pillars.

### 4.3 Strip composition

- **Short end**: a deposit (or the first stub to the first IMM) anchors the origin → first IMM.
- **Front-to-mid**: an SR3 strip across the contiguous IMM quarters (and/or SR1 for the front
  months).
- **Mid-to-long**: par swaps take over (2y+).
- **Overlap rule**: SR1 (monthly) and SR3 (quarterly) windows can overlap and over-determine the
  front. Convention: use SR1 for the front months and SR3 beyond, or hand both to
  `GlobalSolver` as a least-squares fit. The calibrator already rejects overlapping pillars for
  the sequential `Bootstrapper`; document the mixed-strip case as global-only.

---

## 5. Convexity adjustment (the part that lands in the curve)

Futures are **daily margined** (mark-to-market), so the futures rate exceeds the forward rate
implied by a (non-margined) curve. The relationship, in rate space:

```
forward_rate  =  futures_rate  −  ConvexityAdjustment        (CA ≥ 0)
```

**Why it "lands in the curve":** the calibrator builds the curve from *forwards*. By targeting
`futures_rate − CA` (§4.1), the calibrated curve's forward over each IMM window sits `CA` below
the quoted futures rate. So **all downstream pricing off the curve uses true forwards**, and the
futures quote was merely a convexity-corrected input. The adjustment is consumed at calibration
time and never re-applied later.

### 5.1 How CA is supplied (optional, defaults to 0)

The helper's `convexity` is either:

- a **float** — an exogenous adjustment (vendor value, or a desk number) per contract; or
- a **`ConvexityModel`** — a callable `model(market) -> float` that computes CA from a vol model
  at calibration time.

Default `0.0` means "treat the futures rate as the forward" — fine for the very front contracts
where CA is sub-bp, wrong by the 2y point where it reaches a couple of bp.

### 5.2 Model seam

`ConvexityModel` is a designed seam, not a hardcoded formula. Two standard choices:

- **Ho-Lee (leading order)**: for a forward over `[T₁, T₂]`, `CA ≈ ½ · σ² · T₁ · T₂` — a quick,
  closed-form first cut driven by a single normal vol `σ`.
- **Hull-White (one-factor)**: the production form, with mean reversion `a` and vol `σ`; the
  futures-vs-forward adjustment has a known closed form over the reference period. This is the
  recommended default once a vol surface exists (it shares the `VolNamespace` seam from the
  pricing architecture).

The model receives the `market` (for the curve and, when present, the vol surface), so a
self-consistent calibration can iterate curve ↔ CA if desired; the first cut uses an exogenous σ
and a single pass (CA depends weakly on the curve).

---

## 6. Work items

Independent features, in the style of [numba_engine.md](numba_engine.md#8-sequencing):

| # | Feature | Depends on | Notes |
|---|---|---|---|
| **G1** | `imm_date` / `next_quarterly_imm` + serial-month helpers; `Roll.IMM` | — | the only new dating logic (§2) |
| **G2** | `ResolvedSofrFuture` + `SofrFuture(...)` builder + compiler lowering | G1 | reuses the obs grid / `RateKind` (§1) |
| **G3** | `FuturesHelper` + `QuoteKind.FuturesRate` into `CurveCalibrator` | G2 | residual = `implied − (futures − CA)` (§4) |
| **G4** | `convexity` as float (exogenous), default 0 | G3 | the optional-adjustment hook (§5.1) |
| **G5** | `ConvexityModel` seam (Ho-Lee first cut → Hull-White) | G4 + vol seam | model-derived CA (§5.2) |
| **G6** | Futures P&L / DV01 (mark-to-market) through the linear engine | G2 | `point_value`-scaled (§3) |

**G1 → G2 → G3 → G4** is the critical path to calibrating a curve from a SOFR futures strip
with an optional exogenous convexity number. **G5** (model CA) and **G6** (futures P&L) layer on
top independently.
