# YieldCurveGroup (YCG) — plan

Status: Y0 and Y1 built on `feature/carlos/yield_curve_group` (Oct 2026); Y2+ design only. Revised after a design
review and again as Y1 landed.

## 1. Why

`MarketContext` registers every curve under a flat string key and does too much: container, market
data, key validation, ownership analysis, scenarios. Three things it handles badly:

1. **Roles are implicit.** `market["USD.SOFR.1D"] = sofr` and `market["USD.STDCSA"] = sofr` is how a
   curve "projects and discounts"; nothing says so, and nothing stops a curve landing in the wrong slot.
2. **One discounter is assumed.** USD already needs several: LCH (`STDCSA`), CME, Fed Funds, UST.
3. **No multi-currency.** A GBP trade collateralised in USD has no place to hang its dependency on
   the USD curves or on FX.

A `YieldCurveGroup` is one currency's curves and their roles, plus the links to other currencies'
groups that its collateralised discounters are implied off. `MarketContext` becomes `as_of` +
fixings + vols + conventions + `{ccy: YCG}`, and keeps the `market["KEY"]` facade.

## 2. Objects

```
MarketContext      as_of, fixings, vols, conventions, {ccy: YieldCurveGroup}; market-wide ownership
YieldCurveGroup    one currency: YieldCurve views by name + fx_rate + collateral links
YieldCurve         unchanged: origin + key + conventions + fixings over a DiscountCurve
ForYc              link: (group, funding name) — the foreign discounter an implied curve hangs off
FxQuote            convention per currency pair: quote style (American/European), spot lag, joint calendars
FxQuoteStyle       enum: American (base per 1 local) | European (local per 1 base)
```

`YieldCurve` is unchanged. A curve's **projection** identity is its own key (`sofr.key ==
USD.SOFR.1D`); its **discount** names are given at registration. The group stores one `YieldCurve`
view per name over a shared `zero_curve` (`YieldCurve.with_key`), which is what `MarketContext.yield_curve`
derives and caches today; a discount-name view carries no conventions and no fixings, so an alias
never behaves like an index.

## 3. YieldCurveGroup

```python
class YieldCurveGroup:
    currency: str                 # constructor argument; every key registered must be in it
    fx_rate: float = 1.0          # spot, as quoted in `fx_quote` style; stored normalised (base per 1 local)
    fx_quote: FxQuoteStyle | None # American (base per 1 local) or European (local per 1 base); None -> pair convention
    default_funding: str | None   # funding name used when a trade names none

    def add(self, yield_curve, *, funding_names=(), collateral: ForYc | None = None) -> None: ...
    def yield_curve(self, name) -> YieldCurve      # by full key, either kind
    def projection(self, index, tenor=...) -> YieldCurve
    def discount(self, funding=None) -> YieldCurve  # None -> default_funding
    def role(self, zero_curve) -> projects / discounts / both   # derived from the maps
    def curves(self) -> tuple[DiscountCurve, ...]   # distinct zero_curves, each once
    def links(self) -> dict[str, ForYc]             # funding name -> the foreign discounter it is implied off
```

### `add`

```python
usd = YieldCurveGroup()
usd.add(sofr,    funding_names=["USD.STDCSA", "USD.USDSOFR"])   # key USD.SOFR.1D -> projects; two discount names
usd.add(fedfund, funding_names=["USD.FEDFUND"])                 # projects + discounts
usd.add(cme)                                                    # key USD.CMECSA is a discount key -> discounts only
usd.add(ftp)                                                    # key USD.XXX-FTP.1D -> projects only
```

Role is derived, never declared:

- the curve's key is a projection key → it projects, under that key;
- `funding_names` non-empty, or the curve's key is a discount key → it discounts, under those names
  (a discount-keyed curve with no `funding_names` registers under its own key);
- both → both. The `zero_curve` is stored once; each name gets its own view.

One map of views over shared `zero_curve`s, `{name: YieldCurve}`; the key grammar of the name says
which role the view plays, so `projection("SOFR")` and `discount("STDCSA")` are key constructions
followed by a lookup, not separate stores (as built: one dict, not two).

`add` is the registration path; `market["KEY"] = curve` **stays as one-line sugar** for `add` under
that single name (as built: 41 test sites and the Excel API assemble markets as `{name: curve}`
dicts, and the role comes from the key grammar either way, so removing it was churn without a
semantic gain). What `add` uniquely gives is saying "projects *and* discounts" in one call. A second
view over an already-registered object under a new name is fine — it is also how `with_curve`
aliases — so nothing needs merging. `market["KEY"]` for *lookup* stays permanently (decided): it
parses the key, picks the group by currency and returns the view, so engines and `RateGenerator`
are untouched.

### Validation at `add`

Objects carry no currency (a `ZeroCurve` or composite has none, and a collateralised curve legitimately
contains a foreign atom), so validation is on the **names**:

- every key (the curve's and each funding name) is in the group's currency (the group's constructor
  argument; the market creates a group on the first key in a new currency);
- a projection key's `IndexFamily` is registered and publishes the tenor (what `YieldCurve` already checks);
- a funding name is `CCY.NAME`, one slot after the currency, no tenor;
- no duplicate projection key or funding name;
- a `YieldCurve`'s origin equals the market's `as_of` (as `_coerce` checks today);
- `collateral` given → the linked group and funding name exist and the link is acyclic;
- `fx_rate > 0`. The root is **named by the market** (`MarketContext(base_currency="USD")`), not
  inferred from the number: `fx_rate` defaults to 1.0 on every group, so counting 1.0s cannot find it.
  The market checks the base group's `fx_rate` is 1.0. A non-base group left at the default is not
  flagged (decided): it reports local PV unconverted, which is the caller's choice. `fx_rate` lives on
  the group only; `YieldCurve` carries no FX.

### Ownership and bumps: market-wide, not per group

`leaf_owner` / `owned` / `bump` / `with_params` stay on `MarketContext` and run over every group's
curves at once. They cannot be scoped to a group: a collateralised GBP discounter (4a) and a non-G10
discounter (4b) are composites containing a USD atom, and `engines/layout.py` already lays blocks out
across every curve a program names. Groups are a namespace over the same identity-based machinery;
bumping USD SOFR moves every curve built on it in every group, by object identity, exactly as today.

The group follows the market's assemble-then-freeze pattern: `add` while assembling, frozen copies
from `with_curve` / `bump` / `with_params` (which rebuild the affected views, replacing today's
`_yield_cache` bookkeeping).

## 4. FX

Two separate things, previously conflated under one `for_yc`:

**`fx_rate` — translation, one per group.** `YieldCurveGroup(fx_rate=1.27)`: the value of one unit of
the group's currency in the market's base currency (GBP group, USD base: USD per GBP). Defaults to
1.0, which is right for the base group only.

**`fx_quote` — quote style.** `YieldCurveGroup(fx_rate=150.0, fx_quote=FxQuoteStyle.European)` for JPY.
Defined relative to the market's base currency: *American* = base per one unit of local (GBPUSD 1.27),
*European* = local per one unit of base (USDJPY 150). The group normalises to American on
construction (`European` → `1 / fx_rate`) and everything downstream reads one direction. `None` takes
the pair's market convention from the `FxQuote` registry (EUR, GBP, AUD, NZD American against USD;
most others European), so the common case needs no argument and an explicit style wins when a desk
quotes the other way. No silent default to American: a wrong direction inverts a PV translation. Used only to translate a PV between currencies
(`PV_root = fx_rate * PV_local`; FX delta is `PV_local`). It is **not** in the engine parameter vector
and is never differentiated through: the implied discounters are calibrated to quantities in which the
spot cancels (4a), so the curves are spot-free. `FxQuote` normalises the market quote (USDJPY is JPY
per USD) into the group's direction on ingestion; no engine code sees a raw quote.

**`ForYc` — collateral links, zero or more per group.** Each collateralised discounter the group holds
hangs off one foreign discounter, named at registration:

```python
gbp = YieldCurveGroup()
gbp.add(sonia, funding_names=["GBP.STDCSA"])                                              # local collateral
gbp.add(gbp_usd_disc, funding_names=["GBP.USDSTDCSA"], collateral=ForYc(usd, "USD.STDCSA"))  # USD collateral
gbp.add(gbp_eur_disc, funding_names=["GBP.EURSTDCSA"], collateral=ForYc(eur, "EUR.STDCSA"))  # EUR collateral
```

A GBP book can face GBP, USD and EUR CSAs, so one link per collateral currency is normal and bounded.
The link is metadata the group needs for three things: forward FX derivation (which two curves form
`F/S`), validation, and documentation of the dependency; bump propagation does not need it, identity
does that. A `local_funding_id` is not part of the link: in the standard construction the implied
curve hangs off the foreign discounter plus FX points / MTM basis, and the calibrating xccy swap is
itself foreign-collateralised, so the local discounter does not enter.

**Derived, never stored.** `spot_date = FxQuote.spot_date(as_of)` (spot lag, joint calendars of both
currencies plus USD; T+2, T+1 for USD/CAD). Forward FX for a link: `F(T) = S * DF_local_coll(T) /
DF_foreign(T)`, with `DF_local_coll` the linked curve, `DF_foreign` the `ForYc` target, both re-based
to the spot date. Calendars are stand-ins (`no_holidays`) today; FX spot dates are only right around
real holidays once real calendars exist.

**No SuperYieldCurveGroup.** The combination (projection index × collateral currency × CSA flavour)
is selected per trade by `rate_index` + `funding_id`; the facade resolves `GBP.USDSTDCSA` to whatever
the group holds. A layer that lets the same funding name mean different curves for different books is a
market-per-book problem, not a curve one.

## 4a. Calibrating a collateralised discounter

`gbp_usd_disc` is built from a mixed strip, like SOFR from deposits, futures and swaps:

- **FX swap points** (to about 1Y): forward minus spot; the spot cancels in `F/S`.
- **MTM cross-currency basis swaps** (1Y and out): spread on the non-USD float leg (SONIA + basis vs
  SOFR flat), USD notional resetting to spot each period, so no spot exposure beyond the initial
  exchange (standard formulation; re-derive for the exact conventions before relying on it).

Outright forwards or constant-notional basis swaps would bring the spot level into the curve; do not
calibrate to them. Needed: an FX-swap helper and an xccy-basis helper with their own `QuoteKind` (the
seam exists in `calibration/instruments.py`); a cross-currency basis swap convention per pair (spread
leg, payment/reset frequencies, reset lag, calendars) next to `FxQuote`; an MTM-swap pricer whose
per-period notionals use forward FX from the two curves.

## 4b. Currencies with no local rates market (non-G10)

No IRS market; the derivatives market is FX forwards/swaps/NDFs only. The group has **one discounter,
one link, and no projection curves**:

```python
brl = YieldCurveGroup()
brl.add(brl_disc, funding_names=["BRL.USDSTDCSA"], collateral=ForYc(usd, "USD.STDCSA"))
```

`brl_disc = usd_stdcsa + fx_implied_spread`: a composite on the shared USD object, the spread solved so
model forwards match the full strip of point quotes. Consequences:

- the projection map is empty; a trade resetting off a local index raises a clear error;
- no `IndexFamily` is registered for the currency;
- `default_funding` is the single discounter automatically;
- bumping USD SOFR moves the BRL discounter; the implied-spread block is owned by the BRL key
  (market-wide `leaf_owner`, section 3).

Points are quoted in absolute price units, so `F/S = 1 + points/S`: the calibration input uses the spot
level, and calibration and valuation must use the same `S` (the group's `fx_rate`); a mismatch is a bug.

## 5. Worked example: the USD set including FTP

```python
usd = YieldCurveGroup()

sofr    = YieldCurve(..., key="USD.SOFR.1D")       # base
fedfund = sofr + ff_spread                         # composite on sofr, key USD.FEDFUND.1D
cme     = sofr + lch_cme_basis                     # key USD.CMECSA
ust     = sofr + ust_asw_spread                    # UST over SOFR (asset-swap spreads), key USD.UST-OTR.1D
ftp     = ust  + ftp_credit_spread                 # bank debt over UST (ASW spreads for now), key USD.XXX-FTP.1D

usd.add(sofr,    funding_names=["USD.STDCSA"])     # Both, LCH default CSA
usd.add(fedfund, funding_names=["USD.FEDFUND"])    # Both
usd.add(cme)                                       # Discount only
usd.add(ust,     funding_names=["USD.UST-OTR"])    # Both
usd.add(ftp)                                       # Projection only
```

| Curve | Built as | Role | Projection name | Discount name |
|---|---|---|---|---|
| SOFR | base | Both | `USD.SOFR.1D` | `USD.STDCSA` |
| Fed Funds | `sofr + spread` | Both | `USD.FEDFUND.1D` | `USD.FEDFUND` |
| CME basis | `sofr + basis` | Discount | | `USD.CMECSA` |
| UST | `sofr + ASW spread` | Both | `USD.UST-OTR.1D` | `USD.UST-OTR` |
| FTP | `ust + credit spread` | Projection | `USD.XXX-FTP.1D` | |

All spreads are composites over shared objects (`CompositeCurve`, nesting allowed), so:

- bump SOFR → moves SOFR, Fed Funds, CME, UST, FTP;
- bump the UST ASW spread → moves UST and FTP, not SOFR;
- bump the FTP credit spread → moves FTP only;
- `ln DF_ftp = ln DF_sofr + ln DF_ust_spread + ln DF_ftp_spread`: spreads are exactly additive in
  continuous space.

Conventions registrations needed (`definitions.py`): `IndexFamily` for `UST-OTR` and `XXX-FTP`, both
`tenors=(OVERNIGHT,)`. `1D` and `ON` are the same tenor in the key code (`USD.UST-OTR.1D` prints as
`USD.UST-OTR.1D`), which is accepted. `XXX-FTP` is a placeholder name for the funds-transfer-pricing index.

**UST-OTR is `Both`, Act/360 (decided).** `spread` converts each full rate through the *curve*
argument's index day count, so `rg.spread("USD.UST-OTR.1D", over="USD.SOFR.1D", ...)` — the SOFR/GV
spread window by window — needs UST to have a projection identity. An overnight UST rate has no market
definition; Act/360 is a choice that puts the spread in SOFR's basis so SOFR → UST → FTP chains
additively. Nothing resets off it.

**The UST curve is the ASW problem inverted.** The forward direction (Bloomberg) takes a bond, builds a
par swap on its dates and reads `ASW = bond par coupon − swap par rate`. Here the quoted ASW per tenor
is the *input*: `ust_asw_spread` is calibrated so a par swap projecting off `USD.UST-OTR.1D` has par
rate `SOFR par + ASW_T`, and the market-implied UST par yield is read *back* as `par_swap_rate` on a
fixed-vs-`UST-OTR.1D` swap. The bond never enters (no cash flows, no Act/Act ICMA, no accrued), which
is why Y3 does not wait on the bond work; when bonds land, the forward direction becomes a test oracle
for the curve, not its source. The quote sits on the float leg, Act/360, matching `UST-OTR`'s basis.

Reading the built curve needs no helper: `spread("USD.UST-OTR.1D", over="USD.SOFR.1D")` is the
SOFR/GV spread per window, `simple_rate("USD.UST-OTR.1D")` the forward rate (`par_swap_rate` if a par
yield is wanted). Building `ust_asw_spread` from quoted ASW levels reuses the existing `SwapHelper`
quoted on the float leg's `spread` field (`QuoteKind.FloatSpread` next to `ParRate`; the swap builder
already carries `spread`), no new helper type.

## 6. RateGenerator target API

```python
rg = RateGenerator(market)
rg.simple_rate("USD.XXX-FTP.1D", starts, ends)         # exists today; unchanged, the facade resolves it
rg.spread("USD.XXX-FTP.1D", over="USD.UST-OTR.1D", starts=starts, ends=ends, convention=...)
```

`spread` (new): work in continuous space, convert at the end.

1. Continuous forward rate of each curve over the window (Act/365 clock, as `continuous_forward_rate`).
   `s_c = r_c(curve) - r_c(over)` is exact because log-DFs add.
2. To report in a convention (simple on a day count), convert **each full rate** and subtract:
   `r_conv(x) = (exp(x * dt_cont) - 1) / tau`, `spread_conv = r_conv(r_c_curve) - r_conv(r_c_over)`.
   Converting `s_c` alone drops the `r_over * tau` cross term and is wrong.
3. Both keys must be in one currency (a cross-currency spread means nothing without FX).
4. Past windows need fixings on both curves, so `over` must be a projection key for realised windows; a
   discount-name `over` is allowed for windows at or after the origin only.

The market-quoted FTP spread is **not** this quantity. It is a spread of bank-issued debt over UST
yields (a bond/par-style quantity). For now the FTP and UST legs come from ASW spreads, an
annuity-weighted par quantity that belongs to calibration (`par_swap_rate`-style), not to
`RateGenerator`. `spread` above answers "how far apart are two curves over a window"; both are needed.

## 7. Phases

| Phase | Content | Depends on |
|---|---|---|
| Y0 ✔ | Conventions cleanup, done while Y1 touches the files: (a) `CurveKey` (or its string) is the one canonical name into the registry — remove `split_index_name`, `_strip_currency`, `resolve_conventions`, the `index_name`/`tenor` dual arguments and the `key`/`currency`/`index_name` triple on `YieldCurve.from_registry`/`build`; (b) `RateIndex` becomes a `(family, tenor)` tuple, the engines read `family.day_count_method`; (c) drop the `ON` label: keys spell the overnight tenor `1D` (`USD.SOFR.1D`), so every tenor slot is `Term.from_str`-parsable and `keys.py` loses `_ON`, the two `ON` branches and the separate overnight predicate; `OVERNIGHT = Term(1, Days)` stays as a constant; `.ON` → `.1D` is a sed over ~53 files. `ConventionRegistry` and the common `Registry` stay as they are | — |
| Y1 ✔ | `YieldCurveGroup` (`add`, one `{name: YieldCurve}` map, name validation, `default_funding`, `fx_rate` as a float); `MarketContext` holds `{ccy: YCG}`, keeps the lookup facade and `__setitem__` as sugar, drops the unused `fixings=` constructor argument and the `_yield_cache` (views are stored, not derived); ownership and scenarios stay on the market and rebuild views | Y0 |
| Y2 | `fx_rate`, `ForYc` links, `FxQuote` registry, derived spot date and forward FX; FX-swap and xccy-basis helpers; MTM-swap pricer; multi-ccy discounting in one engine | Y1 |
| Y3 | `UST-OTR` (Act/360) and `XXX-FTP` families; `QuoteKind.FloatSpread` on the existing `SwapHelper` for ASW-spread calibration of both; the FTP chain end to end | Y1 only (not Y2) |
| Y4 | `RateGenerator.spread` | Y3 |
| Later | Serialisation of a market by date (graph with ids so object sharing survives a round trip); scenarios as values | |
| Later, after bonds | Proper FTP calibration: the bank's own bonds and comparable corporates priced off `USD.UST-OTR` + spread, solve the spread. Needs price↔yield, accrued, Act/Act ICMA, settlement rules — the same block Treasury futures need (`risk_and_engine_design.md` phase B). Until then the Y3 credit spread comes from ASW-style quotes | phase B |

Y3 is the shorter and more certain path and does not wait on FX.

## 8. Open questions

1. ~~Does the facade stay?~~ Decided: lookup stays permanently; assignment is removed.
2. ~~Where do fixings attach?~~ Decided: on the market by projection key, as today.
3. ~~Is `fx_rate` in the parameter vector?~~ Decided: no; translation only (section 4).
4. ~~Chains?~~ Decided: allowed, resolved transitively to the unique root; only cycles are rejected.
5. `spread`: default convention when none is given — continuous, or the over-curve's index day count?
6. ~~Where does `leaf_owner` live?~~ Decided: on the market, market-wide (section 3).
7. ~~`UST-OTR` projection day count?~~ Decided: `Both`, Act/360, so `spread(ust, over=sofr)` works; no ASW helper type, `spread`/`simple_rate` read it and `SwapHelper` with a float-spread quote builds it (section 5).
8. `default_funding`: the group's default should be the only one; the builders' `funding_id="STDCSA"`
   default would resolve `None` through the group instead, so the two cannot drift. Confirm.
