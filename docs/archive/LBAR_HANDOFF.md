# HANDOFF: L̄ (Wage Index) Treatment

**To:** labour income agent
**From:** validation session (2026-04-16)
**Status:** Current treatment documented and defensible. One open question
worth investigating before sign-off.

---

## What L̄ is

L̄_t is the **SSA Average Wage Index** at time t — a time-varying real-world
quantity (2019 value ≈ $54,100). In Catherine (2025) it shows up in three
places:

1. **Normalizer**: earnings, wealth, brackets are all expressed as multiples of
   L̄_t. Raw earnings L_it become L̃_it = L_it / L̄_t.
2. **Indexes payroll cap & PIA bend points**: eq. 17 has
   `T_it = 0.106 × min{L_it, 2.5 × L̄_t}`; bend points scale with L̄_t each
   year (matches real SSA behaviour — taxable max and bend points are
   re-indexed annually).
3. **AIYE formula** (eq. 20): `AIYE = L̄_t × Σ min{L̃_is, 2.5}`. Terminal L̄_T
   scales career-average back from normalized to level units.

## Current treatment: L̄ = 1 for all t

The model assumes a **stationary economy** and sets L̄_t = 1 ∀ t.
Consequences in code:

- Tax brackets, payroll cap (2.5), bend points (0.21, 1.25) are **constants**
  in model units. See [model.py:268-289](model.py#L268-L289) and
  [model.py:315-319](model.py#L315-L319).
- No L̄ state variable. Dimensionality = (wealth, z, VAR state) only.
- Pension is **constant in model units** from age 68 onward
  ([precompute.py:387](precompute.py#L387) tiles `base_pension` across ages).
- AIME approximation `exp(z) * avg_det` needs no L̄_T multiplier.

Documented at [LABOUR.md:374-378](LABOUR.md#L374-L378) and motivated in
[LABOUR.md:17-22](LABOUR.md#L17-L22).

## What the L̄ = 1 assumption drops

Three channels, in order of economic importance:

| Channel | Effect of dropping | Matters here? |
|---------|-------------------|---------------|
| Deterministic wage growth trend | Young agents' PV of human capital understated | Small in stationary calibrations; absorbed by unit normalization |
| Cohort effects | Different birth cohorts face different L̄ paths | N/A — single-agent problem |
| **L̄ ↔ stock return correlation** | Human capital loses aggregate-market co-movement → agent may overweight stocks relative to full model | **The one that could actually bite** |

The first two are benign for this thesis. The third is the real question.

## Why the L̄↔stock correlation channel could matter

Empirically aggregate US wage growth is mildly procyclical and positively
correlated with equity returns. In a model with time-varying L̄ this makes
human capital slightly stock-like and pushes optimal portfolios toward bonds.
Setting L̄ = 1 removes this hedging motive at the aggregate level.

Whether this is a problem depends on whether the thesis's **idiosyncratic**
income process (`z` with its Guvenen mixture innovations) already correlates
with stock returns through the VAR structure. If `Sigma_rs` (state–return
covariance) implies `z` moves with equity returns, then some of the channel
is already in. If not, it's fully shut down.

## Action items for the labour income agent

1. **Check the VAR structure** (`Sigma_rs` in
   [model.py:77](model.py#L77) and how `z` is partitioned). Is the `z`
   innovation assumed orthogonal to return innovations? If yes, the
   L̄↔returns channel is fully absent. If `z` sits inside the VAR state with
   cross-covariance, some of it is captured already.
2. **Decide whether to strengthen the LABOUR.md note** at lines 374-378 with
   an explicit mention of the aggregate-wage/return correlation tradeoff.
   Standard precedent: Cocco-Gomes-Maenhout (2005) also shut this down —
   defensible.
3. **Quantitative sensitivity (optional)**: not necessary for thesis
   correctness, but if time permits, a back-of-envelope check of how much
   human-capital valuation changes under a 2% deterministic L̄ growth path
   vs L̄ = 1 would confirm the magnitude is small.

## What NOT to do

- Do **not** add L̄ as a state variable. The dimensionality cost is huge
  and Catherine herself does not treat it stochastically in her calibration.
- Do **not** introduce time-indexed tax brackets or bend points. Stationarity
  is the correct modelling choice for a lifecycle portfolio problem of this
  scope.

## Reference

Catherine, S. (2025). "Interest-Rate Risk and Household Portfolios."
Equations 17, 19, 20 (payroll, PIA, AIYE respectively).

User-validated items related to L̄ already checked off in
[LABOUR.md Section 5](LABOUR.md#L336). This handoff only concerns the
residual open question about the aggregate-correlation channel.
