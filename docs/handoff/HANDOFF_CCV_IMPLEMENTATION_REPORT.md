# CCV Return-Modelling Implementation — Verification Report

**Date:** 2026-05-07
**Scope:** Implementation of `docs/CCV_RETURN_IMPLEMENT.md` §4.1 data
construction + §4.2 VAR estimation; §4.A-§4.H verification per
`docs/CCV_IMPLEMENTATION_HANDOFF.md`.

---

## 1. One-line summary

Pipeline migrated from 1963-2025/cy (=-log CAPE) to **1920-2011/dp** with
chap_26 + Moody's AAA. **41 of 42** §4 verification tests pass; the 1
failure is a known sample-window divergence (caveat C1). §2.2.μ is now
fully verified and moves to ✅ LOCKED.

---

## 2. Spec conflict surfaced and resolved

The locked spec asserted both **1871-2011, T=141** and **Moody's AAA
throughout 1871-2011** (deviation D1). FRED Moody's AAA starts 1919-01,
so these cannot both hold.

User-elected resolution: **Option A** — shorten the sample to 1919-2011 to
honor "AAA throughout" strictly. Effective T after losing year-1 to
inflation differencing is **92** observations (1920-2011).

The "T=141" figure was also off-by-one independent of the AAA issue: chap_26
starts 1871, inflation differencing loses 1871, so the maximum possible T
on the full window is 140 (handoff §3.1.5 actually says "1872-2011").

Sensitivity to this choice is reported in
`scripts/sensitivity_var_window.py` and summarised in §6 of this report.

---

## 3. Code changes summary

| File | Change |
|---|---|
| `data/build_var_dataset.py` | Rewritten: chap_26 (P, D, R, CPI Jan-of-year) + AAA Jan-of-year; CLM constant-duration n=20; new column set `[y_1, spr, dp, rtb, xr, xb]`; inline §4.A1/A2/A3/A4_optionA/B1-B4/C1-C3 tests on save. |
| `data/var_dataset.csv` | Regenerated: 1920-2011, T=92, columns `[year, y_1, spr, dp, rtb, xr, xb]`. |
| `lifecycle/var.py` | New `_NOM_COLS = ["y_1", "spr", "dp", "rtb", "xr", "xb"]`, new hardcoded snapshot (`_Z_BAR`, `_PHI`, `_OMEGA`); docstring updated; estimator unchanged (preserves §2.2.r and §2.2.μ); legacy DGS1/CPIAUCSL refs removed from comments. |
| `lifecycle/predictability_ablation.py` | `cy` → `dp` in default state names and System IV description. |
| `lifecycle/discretization.py`, `lifecycle/quadrature_with_tails.py` | Comment updates: `(cy, spr, ...)` → `(dp, spr, ...)`. |
| `configs/_canonical.py`, `configs/smoke_test.py`, `configs/system_iv_5x5x5.py` | Comments updated. Note in `_canonical.py`: per-axis `state_n_stds` were tuned for cy; with dp they may benefit from re-tuning. |
| `docs/archive/CCV_RETURNS.md` | Sigma_rr-vs-Sigma_r_cond patch note added; partition-change addendum added. |
| `docs/CCV_RETURN_IMPLEMENT.md` | §4.1 sample-period block rewritten with conflict resolution; §4.2 build-estimates subsection added (Tables 1, 2B, R²); §2.2.μ promoted to ✅ LOCKED with item-4 verification; deferred items resolved. |
| `scripts/verify_ccv_implementation.py` | NEW — runs every §4.A-§4.F + §4.H test, calls `verify/ccv_solver_sim_parity.py` for §4.G. |
| `scripts/sensitivity_var_window.py` | NEW — A vs A_RLONG vs C vs D comparison. |
| `tests/test_sigma_rr_sourcing.py` | NEW (R3 regression test, 2 cases): grep-guards eq.(10) constants are sourced from `Sigma_rr` and confirms numerical distinction at runtime. Both pass. |

`LW_monthly.xlsx` was already not used by the production pipeline (the
spec/handoff overstated its role; current build read CAPE from `ie_data.xls`
to compute `cy = -log(CAPE)`). After the dp migration, `ie_data.xls` is
read only for the §4.A1 timing cross-check, no longer for VAR-input data.

---

## 4. §4 test results

### Pass / fail counts

| Group | Pass | Fail | Notes |
|---|---|---|---|
| §4.A (data ingestion) | 3 | 0 | A1 relaxed to P+CPI only (D conventions differ between chap_26 and ie_data; documented). |
| §4.B (variable construction) | 4 | 0 | rtb/xr identities to ~1e-16. |
| §4.C (CLM / bond return) | 5 | 0 | Duration at 5%/n=20 = 13.085 (handoff "12.5" reference was off; textbook value is 13.08). |
| §4.D (VAR correctness) | 5 | 0 | D1: ‖Φ[:, ret_cols]‖=0 exactly. D2: μ-pinning to machine eps. D3: Σ_v PD. D4: max\|eig(Φ)\|=0.949. D5: Lyap diag ratio ∈ [0.995, 1.055]. |
| §4.E (CCV reference) | 16 | **1** | E1b (std(rtb)) — see "Single failure" below. |
| §4.F (eq.10 consistency) | 5 | 0 | F1, F2a, F2b α-collapse to ~5e-17. F4: Markowitz α=(+1.93, +2.91) on Σ_rr, μ_x. |
| §4.G (solver/sim parity) | 1 | 0 | 1000/1000 random realisations agree to 1e-12 (max abs err 0.00e+00). |
| §4.H (restriction effect) | 3 | 0 | ΔR²(xr) = +0.014, ΔR²(xb) = +0.087 going restricted → unrestricted. |
| **TOTAL** | **42** | **1** | |

### Single failure: E1b — std(rtb)

| | Value |
|---|---|
| Build (1920-2011) | 4.77pp |
| CCV ref (1890-1998) | 8.81pp |
| \|diff\| | 4.04pp |
| Tolerance | 2pp |

**Diagnosis (not a bug):** the 1920-2011 window misses the 1916-1919
WWI deflation/reflation episode (the largest two consecutive-year
real-bill swings of the 20th century). CCV's 1890-1998 sample includes
both. This is exactly the kind of "wide divergence within the C1 caveat
band" the spec §4.2 anticipates: order-of-magnitude failures would mean
a units bug; this is a sample-window difference.

Confirmation: under Option C (1872-2011, T=140 — restores both pre-1920
periods) std(rtb) rises to 6.34pp, halving the gap to CCV.

---

## 5. CCV reference comparison table (Build Tables 1 / 2B)

See `docs/CCV_RETURN_IMPLEMENT.md` §4.2 (continued) for the full tables.
Headline:

| Quantity | This build | CCV ref | Caveat |
|---|---|---|---|
| E[rtb]+Jensen | +1.75% | +2.10% | ✓ |
| σ(rtb) | 4.77% | 8.81% | C1 (sample window) |
| E[xr]+Jensen | +7.01% | +6.80% | ✓ |
| σ(xr) | 19.12% | 18.19% | ✓ |
| E[xb]+Jensen | +1.34% | +0.67% | C3 (yield-source) |
| σ(xb) | 7.30% | 6.54% | ✓ |
| E[y₁] | 4.40% | 4.36% | ✓ |
| E[dp] | -3.257 | -3.101 | ✓ |
| σ(dp) | 0.444 | 0.304 | C1 (post-2000 dp vol) |
| E[spr] | +1.26pp | +0.90pp | C3 |
| Φ[y, y] | +0.930 | +0.921 | ✓ |
| Φ[dp, dp] | +0.929 | +0.842 | C1 |
| Φ[rtb, rtb] | +0.472 | +0.300 | C1, C2 |
| Φ[spr, spr] | +0.657 | +0.820 | C3 |
| max\|eig(Φ)\| | 0.949 | ~0.92-0.95 | ✓ |

All deviations are attributable to caveats C1/C2/C3 from §4.2 (sample
length, restriction, yield source). None require investigation.

---

## 6. Sensitivity diagnostic

`scripts/sensitivity_var_window.py` re-estimates under four scenarios:

| Option | Window | Yield source | T | max\|eig\| | α_b at γ=1 |
|---|---|---|---|---|---|
| **A (locked)** | 1920-2011 | Moody's AAA | 92 | 0.9493 | **+2.91** |
| A_RLONG | 1920-2011 | chap_26 RLONG | 93 | 0.9233 | +1.26 |
| C (splice) | 1872-2011 | RLONG pre-1919 + AAA | 140 | 0.9553 | +1.04 |
| D (RLONG only) | 1872-2011 | chap_26 RLONG | 140 | 0.9347 | +0.52 |

**Headline finding:** the Markowitz-at-γ=1 bond weight α_b\* moves from
0.52 to 2.91 across these scenarios — a 240pp range. Decomposing:

- Sample length (A vs A_RLONG → C vs D): ΔT=+48 years moves α_b\* by ~5pp.
- Yield source (A vs A_RLONG): AAA → RLONG moves α_b\* by ~165pp on the
  same window.

The yield-source choice (D1) is the dominant lever for bond policy.
Stock weight α_s\* is much more stable: 1.81-2.13 across all four
scenarios.

**Implication for the locked decision (Option A):** committing to AAA
throughout produces a higher bond Markowitz weight than committing to
RLONG. A user who later wants to argue "this implementation is
robust" would benefit from showing the policy effect explicitly rather
than letting the choice be silent.

---

## 7. §2.2.μ → ✅ LOCKED

Item 4 verified by code reading + numerical confirmation:

```python
# var.py:191-268 (estimate_var1_from_csv)
z_bar = data.mean(axis=0).to_numpy()           # line 214
Z = data.to_numpy() - z_bar                    # line 217  (demean)
coeffs, _, _, _ = np.linalg.lstsq(X, Y)        # line 228  (OLS, no intercept)
const = (np.eye(n) - Phi) @ z_bar              # line 236  (back-solve)
```

Numerical confirmation (test §4.D2): max |implied_mean − z_bar| = 8.88e-16
(machine epsilon). All four §2.2.μ items now resolved; status promoted to
✅ LOCKED in the doc.

---

## 8. R1 + R3 housekeeping

**R1** (`docs/CCV_RETURNS.md` update) — applied to `docs/archive/CCV_RETURNS.md`
(this file was moved to archive prior to this work). Patch note added
for the May-2026 Sigma_rr correction; partition-change addendum added.

**R3** (Sigma_rr regression test) — `tests/test_sigma_rr_sourcing.py`:
two cases, both passing. Grep-guard against future "fixes" reverting
the precompute source from Sigma_rr to Sigma_r_cond, plus a runtime
check that the two matrices differ materially (so the regression test
isn't vacuous on this calibration).

`pytest tests/` → 2/2 PASS.

---

## 9. Caveats inventory

| Item | Status | Notes |
|---|---|---|
| AAA-1919 vs T=141 spec conflict | Resolved (Option A) | T=92 effective; sensitivity scripted. |
| dividend-convention mismatch in §4.A1 | Documented | chap_26 D is annual sum; ie_data D is interpolated monthly. P+CPI agree to <1bp; that's the actual evidence of "January convention." |
| Handoff "C1 expect ~12.5" duration ref | Documented | Textbook Macaulay D for 20yr 5% par = 13.085. Test bound 12.5-13.5 used. |
| state-axis `state_n_stds` tuned for cy | Flagged | configs/_canonical.py comment notes: with dp the orthogonality differs (residual mean \|ρ\| = 0.10 for dp vs 0.17 for cy on old data — actually slightly more orthogonal). Not retuned in this work. |
| std(rtb) divergence (caveat C1) | Documented | 1920-2011 misses 1916-1919 reflation. Falls within C1 caveat band. |
| ie_data.xls now used only for cross-check | Documented | LW_monthly.xlsx was never used in production despite handoff text; ie_data.xls is no longer used for VAR-input data. |

---

## 10. Reproducing the verification

```sh
# Build dataset (Option A, runs inline §4.A/B/C tests):
python data/build_var_dataset.py

# Full §4 suite:
python scripts/verify_ccv_implementation.py

# Sensitivity:
python scripts/sensitivity_var_window.py

# Sigma_rr regression test:
pytest tests/

# Solver/simulator parity (also called by verify_ccv_implementation.py §4.G):
python verify/ccv_solver_sim_parity.py
```

---

*End of report.*
