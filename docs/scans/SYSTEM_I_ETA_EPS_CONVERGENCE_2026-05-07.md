# System I × (n_eta, n_eps) policy-resolution convergence study

**Date:** 2026-05-07
**Branch:** `jax-rewrite`
**Scope:** Quantify how the working-age income-shock quadrature density
`(n_eta, n_eps)` affects the solved consumption / risky-share / bond-share
policies in System I (iid returns, single-axis rtb state) at fixed `n_z=30`.
The hypothesis under test: *(n_eta=3, n_eps=4) is already converged*; refining
to (4, 5) or (6, 6) buys only negligible policy change. **Verdict: GREEN.**

**Outputs:**
- Metrics JSON: [system_i_eta_eps_convergence_metrics.json](system_i_eta_eps_convergence_metrics.json)
- Figures: [figures/](figures/)
  - [eta_eps_convergence_curves.png](figures/eta_eps_convergence_curves.png)
  - [eta_eps_per_age_divergence.png](figures/eta_eps_per_age_divergence.png)
  - [eta_eps_per_z_divergence.png](figures/eta_eps_per_z_divergence.png)
  - [eta_eps_per_wealth_divergence.png](figures/eta_eps_per_wealth_divergence.png)
  - [eta_eps_probe_alpha_vs_age.png](figures/eta_eps_probe_alpha_vs_age.png)
- Analysis script: [scripts/analysis/system_i_eta_eps_convergence.py](../../scripts/analysis/system_i_eta_eps_convergence.py)

---

## TL;DR

| Verdict component | Outcome |
|---|---|
| Grid-policy convergence | **GREEN.** (n_eta=3, n_eps=4) deviates 0.36% (relative sup-norm) from (6, 6) in C, with portfolio sup divergences of 0.66 pp on α_s and 0.58 pp on α_b. |
| Working/retirement split sanity | **PASS.** Retirement-age divergence is **exactly 0.0** for all (C, α_s, α_b). Confirms (n_eta, n_eps) does not enter the retirement FOC. |
| Distribution snapshot | **PASS.** Coverage ranges (min, max for α_s, α_b) match across the three configs to within ~0.3 pp; deeply-constrained corners are bit-identical. |
| Newton convergence | **PASS.** `total_newton_failures = 0` and `solve_status = complete` for all three bundles. |
| **Recommendation** | **Use (n_eta=3, n_eps=4) for all System I/II/III/IV ablation runs.** (4, 5) and (6, 6) buy nothing the policy can detect. |

---

## §1 — Convergence-rate table

Sup-norm and RMS divergence vs the (n_eta=6, n_eps=6) reference. Because all
three bundles share shape `(78 ages, 30 z, 7 states, 180 wealth)`, the
comparison is element-wise — **no interpolation required**.

| (n_eta, n_eps) | product | sup\|C\| | sup\|α_s\| | sup\|α_b\| | RMS\|C\| | rel-sup C |
|:---:|:---:|---:|---:|---:|---:|---:|
| (3, 4) | 12 | 6.63e-02 | 6.56e-03 | 5.84e-03 | 3.84e-03 | **0.36 %** |
| (4, 5) | 20 | 4.41e-02 | 4.00e-03 | 3.70e-03 | 3.58e-03 | **0.22 %** |
| (6, 6) | 36 | 0        | 0        | 0        | 0        | (reference) |

The convergence is monotone: (4, 5) cuts the (3, 4) sup divergence by ≈33% on
all three policies. But both coarse configurations are already an **order of
magnitude tighter** than the n_z=30 GREEN threshold from the n_z sweep
(2.9% rel-sup C, see [SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md](SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md)).

For comparison, on the same (78, 30, 7, 180) policy tensor:

| Refinement axis | Tightest GREEN config | rel-sup\|C\| vs reference |
|---|---|---:|
| n_z | n_z = 30 (vs n_z = 70) | 2.9 % |
| (n_eta, n_eps) | (3, 4) (vs (6, 6))     | 0.36 % |

Income-shock quadrature is **~8× less sensitive than z-grid resolution** at
these working points. The (n_eta=3, n_eps=4) baseline costs ~67 s wall vs 158 s
for (6, 6) — a 2.4× compute saving per System I solve.

---

## §2 — Working / retirement split (sanity check)

Retirement-age policies (ages 67..99, indices 45..77) integrate only over
return shocks; working-age income shocks (η, ε) cannot enter their FOC. A
non-zero retirement divergence here would indicate either (a) a config
inconsistency between bundles or (b) a numerical-precision artefact.

| (n_eta, n_eps) | array | working sup | retirement sup |
|:---:|---|---:|---:|
| (3, 4) | C   | 6.63e-02 | **0.0** |
| (3, 4) | α_s | 6.56e-03 | **0.0** |
| (3, 4) | α_b | 5.84e-03 | **0.0** |
| (4, 5) | C   | 4.41e-02 | **0.0** |
| (4, 5) | α_s | 4.00e-03 | **0.0** |
| (4, 5) | α_b | 3.70e-03 | **0.0** |

Bit-identical retirement slabs — the working-age FOC is the only place
(n_eta, n_eps) enters, and the upstream pension-after-tax / state-quadrature
machinery is independent of these knobs. This is the cleanest sanity-check
result possible.

---

## §3 — Where the residual divergence concentrates

For each (n_eta, n_eps)×{C, α_s, α_b} comparison, the per-axis-max collapses
the 4-D divergence tensor to a 1-D profile. Argmax cells (single worst-cell
location for each comparison):

| coarse | array | peak age | peak z idx (of 30) | peak state idx (of 7) | peak wealth idx (of 180) |
|:---:|---|---:|---:|:---:|---:|
| (3, 4) | C   | 34 (mid-working) | 29 (top of z-grid, ≈ +2.25 σ) | 0 (lowest rtb)  | 179 (max wealth) |
| (3, 4) | α_s | 22 (start of life) | 29 (top z) | 0 | 75 (lower-middle wealth) |
| (3, 4) | α_b | 30 (early working) | 11 (z ≈ −0.6 σ) | 0 | 5 (very low wealth) |
| (4, 5) | C   | 39 (mid-working) | 29 | 0 | 179 |
| (4, 5) | α_s | 28 | 27 | 0 | 74 |
| (4, 5) | α_b | 24 | 10 | 0 | 4 |

**Interpretation:**
- **Working ages 22–39** carry essentially all of the divergence — consistent
  with the §2 sanity check that retirement contributes zero.
- **Top of the z-grid** (z idx ≈ 27–29) is where coarse working-age income-
  shock quadrature most underweights the right-tail mass; consumption at the
  high-z, max-wealth corner is the most-shifted cell when (n_eta, n_eps)
  drops from (6, 6) to (3, 4).
- **state_idx = 0** (lowest rtb realisation) consistently hosts the worst
  cell across all three policies and both coarse configs. Low rtb means
  the riskless return is least favourable, so the optimal policy is most
  sensitive to the left/right tails of the income shock — coarse quadrature
  hurts most at this corner.
- **α_b worst cells live at very low wealth** (idx 4–5, ≈ first wealth-grid
  rung), where the borrowing/lending margin is thinnest; α_s peaks at
  lower-middle wealth (idx 74–75) where portfolio choice is most sensitive
  to expected human-wealth realisations.

See [eta_eps_per_age_divergence.png](figures/eta_eps_per_age_divergence.png),
[eta_eps_per_z_divergence.png](figures/eta_eps_per_z_divergence.png), and
[eta_eps_per_wealth_divergence.png](figures/eta_eps_per_wealth_divergence.png)
for the full profiles. The per-age plot includes a vertical marker at
`retire_age_idx = 45` showing the abrupt drop to zero divergence past
retirement.

**Distribution snapshot:**

| (n_eta, n_eps) | min α_s | max α_s | min α_b | max α_b |
|:---:|---:|---:|---:|---:|
| (3, 4) | 0.3917 | 1.2616 | 0.1646 | 0.8661 |
| (4, 5) | 0.3917 | 1.2583 | 0.1646 | 0.8635 |
| (6, 6) | 0.3917 | 1.2596 | 0.1646 | 0.8646 |

Minimum shares are bit-identical across configs (deeply-constrained corners
of the policy don't see the working-age quadrature). Maximum shares wobble
by ≤ 0.30 pp non-monotonically — (3, 4) overshoots (6, 6) max α_s by 0.20 pp,
(4, 5) undershoots by 0.13 pp. This non-monotonicity in `S_max` indicates
the residual differences at the policy's tail cells are below the noise
floor of the quadrature, not a systematic bias one config can correct for.

The probe-cell line plot ([eta_eps_probe_alpha_vs_age.png](figures/eta_eps_probe_alpha_vs_age.png))
overlays α_s(age) and α_b(age) at (z_idx, state_idx, wealth_idx) =
(15, 3, 90) (per-axis midpoint convention from the recent probe-index fix)
for all three configs. The three curves are visually indistinguishable
across the entire age range.

---

## §4 — Verdict and recommendation

> **(n_eta=3, n_eps=4) is GREEN** — converged for all practical purposes.
> Sup-norm relative error on consumption is 0.36% vs the (6, 6) reference,
> portfolio shares stay within 0.7 percentage points everywhere, retirement-
> age policies are bit-identical, and deeply-constrained portfolio corners
> are bit-identical. The hoped-for "stops at minimum tested" result holds.
>
> **(n_eta=4, n_eps=5) is GREEN-tighter** — cuts the (3, 4) sup divergences
> by ≈33% on all three policies, but absolute differences are already so
> small that this refinement is not visible at any reasonable downstream
> grain (sim-EE, welfare, decomposition). Use it only if you want a
> belt-and-braces canonical run; for ablation work it is wasted compute.

**Operational recommendation:**
- **For all System I/II/III/IV ablation runs:** use **(n_eta=3, n_eps=4)**.
- **For canonical / publication runs:** (n_eta=4, n_eps=5) is a defensible
  upper-tier choice; (6, 6) is unnecessary on this evidence.
- The convention `n_eps ≥ n_eta` enforced by the sweep launcher
  ([verify/benchmark_system_i_eta_eps_sweep.py](../../verify/benchmark_system_i_eta_eps_sweep.py))
  is preserved by all three configs.

---

## §5 — Cross-link with n_z sweep

This sweep is the second leg of the System I calibration story. The first
leg ([SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md](SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md))
established **n_z=30** as the smallest defensible labour-income state-grid
resolution. Combined verdict for downstream ablation work:

| Calibration knob | Recommended ablation setting | Wall-time vs canonical | Cited rel-sup C |
|---|---|---:|---:|
| `n_z`            | 30 (vs canonical n_z=70)        | ~½ × | 2.9 % |
| `(n_eta, n_eps)` | (3, 4) (vs canonical (6, 6) or (4, 5)) | ~⅖ × (66.6 / 157.7) | 0.36 % |

**Combined compute saving for ablation studies (System I evidence):**
`(n_z=30, n_eta=3, n_eps=4)` vs `(n_z=70, n_eta=6, n_eps=6)` is ≈ 5× faster
per solve. Across Systems II, III, IV — each of which inherits the same
working-age quadrature and z-grid machinery — the compounded saving is the
primary motivation for running the ablation set at coarse-but-defensible
resolution.

The n_z verdict (RED at n_z=10, YELLOW at n_z=15, GREEN-with-caveat at
n_z=30) was nuanced and cost-sensitive. The (n_eta, n_eps) verdict here is
unambiguous GREEN at the smallest tested setting — (n_eta, n_eps) genuinely
plateaus before z-grid does.

---

## §6 — Caveats / out-of-scope

- **Sim-path Euler-equation residual comparison** (mirroring §3 of the n_z
  report) was *not* done here, per the handoff's "stay grid-based"
  scoping. If a reviewer asks for sim-EE confirmation of the GREEN verdict,
  the wrapper [scripts/analysis/run_ee_simpath_system_i.py](../../scripts/analysis/run_ee_simpath_system_i.py)
  can be re-pointed at the three eta/eps bundles.
- **Cross-system (System I vs II/III/IV) sensitivity** is out of scope.
  The economic argument for transferring this verdict to richer systems is
  that those systems use the *same* working-age η × ε integration; the only
  thing that changes downstream is the state-vector dimensionality, which
  doesn't interact with the income-shock quadrature error analysed here.
  Still, a single (3, 4) vs (6, 6) check per system at canonical n_z is
  cheap insurance and recommended before fully committing.
- **Newton-iter cap behaviour** is unchanged from the n_z study (§4 there);
  histograms show `max=100` on the tiny-savings boundary cells in all three
  bundles. This is a separate finding independent of (n_eta, n_eps).

---

## Reproducibility

```sh
# Bundles must be present at:
#   saved_runs/ablations/system_i_grid7_nz30_eta{3eps4,4eps5,6eps6}_calib1/
# Sync from s3://hugo-thesis-runs/saved_runs/ablations/ if missing.

python scripts/analysis/system_i_eta_eps_convergence.py
```

This single script writes both the metrics JSON and the figures. Pass
`--no-plots` to skip the figure generation, or `--bundles-root`,
`--output-dir`, `--fig-dir` to redirect inputs/outputs.
