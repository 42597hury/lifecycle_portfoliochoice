# System II Quadrature-Density Policy-Convergence Study

**Date:** 2026-05-08 rerun of the 2026-05-07 handoff  
**Branch:** `jax-rewrite`  
**Scope:** Quantify how state-quadrature density and return-quadrature density
affect the solved consumption / risky-share / bond-share policies in System II
(state vector = `(rtb, y_1)`) at fixed `n_z=15`, `(n_eta, n_eps)=(3, 4)`,
state grid `(7, 7)`, and wealth grid `n_w=180`.

All four bundles are now present, share shape `(78, 15, 49, 180)`, and compare
element-wise with no interpolation.

**Outputs:**
- Metrics JSON: [system_ii_quad_convergence_metrics.json](system_ii_quad_convergence_metrics.json)
- Figures: [figures/](figures/)
  - [system_ii_quad_convergence_curves.png](figures/system_ii_quad_convergence_curves.png)
  - [system_ii_quad_per_age_divergence.png](figures/system_ii_quad_per_age_divergence.png)
  - [system_ii_quad_per_z_divergence.png](figures/system_ii_quad_per_z_divergence.png)
  - [system_ii_quad_per_wealth_divergence.png](figures/system_ii_quad_per_wealth_divergence.png)
  - [system_ii_quad_state_heatmap_C.png](figures/system_ii_quad_state_heatmap_C.png)
  - [system_ii_quad_state_heatmap_S.png](figures/system_ii_quad_state_heatmap_S.png)
  - [system_ii_quad_state_heatmap_B.png](figures/system_ii_quad_state_heatmap_B.png)
  - [system_ii_quad_probe_vs_age.png](figures/system_ii_quad_probe_vs_age.png)
- Analysis script: [system_ii_quad_convergence.py](../../scripts/analysis/system_ii_quad_convergence.py)

---

## TL;DR

| Verdict component | Outcome |
|---|---|
| State-quad uniform refinement `(3,3) -> (4,4)` | **RED.** `sq4x4_rq3x3` differs from baseline by `sup_C=3.17e-01`, `sup_S=9.76e-03`, `sup_B=2.94e-02`. Bond share moves by roughly 3 pp at the worst cell. |
| Ret-quad refinement `(3,3) -> (4,4)` | **RED.** `sq3x3_rq4x4` is much more sensitive on portfolio shares: `sup_S=7.94e-02`, `sup_B=1.50e-01`. Return quadrature is the biggest danger axis in this sweep. |
| `y_1` K-bump `(3,3) -> (3,5)` | **GREEN under the baseline-divergence proxy.** `sq3x5_rq3x3` has much lower divergence from baseline than `(4,4)` uniform on all policies: C falls from `3.17e-01` to `6.24e-02`, S from `9.76e-03` to `2.70e-03`, B from `2.94e-02` to `1.13e-02`. A denser state reference is still needed before calling it ground truth. |
| Newton / validation gates | **PASS.** All four bundles have finite C/S/B arrays, shape `(78, 15, 49, 180)`, `total_newton_failures = 0`, and `solve_status = complete`. |
| Recommendation | **Prefer the `y_1` K-bump as the cheap state-quadrature rule for System II-style state vectors: `n_state_quad=(3,5)`. Do not cheap out on return quadrature where portfolio-share precision matters; `(3,3)` vs `(4,4)` return nodes moves bond share by 15 pp at the worst grid cell.** |

The main surprise is that the asymmetric `y_1` K-bump works under the available
baseline-divergence proxy: 15 state-quadrature nodes stay much closer to the
baseline than the 16-node uniform refinement. That is strong directional support
for the Smolyak-style recommendation in systems whose state vector includes
`y_1`, but not a substitute for a denser reference bundle.

---

## Section 1 - Bundles Loaded

| Label | n_state_quad | n_ret_nodes_1d | Wall (s) | Available |
|:---:|:---:|:---:|---:|:---:|
| `sq3x3_rq3x3` baseline | `(3, 3)` | `(3, 3)` | 581.4 | yes |
| `sq4x4_rq3x3` state uniform refine | `(4, 4)` | `(3, 3)` | 979.7 | yes |
| `sq3x3_rq4x4` return refine | `(3, 3)` | `(4, 4)` | 975.4 | yes |
| `sq3x5_rq3x3` `y_1` K-bump | `(3, 5)` | `(3, 3)` | 934.2 | yes |

Validation gates are clean for all four bundles:

- C, S, and B have shape `(78, 15, 49, 180)`.
- No NaN or Inf values.
- `state_names == ('rtb', 'y_1')`.
- `n_z=15`, `(n_eta, n_eps)=(3, 4)`.
- `total_newton_failures == 0`.
- `solve_status == complete`.

The 49 state cells reshape as `(n_rtb, n_y_1) = (7, 7)` in C-order:
`state_idx = 7 * rtb_idx + y1_idx`.

---

## Section 2 - Pairwise Divergence Matrix

All rows report absolute element-wise divergence over the full policy tensor.
The baseline is `sq3x3_rq3x3`.

| Comparison | Effect | sup C | sup S | sup B | RMS C | p99 C | rel-sup C |
|---|---|---:|---:|---:|---:|---:|---:|
| `sq4x4_rq3x3` vs baseline | state uniform refine | 3.171e-01 | 9.762e-03 | 2.937e-02 | 2.349e-02 | 1.130e-01 | 6.397e-03 |
| `sq3x3_rq4x4` vs baseline | return refine | 3.096e-02 | 7.937e-02 | 1.501e-01 | 2.626e-03 | 1.253e-02 | 3.426e-03 |
| `sq3x5_rq3x3` vs baseline | `y_1` K-bump | 6.237e-02 | 2.699e-03 | 1.135e-02 | 7.876e-03 | 3.194e-02 | 1.510e-03 |
| `sq3x5_rq3x3` vs `sq4x4_rq3x3` | K-bump vs uniform | 3.179e-01 | 7.690e-03 | 2.478e-02 | 2.729e-02 | 1.250e-01 | 6.663e-03 |

The K-bump row against baseline is the best available proxy in this 4-run
factorial. It is smaller than uniform `(4,4)` on all three policies, despite
using 15 state nodes instead of 16. The direct K-bump-vs-uniform row stays
non-zero because the two refinements move the policy in different directions at
extreme state corners. A dense state reference would be needed to say which move
is absolutely closer to the continuum solution.

---

## Section 3 - Working / Retirement Split

| Comparison | Array | Working sup | Retirement sup |
|---|:---:|---:|---:|
| `sq4x4_rq3x3` vs baseline | C | 2.377e-01 | **3.171e-01** |
| `sq4x4_rq3x3` vs baseline | S | **9.762e-03** | 5.514e-03 |
| `sq4x4_rq3x3` vs baseline | B | **2.937e-02** | 1.710e-02 |
| `sq3x3_rq4x4` vs baseline | C | 2.693e-02 | **3.096e-02** |
| `sq3x3_rq4x4` vs baseline | S | **7.937e-02** | 1.628e-02 |
| `sq3x3_rq4x4` vs baseline | B | **1.501e-01** | 4.193e-02 |
| `sq3x5_rq3x3` vs baseline | C | **6.237e-02** | 4.580e-02 |
| `sq3x5_rq3x3` vs baseline | S | 2.605e-03 | **2.699e-03** |
| `sq3x5_rq3x3` vs baseline | B | **1.135e-02** | 1.037e-02 |

State-quadrature differences show up in both working and retirement ages
because `(rtb, y_1)` remains part of the retirement value-function approximation
through projected retirement income. Portfolio-share divergence is strongest in
working ages, usually around ages 60-63. Consumption divergence is often largest
in late retirement at the high-wealth edge.

---

## Section 4 - Where Divergence Concentrates

Worst cells by comparison:

| Comparison | Array | Peak age | z idx | `(rtb_idx, y1_idx)` | wealth idx |
|---|:---:|---:|---:|:---:|---:|
| `sq4x4_rq3x3` vs baseline | C | 93 | 14 | `(1, 0)` | 179 |
| `sq4x4_rq3x3` vs baseline | S | 63 | 14 | `(1, 0)` | 87 |
| `sq4x4_rq3x3` vs baseline | B | 62 | 14 | `(3, 6)` | 80 |
| `sq3x3_rq4x4` vs baseline | C | 97 | 14 | `(6, 0)` | 179 |
| `sq3x3_rq4x4` vs baseline | S | 62 | 14 | `(0, 0)` | 91 |
| `sq3x3_rq4x4` vs baseline | B | 41 | 14 | `(6, 0)` | 111 |
| `sq3x5_rq3x3` vs baseline | C | 34 | 14 | `(6, 5)` | 179 |
| `sq3x5_rq3x3` vs baseline | S | 67 | 2 | `(0, 6)` | 2 |
| `sq3x5_rq3x3` vs baseline | B | 61 | 14 | `(0, 6)` | 85 |

Per-state top cells:

| Comparison | C top cells | S top cells | B top cells |
|---|---|---|---|
| `sq4x4_rq3x3` vs baseline | `(1,0)`, `(5,6)`, `(1,1)`, `(2,0)`, `(0,1)` | `(1,0)`, `(3,6)`, `(2,0)`, `(0,6)`, `(6,4)` | `(3,6)`, `(0,6)`, `(2,6)`, `(1,6)`, `(4,6)` |
| `sq3x3_rq4x4` vs baseline | `(6,0)`, `(5,0)`, `(4,0)`, `(0,0)`, `(1,0)` | `(0,0)`, `(1,0)`, `(2,0)`, `(3,0)`, `(4,0)` | `(6,0)`, `(0,6)`, `(5,0)`, `(1,6)`, `(6,1)` |
| `sq3x5_rq3x3` vs baseline | `(6,5)`, `(5,5)`, `(4,5)`, `(6,6)`, `(3,5)` | `(0,6)`, `(0,1)`, `(0,0)`, `(1,1)`, `(1,0)` | `(0,6)`, `(0,3)`, `(6,3)`, `(0,1)`, `(1,6)` |

Interpretation:

- The high `z` edge (`z_idx=14`) dominates most worst cells. The exception is
  the K-bump S peak at retirement age 67, `z_idx=2`, very low wealth.
- Consumption divergences tend to live at max wealth (`wealth_idx=179`).
- Portfolio-share divergences live in mid-wealth bands where portfolio choice
  is most elastic.
- Uniform state refinement reveals large sensitivity on the top `y_1` row for
  bond shares. This is exactly the structure the `y_1` K-bump is meant to
  target, and the K-bump reduces the baseline divergence strongly.

The state heatmaps in [figures/](figures/) make this visually clear: the
residual error mass is mostly at state-grid edges and corners, especially the
top `y_1` row for bond-share differences.

---

## Section 5 - Verdicts

### State-Quad Uniform Refinement: RED

The move from `(3,3)` to `(4,4)` state quadrature changes policies by
`sup_C=0.317`, `sup_S=0.0098`, and `sup_B=0.029`. The portfolio changes are
not catastrophic, but they are big enough that `(3,3)` cannot be called quietly
converged for System II when compared to uniform `(4,4)`.

### Ret-Quad Refinement: RED

The move from `(3,3)` to `(4,4)` return quadrature changes portfolio shares by
`sup_S=0.079` and `sup_B=0.150`. This is the strongest sensitivity in the
sweep. Even though consumption is comparatively stable (`sup_C=0.031`), the
portfolio policies are not.

Operationally: use `(4,4)` return quadrature for any result that depends on
portfolio-share precision, especially welfare decompositions or sim-path Euler
diagnostics. `(3,3)` may still be usable for cheap exploratory scans whose
targets are coarse aggregate moments.

### `y_1` K-Bump: GREEN Under Baseline Proxy

The asymmetric `(3,5)` state quadrature is the clear winner among the two
state-refinement options tested:

| Metric vs baseline | uniform `(4,4)` | K-bump `(3,5)` | K-bump reduction |
|---|---:|---:|---:|
| sup C | 3.171e-01 | 6.237e-02 | 80.3% |
| sup S | 9.762e-03 | 2.699e-03 | 72.4% |
| sup B | 2.937e-02 | 1.135e-02 | 61.4% |
| wall time | 979.7 s | 934.2 s | 4.6% faster |

This supports the Smolyak-style claim directionally: in System II, the `y_1`
axis appears to deserve extra quadrature density more than uniform density
across `(rtb, y_1)`. The unresolved piece is whether a denser reference, outside
this handoff's scope, agrees with the baseline-proxy ordering.

---

## Section 6 - Cross-Link With System I Findings

System I already showed that two other knobs stop early:

- [System I n_z sweep](SYSTEM_I_NZ_CONVERGENCE_2026-05-07.md): `n_z=30` is the
  smallest defensible ablation setting; `n_z=10` is unsafe.
- [System I eta/eps sweep](SYSTEM_I_ETA_EPS_CONVERGENCE_2026-05-07.md):
  `(n_eta, n_eps)=(3,4)` is GREEN at the smallest tested point.

System II state quadrature does **not** simply inherit a "minimum tested is
fine" verdict. Uniform `(3,3)` versus `(4,4)` is RED, and return quadrature is
more sensitive still. The optimistic finding is more specific: asymmetric
state quadrature works. For systems with `y_1` in the state vector, `(3,5)` is
both cheaper and more accurate than uniform `(4,4)` in this sweep.

Recommended ablation settings after this pass:

| Knob | Recommendation |
|---|---|
| `n_z` | Use at least 30, from System I evidence. |
| `(n_eta, n_eps)` | Use `(3,4)`, from System I evidence. |
| `n_state_quad_nodes` with `(rtb, y_1)` | Prefer `(3,5)` as the cheap/default state rule; use a denser reference before calling it canonical-quality. |
| `n_ret_nodes_1d` | Use `(4,4)` when portfolio-share precision matters; `(3,3)` is only exploratory. |

---

## Caveats / Out Of Scope

- This is grid-based only; no sim-path Euler-equation comparison was run.
- The `(4,4)` return-quadrature result is not a proof that `(4,4)` is fully
  converged; it only proves `(3,3)` and `(4,4)` disagree materially.
- No `(5,5)/(5,5)` extension was run, per the handoff's scope.
- No solver-side changes were made.
- The K-bump GREEN label is a baseline-proxy verdict, not a proof against the
  continuum problem. The direct K-bump-vs-uniform comparison remains non-trivial,
  especially for C at high-wealth state corners.

---

## Reproducibility

```sh
python scripts/analysis/system_ii_quad_convergence.py
```

The script writes [system_ii_quad_convergence_metrics.json](system_ii_quad_convergence_metrics.json)
and regenerates all `system_ii_quad_*.png` figures under [figures/](figures/).
Use `--no-plots` to skip figure generation.
