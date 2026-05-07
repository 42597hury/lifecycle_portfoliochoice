# Mixed-Precision Cast Site Review

**Date:** 2026-05-06
**Branch:** `jax-rewrite`
**Reviewer:** Claude (Opus 4.7, 1M ctx)
**Handoff under review:** [docs/handoff/HANDOFF_MIXED_PRECISION_GATHER.md](../handoff/HANDOFF_MIXED_PRECISION_GATHER.md)

This is the §7 deliverable: review the proposed cast list before implementation. Implementation does not start until the user signs off on the final list.

---

## Executive summary

The proposal is sound. The fp32-gather / fp64-FOC boundary is the right design and the bulk of §3.1 / §3.2 holds up under line-by-line review. I have **two corrections** and **one site missed**:

1. **§3.1 row 5 ("`c_next` itself if cast at build time"):** misleading — `c_next` *storage* must remain fp64 (the warm-start `S_list[t+1]` lives in the same dtype convention). Cast at the gather entry inside `_solve_*_at_cell`, not at policy storage time. Bandwidth win still applies because the gather + cast fuses in XLA.
2. **§3.3 row 3 (`s_grid` and `wealth_grid`):** split the recommendation. `wealth_grid` IS used inside the interp bracket — cast it. `s_grid` is consumed only in `_egm_scan_cell` and never reaches the gather/interp; **leave it alone**.
3. **Site missed in §3.1:** the inline multilinear interp inside `retirement_foc_jac_ccv` ([solver.py:868-887](../../lifecycle/solver.py#L868-L887)) is a *separate* implementation from `_interp_c_and_mpc_at_cell`. The handoff focuses on `_interp_c_and_mpc_at_cell` (used in working), but retirement uses an inlined `per_kv_kr` that needs the identical fp32-gather/fp64-FOC treatment. Both must be plumbed.

Everything else in the handoff stands. Compound error across 33 ages is bounded by the fact that `S_list[t+1]` / `C_list[t+1]` are stored in fp64 (output of `_lift_to_wealth_grid` → `jnp.interp`, which inherits fp64 from `c_egm`/`a_s_egm` built in fp64 inside `_egm_scan_cell`). No fp32 contamination leaks across age boundaries.

---

## §3.1 sites — agent agrees / disagrees / proposes changes

| Site | Verdict | Notes |
|---|---|---|
| **`c_corners` gather in retirement** ([_solve_retirement_at_cell:1178](../../lifecycle/solver.py#L1178)) | **Agreed.** Cast `c_corners_at_z` to fp32 right after the advanced gather. Values are policy consumption, bounded in [`min_consumption=1e-10`, ~10]; fp32's 7-digit mantissa is ample for the multilinear weighting. The `min_consumption` floor is reapplied in fp64 after the cast back, so floor cells round-trip cleanly. | Hot site. |
| **`c_corners_T` gather + transpose** in working ([_solve_working_at_cell:1235-1236](../../lifecycle/solver.py#L1235-L1236)) | **Agreed.** Same as above plus the extra `n_z` axis. Largest bandwidth-bound site in the per-FOC trace. | Hot site. |
| **Multilinear-interp corner values** (`c_per_corner`, `slope_per_corner`) inside `_interp_c_and_mpc_at_cell` ([solver.py:790-834](../../lifecycle/solver.py#L790-L834)) | **Agreed.** Pure local arithmetic on bounded values. fp32 has 7 digits; we use 4–5. | |
| **Multilinear-interp corner weights** (`w_kv * c_per_corner`) | **Agreed.** Weights in [0,1] from frac products. fp32 ample. | |
| **`c_next` itself if cast at build time of the kernel closure** | **Disagree (as written) — REPLACE with: cast at gather entry, not at storage.** Storage of `C_list[t+1]` must remain fp64 because (a) warm-start init reads from `S_list[t+1][z, i_s, w_ref_idx]` and the warm-start invariant says fp64 (per §3.2 row 11 of the handoff), and (b) all stored policies enter the next age's `c_next` argument. The bandwidth win from a fp32 gather is achieved by XLA fusing the gather → cast → reduce chain (verifiable in HLO; gate 3). If HLO shows the gather still loads fp64 bytes, *then* we explore casting `c_next` once at kernel entry inside `_solve_*_at_cell` as an optimization, but **not** as policy storage. | Implementation-relevant. |
| **Wealth-grid `searchsorted` index arithmetic** | **Agreed.** Already int32. No change. | |
| **`x_next = s × R_p + pension_next_z` if computed in fp32** | **Agreed with caveat.** Cast `x_next` to fp32 *only inside* the gather/interp closure, after the FOC builds it in fp64. The bequest path uses `sR_p` directly in fp64 — that path is unaffected. Magnitudes 0.1–225 are well within fp32. | Cast inside `_interp_c_and_mpc_at_cell`/`per_kv_kr`, not at the FOC entry. |

---

## §3.2 sites — agent agrees / disagrees / proposes changes

All entries **agreed**. Spot-checks of the math:

- `c^(-γ)` at γ=10 and `c=0.01`: 1e20. fp32 max is 3.4e38 so it fits, but mantissa-relative error of 1e-7 corresponds to absolute error of 1e13 in mu. In the FOC sum that's catastrophic. **fp64 mandatory** — agreed.
- `bequest_mu_and_mup` at high wealth: with `b_bar=10`, `gamma=10`, `A≈4`, `C_bar≈187.5` → `mu ≈ 1.6e-24`. fp32's smallest normal is 1.2e-38, smallest subnormal 1.4e-45 — `mu` would be subnormal with 0–7 mantissa bits. **fp64 mandatory** — agreed.
- Newton residual at `tol=1e-7`: residual is sum of products with each term order ~1; the threshold lives at ~1e-7 of the term magnitude. fp32's 7-digit precision is *exactly* the tolerance, no margin. **fp64 mandatory** — agreed.
- Jacobian `J_ss = sum(jac_lin * dRp_das² + extra_ss)` where `extra_ss = wmu * R_p * (dr_da_s² - sigma2_xr)`: explicit subtraction of similar-magnitude terms. **Cancellation hazard in fp32** — agreed.
- CCV variance correction in `_ccv_log_return_and_grad` ([solver.py:665-675](../../lifecycle/solver.py#L665-L675)): `r_p = log_R_bill + α·log_x + 0.5·α·diag(Σ) - 0.5·(α·Σ·α)`. The two 0.5-prefactor terms have similar magnitudes when Σ is well-conditioned and αs are O(1); cancellation real. **fp64 mandatory** — agreed.
- `exp(r_p)`: `R_p ≈ 1`, so a fp32 r_p error of 1e-7 maps to R_p error of 1e-7 absolute, which combined with `c = (β·V_dot)^(-1/γ)` and γ=10 amplifies to a c-error of (1+1e-7)^10 ≈ 1+1e-6. **Keep fp64** — agreed.
- Newton step `step_s = -(Jbb·fs - Jsb·fb) / det`: `det = Jss·Jbb - Jsb²` is itself a cancellation. `singular_det = 1e-15` is below fp32's smallest normal at typical Jacobian magnitudes. **fp64 mandatory** — agreed.
- Backward-age warm-start `init_a_s_arr[z_idx, i_s, w_ref_idx]`: stored fp64, gathered fp64. **Agreed** — and any precision change here would contaminate the warm-start chain across 33 ages.

---

## §3.3 ambiguous sites — agent's recommendation

| Site | Recommendation | Reasoning |
|---|---|---|
| **`log_R_bill, log_x_s, log_x_b` scenario tensors** | **fp64.** | They feed `_ccv_log_return_and_grad`, which has the cancellation in the variance correction. Casting them fp32 would push fp32 noise into r_p before the cancellation, amplifying through `exp`. The bandwidth saving is small (a (n_state_quad, n_ret_quad) tensor — KB-scale at canonical). |
| **`weight_kv_kr` (state quad × ret quad)** | **fp64.** | Same shape as log_R_bill (KB-scale, no bandwidth gain) and it multiplies `mu_comb` directly in the FOC sum. fp32 weight noise of 1e-7 propagates 1:1 into the FOC residual, which is precisely the convergence tolerance. Tighter than I'd accept. |
| **`s_grid` and `wealth_grid`** | **Split: cast `wealth_grid` inside interp, leave `s_grid` fp64.** | `s_grid` is consumed only by `_egm_scan_cell` (the savings sweep that calls `foc_factory(s_val)`); `s_val` enters the FOC as fp64 and never reaches the gather/interp. `wealth_grid` IS used inside the interp bracket (`searchsorted`, `1/(x1-x0)`, the fw weight) — cast it inside the interp closure to keep the bracket arithmetic in fp32 with everything else. |
| **eta/eps inner-quadrature path in `_solve_working_at_cell`** | **fp64 in the FOC sum; fp32 only at the gather/interp boundary.** | The (k_v, k_r, k_eta, i_e) reduction is a fp64 sum — that's untouched by this proposal. The interp inside the inner loop returns fp64 `c_at_xn`/`mpc_at_xn` (cast back before CRRA), so the FOC sum is unaffected. |
| **Per-cell `psi_z` (survival probability)** | **fp64.** | Scalar — zero bandwidth benefit. Multiplied with `mu_comb` in the FOC sum, where any noise is residual-tolerance-tight. |
| **`A_is` (annuity factor)** | **fp64.** | Agreed with handoff. Used in bequest, which is fp64-mandatory. |

---

## §3.4 worries — agent's mitigation

- **Compound error across 33 ages with backward-age warm-start.** *Addressed by design.* Stored policies pass through `_lift_to_wealth_grid` → `jnp.interp(wealth_grid, x_sorted, c_sorted)` where `c_sorted` is built in fp64 inside `_egm_scan_cell` (`c_opt = jnp.maximum((beta·V_dot)^(-1/γ), min_consumption)`, all fp64). `S_list[t+1]` and `C_list[t+1]` are therefore fp64 regardless of the gather precision. The warm-start init scalar `init_a_s_arr[z_idx, i_s, w_ref_idx]` is fp64. **No fp32 contamination crosses age boundaries.** I will still verify gate 5 (tail-cell spot check) at age 67 vs age 99 to confirm.
- **`c_at_xn` lower bound.** *Addressed by ordering* in §5.3 of the handoff: cast back to fp64 first, then `jnp.maximum(c_at_xn, min_consumption)`. The floor is enforced in fp64. ✓
- **XLA fusion eliding the cast boundary.** *Real concern, mitigated by gate 3.* Plan: inspect HLO after building the working kernel; look for `convert(f64←f32)` ops at the boundary, and confirm `**` / `multiply(c, c)` operate on f64 operands. If XLA elides the cast (e.g. by promoting the gather back to f64), insert `jax.lax.optimization_barrier(c_at_xn_g)` before the cast back. Document the HLO snippet in the implementation commit message.

---

## Additional sites the agent identified

1. **The inline `per_kv_kr` interp inside `retirement_foc_jac_ccv`** ([solver.py:868-887](../../lifecycle/solver.py#L868-L887)).
   The handoff §5.3 modifies `_interp_c_and_mpc_at_cell` (used by working). Retirement uses a *separate* inline interp because z is frozen at retirement and the (n_z, frac_z) bilinear axis is gone — it's a (n_corners, n_w) bilinear instead of trilinear. **The cast plumbing must apply to both paths.** Concretely: thread `gather_dtype` into `retirement_foc_jac_ccv` and cast `c_corners_at_z`, `wealth_grid`, `x_next`, the per-corner values, and the slope inside `per_kv_kr`. Cast the `c`/`mpc` outputs back to fp64 before the `min_consumption` floor.

2. **`frac_z`, `eta_iz_lo`, `eta_frac_z` weights** ([_solve_working_at_cell:1832-1836 / boundary path](../../lifecycle/solver.py#L1832-L1836)). These are bounded [0,1] bracket weights computed in fp64 from `bracket_uniform`. Cast to fp32 *inside* the interp closure (alongside x_next). Outside the closure, the FOC handlers don't read them.

3. **The `iw`, `iz_lo`, `iz_hi` integer indices.** Already int32. No change needed; explicitly noting so the implementation doesn't accidentally cast them.

---

## Final cast list (after review) — for user approval

### Cast TO fp32 (inside the interp/gather closure)

1. `c_corners_at_z` (retirement) — cast after the advanced gather at [solver.py:1178](../../lifecycle/solver.py#L1178), before passing into `retirement_foc_jac_ccv`'s inline `per_kv_kr`.
2. `c_corners_T` (working) — cast after the gather + transpose at [solver.py:1235-1236](../../lifecycle/solver.py#L1235-L1236), before passing into `working_foc_jac_ccv`'s `per_kv` closure.
3. **Inside `_interp_c_and_mpc_at_cell` (working path):** cast `c_corners_kv`, `w_corners_kv`, `x_next_scalar`, `wealth_grid`, `frac_z`. Compute the multilinear interp (corner values, slopes, weighted sums) in fp32. Cast `c_at_xn`, `mpc_at_xn` back to fp64 BEFORE applying `jnp.maximum(c, min_consumption)` and `jnp.clip(mpc, 0, 1)`.
4. **Inside the inline `per_kv_kr` of `retirement_foc_jac_ccv` (retirement path):** same recipe as #3 — cast `c_kv`, `w_kv`, `x_scalar`, `wealth_grid` to fp32; do the bilinear interp in fp32; cast `c`, `mpc` back to fp64 before the floor/clip.
5. `x_next` inside both interp closures (cast inside the closure, not at the FOC level — the bequest computation in working/retirement FOCs uses `sR_p` directly in fp64).

### KEEP fp64 (no change)

- All FOC summation: `wmu * dRp_das`, `wmu * R_p`, `wmu * mup_comb * s_val`, the bequest/alive split sums.
- `_ccv_log_return_and_grad` (r_p, R_p, dr/da_s, dr/da_b) and all variance-correction subtractions.
- `bequest_mu_and_mup`, `mu_alive`, `mup_alive`, all CRRA `**(-gamma)`.
- Newton residuals, Jacobian terms, step calculation, line search, convergence test.
- `weight_kv_kr` (cancellation-tight in FOC sum).
- `log_R_bill`, `log_x_s`, `log_x_b` (feed the variance correction).
- `psi_z`, `A_is` (scalars, no bandwidth gain).
- `s_grid`, `s_val`, EGM inverse `(beta·V_dot)^(-1/γ)`.
- Stored policies `C_list`, `S_list`, `B_list` (fp64 invariant).
- Warm-start init `init_a_s_arr`, `init_a_b_arr` (read at fp64).
- The `static` tuple (Python ints/floats).

### Implementation differences from §5

- §5.3 covers `_interp_c_and_mpc_at_cell` only. **Add a parallel diff for the inline `per_kv_kr` inside `retirement_foc_jac_ccv`.**
- The plumbing in §5.4 reads `gather_dtype` from `sc.gather_precision`. That's correct, but the `static` tuple needs `gather_dtype` to NOT be a JAX array — `jnp.float32`/`jnp.float64` are dtype objects (Python values), so they bake into the trace cleanly. Confirmed.
- Skip casting `c_next` at storage time. The HBM-bandwidth saving comes from XLA fusing the advanced gather with the dtype convert. Verify in HLO (gate 3).

### Verification gates — agreed in order

1. Default unchanged (`gather_precision="f64"` bit-identical).
2. `verify/mixed_precision.py` smoke: max relative error on (C, S, B) < 1e-4, no NaN.
3. HLO inspection: f32 ops in gather/interp section, f64 ops in CRRA/FOC; `convert(f64←f32)` op visible at the boundary right before any `**(-gamma)`.
4. `verify/canonical_small.py` at fp32: alphas sane, no NaN.
5. Tail-cell spot check at edges (top wealth, bottom z, edge state-grid points): rel err < 1e-4.

If any gate fails, stop and report.

---

## What changes vs. handoff §3 — summary for sign-off

| # | Change | Rationale |
|---|---|---|
| 1 | §3.1 row 5: clarify "cast at gather entry, not at storage" | Storage must remain fp64 to preserve the warm-start invariant. |
| 2 | §3.3 row 3 (`s_grid` + `wealth_grid`): split — cast `wealth_grid` inside interp, leave `s_grid` fp64 | `s_grid` never enters the gather/interp; `wealth_grid` does. |
| 3 | NEW: cast plumbing must reach the inline `per_kv_kr` inside `retirement_foc_jac_ccv` | Retirement uses a separate interp, not `_interp_c_and_mpc_at_cell`. |
| 4 | NEW (clarification): `frac_z`, `eta_frac_z` cast inside the interp closure (not at the FOC entry) | Keeps the FOC arithmetic fp64 while making the interp arithmetic fp32-uniform. |

Everything else from §3.1 / §3.2 / §3.4 stands as written.

---

## Awaiting user confirmation

Per handoff §7: **do not implement until the user confirms the final cast list above.**
