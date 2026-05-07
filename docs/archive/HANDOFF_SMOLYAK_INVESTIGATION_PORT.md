# Handoff: Smolyak-Hermite quadrature investigation — port from numba branch

**Status:** Investigation complete in numba branch. Concluded NOT suitable for production drop-in. Lobatto-Smolyak partial recovery exists with ~3× speedup at ~3% policy bias on deep-tail near-cap cells. Document this here so the JAX branch can either (a) replicate the negative result on its 4-state/2-return geometry, or (b) deliberately not pursue it without re-deriving the same conclusions.

**Source of truth (read these first):** all investigation code lives in the numba branch:

```
Repo:    C:\Users\carlh\Projekt\thesisscripts
Branch:  main
Scripts: scripts/_investigate_smolyak_returnstate.py    — node counts + MGF integration accuracy
         scripts/_check_smolyak_arbitrage.py            — discrete-cloud arbitrage check
         scripts/_benchmark_smolyak_foc.py              — FOC integrand level error (trilinear V)
         scripts/_benchmark_smolyak_alpha_star.py       — α* root benchmark (production-relevant)
         scripts/_benchmark_lobatto_smolyak.py          — prescribed-tail Smolyak
Existing arbitrage routine reused: scripts/_check_ccv_arbitrage.py
```

Each script is standalone, runs from project root with `python -m scripts._<name>`, and does not modify production code.

---

## 1. Why this investigation was launched

The numba canonical config uses:
- `n_state_quad_nodes = (3, 4, 4)` — 48 state-innovation nodes (3D state)
- `n_ret_nodes_1d     = (3, 5, 5)` — 75 return-residual nodes (3D return)
- Joint per-cell expectation: 48 × 75 = **3,600** outer-product nodes per state cell, evaluated at every Newton iteration of every age × state × wealth grid point.

The proposal: replace the nested 3+3 tensor product with a single 6-D anisotropic Smolyak-Hermite rule. Theoretical speedup: ~5×–12× depending on Smolyak level. The hypothesis was that the integrand has enough smoothness for sparse-grid polynomial-exactness arguments to hold.

The setup is fundamentally the same in the JAX branch except the dimensions are redistributed: **4D state** `(cy, spr, rtb, y_1)` × **2D return** `(xr, xb)`. Total joint dimension is still 6. Same arithmetic question applies.

---

## 2. Investigation summary (numba branch findings)

### 2.1 The four failure modes encountered, in order

1. **Naive anisotropic Smolyak** (`_investigate_smolyak_returnstate.py`)
   - L=4 anisotropic with weights `w = (2, 4/3, 4/3, 2, 1, 1)`: 286 signed nodes (12.6× speedup)
   - L=5 same weights: 794 signed nodes (4.5× speedup)
   - **Polynomial exactness on monomials up to degree 6**: matches baseline tensor.
   - **MGF stress integrand (`E[exp(α·log_R_excess)]`)**: matches baseline to ~5e-5 absolute at α=(6,6).
   - **Looked promising.** Recommended L=5 anisotropic as the candidate.

2. **Spurious arbitrage** (`_check_smolyak_arbitrage.py`)
   - Tested via the project's existing CCV arbitrage diagnostic (`scripts/_check_ccv_arbitrage.py:50`, `check_state_arbitrage`).
   - **Smolyak L=5 anisotropic admits arbitrage** at 116/729 (16%) of canonical state cells, with `max_t = +0.245` (24% guaranteed log-excess return at the worst cell).
   - **Smolyak L=4 anisotropic** is worse: 242/729 (33%) of cells admit arbitrage.
   - Root cause: Smolyak's combinatorial truncation skips joint-extreme corners (e.g. spr-tail × y_1-tail × xb-residual-tail simultaneously). The full tensor has all 3,600 corners; Smolyak doesn't.

3. **Anchor-enriched Smolyak** (same script, augmented variants)
   - Added 64 = 2^6 binary-corner anchor nodes at canonical max-K tail z values per axis, with **zero weight** (support enrichment, no integration contribution).
   - **Arbitrage closed**: max_t = +6.8e-14 on full 729-cell scan. State-independent fix.
   - **Speedup retained**: L=4 + 64 anchors = 305 unique nodes (11.8× speedup); L=5 + 64 anchors = 713 unique nodes (5.0× speedup).
   - **But integration broke** at high γ × |α|. See next.

4. **FOC integrand error at near-cap α** (`_benchmark_smolyak_foc.py`)
   - Tested the production CCV FOC kernel `f(z) = u'(W_next) · V_next · exp(r_p^CCV)` with γ=5, synthetic-but-realistic V_next on the canonical state grid.
   - **At α=(0,0)**: ~1e-4 relerr ✓
   - **At α=(1,1)**: ~1e-3 relerr ✓
   - **At α=(3,2)**: ~7% relerr — borderline
   - **At α=(5,-3)**: ~25% relerr ✗
   - **At α=(6,±6) cap-bound**: **63–77% relerr** ✗

   Reason: the integrand has effective exponent `(1−γ) · r_p^CCV` ≈ −4 · (linear-in-z with α-amplified loadings) at γ=5. At α=(6,6), per-axis effective exponent is ~24× the residual std-dev. The integrand peaks at exactly the joint-extreme corners that Smolyak skips. Anchors with zero weight don't help integration — they add support, not mass.

5. **α\* root benchmark** (`_benchmark_smolyak_alpha_star.py`)
   - The 70% level error does NOT translate to 70% policy error. Errors largely cancel between FOC numerator terms.
   - **Smolyak L=5 + 64 anchors**: max |Δα\*| = **0.088** across 8 stress cells (mean 0.016).
   - **Smolyak L=4 + 64 anchors**: max |Δα\*| = **0.152** (mean 0.036).
   - 4/8 cells pass strict 1e-3 criterion for L=5+anchors, 3/8 for L=4+anchors.
   - Worst cell is consistently a deep-tail cy state (cy ≈ −4.5, far in stationary tail) where α_b is near the cap.

### 2.2 Lobatto-Smolyak partial recovery (`_benchmark_lobatto_smolyak.py`)

Built a Smolyak rule from per-level Lobatto-prescribed-tail univariate rules instead of standard Gauss-Hermite. At every Smolyak level ≥ 2 on a Lobatto axis, the univariate rule has prescribed tail nodes at ±Z **with non-zero Gauss-style weights** (e.g. `gauss_hermite_prescribed_tails(K=5, Z=2.86)` yields tail weights ≈ 0.011 each). Smolyak combinations then carry real probability mass into joint corners.

| Rule | #Nodes | Speedup | Arb max_t (729 scan) | mean \|Δα\*\| | max \|Δα\*\| | Pass(1e-3) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline tensor (3,600) | 3,600 | 1.0× | +6.6e-14 | 0 | 0 | 8/8 |
| Smolyak L=5 + 64 anchors | 713 | 5.0× | +1.1e-13 | 1.6e-2 | 8.8e-2 | 4/8 |
| **Lobatto-Smolyak L=4** | **1,145** | **3.1×** | **+3.3e-14** | **4.3e-3** | **3.3e-2** | **7/8** |
| Lobatto-Smolyak L=5 | 2,777 | 1.3× | +2.1e-14 | 4.4e-3 | 3.3e-2 | 7/8 |

**Best practical candidate**: Lobatto-Smolyak L=4. Arbitrage-clean, 3.1× speedup, max policy bias 0.033 (≈ 0.5% of the leverage cap). 7/8 cells pass strict 1e-3.

L=5 doesn't improve over L=4 — diminishing returns, no point in 1,632 extra nodes.

### 2.3 Per-axis schedule that worked (numba branch, 3+3 geometry)

```
axis  channel   kind     Z       why
z_v0  cy        gh       —       M[xb, cy] ≈ -0.005 (weak; no benefit from Lobatto)
z_v1  spr       lobatto  2.93    M[xb, spr] ≈ -8.51 (dominant)
z_v2  y_1       lobatto  2.93    M[xb, y_1] ≈ -8.72 (dominant)
z_r0  rtb       gh       —       small residual variance (return-block)
z_r1  xr        lobatto  2.86    matches GH-5 max tail
z_r2  xb        lobatto  2.86    matches GH-5 max tail
```

Z = 2.93 on state axes matches the canonical `state_n_stds=2.93`. Z = 2.86 on return axes matches the GH-5 max-tail node position. Both produce arbitrage-free clouds at machine-epsilon level.

### 2.4 Architectural cost in the numba branch

Counted **9 separate `@njit` kernels** in `lifecycle/solver.py` and `lifecycle/inf_horizon_solver.py` that consume the nested `(k_v, k_r)` pattern. Each independently uses `v_nodes[k_v]`, `ret_nodes[k_r]`, `M_v_nodes[k_v]`. Smolyak couples (z_v, z_r) into a single 6-D node, breaking the independence — so each kernel needs adaptation. Verdict: "moderate refactor, not easy" (3-5 days implementation + 1-2 days validation).

---

## 3. Adaptations required for the JAX branch

The JAX branch has **4D state** `(cy, spr, rtb, y_1)` × **2D return** `(xr, xb)` per [docs/handoff/IMPLEMENTATION_HANDOFF_rtb_as_state.md](docs/handoff/IMPLEMENTATION_HANDOFF_rtb_as_state.md). Total joint dimension is still 6, so the conclusions of the numba investigation should largely transfer. But several details must change.

### 3.1 Quadrature dimensions

| Block | numba branch | JAX branch |
|---|---|---|
| State innovation | 3D `(cy, spr, y_1)` | **4D `(cy, spr, rtb, y_1)`** |
| Return residual | 3D `(rtb, xr, xb)` | **2D `(xr, xb)`** |
| Total joint | 6D | **6D (unchanged)** |
| `n_state_quad_nodes` | length-3 tuple | **length-4 tuple** |
| `n_ret_nodes_1d` | length-3 tuple | **length-2 tuple** |
| Σ_ss shape | 3×3 | **4×4** |
| Σ_r_cond shape | 3×3 | **2×2** |
| M matrix shape (Σ_rs Σ_ss⁻¹) | 3×3 (return × state) | **2×4 (return × state)** |

### 3.2 Recommended per-axis Lobatto-Smolyak schedule for the JAX setup

Per the IMPLEMENTATION_HANDOFF rationale — `M[xb, y_1]` remains the dominant entry (~8.85) under the 4-state ordering, and y_1 still lives at the last state axis. Returns have lost the rtb axis (now a state) so the return block is now just `(xr, xb)`.

**Proposed assignment** (must verify empirically before adopting; numbers below are extrapolated from the numba branch):

```
6-D z order: (z_v0, z_v1, z_v2, z_v3, z_r0, z_r1)

axis  channel   kind          Z      why
z_v0  cy        gh            —      same as numba: low loading on bond
z_v1  spr       lobatto       2.93   moderate loading
z_v2  rtb       lobatto OR gh ?      NEW — needs empirical decision (see §3.3)
z_v3  y_1       lobatto       2.93   dominant loading via M[xb, y_1]
z_r0  xr        lobatto       2.86   matches GH max tail
z_r1  xb        lobatto       2.86   matches GH max tail
```

### 3.3 The rtb axis (NEW state axis) — open empirical question

The bill return rtb is now a state innovation (axis z_v2). Under the JAX rtb-as-state migration, log_R_bill is read directly from `state_{t+1}[rtb_index_in_state]`, not synthesized from a return residual. Implications for Smolyak:

- The realised log_R_bill is a **linear function** of the state innovation z_v (specifically the rtb component after Cholesky). No CCV exp/Itô nonlinearity on this channel.
- The integrand depends on rtb mostly through `r_p^CCV = log_R_bill + α_xr · x_xr + α_xb · x_xb + Itô/Jensen`. The rtb contribution is roughly linear in z_v through Cholesky, so the integrand has **lower** sensitivity to z_v[rtb] than to z_v[y_1] (which enters through M[xb, y_1] = −8.85 amplified by α_b).
- **Best guess**: GH on z_v2 (rtb) is sufficient — same logic as cy (low integrand sensitivity, no benefit from prescribed tails). But this should be validated.

**Required experiment in the JAX branch**: run the arbitrage check with both variants:
- A: rtb axis = GH (3 axes Lobatto: spr, y_1, xr, xb; 2 axes GH: cy, rtb)
- B: rtb axis = Lobatto (5 axes Lobatto: spr, rtb, y_1, xr, xb; 1 axis GH: cy)

If A passes the 729-cell arbitrage check, prefer A (smaller node count). If A admits any arbitrage, fall back to B.

### 3.4 Anchor count if pursuing the anchored-Smolyak (failed) variant

If the JAX investigation needs to reproduce the anchored-Smolyak result for completeness (likely will fail same as numba branch), the anchor count is **2^6 = 64** binary corners — same as numba. The 6-D dimensionality is unchanged; what changed is which 3 axes are state vs return.

The anchor positions per axis are the GH-K_max tail z values. With suggested K_max:

```
K_max_jax = (3, 5, 5, 5, 5, 5)    # cy at K=3; spr/rtb/y_1 at K=5; xr/xb at K=5
```

(The numba canonical was K_max = (3, 4, 4, 3, 5, 5) — the JAX version raises the spr/rtb/y_1 axes to K=5 because Lobatto requires K ∈ {3, 5, 7}, and K=4 is not allowed on Lobatto axes. This adds nodes vs numba but the JAX baseline tensor is also bigger because it's now 4D state.)

---

## 4. Numerical contracts the JAX port must satisfy

These are validation gates. Any candidate quadrature must hit ALL of them before being declared production-viable.

### 4.1 Arbitrage scan (gate)

Run `check_state_arbitrage` (or its JAX equivalent — port the routine from `scripts/_check_ccv_arbitrage.py:50` in the numba branch) on **all canonical state grid cells** (likely 9^4 = 6,561 if state_grid_sizes = (9, 9, 9, 9), or whatever the JAX canonical is).

Pass criterion: `max_t < 1e-12` across all cells. The full tensor product baseline reaches ~1e-13. Any candidate with `max_t > 1e-5` at any cell is structurally broken (not just numerical noise).

### 4.2 α\* root benchmark (gate)

For ≥ 8 stress cells (mix of bond-stress corners, body cells, and at least 2 deep-cy-tail cells where α_b approaches cap), find the FOC root α\* under both the candidate rule and the baseline tensor. Compare component-wise.

Pass criterion: `max |Δα\*| < 0.05` across all stress cells, with mean ≤ 0.01. The numba Lobatto-Smolyak L=4 hit max 0.033 / mean 0.004 — set this as the empirical target.

### 4.3 Moment-matching (sanity gate)

Verify that `sum(weights) = 1.0` exactly (probability measure). For a multivariate normal, also check that low-order moments are recovered to machine precision:

```
E[z_k]       = 0     for all k       (level-1 GH exact)
E[z_k z_j]   = δ_{kj}  for all (k, j)  (level-2+ GH/Lobatto exact)
E[z_k^2 z_j^2] = 1 + 2δ_{kj} - δ_{kj}  (level-2+ exact under all rules tested)
```

Any candidate that fails moment-matching to ~1e-10 is buggy regardless of node count.

### 4.4 End-to-end policy regression (final gate)

Solve the canonical 4-state lifecycle problem with both the candidate and the baseline tensor. Compare policy arrays `C_mat`, `S_mat`, `B_mat`.

Pass criterion: max relative difference < 1e-3 at all (age, z, state, wealth) cells in the body of the distribution; ≤ 1e-2 acceptable in the deep-tail region (low stationary probability).

---

## 5. Architectural assessment for JAX

### 5.1 Why JAX may be friendlier than numba for this swap

The numba branch's blocker was 9 separate `@njit` kernels each hardcoding the `(k_v, k_r)` nested loop. JAX doesn't have that pattern:

- **`lax.scan` over a single quadrature node index** is the natural shape. No nested scans needed for a flat 6-D Smolyak rule.
- **`vmap` over states** doesn't care about the inner quadrature structure as long as the per-cell function is dimension-flexible.
- **Static shapes** are more tractable when the quadrature node count is a compile-time constant — Smolyak's `n_smolyak` is a fixed integer once the rule is built.
- **No type-stability concerns** like numba's njit.

### 5.2 But JAX has its own concerns

- **JIT compilation cache**: changing quadrature size triggers a recompile. For research-mode flexibility, this is fine; for production, it's why the numba branch caches the 3,600-node kernel aggressively.
- **GPU memory layout**: a 6-D Smolyak rule with non-paired (z_v, z_r) breaks the existing nesting. The per-cell `M_v_nodes` precompute (currently `(n_state_quad, n_ret)`) becomes `(n_smolyak, n_ret)` — same memory, different access pattern, may impact GPU coalescing.
- **Per-cell amortization**: the numba branch reuses 8 trilinear-V corner indices and `M_v_nodes[k_v]` across n_ret_quad inner iterations. Smolyak destroys that nesting. Net wall-clock speedup is ~2.5–3× rather than the naive node-count ratio of 3.1×. This applies equally to JAX.

### 5.3 Suggested implementation path for JAX

**MVP shape** if you decide to pursue:

1. Add to `DiscretizationConfig` (in `configs/_canonical_jax.py` or equivalent):
   ```python
   quad_mode: str = "tensor"        # "tensor" | "lobatto_smolyak"
   smolyak_level: int = 4
   smolyak_lobatto_kinds: tuple[str, ...] = (
       "gh", "lobatto", "gh", "lobatto",  # state axes (cy, spr, rtb, y_1)
       "lobatto", "lobatto",              # return axes (xr, xb)
   )
   smolyak_lobatto_Z: tuple[float | None, ...] = (
       None, 2.93, None, 2.93,
       2.86, 2.86,
   )
   ```
2. In `precompute.py`, branch the quadrature build:
   - `if quad_mode == "tensor"`: existing path, builds `v_nodes, ret_nodes, v_weights, ret_weights, M_v_nodes` as separate arrays.
   - `elif quad_mode == "lobatto_smolyak"`: build a paired joint rule. Pack as either:
     - **Option A**: separate `v_nodes_joint (n_smolyak, n_state)` and `ret_nodes_joint (n_smolyak, n_ret)` with `v_weights_joint = smolyak_weights` and `ret_weights = ones(1)`. Solver kernels read `ret_nodes[k_v]` instead of `ret_nodes[k_r]` when in joint mode.
     - **Option B**: single `joint_z (n_smolyak, 6)` table with `joint_w (n_smolyak,)`. Solver does flat scan over n_smolyak and slices z internally.
   - Recommend **Option B** for JAX (cleaner with `lax.scan`).
3. In `solver.py`, add the joint-mode path to the FOC kernels. The hot loop becomes a single `lax.scan` over n_smolyak, with the integrand body unchanged. Bracketing is computed per-node (no longer amortized across return-quad inner iterations).
4. Validate against the four numerical contracts in §4.

### 5.4 If you do not pursue this

The handoff is here as a decision record. The investigation conclusion is:

> "Anisotropic Smolyak-Hermite, even with prescribed-tail Lobatto refinement, is a borderline win at best for this CCV problem. The integrand is heavy-tailed at near-cap leverage, the joint-extreme corners that Smolyak skips carry the dominant probability mass at high α, and the architectural cost of swapping into a 9-kernel solver (or its JAX equivalent) is non-trivial. The 3,600-node tensor product (or the JAX-equivalent canonical) is the right tool. **Cheaper alternatives to explore first**: reducing canonical K (e.g. (3,3,4) instead of (3,4,4)) for ~1.3× speedup at zero architectural cost; or profiling the wall-clock budget to confirm quadrature is actually the bottleneck before refactoring."

---

## 6. Concerns, pitfalls, and lessons learned

### 6.1 Don't trust the MGF benchmark

The first investigation script (`_investigate_smolyak_returnstate.py`) tested integration accuracy with `f(z) = exp(α · log_R_excess)` — the bare moment generating function. This is **misleading** because it omits the CRRA u'(W_next) factor. Including u' amplifies tail dependency dramatically:

- MGF integrand effective z-loading at α=(6,6) ≈ 6× residual std-dev
- Full FOC integrand `f = u' · V · exp(r_p)` effective loading at α=(6,6) γ=5 ≈ 24× residual std-dev (factor (1−γ) = −4 amplifies)

**Lesson**: any future quadrature investigation must test against the actual production FOC kernel, not a stylized exponential.

### 6.2 Anchor enrichment is a support trick, not an integration trick

64 zero-weight binary corners closed the arbitrage failure but did NOTHING for integration accuracy. This was empirically clear from the FOC integrand benchmark. The intuition is simple: arbitrage depends only on whether nodes EXIST at certain corners; integration depends on whether nodes have WEIGHT at those corners.

**Lesson**: when a discretization attempt fails arbitrage but you patch it with anchors, run the integration benchmark before declaring victory.

### 6.3 Level error vs root error

The 70% relerr on the integrand level at α=(6,6) translated to only 8.8% relerr on the FOC root — because both Smolyak and baseline approximate the same heavy-tailed integral, and errors largely cancel in the FOC numerator. The user (Hugo) called this out correctly when I first dismissed Smolyak prematurely.

**Lesson**: production-relevance is the FOC ROOT, not the integrand level. Always run the α\* benchmark before final go/no-go.

### 6.4 The Lobatto K-constraint is annoying

`gauss_hermite_prescribed_tails(K, Z)` requires K ∈ {3, 5, 7} (in the numba branch's `lifecycle/quadrature_with_tails.py`). For Smolyak this means Lobatto axes step through K = 1, 3, 5, 7 at levels 1, 2, 3, 4 — unable to use K=2, 4. This skews the level-vs-K mapping vs GH axes (where K=i at level i). It works but limits granularity.

**Lesson**: if the JAX branch already has a more flexible Lobatto routine that accepts K=2, 4, 6, that's a real win. If not, accept the K ∈ {1, 3, 5, 7} constraint and design the level vector around it.

### 6.5 Deep-cy-tail cells are the bottleneck

Across ALL Smolyak variants tested, the worst-case |Δα\*| was at deep cy-tail cells (cy ≈ −4.5, far in the canonical n_stds=2.93 envelope's tail). These are the hardest cells for any quadrature reduction. The JAX 4-state geometry adds rtb to the state — whether deep-rtb-tail cells are similarly hard depends on the integrand sensitivity through M and the canonical n_stds for the rtb axis.

**Lesson**: stress-test your candidate at deep-tail cells, not just at the median state.

### 6.6 The 64 binary corners may not be enough at higher dimensions

The numba branch is 6-D with 2^6 = 64 corners. If the JAX branch ever extends to higher state dimensionality (e.g. 5-state for some extension), the 2^6 = 64 → 2^7 = 128 → 2^8 = 256 corners scales fast. For 6-D it's fine; check before assuming for higher d.

---

## 7. Inventory of investigation files (numba branch)

For each file, lists what it does, its key entry point, runtime, and the most important output.

### 7.1 `scripts/_investigate_smolyak_returnstate.py`
- **What**: Counts nodes for various Smolyak rules; tests polynomial-exactness on monomials up to degree 6; tests MGF integration accuracy at stress α.
- **Entry**: `python -m scripts._investigate_smolyak_returnstate` (no flags)
- **Runtime**: ~30 s
- **Key classes**: `SmolyakRule.build(L, w, K_max)` — Wasilkowski-Wozniakowski combination technique.
- **Output**: Node-count table (e.g. L=4 anisotropic = 286 signed nodes); polynomial-exactness max error; stress-integrand error table.

### 7.2 `scripts/_check_smolyak_arbitrage.py`
- **What**: Runs `check_state_arbitrage` (from `scripts/_check_ccv_arbitrage.py`) on Smolyak rules' unique-z node positions. Tests both unanchored and 64-anchor variants.
- **Entry**: `python -m scripts._check_smolyak_arbitrage [--n_states N]`
- **Runtime**: ~60 s for n_states=80; ~17 min for full 729-state scan
- **Key output**: max_t over scanned states; flagged-state count.
- **Conclusion**: 64 anchors close arbitrage at machine epsilon, state-independent.

### 7.3 `scripts/_benchmark_smolyak_foc.py`
- **What**: Per-cell FOC integrand level error with synthetic-but-realistic V_next (`V(s) = -W^{1-γ}/(1-γ) · exp(ρ·s)` with ρ = (0.3, 1.5, 1.2)) trilinearly-interpolated on the canonical state grid.
- **Entry**: `python -m scripts._benchmark_smolyak_foc [--n_cells N]`
- **Runtime**: ~30 s for 6 cells × 7 alphas
- **Key output**: relerr table at each (cell, alpha) combination.
- **Conclusion**: Anchored Smolyak fails badly at near-cap α despite being arbitrage-clean.

### 7.4 `scripts/_benchmark_smolyak_alpha_star.py`
- **What**: For each stress cell, finds the FOC root α\* under each rule via `scipy.optimize.root` (Powell hybrid) with multiple initial seeds and a box-constrained fallback. Compares α\* to baseline.
- **Entry**: `python -u -m scripts._benchmark_smolyak_alpha_star [--n_cells N]`
- **Runtime**: ~60 s for 8 cells × 3 rules
- **Key output**: per-cell α\* table and |Δα\*| summary.
- **Conclusion**: Smolyak L=5 + 64 anchors max |Δα\*| = 0.088, mean 0.016. 4/8 pass strict 1e-3.

### 7.5 `scripts/_benchmark_lobatto_smolyak.py`
- **What**: Builds Lobatto-Smolyak rules (per-level Lobatto univariate rules with prescribed tails and non-zero weights). Runs all three benchmarks (arbitrage, integrand, α\*) in one invocation.
- **Entry**: `python -u -m scripts._benchmark_lobatto_smolyak`
- **Runtime**: ~3 min total
- **Key output**: combined verdict table. **Lobatto-Smolyak L=4 is the best variant tested**.
- **Conclusion**: arbitrage-clean, 3.1× speedup, max |Δα\*| = 0.033 — borderline production-viable.

---

## 8. References and further reading

- Numba branch source of truth: `C:\Users\carlh\Projekt\thesisscripts\scripts\_investigate_smolyak_returnstate.py` and siblings.
- Project documentation referenced during the investigation:
  - `docs/notes/LOBATTO_CONFIG_TRACKER.md` — historical K-tuning record
  - `docs/CCV_RETURNS.md` — CCV log-portfolio formula
  - `docs/handoff/IMPLEMENTATION_HANDOFF_CVC_RETURNS.md` — wealth dynamics specification
- External: Wasilkowski & Wozniakowski (1995) "Explicit cost bounds of algorithms for multivariate tensor product problems" — the Smolyak combination technique used in the investigation scripts.
- External: Heiss & Winschel (2008) "Likelihood approximation by numerical integration on sparse grids" — Smolyak-Hermite background.
- External: Genz & Keister (1996) — nested Hermite rules for Smolyak (unused here; the numba investigation used non-nested GH for simplicity, accepting the larger total node count).

---

## 9. Decision recommendation

**Default**: do NOT port Lobatto-Smolyak to the JAX branch as a config option. The 3.1× speedup at 3% policy bias on deep-tail cells is not worth the ~1 week of implementation + validation.

**Optional**: port as a research-mode flag (`quad_mode="lobatto_smolyak"`) for ablation studies, with `quad_mode="tensor"` as the default. This makes "what would the policy look like under sparse-grid quadrature?" a one-line config change, which has scientific value for the thesis even if production never adopts it.

**Cheaper first**: profile the JAX solver to confirm quadrature is the wall-clock bottleneck (it might not be — JAX's `vmap` over states + EGM root-find may dominate). If quadrature isn't the bottleneck, no speedup work on quadrature pays off regardless.

**Conservative alternative**: reduce canonical K. The numba canonical is `n_state_quad_nodes=(3,4,4)` — try `(3,3,4)` for ~1.3× speedup at zero architectural cost. The JAX-canonical equivalent is `n_state_quad_nodes=(3,3,?,4)` with rtb at axis 2 — needs empirical testing of what minimum K on rtb is acceptable.

---

*Investigation conducted by Hugo + Claude (Opus 4.7) on the numba branch, May 2026. Source files preserved at `C:\Users\carlh\Projekt\thesisscripts\scripts\_*.py`. Reproduce by running each script standalone from the numba branch project root.*
