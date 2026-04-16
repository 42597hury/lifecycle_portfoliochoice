# HANDOFF: On-the-Fly Income — Runtime Complexity Estimate

**To:** compute-complexity agent
**From:** labour-income validation session (2026-04-16)
**Scope:** Estimate the **percentage increase in solver wall-time** if
the solver replaces its current z-interpolated income lookup with an
on-the-fly scalar computation of disposable income. Do **not** implement
the change. Do **not** run the full solver to benchmark. Produce an
analytic / profile-informed complexity estimate with a single % figure
and a confidence range.

---

## 1. Why the change is being considered

The z-grid has `n_z = 11` points covering `[-5.6, +5.6]` in log-income
with `dz ≈ 1.122`. The solver's continuation step computes
`z_next = ρ·z_i + η_node` (continuous, rarely on the grid) and
**linearly interpolates** income from a precomputed table:

```python
# solver.py:853-854
income_next = (1 - frac_z) * income_table[iz_lo,   ie]
            +      frac_z  * income_table[iz_lo+1, ie]
```

Measured (23,100 off-grid points): median relative error ≈ 14%, max
≈ 16.6%. Root cause is analytic — income is approximately exponential
in z, so a chord between two grid points systematically overshoots the
convex curve. Closed form: `(interp - true)/true = cosh(dz/2) - 1 =
16.05%` at `dz = 1.122`, matching the observation.

This error is almost certainly the same phenomenon as the ~17%
solver-simulation gap flagged in `issues.md`, since the simulation
evaluates income directly from continuous z.

Consumption policy `c_next_full` is also z-interpolated but will be
left in place for now (a separate refactor may replace it with a
polynomial basis à la Cocco–Gomes–Maenhout 2006). This handoff is
specifically about the **income** lookup.

---

## 2. The proposed code change (not to be implemented)

Replace the two lines above with:

```python
y_gross_next = exp(log_det_profile[t+1] + z_next + eps_nodes[ie])
income_next  = disposable_income_working_scalar(y_gross_next)
```

A scalar version of the tax/payroll function already exists as
`_scalar_disposable_income` in [simulation.py](simulation.py) — see
item "Relevant code locations" below.

This eliminates the interpolation error entirely for the income
component (policy interpolation error remains separately).

---

## 3. What the work per lookup becomes

**Current lookup (linear interp):**
- 2 array reads `income_table[iz_lo, ie]`, `income_table[iz_lo+1, ie]`
- 1 fused multiply-add: `(1-frac_z)*a + frac_z*b`
- ≈ 2–3 elementary ops, ~3 ns on commodity x86

**Proposed lookup (on-the-fly):**
- 1 `exp()` call
- 1 tax-bracket walk: piecewise linear 7-bracket schedule (income tax)
  preceded by a payroll step (`0.106 * min(y, 2.5)`). See [model.py:268](model.py#L268).
- ≈ 7–8 branch compares + 3–5 FMAs + 1 `exp()` + (rarely) a 2nd branch
  evaluation for very high income
- ~30–50 ns unoptimized; faster under Numba / ahead-of-time inlining

**So per income lookup: ~10× arithmetic work.** Whether that
propagates to ~10% total wall-time or something very different depends
entirely on how much of solver runtime is spent in this lookup vs
everywhere else.

---

## 4. Solver nested-loop structure (context for the estimate)

The solver iterates, roughly (see [solver.py](solver.py)):

```
for age t in working ages (45 ages):                        # outer
    for wealth point iw in wealth grid (n_w = 150):
        for joint state s (n_state_grid = 7×7×7 = 343):
            for savings point j_s in savings grid (n_s = 150):   # EGM
                Newton iterations (≤ 20 for constrained run):
                    compute FOC and Jacobian:
                        for k_eta in η nodes (2·n_eta_nodes = 10):
                            for i_e in ε nodes (2·n_eps_nodes = 10):
                                ... income lookup HERE ...
                                + c_next bilinear interp
                                + CRRA marg util + derivative
                                + accumulate into foc_s, foc_b,
                                  J_ss, J_bb, J_sb, euler_sum
```

Per FOC evaluation: **100 inner-node iterations** (η × ε). Each does
one income lookup (current: table read + blend; proposed:
exp + bracket walk).

For the saved run (`constrained_grid7x7x7_nz11`): `n_eta = 10` (K_eta=5
per comp), `n_eps = 10` (K_eps=5 per comp). At config defaults `n_eta
= n_eps = 6` (K=3 per comp), so 36 inner nodes instead of 100. The
runtime-per-FOC scales ~linearly in the inner-node count, and the
income-lookup share scales with it too.

---

## 5. Other per-node work (denominator of the % estimate)

For each (η, ε) inner-loop iteration, beyond the income lookup, the
solver also does (from solver.py:849–885):

- 1 wealth bracket search via `find_bracket` (binary search in a
  150-point wealth grid — ~7–8 compares, ~10 ns)
- 4 reads from `c_next_full[iz, j_s, iw]` (the 2×2 corners for
  bilinear interp in (z, wealth))
- 4 multiplies + 2 adds for the bilinear blend of `c_next`
- 1 FMA each for `mpc_lo`, `mpc_hi`, then 2 more for the blend
- 1 branch clamp on `c_next` and `mpc`
- CRRA marginal utility: `c_next ** (-gamma)` (1 power)
- `mup_alive` derivative term: 4 ops
- 6 weighted-sum accumulators: 6 FMAs
- 3 Jacobian accumulators: 3 mults + 3 FMAs

Rough flop count per inner iter (excluding income): ~40–60 ops, ~60–80
ns unoptimized. **This is the denominator you'd compare income-lookup
cost against.**

---

## 6. Is the solver JIT-compiled?

Need to check. Standard patterns in this codebase:
- If `compute_foc_jac_working` is decorated with `@numba.njit` (or
  called inside a njit function), then both current lookup and the
  proposed on-the-fly call get aggressively inlined, and `exp` lowers
  to an SIMD intrinsic. Overhead likely <10%.
- If it's pure Python/Numpy, the scalar function call overhead for
  `_scalar_disposable_income` would dominate on the proposed path
  (Python function-call overhead alone ≈ 100 ns).

Please **check the solver code** for `@njit` / `numba.jit` /
`numba.jitclass` decorators and base the estimate on what's actually
in use. If the code is JITed, also check whether the scalar income
function is JITable (i.e. branch-heavy piecewise code is fine in
Numba; numpy masking-style code is not).

---

## 7. Relevant code locations

- **Current interpolation:** [solver.py:849-855](solver.py#L849-L855)
  — inner loop containing the income-lookup line.
- **Income function (vectorized, used by precompute):**
  `disposable_income_working` at
  [model.py:268](model.py#L268) — 7-bracket piecewise schedule + payroll.
- **Income function (scalar, already written for simulation):**
  `_scalar_disposable_income` in [simulation.py](simulation.py) —
  same formula, scalar inputs, Python if-elif branches. This is the
  function that would be called on-the-fly.
- **Precompute (would be bypassed):**
  `_precompute_working_income` at
  [precompute.py:366](precompute.py#L366) — builds `(n_age, n_z,
  n_eps)` table via broadcasting + vectorized disposable-income call.
  Could be kept for simulation use or removed; table is only consumed
  by the solver's interpolation path.

---

## 8. What we want back

A single numeric estimate:

> **Switching the solver from table+interp lookup to on-the-fly scalar
> income increases solver wall-time by X% ± Y.**

And a short (≤ 200 word) justification, covering:

1. Per-node cost delta (current vs proposed, in ns or flops).
2. Inner-node share of total FOC cost (% of per-iteration work spent
   on income lookup today).
3. Whether the solver is JIT-compiled and how that changes the
   estimate.
4. Whether ` _scalar_disposable_income` is suitable for being called
   inside the hot loop (Numba-compatible, no Python-level overhead,
   etc.) or whether a small rewrite is needed for the comparison to be
   fair.
5. Any caveat about `exp()` throughput on the target hardware.

An analytic estimate is fine — you do **not** need to run the full
solver. If micro-benchmarking is useful (e.g. time a million calls to
each function), that's helpful but not required.

---

## 9. What NOT to do

- Do NOT implement the change.
- Do NOT propose code changes to the solver's logic beyond the
  one-line replacement described in §2.
- Do NOT run the full solver (expensive; several minutes).
- Do NOT re-audit income or the tax schedule — already validated;
  see [LABOUR.md Section 5](LABOUR.md#L336).
- Do NOT touch consumption-policy interpolation — that's a separate
  refactor (likely polynomial basis, à la Cocco–Gomes–Maenhout 2006).

---

## 10. User preferences (relayed)

- Terse; no trailing summaries
- Markdown code references as `[file.py:42](file.py#L42)`
- Thesis deadline 2026-05-18; this is a cost estimate to inform a
  refactor decision, not research
