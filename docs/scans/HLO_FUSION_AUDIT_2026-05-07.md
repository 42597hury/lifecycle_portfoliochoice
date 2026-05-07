# HLO Fusion Audit — Per-Age Single-Device Kernels

**Branch:** `jax-rewrite`
**Date:** 2026-05-07
**Scope:** [`_build_per_age_terminal_kernel_vmap_only`](../../lifecycle/solver.py#L1719),
[`_build_per_age_retirement_kernel_vmap_only`](../../lifecycle/solver.py#L1897),
[`_build_per_age_working_kernel_vmap_only`](../../lifecycle/solver.py#L2119).
**Method:** lower each `@jit`'d `per_chunk` to StableHLO + post-XLA HLO at a tiny config; read structure.

## Verdict: GREEN

The FOC + Newton + backtracking + EGM stack compiles into the expected
fused HLO graph in all three kernels. No stray boundaries, no `jnp.array`
materializations, no gathers leaking into the Newton inner loop that
shouldn't be there.

One informational note (working kernel only): XLA rematerializes the
[`c_corners_T` transpose at solver.py:1338](../../lifecycle/solver.py#L1338)
six times across consumer regions instead of sharing a single copy. This
is XLA's layout-CSE choice, not a Python-side bug, and the cost is bounded
(a few µs of redundant transpose per per-cell solve at full scale). Not
worth a fix at this layer.

---

## Method

Audit script: [`scripts/scratch/audit_hlo_fusion.py`](../../scripts/scratch/audit_hlo_fusion.py)
(reuses the closure-walk + `lower().compile().as_text()` pattern from
[`verify/mixed_precision_tiny.py`](../../verify/mixed_precision_tiny.py)).

Tiny config (matches `verify/smoke.py` scale):
- `state_grid_sizes=(2, 3, 2, 3)` → `n_state=4`, `N_state=36`
- `n_z=3`, `n_w=12`, `n_savings=12`
- `n_state_quad_nodes=(2, 3, 2, 3)` → 36 state-quad nodes
- `n_ret_nodes_1d=(2, 2)` → 4 return-quad nodes
- `n_eta=2`, `n_eps=2` (working only)
- `max_iter=30`, `gather_precision='f64'`, `use_fori_newton=True`

Dumps:
- `docs/scans/hlo_dumps/{terminal,retirement,working}_kernel.stablehlo.txt`
  — pre-XLA, what JAX submitted.
- `docs/scans/hlo_dumps/{terminal,retirement,working}_kernel.hlo.txt`
  — post-XLA, what XLA actually compiled. (Single-device CPU; only
  `kind=kLoop` fusions emitted.)

---

## Per-kernel structure summary

| kernel     | size     | fusions | while | reduce | copy | transpose | bitcast | convert | gather |
|------------|---------:|--------:|------:|-------:|-----:|----------:|--------:|--------:|-------:|
| terminal   |  324 KB  |     151 |     3 |     21 |   11 |         0 |      54 |      11 |     13 |
| retirement |  619 KB  |     219 |     7 |     28 |   35 |         0 |     189 |      57 |     44 |
| working    |  874 KB  |     276 |     7 |     47 |   42 |         6 |     260 |      72 |     53 |

(Counts from `grep -cE` against the post-XLA HLO. Bitcasts are
zero-cost view ops; copies break SSA/aliasing for `lax.while_loop`
state. None are HBM→HBM moves of payload data.)

---

## while-loop inventory

The handoff predicted the per-cell solve should compile to ONE big
`while` for Newton, ONE for backtracking, plus `searchsorted` whiles.
Confirmed:

### terminal (3 whiles)
| op_name | role |
|---|---|
| `vmap(vmap())/while` | **Newton outer** ([solver.py:684](../../lifecycle/solver.py#L684)) |
| `vmap(vmap())/while/body/closed_call/while` | **Backtracking inner** ([solver.py:564](../../lifecycle/solver.py#L564)) |
| `vmap(jit(_interp))/jit(searchsorted)/vmap()/while` | EGM-lift `jnp.interp` searchsorted (post-Newton) |

### retirement (7 whiles)
Adds:
- Two `vmap(vmap(vmap(vmap(jit(searchsorted)))))/while` — state-bracket
  binary searches in `_build_step_state_brackets` (per-cell setup,
  outside Newton; trip count 4 = log₂(state-axis size)).
- Two `vmap(vmap())/while/body/closed_call/.../searchsorted/while` —
  wealth-grid searchsorted INSIDE the Newton body and the backtracking
  body (necessary: `iw` depends on `x_next`, which depends on the
  Newton iterate via `s_val * R_p + pension`).

### working (7 whiles)
Same shape as retirement; the searchsorted whiles pick up two extra
vmap layers because of the `(eta, eps)` integration.

`max_iter` is non-static at lowering time (it's threaded through the
SolverConfig static-arg cache key, but XLA still keeps it as a `while`
rather than unrolling — confirmed by `known_trip_count` metadata being
absent on the Newton/backtracking whiles, present only on the
fixed-trip searchsorted ones). This is the desired behaviour:
unrolling 30 Newton iters into 30 fusions would dominate compile time
and HLO size.

---

## FOC arithmetic + reductions: actually fused

Spot-check on the retirement Newton-body reductions:

```
%multiply_multiply_fusion.2 = f64[108,12,36,4] fusion(...)   <- (n_cells, n_savings, n_kv, n_kr) tensor
                                                                of e.g. wmu * R_p * dr_da_s * dr_da_b
%wrapped_reduce-window.14 = f64[108,12,2,1]    fusion(%multiply_multiply_fusion.2, ...)
%wrapped_reduce.10        = f64[108,12]        fusion(%wrapped_reduce-window.14, ...)
   metadata={op_name=".../while/body/closed_call/reduce_sum"}
```

XLA fuses the multiply chain into one fusion that produces the
unreduced `(kv, kr)` tensor, then splits the `jnp.sum` into a
`reduce-window` + final `reduce` to fit the CPU L2 tile. The unreduced
tensor is **not** materialized to HBM — it lives in the loop body of
`%wrapped_reduce-window`.

Fusion-name prefix histogram (top 10, retirement) confirms the right
patterns are firing:

```
19 bitcast_gather_fusion    11 multiply_add_fusion       11 compare_select_fusion
 9 or_select_fusion          9 multiply_divide_fusion     8 reduce_select_fusion
 8 multiply_multiply_fusion  7 pad_gather_fusion          6 multiply_reduce_fusion
 6 copy_gather_fusion        ...
```

`multiply_reduce_fusion` (6×) and `multiply_add_fusion` (11×) are the
ones we cared about — the FOC `wmu * dRp_das + extra` chains and the
`jnp.sum(wmu * ...)` reductions both fuse with their producers.

---

## Gather hoisting: clean

The handoff flagged "repeated gathers inside the Newton loop" as a
potential perf bug (the `c_corners` gather should be hoisted to
per-cell, not per-Newton-iter). Confirmed clean:

### retirement, by location
| where | count | what |
|---|---:|---|
| OUTSIDE any while | 26 | per-cell setup: `c_next[z_idx, j_corners_i, :]` ([1271](../../lifecycle/solver.py#L1271)), `log_R_bill_all[i_s]`, `j_corners_all[i_s]`, `psi_per_z[z_idx]`, etc., plus `_lift_to_wealth_grid` `jnp.interp` gathers |
| INSIDE Newton body (not backtracking) | 4 | wealth-grid `c_kv[:, iw]` reads inside `per_kv_kr` (necessary — `iw` depends on the Newton iterate); searchsorted's binary-search gathers |
| INSIDE backtracking body | 4 | same pattern, repeated for the backtracked `α`-step `foc_fn` call (necessary) |

`c_corners_at_z = c_next[z_idx, j_corners_i, :]` is hoisted out of the
Newton loop ([solver.py:1271](../../lifecycle/solver.py#L1271)). Confirmed.

### working, by location
Same shape, plus the `c_corners_T` transpose at
[solver.py:1338](../../lifecycle/solver.py#L1338) which is what produces
the 6 transposes flagged in the summary table — see next section.

### terminal
Zero gathers inside the Newton or backtracking body, as expected
(terminal FOC has no wealth-grid interp; `x_next = sR_p` only).

---

## One borderline observation: `c_corners_T` transpose rematerialization (working kernel)

`c_corners_T = jnp.transpose(c_corners, (1, 0, 2, 3))` at
[solver.py:1338](../../lifecycle/solver.py#L1338) appears 6 times in
the working HLO:

```
%transpose.3 = f64[108,36,3,16,12]{4,2,3,1,0} transpose(%bitcast.246), dimensions={0,1,3,2,4}
%copy.25     = f64[108,36,3,16,12]{4,3,2,1,0} copy(%transpose.3)
... (consumed by gather.366 .. gather.369) ...

%transpose.4 = f64[108,36,3,16,12]{4,2,3,1,0} transpose(%bitcast.324), dimensions={0,1,3,2,4}
%copy.28     = f64[108,36,3,16,12]{4,3,2,1,0} copy(%transpose.4)
... (consumed by gather.372) ...

(transpose.5 .. transpose.8, similar pattern)
```

Each transpose is followed by a layout-copy and consumed by 1–4 gather
ops. XLA is rematerializing the same transpose into each consumer's
sub-region rather than maintaining a single shared buffer. On CPU this
is a cache-locality choice (better to recompute the transpose locally
than read it from HBM); on GPU it would be a few µs of redundant work
per per-cell solve.

**Verdict: not a bug, not a clear win to fix.** Consumers are inside
the alive-contribution chain (`vmap(per_kv)(c_corners_T, w_corners,
x_next)` in `working_foc_jac_ccv`), and the per-cell working memory at
full scale is already large — XLA's choice to rematerialize avoids
keeping (n_cells, n_state_quad, n_z, n_corners, n_w) live across the
whole alive-contribution computation. Could be revisited with
`jax.lax.optimization_barrier` if a future GPU profile shows
transpose-time as a hot spot, but no signal to act on now.

---

## Things explicitly checked and ruled out

- **No stray `jnp.array(...)` materializations inside hot-path bodies.**
  Searched HLO for unattached `constant.NN` allocations inside Newton
  bodies; none found.
- **No D→H copies.** No `host-transfer` or `send`/`recv` ops anywhere.
- **No standalone reduction kernels for the FOC sums.** All
  `jnp.sum(wmu * dRp_das)` etc. fuse into `multiply_reduce_fusion` /
  `multiply_multiply_fusion` chains with their producing multiplies
  (CPU XLA splits the final reduce into reduce-window + reduce, but
  the multiply-then-sum chain is one fusion).
- **No per-Newton-iter materialization of `(n_state_quad, n_ret_quad)`
  tensors.** The intermediate `(108, 12, 36, 4)` tensors live inside
  fusion bodies, not in HBM. The only intermediate that does land in
  HBM is the `(108, 12, 2, 1)` partial-reduce tile from XLA's
  reduce-split, which is a 72× shrink.
- **`bitcast` and `copy` counts are layout/SSA bookkeeping, not data
  movement.** None carry payload-sized data between unfused producers
  and consumers.

---

## Where the audit is silent

This was HLO inspection on **CPU XLA**, single-device, at tiny config.
Not covered:

- **GPU XLA fusion can differ.** GPU emits `kInput`/`kOutput` fusions
  (reduction-style with shared memory) where CPU only emits `kLoop`.
  The CPU HLO is a faithful proxy for *whether the JAX trace structure
  permits fusion*; it does not predict the absolute number of GPU
  kernel launches, which the GPU regression run should measure
  directly via NSight or `--xla_dump_to`.
- **Pmap kernels not inspected.** Per the handoff scope.
- **Working-kernel `use_pension_next=True` (work→retire boundary)
  not separately inspected** — same trace structure as
  `use_pension_next=False` modulo a small per-cell `pension_at_eta`
  computation; likely identical fusion picture.

---

## Closing

Inner-loop fusion structure is what we expected after the recent
hot-path tuning. The `_ccv_log_return_and_grad` + FOC arithmetic + 2D
Newton + backtracking + EGM lift are not leaving table-stakes wall
time on the floor at the fusion-structure level. Optimisation
headroom in this layer is now bounded by what XLA can in principle do,
not by any unfused boundary in the Python source.

**No follow-up handoff recommended at this layer.** Next perf work
(if any) is downstream: GPU profile to confirm the same picture
on-device, or move up the stack to outer-loop / inter-age scheduling.
