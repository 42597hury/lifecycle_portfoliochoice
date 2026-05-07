# Handoff: Restore CCV Log-Wealth Dynamics in `simulate_lifecycle`

**Branch:** `jax-rewrite`
**Status when this doc was written:** [lifecycle/simulation.py](../../lifecycle/simulation.py) hardwires the wealth law to **arithmetic returns**:

```python
# simulation.py:354-359 (current)
a_bill_t = 1.0 - a_s_t - a_b_t
R_port = a_s_t * R_stock + a_b_t * R_bond + a_bill_t * R_bill
estate_t = savings_t * R_port
```

This contradicts the solver, which uses CCV log-wealth dynamics
([_ccv_log_return_and_grad](../../lifecycle/solver.py#L657)). The `wealth_dynamics_spec`
kwarg ([simulation.py:496](../../lifecycle/simulation.py#L496)) is accepted "for compat"
but silently ignored. The file-level docstring ([simulation.py:21-25](../../lifecycle/simulation.py#L21-L25))
claims the divergence is "INTENTIONAL per handoff-3 spec" — there is no such
authorising handoff in `docs/handoff/` or `docs/archive/`, and the project's
own design contract in [docs/CCV_RETURNS.md](../CCV_RETURNS.md) explicitly
mandates parity:

> "Consistency between solver and simulator is mission-critical: if they
> disagree on R_p at any quadrature node, every Euler-residual diagnostic
> becomes meaningless." — `CCV_RETURNS.md` §2.3

**Target:** restore the CCV branch as the simulator's wealth law and
honour `wealth_dynamics_spec` so callers cannot silently get the wrong
law. Simulator and solver must compute `R_p` identically at every
realisation.

**Out of scope:** changing solver math, changing diagnostics' deliberate
arithmetic-vs-CCV comparator path
([diagnostics.py:1198-1200](../../lifecycle/diagnostics.py#L1198-L1200)),
or touching the rtb-as-state machinery already in place.

---

## 1. Why this matters

The solver chose `(c*, α_s*, α_b*)` under the assumption that
`x_{t+1} = s · exp(r_p^CCV) + π`. If the simulator advances wealth by
`x_{t+1} = s · arithmetic_R_p + π` instead:

1. **The simulated agent is solving a different problem than was optimised.**
   The optimum `(c*, α_s*, α_b*)` is not the optimum of the wealth law
   the simulator advances under. Any policy-evaluation moment is
   measuring the policy on the wrong dynamic.

2. **Euler-residual diagnostics are biased by the convexity wedge.**
   `arithmetic_R_p − exp(r_p^CCV)` is the Jensen-minus-Itô residual of the
   Taylor expansion. It scales with `α'·Σ·α`. At the calibrated
   `Sigma_rr` values, the wedge is on the order of tens to hundreds of bps
   per period and grows quadratically in α — it dwarfs the numerical
   FOC residual at leveraged α.

3. **Welfare and moment statements computed off simulated paths are
   mis-measured by the same wedge,** compounding over the lifecycle
   (~33 retirement years on top of the working life).

The diagnostics module in [diagnostics.py:1198-1200](../../lifecycle/diagnostics.py#L1198-L1200)
deliberately uses arithmetic returns as a **ground-truth comparator**
against the CCV objective — that is the *correct* place for arithmetic
returns to live, and that path stays untouched. The simulator is the
wrong place.

---

## 2. The math (what the patch must compute)

CCV w8566 eq. (10), already implemented in
[solver.py:657-679](../../lifecycle/solver.py#L657-L679):

```
r_p^CCV = log_R_bill
        + α_s · log_x_s + α_b · log_x_b
        + ½ · (α_s · σ²_xr + α_b · σ²_xb)                      ← Jensen lift
        − ½ · (α_s² · σ²_xr + 2 · α_s · α_b · σ_xrxb
               + α_b² · σ²_xb)                                  ← Itô vol-drag

R_p^CCV = exp(r_p^CCV)
```

Symbols (matching solver naming so reviewers can diff line-for-line):
- `log_R_bill = rtb_{t+1}` — read from `s_next[rtb_idx]` (already done in
  the simulator post the rtb-as-state migration; see line 334).
- `log_x_s = log(R_stock / R_bill)` — log excess equity return realisation.
- `log_x_b = log(R_bond / R_bill)` — log excess bond return realisation.
- `σ²_xr, σ²_xb, σ_xrxb` — entries of the unconditional return-block
  covariance `Sigma_rr`, already computed and stored in the Precompute
  pytree ([precompute.py:312-314](../../lifecycle/precompute.py#L312-L314),
  fields `pc.sigma2_xr`, `pc.sigma2_xb`, `pc.sigma_xrxb`).

The σ² choice (`Sigma_rr`, not `Sigma_r_cond`) is the same one the solver
uses; do not introduce a different source. The audit-trail rationale lives
at [precompute.py:303-311](../../lifecycle/precompute.py#L303-L311).

---

## 3. Implementation

### 3.1 Validate `wealth_dynamics_spec`

[simulation.py:483-498](../../lifecycle/simulation.py#L483-L498). The kwarg
default is already `"ccv_log"`. The simulator should accept `"ccv_log"`
only — `"simple_clamp"` was removed from the JAX solver, and arithmetic
is not a real choice for production simulation.

Replace the current "accepted for compat; ignored" semantics with an
explicit assert at the top of `simulate_lifecycle`:

```python
if wealth_dynamics_spec != "ccv_log":
    raise ValueError(
        f"wealth_dynamics_spec must be 'ccv_log' (got {wealth_dynamics_spec!r}); "
        "the JAX simulator only supports CCV log-wealth dynamics, matching the "
        "solver. The 'simple_clamp' branch was removed in the JAX rewrite."
    )
```

Update the docstring at [simulation.py:516-517](../../lifecycle/simulation.py#L516-L517):

```python
wealth_dynamics_spec : "ccv_log" only. Simulator computes R_p with
    the same CCV log-portfolio formula the solver uses (see solver.py
    _ccv_log_return_and_grad). Anything else raises ValueError.
```

### 3.2 Fix the file-level docstring

[simulation.py:21-31](../../lifecycle/simulation.py#L21-L31) is misleading.
Replace with:

```python
Wealth-dynamics in the simulated economy:
    CCV log-portfolio return — matches the solver's CCV w8566 eq. (10)
    implementation in solver._ccv_log_return_and_grad. Solver and
    simulator MUST compute R_p identically at every realisation,
    otherwise every Euler-residual and policy-evaluation diagnostic
    is biased by the Jensen-minus-Itô wedge (see docs/CCV_RETURNS.md).

Bankruptcy: NO clamp on estate_t (under CCV, R_p = exp(r_p^CCV) > 0,
    so s · R_p > 0 whenever s > 0; the clamp is unnecessary).
```

The "intentionally divergent / handoff-3 / don't harmonise" lines are
deleted.

### 3.3 Thread σ scalars through the kernel

[`_build_simulate_kernel`](../../lifecycle/simulation.py#L242) needs three
new closure values. Add to the signature:

```python
def _build_simulate_kernel(
    ...
    rho, pz, mu_eta1, sigma_eta1, sigma_eta2, mu_eta2_eff,
    pe, mu_eps1, sigma_eps1, sigma_eps2, mu_eps2_eff,
    n_age, n_z, n_ret, n_state,
    rtb_idx, xr_pos, xb_pos,
    sigma2_xr, sigma2_xb, sigma_xrxb,    # NEW
    use_mc_returns,
):
```

And in the kernel-build call inside `simulate_lifecycle`
([simulation.py:664-702](../../lifecycle/simulation.py#L664-L702)) pass them
from `pc`:

```python
kernel = _build_simulate_kernel(
    ...
    rtb_idx=rtb_idx, xr_pos=xr_pos, xb_pos=xb_pos,
    sigma2_xr=jnp.float64(pc.sigma2_xr),       # NEW
    sigma2_xb=jnp.float64(pc.sigma2_xb),       # NEW
    sigma_xrxb=jnp.float64(pc.sigma_xrxb),     # NEW
    use_mc_returns=use_mc_returns,
)
```

### 3.4 Replace the wealth-law block

In `step_fn`, replace [simulation.py:350-359](../../lifecycle/simulation.py#L350-L359):

```python
# --- BEFORE ---
R_bill = jnp.exp(log_R_bill)
R_stock = R_bill * jnp.exp(log_x_s)
R_bond = R_bill * jnp.exp(log_x_b)

a_bill_t = 1.0 - a_s_t - a_b_t
R_port = a_s_t * R_stock + a_b_t * R_bond + a_bill_t * R_bill
# No clamp on estate — the simulator reflects the truth, including
# rare catastrophic realisations under uncapped leverage. The
# offgrid diagnostic surfaces these post-hoc.
estate_t = savings_t * R_port
```

with:

```python
# --- AFTER ---
# CCV log portfolio return (Campbell-Viceira w8566 eq. 10).
# Must agree node-by-node with solver._ccv_log_return_and_grad —
# verified in tests/test_ccv_solver_sim_parity.py.
log_R_port = (
    log_R_bill
    + a_s_t * log_x_s + a_b_t * log_x_b
    + 0.5 * (a_s_t * sigma2_xr + a_b_t * sigma2_xb)
    - 0.5 * (
        a_s_t * a_s_t * sigma2_xr
        + 2.0 * a_s_t * a_b_t * sigma_xrxb
        + a_b_t * a_b_t * sigma2_xb
    )
)
R_port = jnp.exp(log_R_port)
# Under CCV: R_port = exp(r_p) > 0 by construction, so s · R_port > 0
# whenever s > 0. No clamp needed.
estate_t = savings_t * R_port
```

`log_R_bill`, `log_x_s`, `log_x_b` are already in scope from
[simulation.py:334, 343-348](../../lifecycle/simulation.py#L334-L348).
`R_stock` and `R_bond` were only computed for the arithmetic combination
and become unused — delete those two lines as part of the patch (the
upstream `R_bill = jnp.exp(log_R_bill)` is also unused now; delete it
too, or keep only if a downstream output references it — confirm by
grepping `R_bill` inside the kernel after editing).

### 3.5 Output panel

The panel keys at [simulation.py:408-422](../../lifecycle/simulation.py#L408-L422)
include `"R_port"`. Continue exporting `R_port` (now CCV-derived) — the
field semantics (gross portfolio return) are unchanged; only the
formula changed. No downstream code that reads `panel["R_port"]` needs
to know.

### 3.6 The `_wealth_offgrid_diagnostics` `negative_frac` field

[simulation.py:447-480](../../lifecycle/simulation.py#L447-L480) reports the
share of households with `x_t < 0`. Under CCV, `s · R_port > 0` whenever
`s > 0`, and `x_{t+1} = s · R_port + π_{t+1}` with `π ≥ 0` — so
`x_{t+1} ≥ 0` always. The `negative_frac` output should be ≈ 0 after
the fix; if it's not, something else is wrong (e.g. a stale policy
saved from the arithmetic-era simulator). Keep the field for
diagnostic/regression value, but note in its docstring that under CCV
it is expected to be 0.

---

## 4. Verification

### 4.1 Parity unit test (REQUIRED)

Add a test that fixes the solver and simulator return formulas to give
the same `R_p` at the same `(α_s, α_b, log_R_bill, log_x_s, log_x_b, σ)`.

Path: `verify/ccv_solver_sim_parity.py` (top level, matching the
existing `verify_*.py` style — there is no `tests/` directory in this
branch), or under `scripts/diagnostics/_diag_ccv_parity.py` if that's
where the team wants it. Suggested content:

```python
"""Verify solver and simulator agree on R_p at every realisation."""
import numpy as np
import jax.numpy as jnp
from lifecycle.solver import _ccv_log_return_and_grad

def simulator_R_p(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
                  sigma2_xr, sigma2_xb, sigma_xrxb):
    log_R_port = (
        log_R_bill
        + alpha_s * log_x_s + alpha_b * log_x_b
        + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
        - 0.5 * (alpha_s * alpha_s * sigma2_xr
                 + 2.0 * alpha_s * alpha_b * sigma_xrxb
                 + alpha_b * alpha_b * sigma2_xb)
    )
    return jnp.exp(log_R_port)

rng = np.random.default_rng(0)
for _ in range(1000):
    a_s = rng.uniform(-2.0, 3.0)
    a_b = rng.uniform(-2.0, 3.0)
    rb = rng.normal(0.01, 0.005)
    xs = rng.normal(0.05, 0.18)
    xb = rng.normal(0.02, 0.06)
    s2_s = rng.uniform(0.01, 0.05)
    s2_b = rng.uniform(0.001, 0.01)
    sxs = rng.uniform(-0.005, 0.005)

    R_solver, _, _ = _ccv_log_return_and_grad(
        a_s, a_b, rb, xs, xb, s2_s, s2_b, sxs
    )
    R_sim = simulator_R_p(a_s, a_b, rb, xs, xb, s2_s, s2_b, sxs)
    assert abs(float(R_solver) - float(R_sim)) < 1e-12, (
        f"solver/simulator R_p disagree at α=({a_s:.3f},{a_b:.3f}): "
        f"solver={float(R_solver):.10e} sim={float(R_sim):.10e}"
    )

print("PASS: 1000/1000 random realisations agree to 1e-12")
```

If this test ever fails, **the patch is wrong** — both formulas must
compile to the same arithmetic ops on the same σ-scalars.

### 4.2 End-to-end smoke (REQUIRED)

Run [verify/smoke.py](../../verify/smoke.py) (or the canonical small
cell of [verify/canonical_small.py](../../verify/canonical_small.py)).
Compare against the pre-patch reference run on the same seed:

- `sim_R_port` should change (it was arithmetic, now CCV) — the per-age
  mean should shift by roughly `0.5 · (α_s · σ²_xr + α_b · σ²_xb − α'·Σ·α)`
  evaluated at the policy α at each age. At calibrated values this is
  on the order of 1–10 bps per age, but **in the direction of CCV**
  (which is what we want; the pre-patch arithmetic was the bug).
- `sim_estate` and `sim_x` should likewise shift; magnitudes
  differ by the cumulative wedge.
- The `wealth_offgrid["negative_frac"]` series should drop to (near) 0.
- All other outputs (`sim_c`, `sim_alpha_s`, `sim_alpha_b`, `sim_z`,
  `sim_alive`, `death_age`) come from the unchanged policy lookup +
  income transition + mortality — they should match the pre-patch run
  bit-for-bit at any age before the FIRST period where the wealth-law
  divergence has had time to propagate. (At t=0 the carry is still
  the initialised wealth, so panels at t=0 should be identical.)

### 4.3 Euler-residual sanity (RECOMMENDED)

If `scripts/diagnostics/_diag_euler_errors.py` exists in this branch
([referenced in CCV_RETURNS.md:257](../CCV_RETURNS.md#L257)), run it on
a fresh bundle:

- Pre-patch: residuals will conflate FOC numerical error with the
  arithmetic-vs-CCV wedge.
- Post-patch: residuals should be dominated by quadrature error
  (the FOC-numerical-error scale) and shrink by an order of magnitude
  at leveraged α cells.

A diagnostic comparison (residual histogram pre vs post) is the cleanest
way to demonstrate the fix has the intended economic effect.

### 4.4 Existing CCV verifies (RECOMMENDED)

[verify_ccv_theory_to_code.py](../../) — if there's a top-level CCV
theory verifier, run it post-patch to confirm no regressions in
solver-side CCV math (this patch shouldn't touch solver, but a regression
test never hurts).

---

## 5. Risks and edge cases

1. **Negative wealth disappearing.** The pre-patch arithmetic path could
   produce `s · R_port < 0` at leveraged α with bad return realisations
   (the docstring at line 27-31 calls this out). Under CCV, `R_port > 0`
   always. The `_wealth_offgrid_diagnostics["negative_frac"]` field will
   drop to zero. **This is the intended behaviour change** — the agent
   was insolvent in the simulator only because the simulator was using
   a different wealth law than the solver. Under matched dynamics, the
   solver's bequest-with-clamp handles all bankruptcy logic at the
   solve stage.

2. **Saved bundles.** Bundles produced before this patch contain the
   policy `(C_mat, S_mat, B_mat)` which the solver computed under CCV
   anyway — the bundle is fine, only the simulator was wrong. After
   patch, re-running `simulate_lifecycle` on an existing bundle gives
   the **correct** simulated paths for that policy. No re-solve needed.
   `policy_io.py` metadata records `wealth_dynamics_spec="ccv_log"`
   already; nothing to change there.

3. **Backwards-compatibility flag.** Do NOT add a `use_arithmetic`
   escape hatch. The whole point of this patch is to enforce parity.
   If a future user genuinely needs arithmetic-return path simulations
   for a comparator (the way `diagnostics.py:1198-1200` does), they
   should write a separate function — keeping it out of the production
   simulator forces the parity.

4. **`R_stock` / `R_bond` removal.** Confirm by grep inside `step_fn`
   that nothing downstream references the old `R_stock`, `R_bond`,
   `R_bill`, `a_bill_t` after the patch. The output panel key
   `R_port` is the only consumer; everything else uses `log_x_s`,
   `log_x_b`, `log_R_bill` directly.

5. **`use_mc_returns` paths.** The CCV formula is identical regardless
   of whether `log_x_s`, `log_x_b` came from quadrature nodes or
   Monte-Carlo Cholesky residuals — both branches at
   [simulation.py:340-348](../../lifecycle/simulation.py#L340-L348)
   produce the right `log_x_s, log_x_b`. The patch lives below that
   branch and is mode-agnostic. ✓

6. **Consumption-rescaling block.** The `c_t = jnp.where(x_t > wealth_max_grid, ...)`
   rescaling at [simulation.py:317-321](../../lifecycle/simulation.py#L317-L321)
   is independent of the wealth law; do not touch it.

---

## 6. File checklist

- [ ] [lifecycle/simulation.py](../../lifecycle/simulation.py) —
  docstring (21-31), `wealth_dynamics_spec` validation (~498),
  kernel signature (~256), kernel-build call (~700), wealth-law
  block (~350-359), `R_port` upstream cleanup, `negative_frac`
  docstring note.
- [ ] `verify/ccv_solver_sim_parity.py` (NEW) — parity unit test.
- [ ] Re-run `verify/smoke.py` (or `verify/canonical_small.py`) and
  confirm panel changes match the expected wedge sign/magnitude.
- [ ] (Optional) `scripts/diagnostics/_diag_euler_errors.py` —
  pre/post residual comparison.

No changes to: `solver.py`, `precompute.py`, `inf_horizon_solver.py`,
`diagnostics.py`, `policy_io.py`, `model.py`, `var.py`, `mortality.py`,
`discretization.py`, `numerics.py`, `predictability_ablation.py`.

---

## 7. Rollback

If the parity test passes but downstream verifications regress
unexpectedly, the rollback is a single-file revert. Bundles produced
under the old simulator do not need to be regenerated to roll back —
the policy `(C_mat, S_mat, B_mat)` was always CCV-correct on the solver
side; it is only the *simulated paths* that change with this patch.
