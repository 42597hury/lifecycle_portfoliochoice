# HANDOFF - Catmull-Rom z interpolation in the working-age solver

> **Status:** investigation handoff only. No code changes have been made in this pass.
>
> **Question:** is the current Catmull-Rom cubic interpolation in `z` a real problem, how empirically relevant are its failure modes, and how much does it improve on plain linear interpolation?

## 1. Executive summary

The current solver does use Catmull-Rom cubic interpolation in the `z` dimension inside the working-age continuation-value lookup, even though some docs still describe the path as linear. The cubic is implemented in the hot helper [`_interp_z_wealth`](solver.py#L299) and is activated for interior `z` intervals only.

The main takeaways from the investigation so far are:

- **Catmull-Rom is real, live, and intentional.** This is not dead code.
- **Theoretical failure modes exist**: it is not monotonicity-preserving, it can overshoot monotone data, and the outermost two `z` intervals still use linear fallback.
- **In a small current-model solved slice, those failure modes looked numerically small**:
  - monotone-stencil overshoot occurred, but the worst relative local deviation was only about `1.7e-3`
  - the raw cubic `mpc` never left `[0,1]` in the scan that was run
  - the cubic/linear branch switch was value-continuous up to machine precision
- **Catmull-Rom appears materially better than linear on smooth monotone functions at the actual current `z` spacing**, so the burden of proof is on any proposal to remove it.

The right next question is **not** "can Catmull-Rom overshoot in principle?" It can. The right question is:

> On this model's actual policy surfaces, does Catmull-Rom improve approximation enough over linear interpolation to justify the extra complexity and the small overshoot risk?

This handoff is designed to guide that comparison.

---

## 2. Where Catmull-Rom lives in the code

### 2.1 Current implementation

The active interpolation helper is:

- [`solver.py:_interp_z_wealth`](solver.py#L299)

The cubic branch is:

```python
if use_cubic:
    c_zm1 = ...
    c_z0  = ...
    c_z1  = ...
    c_z2  = ...
    c_val = Catmull-Rom(c_zm1, c_z0, c_z1, c_z2, frac_z)
```

and the associated `mpc` is built by applying the **same Catmull-Rom operator** to the nodewise wealth slopes:

```python
mpc_zm1 = ...
mpc_z0  = ...
mpc_z1  = ...
mpc_z2  = ...
mpc_val = Catmull-Rom(mpc_zm1, mpc_z0, mpc_z1, mpc_z2, frac_z)
```

This matters because Catmull-Rom is linear in the data. That makes the current `mpc` path exactly consistent with the interpolated `c_next` in the sense tested in [`tests/test_cubic_interp.py`](tests/test_cubic_interp.py#L271).

### 2.2 Where the branch is chosen

The working-age FOC computes:

- `iz_lo = int((z_next - z_grid[0]) / dz)`
- `frac_z = ...`
- `use_cubic = (iz_lo >= 1) and (iz_lo + 2 < n_z)`

See [`solver.py`](solver.py#L639-L685).

So:

- intervals `iz_lo = 0` and `iz_lo = n_z - 2` use **linear**
- interior intervals use **Catmull-Rom**

### 2.3 Stale documentation

The design doc still says working-age consumption is linearly interpolated in `z`:

- [`contextfiles/DESIGN.md`](contextfiles/DESIGN.md#L1068-L1070)

That description is currently stale. Do not use it as ground truth for the implementation.

---

## 3. Known theoretical issues with Catmull-Rom here

These are real concerns, but they need to be separated carefully.

### 3.1 Monotone data can overshoot

Catmull-Rom is not monotonicity-preserving. If `c(z)` is monotone across the 4-point stencil, the cubic can still dip below `min(c_z0, c_z1)` or rise above `max(c_z0, c_z1)` inside the interval.

In a CRRA model this matters because marginal utility amplifies negative `c` errors:

- if `u'(c) = c^(-gamma)`, then a small downward error in `c` becomes a larger upward error in `u'(c)`

### 3.2 The `mpc` clamp is potentially non-smooth

The current code clamps `mpc_val` into `[0,1]` in both the cubic and linear branches:

- [`solver.py`](solver.py#L324-L338)

If the raw cubic `mpc` ever leaves `[0,1]`, then the clamp introduces a kink in the Jacobian path because:

```python
mup_alive = -gamma * mu_alive / c_next * mpc
```

uses the clamped `mpc`.

### 3.3 The cubic/linear branch creates a slope kink in `z`

At the first and last interior intervals, the scheme changes from linear to cubic or cubic to linear. Because both interpolants match the endpoint values, this is **not a value discontinuity**, but it can be a derivative kink with respect to `z`.

### 3.4 The coarse `z` grid amplifies all of this

With `n_z = 11` and `n_stds = 3.0`, the current model has:

- `sigma_z ~= 1.8697`
- `dz ~= 1.1218`
- `dz / sigma_z = 0.6`

So one `z` step is roughly:

- `exp(dz) ~= 3.07x` income

and a 4-point cubic stencil spans roughly:

- `exp(4 * dz) ~= 88.9x` income

That is a coarse stencil, especially near the boundaries.

---

## 4. What was actually observed in this investigation

### 4.1 Existing synthetic cubic tests pass

Running [`tests/test_cubic_interp.py`](tests/test_cubic_interp.py) gave the following substantive results:

- endpoint interpolation passes
- Catmull-Rom is exact on linear and quadratic data
- Catmull-Rom beats linear interpolation on `exp(z)` in the test's smooth synthetic setup
- boundary fallback logic is correct
- no negative `c_next` on the synthetic "realistic" positive policy
- `mpc` Catmull-Rom is exactly consistent with the interpolated `c_next` in the current implementation

The final import check in that file failed only because the script is not run with the project root on `PYTHONPATH`; the numerical tests themselves passed.

### 4.2 Small current-model solve: empirical relevance looked limited

A tiny current-model solve was run with:

- `start_age = 66`
- `retire_age = 67`
- `terminal_age = 68`
- hardcoded current nominal-system VAR
- `n_z = 11`
- `state_grid = (3,3,3)`
- `n_state_quad_nodes = 2`
- `n_eps_nodes = 2`
- `n_eta_nodes = 2`
- constrained mode

That solve completed with:

- `0` Newton failures
- `0` warm resets

on the scanned slice.

### 4.3 Overshoot scan on the solved age-66 working slice

On the solved `C[t=0, :, :, :]` slice:

- monotone cases checked: `1,470,150`
- Catmull-Rom overshoot cases: `18,466`
- overshoot frequency: `0.01256`

Worst local deviations:

- max relative undershoot below the bracket: `1.547e-4`
- max relative overshoot above the bracket: `1.683e-3`

Interpretation:

- overshoot is **present**
- but on this test it looked **small**
- nothing here looked like a solver-breaking pathology on its own

### 4.4 Raw cubic `mpc` never left `[0,1]` in the scan

On the same solved slice:

- total scanned cubic `mpc` cases: `1,603,800`
- out-of-bounds raw `mpc` cases: `0`

So the feared `mpc` clamp activation did **not** occur in the empirical scan that was run.

This does **not** prove it never occurs anywhere, but it does strongly weaken the claim that clamp activation is a dominant current failure mechanism.

### 4.5 Branch-switch continuity

At the cubic/linear switch:

- max value gap at the branch boundary was about `2.66e-15`

So the switch is effectively value-continuous.

There is still a derivative kink in `z`, but the specific claim "the FOC sees a discontinuity as controls move" is misleading:

- the branch is determined by `z_next = rho*z + eta`
- it does **not** switch as Newton moves `alpha_s, alpha_b`

So this is a kink in the `z` axis, not a control-space branch flip.

---

## 5. Fast synthetic evidence that Catmull-Rom helps over linear

To anchor the "is it better than linear?" question, a few synthetic monotone functions were tested at the **actual current `dz ~= 1.1218`**, not the finer `[-2,2]` test grid used in [`tests/test_cubic_interp.py`](tests/test_cubic_interp.py#L170).

Results:

### 5.1 `c(z) = exp(z)`

- max relative error:
  - linear: `16.7%`
  - Catmull-Rom: `4.83%`
- median relative error:
  - linear: `12.99%`
  - Catmull-Rom: `2.84%`
- Catmull-Rom wins share: `100%`

### 5.2 `c(z) = 0.4 + 0.25 * exp(0.35 z)`

- max relative error:
  - linear: `1.38%`
  - Catmull-Rom: `0.0918%`
- median relative error:
  - linear: `0.460%`
  - Catmull-Rom: `0.0236%`
- Catmull-Rom wins share: `100%`

### 5.3 `c(z) = 0.7 + softplus(0.5 z)`

- max relative error:
  - linear: `0.781%`
  - Catmull-Rom: `0.0330%`
- median relative error:
  - linear: `0.348%`
  - Catmull-Rom: `0.0107%`
- Catmull-Rom wins share: `100%`

Interpretation:

- at the current coarse `z` spacing, Catmull-Rom can materially improve over linear on smooth monotone functions
- that does **not** prove it is better on the actual policy surface everywhere
- but it is enough to say "replace it with linear because cubic is scary" is not justified without more targeted evidence

---

## 6. Provisional assessment

This is my current view after the investigation:

1. **Catmull-Rom is not obviously a bug.** It looks like a deliberate accuracy upgrade over linear interpolation.
2. **Theoretical issues are real but currently look numerically small** on the tested solved slice.
3. **The strongest current evidence points toward Catmull-Rom helping more than it hurts**, at least on smooth policy-like functions at the current `dz`.
4. The open empirical question is narrower:

> On actual solved policy slices, especially near low/high `z`, low wealth, and the work-retirement boundary, does Catmull-Rom reduce interpolation error enough relative to linear to matter economically?

That is what the next agent should measure.

---

## 7. What the next agent should do

The next agent should focus on **quantifying the improvement of Catmull-Rom over basic linear interpolation** rather than re-litigating generic cubic-interpolation theory.

### 7.1 First priority: compare CR vs linear on actual policy data

Use current solved policy arrays or small partial re-solves and compare:

- `c_next` under Catmull-Rom
- `c_next` under linear interpolation in `z`

at sampled off-grid `z_next` points.

Recommended focus regions:

- low `z` and high `z` states
- low wealth segments where curvature is strongest
- `age = retire_age - 1`
- states with large `cy` or extreme financial conditions

Metrics to report:

- `|c_CR - c_linear| / c_CR`
- distribution by age
- distribution by `z` interval
- maximum and p95 / p99 differences

### 7.2 Better benchmark: use a finer-`z` reference

The cleanest way to answer "is Catmull-Rom better than linear?" is not to compare CR directly to linear, but to compare **both** against a higher-resolution reference.

Recommended experiment:

1. Build a **small** current-model problem that is cheap enough to solve repeatedly.
2. Solve it at:
   - coarse `n_z = 11`
   - finer `n_z = 21` or `31`
3. At coarse-grid off-grid `z_next` evaluation points, compare:
   - coarse Catmull-Rom
   - coarse linear
   - fine-grid reference

This can be done on a small lifecycle tail or a partial solve if runtime is tight.

What to measure:

- pointwise interpolation error in `c_next`
- resulting Euler residual differences
- resulting policy differences (`c`, `alpha_s`, `alpha_b`) when the coarse solver is run with linear vs cubic

### 7.3 Do not stop at interpolation-error plots

The relevant question is solver impact, not just approximation geometry.

Also report:

- Newton failures by age
- worst FOC residuals by age
- monotonicity violations in EGM outputs
- whether linear interpolation changes policy smoothness or corners

The diagnostic counters already exist in [`solver.py`](solver.py#L2838-L2870).

### 7.4 Boundary-specific comparison

Because only the outermost two `z` intervals use linear fallback, check whether Catmull-Rom's advantage over linear is concentrated in the interior or whether the remaining linear edge intervals dominate the actual error.

In other words, break results out by:

- `iz_lo = 0`
- `iz_lo = 1`
- middle intervals
- `iz_lo = n_z - 3`
- `iz_lo = n_z - 2`

### 7.5 Suggested decision rule

The next agent should aim to answer:

- If Catmull-Rom reduces interpolation error materially and does not create solver instability, keep it.
- If Catmull-Rom barely beats linear on actual policies while adding visible wiggles or bad tail behavior, switching to linear may be justified.

Do not decide based only on synthetic polynomial tests.

---

## 8. Important caveat if alternatives are considered

If the next agent starts exploring monotonicity-preserving Hermite or PCHIP-style alternatives, they must not ignore the current `mpc` invariant:

> The present Catmull-Rom implementation is linear in the node values, so the solver can compute `mpc` by applying the same operator to nodewise wealth slopes.

This is exactly what [`tests/test_cubic_interp.py`](tests/test_cubic_interp.py#L271) checks.

A slope-limited Hermite scheme is generally **not linear** in the node values. That means:

- `interpolate(c)` and then differentiate with respect to wealth
- versus
- interpolate the nodewise wealth slopes directly

are no longer automatically the same object.

If an alternative interpolation rule is ever tested, the next agent must explicitly verify whether the `mpc` used in the Jacobian is still consistent with the interpolated `c_next`. Otherwise a "safer" interpolant could quietly introduce a worse Jacobian inconsistency than the current Catmull-Rom overshoot.

This is not the main task of this handoff, but it is an important guardrail.

---

## 9. Bottom line

The current evidence does **not** support treating Catmull-Rom as a major live bug.

It does support treating it as:

- a real approximation choice,
- with small but genuine monotonicity risks,
- and with an open empirical question about how much it improves on linear interpolation in this model.

The next useful step is a **coarse-vs-fine reference comparison** that measures Catmull-Rom and linear on actual policy slices and actual solver outputs.
