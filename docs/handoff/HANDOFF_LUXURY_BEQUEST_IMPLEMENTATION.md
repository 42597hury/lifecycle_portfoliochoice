# HANDOFF — Implement luxury-bequest shift (De Nardi 2004)

## Scope and verdict

This is the implementation handoff for the De Nardi (2004) luxury-bequest specification, replacing the current pure-CRRA bequest with a shifted form that bounds the marginal-utility integrand. The proposal has been theory-reviewed and green-lit; see [PROPOSAL_LUXURY_BEQUEST_REVIEW.md](PROPOSAL_LUXURY_BEQUEST_REVIEW.md) for the economic justification.

The change resolves a real specification incoherence (the unshifted CRRA-with-clamp has an `∞` discontinuity in perceived bequest payoff at the bankruptcy boundary) and as a happy consequence eliminates a `10³⁰` Lobatto-tail spike that was contaminating the EE diagnostic and biasing the solved policy toward over-cautious leverage.

**Frame the change in the paper as**: "removes the bankruptcy-boundary discontinuity in the unshifted CRRA-with-clamp specification, following De Nardi (2004)" — not as a quadrature workaround.

## What changes

| File | Change | Type |
|---|---|---|
| [lifecycle/model.py](../../lifecycle/model.py) | Update `bequest_utility`, `bequest_marginal`, `bequest_marginal_inv`; add `DELTA_BEQUEST` constant | Modify |
| [lifecycle/solver.py](../../lifecycle/solver.py) | Add `_shifted_bequest_mu_and_mup` njit helper; replace 2-line bequest math at retirement & working-age FOC kernels; **rewrite terminal-step solver** | Modify (significant) |
| [scripts/diagnostics/_diag_euler_errors.py](../../scripts/diagnostics/_diag_euler_errors.py) | Replace 2-line bequest math at retirement & working-age EE kernels (mirror solver) | Modify |
| [docs/DESIGN.md](../../docs/DESIGN.md), [docs/UTILITY.md](../../docs/UTILITY.md), [lifecycle/precompute.py](../../lifecycle/precompute.py) (comments) | Update bequest formula docstrings/comments | Doc |

## What does NOT change

- **Simulator** ([lifecycle/simulation.py](../../lifecycle/simulation.py)). The estate at death is a *realised quantity*, `max(s·R_port, 0)`. No bequest *utility* is computed inside the simulator. Leave it alone.
- **Lobatto tail rule.** Tail coverage at Z=7 prevents finite-K-GH arbitrage; that role is unchanged. Keep canonical `ret_lobatto_Z = (None, 7, 7)` and `state_lobatto_Z = (None, 7, 7)`.
- **The hard bankruptcy clamp.** `max(W, 0)` survives. Bankrupt heirs still inherit zero estate; debt does not pass through. The shift only changes the agent's *preferences*, not the *realised estate*.
- **`min_wealth_inv` floors** in the solver kernels. The existing `if sR_p > 0:` branches keep `mu_bequest = 0` on the bankrupt side; the shift changes only the solvent-arm computation.

## The single source of truth

Add to [lifecycle/model.py](../../lifecycle/model.py) near the existing bequest helpers (line 264):

```python
# Annuity-normalised luxury-bequest shifter (De Nardi 2004).
#
# Bound on marginal bequest utility:  mu_max = b_bar * DELTA_BEQUEST**(-gamma) / A.
# For (b_bar, gamma, A) = (10, 5, 4):
#     DELTA = 0.005  ->  mu_max ~ 8e11   (vs. raw spike ~1e30 in the unshifted spec)
#     DELTA = 0.01   ->  mu_max ~ 2.5e10
#     DELTA = 0.02   ->  mu_max ~ 7.8e8
# Sensitivity sweep gate: ship at 0.005 if optimal alpha is stable to ~5%
# across {0.001, 0.005, 0.01, 0.02}.
DELTA_BEQUEST = 0.005
```

Update the three helpers:

```python
def bequest_utility(W, A, gamma, b_bar, delta=DELTA_BEQUEST):
    C_bar = np.maximum(W, 0.0) / A + delta
    return b_bar * C_bar ** (1.0 - gamma) / (1.0 - gamma)


def bequest_marginal(W, A, gamma, b_bar, delta=DELTA_BEQUEST):
    pos = W > 0.0
    C_bar = np.where(pos, W / A + delta, 1.0)  # placeholder for the W<=0 branch
    mu = b_bar * C_bar ** (-gamma) / A
    return np.where(pos, mu, 0.0)


def bequest_marginal_inv(mu, A, gamma, b_bar, delta=DELTA_BEQUEST):
    """Inverse marginal. Domain: mu in (0, mu_max] with mu_max = b_bar*delta**-gamma/A.
    Above mu_max, the constraint W = 0 binds and the inverse clamps."""
    mu_max = b_bar * delta ** (-gamma) / A
    mu_clamped = np.minimum(mu, mu_max)
    inner = (mu_clamped * A / b_bar) ** (-1.0 / gamma) - delta
    return A * np.maximum(inner, 0.0)
```

These helpers are currently dead code in the live tree (only solver inlines and docs reference them) but updating them is mandatory so the spec is single-sourced. Future diagnostics will rightly reach for these.

## Numba helper for FOC kernels

Add to [lifecycle/solver.py](../../lifecycle/solver.py) near the top, alongside the other njit helpers. Import `DELTA_BEQUEST` from `lifecycle.model`:

```python
from lifecycle.model import DELTA_BEQUEST  # add to imports

@njit(cache=True, inline='always')
def _shifted_bequest_mu_and_mup(W, A, gamma, b_bar, delta):
    """Shifted-bequest marginal utility and its derivative w.r.t. W.

    Caller must guard W > 0 (handled by the existing `if sR_p > 0:` branch
    in every FOC kernel call site). On the bankrupt branch the caller still
    assigns mu_bequest = 0, mup_bequest = 0 directly.

        mu  = b_bar * (W/A + delta)**(-gamma) / A
        mup = -gamma * mu / (A * (W/A + delta))
    """
    C_bar = W / A + delta
    mu = b_bar * C_bar ** (-gamma) / A
    mup = -gamma * mu / (A * C_bar)
    return mu, mup
```

Mirror this helper into [scripts/diagnostics/_diag_euler_errors.py](../../scripts/diagnostics/_diag_euler_errors.py) (cannot import from solver because the diagnostic file isn't in the cache hierarchy — duplicate it, comment that it must stay in lockstep with the solver helper).

## Call sites — replace the two-line bequest math

The four FOC call sites all share the same pattern. **They live inside an existing `if sR_p > 0.0:` arm**, so the helper is called only on the solvent branch. The bankrupt-arm code (`mu_bequest = 0.0; mup_bequest = 0.0` etc.) stays unchanged.

### (a) Retirement, [lifecycle/solver.py:873-879](../../lifecycle/solver.py#L873-L879)

OLD:
```python
if sR_p > 0.0:
    w_A = sR_p / annuity_factor_is
    mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is
    mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)
else:
    mu_bequest = 0.0
    mup_bequest = 0.0
```

NEW:
```python
if sR_p > 0.0:
    mu_bequest, mup_bequest = _shifted_bequest_mu_and_mup(
        sR_p, annuity_factor_is, gamma, b_bar, DELTA_BEQUEST
    )
else:
    mu_bequest = 0.0
    mup_bequest = 0.0
```

### (b) Working-age, [lifecycle/solver.py:1025-1030](../../lifecycle/solver.py#L1025-L1030)

**Note the structural difference**: working-age computes the bequest contribution *outside* the inner `eta × eps` loop and accumulates directly into `euler_sum / foc_s / foc_b / J_**` because the bequest doesn't depend on next-period income shocks. The replacement is the same two lines, but find them inside this larger Path B block:

```python
sR_p = s_val * R_p
if sR_p > 0.0:
    w_inv = sR_p
    # OLD:
    # w_A = w_inv / annuity_factor_is
    # mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is
    # mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)
    # NEW:
    mu_bequest, mup_bequest = _shifted_bequest_mu_and_mup(
        w_inv, annuity_factor_is, gamma, b_bar, DELTA_BEQUEST
    )
    # ... rest of the death-branch FOC accumulation unchanged ...
    death_mu = p_state_ret * prob_death * mu_bequest
    death_mup = p_state_ret * prob_death * mup_bequest
    euler_sum += death_mu * R_p
    foc_s     += death_mu * Rex_s
    foc_b     += death_mu * Rex_b
    jac_b      = death_mup * s_val
    J_ss      += jac_b * Rex_s * Rex_s
    J_bb      += jac_b * Rex_b * Rex_b
    J_sb      += jac_b * Rex_s * Rex_b
else:
    w_inv = 0.0
# Alive contribution loops over k_eta, k_eps separately, using w_inv
```

### (c) Diagnostic retirement EE kernel, [scripts/diagnostics/_diag_euler_errors.py:734-738](../../scripts/diagnostics/_diag_euler_errors.py#L734-L738)

OLD:
```python
if sR_p > 0.0:
    w_A = sR_p / annuity_factor_cur
    mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_cur
else:
    mu_bequest = 0.0
```

NEW (this kernel doesn't compute `mup_bequest` — only one return needed):
```python
if sR_p > 0.0:
    C_bar = sR_p / annuity_factor_cur + DELTA_BEQUEST
    mu_bequest = b_bar * C_bar ** (-gamma) / annuity_factor_cur
else:
    mu_bequest = 0.0
```

### (d) Diagnostic working-age EE kernel, in `_compute_euler_sum_working_continuous`

Same pattern as (c). Search for the analogous block (`if sR_p > 0.0: ... mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_cur`) and replace.

## Terminal-step re-derivation (the load-bearing change)

The existing terminal step ([lifecycle/solver.py:1180-1546](../../lifecycle/solver.py#L1180-L1546)) is built on the pure-CRRA homogeneity of the bequest:

- Pure CRRA: `b(s·R_p, A) ∝ s^{1−γ} · R_p^{1−γ}`. Optimal `α*` is independent of `s`. Optimal consumption `c = W · ratio / (ratio + 1)` is closed-form linear in `W`.
- Shifted: `b(s·R_p, A) = b_bar · (max(s·R_p, 0)/A + δ)^{1−γ} / (1−γ)`. **Not homogeneous in `s`.** Optimal `α*` depends on `s`. Closed-form is invalid.

We re-derive faithfully via EGM on a savings grid. **No scipy calls**; reuse the solver's own Newton machinery.

### New kernels to write

#### `compute_terminal_foc_jac_shifted` (replaces `compute_terminal_portfolio_foc_jac`)

```python
@njit(fastmath=True)
def compute_terminal_foc_jac_shifted(
    alpha_s, alpha_b, s_val, A_is,
    state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights,
    gamma, b_bar, delta,
    prob_skip=1e-12,
):
    """Terminal-period portfolio FOC and Jacobian under shifted bequest.

    FOC: foc_k = sum_{k_v, k_r} w_v * w_r * mu_bequest(s*R_p) * (R_k - R_bill)
    where mu_bequest is the shifted form. The bankruptcy clamp 1{s*R_p > 0}
    enters explicitly.

    Also returns V_dot = E[mu_bequest * R_p] used by the consumption Euler.

    Returns: (foc_s, foc_b, J_ss, J_bb, J_sb, V_dot)
    """
    foc_s = 0.0; foc_b = 0.0
    J_ss = 0.0; J_bb = 0.0; J_sb = 0.0
    V_dot = 0.0
    a_bill = 1.0 - alpha_s - alpha_b

    for k_v in range(len(state_weights)):
        w_v = state_weights[k_v]
        if w_v < prob_skip:
            continue
        for k_r in range(len(ret_weights)):
            weight = w_v * ret_weights[k_r]
            if weight < prob_skip:
                continue
            R_bill_kr = Rx_bill[k_v, k_r]
            R_s = R_bill_kr * Rx_stock_mult[k_v, k_r]
            R_b = R_bill_kr * Rx_bond_mult[k_v, k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill_kr
            Rex_s = R_s - R_bill_kr
            Rex_b = R_b - R_bill_kr

            sR_p = s_val * R_p
            if sR_p > 0.0:
                mu_b, mup_b = _shifted_bequest_mu_and_mup(
                    sR_p, A_is, gamma, b_bar, delta
                )
                wmu  = weight * mu_b
                wmup = weight * mup_b
                foc_s += wmu * Rex_s
                foc_b += wmu * Rex_b
                jac = wmup * s_val
                J_ss += jac * Rex_s * Rex_s
                J_bb += jac * Rex_b * Rex_b
                J_sb += jac * Rex_s * Rex_b
                V_dot += wmu * R_p
            # else: contributes 0 (bankrupt — zero bequest derivative)

    return foc_s, foc_b, J_ss, J_bb, J_sb, V_dot
```

#### `solve_terminal_portfolio_at_s` (Newton, mirroring the constrained / unconstrained solvers)

For each (`i_s`, `s`) pair, run a damped Newton on `(α_s, α_b)` with the analytical Jacobian from `compute_terminal_foc_jac_shifted`. Mirror the structure of the existing `solve_portfolio_2d_terminal_constrained_njit` / `solve_portfolio_unconstrained_terminal_njit` (same line search, same alpha-box projection, same exit codes). Returns `(α_s*, α_b*, V_dot*, exit_code, foc_resid, n_iter)`.

The structure is mechanically the same as the existing terminal solvers — just call `compute_terminal_foc_jac_shifted(α_s, α_b, s, A_is, ...)` instead of `compute_terminal_portfolio_foc_jac(α_s, α_b, ...)` (note the new `s` and `A_is` arguments).

#### `solve_terminal_age` rewrite

Replace the current closed-form with EGM on a savings grid. Per `(i_s, j_s)`:

1. Pick savings `s = s_grid[j_s]`.
2. Solve portfolio: `(α_s*, α_b*, V_dot*, ...) = solve_terminal_portfolio_at_s(s, A_is, ...)`.
3. Invert consumption Euler: `c* = (β · V_dot*)^{−1/γ}`.
4. Implied wealth: `W_implied = c* + s`.

Then for each `(i_s, i_w)`:

5. Interpolate `(c, α_s, α_b)` from `(W_implied, c*, α_s*, α_b*)` onto `wealth_grid[i_w]`.
6. **Constrained region** (`W < W_implied[0]`): the agent is below the lowest savings level on the grid. The EGM-monotone region. Set `s = s_grid[0]` (or any small value), `c = W − s`, `α = α*[0]`. Or set `s = 0`, `c = W`, `α = α*` from a fallback solve at `s = 0` (which becomes an Inada-style limit; pick `s ≈ wealth_min` to stay within the regularized regime).
7. **Above grid** (`W > W_implied[-1]`): linear extrapolation in `c`, hold `α` constant from the largest `s`.

Reuse `_build_terminal_quad_returns` ([lifecycle/solver.py around line 1499](../../lifecycle/solver.py#L1499)) unchanged — it builds the per-state return arrays.

### Cost estimate

- Old: `N_state ≈ 343` Newton solves (~3.5 ms total).
- New: `N_state × N_s ≈ 343 × 180 ≈ 62k` Newton solves. Each call ~100 µs (10 Newton iters × ~10 µs per FOC eval at 75×75 nodes with numba). Total ≈ 6 s.

Negligible relative to the working-age solve (~minutes). Acceptable.

### Initial guesses for terminal Newton

Hot-start: at each `i_s`, sweep `s_grid[j_s]` from largest to smallest. Use `(α_s, α_b)` from the previous `j_s` as the initial guess for the next. This dramatically improves convergence (the policy is approximately continuous in `s`).

### Sanity check the rewrite against the original

For `δ → 0+` the shifted spec converges to pure CRRA pointwise. Verify at solve time: with `δ = 1e-9`, the new `solve_terminal_age` should produce a `c_vec` that agrees with the old closed-form to ~ppm (limited by Newton tolerance). Add this as a regression test.

## Documentation updates

1. [lifecycle/solver.py](../../lifecycle/solver.py) module docstring (line ~5): update bequest formula reference.
2. [lifecycle/precompute.py:101, 252](../../lifecycle/precompute.py#L101) — comments referring to bequest helpers; update to mention the shifter.
3. [docs/DESIGN.md](../../docs/DESIGN.md) — find the bequest section (around line 637); replace the unshifted formula with the shifted form. Cite De Nardi (2004).
4. [docs/UTILITY.md](../../docs/UTILITY.md) — table around line 163; functional-form section around line 700+. Update both.
5. [configs/_canonical.py](../../configs/_canonical.py) — no change to the data fields, but consider adding a comment near `b_bar = 10` noting that `DELTA_BEQUEST` lives in `lifecycle/model.py`.

## Validation plan

Run in this order; do not move on until each criterion passes.

### Step 1 — Unit / sanity check

Re-run [scripts/diagnostics/_diag_split_rule_sanity.py](../../scripts/diagnostics/_diag_split_rule_sanity.py) on the existing v4_lobatto bundle (the diagnostic kernels will already have the shift, the bundle's policy is the unshifted one — that's fine; we're testing the integrand evaluation).

**Pass criteria** at the worst leveraged retirement cell:
- Top per-node bequest contribution under Lobatto drops from `8.81 × 10¹¹` to `< 0.1`.
- `e_bequest_lobatto - e_bequest_dense_gh` < `0.01` in absolute terms (was `8.81 × 10¹¹`).

### Step 2 — Pure-CRRA convergence

With `DELTA_BEQUEST = 1e-9`, re-solve a small smoke bundle. Compare `c`, `α_s`, `α_b` arrays to a fresh solve from the pre-change branch.

**Pass criterion**: agreement to within Newton tolerance (~`1e-6` relative).

### Step 3 — Sensitivity sweep

Re-solve four small bundles at `δ ∈ {0.001, 0.005, 0.01, 0.02}`. Tabulate `(α_s, α_b)` at the worst-leveraged retirement cell from each.

**Pass criterion**: max policy shift across the band is `< 5%` (relative). If `δ = 0.001` destabilises the solver (Newton failures, EGM violations), document — `δ ≈ 3 × 10⁻⁴` is the theoretical lower bound from the reviewer's analysis.

If pass, **ship at `δ = 0.005`** (closest defensible value to the original CRRA spec).

### Step 4 — Distribution-wide policy-shift table

Solve the canonical bundle with the shift. Run [scripts/diagnostics/_diag_split_rule_sanity.py](../../scripts/diagnostics/_diag_split_rule_sanity.py) modified (or a new diagnostic) to produce, for 50–100 representative `(age, wealth, state_z)` cells, the `|Δα|` between the new shifted bundle and the unshifted reference bundle.

**Report** (do not gate on):
- Mean and median `|Δα_s|`, `|Δα_b|`.
- Number of cells with `|α_b| > 5` or `|α_b| > alpha_max` post-shift.
- Aggregate sim moments: mean retirement leverage, share of bankrupt sim paths, estate-at-death distribution.

These are *informational* — the reviewer asked for them to confirm the worst-cell shift `(α_b: −1.21 → −2.35)` is localised, not pervasive. If the median `|Δα_b|` is `< 5%` of the original `|α_b|` and the worst is large, the spike was localised — ship. If many cells move materially, escalate.

### Step 5 — Full EE diagnostic battery

On the new shifted canonical bundle, run the full EE diagnostic battery (sim-path EE under propagated Lobatto, gridpoint EE).

**Pass criteria** (publication gates):
- `mean log10|EE| < −4.5` at retirement
- `max log10|EE| < −3.0` at retirement

These are the gates the unshifted spec was missing because of the spike.

## References

- De Nardi, M. (2004). "Wealth Inequality and Intergenerational Links." *Review of Economic Studies* 71(3), 743–768.
- De Nardi, M., French, E., Jones, J. (2010). "Why Do the Elderly Save? The Role of Medical Expenses." *Journal of Political Economy* 118(1), 39–75.
- Lockwood, L. (2018). "Incidental Bequests and the Choice to Self-Insure Late-Life Risks." *American Economic Review* 108(9), 2513–2550.

## Companion documents

- [PROPOSAL_LUXURY_BEQUEST_REVIEW.md](PROPOSAL_LUXURY_BEQUEST_REVIEW.md) — original proposal + theory review.
- [bequest_shifted_implementation.py](bequest_shifted_implementation.py) — reviewer's prototype (helper math is correct; line numbers and assumptions are stale; this handoff supersedes its scope).
- [scripts/diagnostics/_diag_split_rule_sanity.py](../../scripts/diagnostics/_diag_split_rule_sanity.py) — empirical evidence of the spike and the required validation harness.
