# FOC + EGM mathematical correctness review (2026-05-10)

**Branch:** `jax-rewrite`. **Reviewer angle:** mathematical / equation-level
correctness. Independent of the parallel implementation review on the same
scope. **Read-only.** Reviewer derives every FOC by hand from the underlying
Bellman equation, transcribes the JAX code, then term-by-term checks them
against each other; finally runs JAX autodiff identity checks on synthetic
inputs to verify the analytic Jacobians.

---

## §1. Scope and angle

This review verifies the equation-level correctness of:

1. The Bellman equation per period (working / boundary / retirement / terminal).
2. CRRA + luxury-bequest derivatives (level, marginal, second derivative).
3. CCV log-portfolio return `R_p(α_s, α_b)` and its first/second derivatives.
4. The four FOC kernels in [solver.py](lifecycle/solver.py):
   `terminal_foc_jac_ccv` ([solver.py:850](lifecycle/solver.py#L850)),
   `retirement_foc_jac_ccv` ([solver.py:1030](lifecycle/solver.py#L1030)),
   `working_foc_jac_ccv` ([solver.py:1140](lifecycle/solver.py#L1140)), plus
   the boundary case (`working_foc_jac_ccv` reused with pension as `income_table`,
   selected via `use_pension_next` in
   [solver.py:2490-2496](lifecycle/solver.py#L2490)) and the variant
   `working_foc_jac_ccv_pi_z` in
   [solver_pi_z_variant.py:125](lifecycle/solver_pi_z_variant.py#L125).
5. EGM mechanics: per-savings scan, Euler inversion, lift-to-wealth-grid.
6. The terminal condition (pure bequest, no continuation).
7. The infinite-horizon iteration in
   [inf_horizon_solver.py](lifecycle/inf_horizon_solver.py).
8. Boundary/edge handling for tiny savings, singular Jacobian, line-search
   exhaustion, EC_NEWTON_FAIL.

The companion implementation review (separate file) verifies code paths,
fori-loop masking, gather precision, etc.

---

## §2. Bellman equation per period

**Notation.** Period-t state: `(z_t, s_t, W_t)` where `z_t` ∈ ℝ persistent
log-income, `s_t` ∈ ℝ^{n_state} financial state vector (post real-yields
pivot: typically `(y_1, spr, xb)` or similar; see
[discretization.py](lifecycle/discretization.py) and
[REAL_YIELDS_PIVOT_REVIEW_2026-05-08.md](docs/scans/REAL_YIELDS_PIVOT_REVIEW_2026-05-08.md)),
`W_t` ∈ ℝ_{>0} cash-on-hand. Choices: consumption `c_t`, savings
`s = W_t − c_t`, portfolio shares `(α_s, α_b)` for stocks and bonds with
bill share `1 − α_s − α_b`. Mortality probability `1 − ψ_z` at end of period.

CCV portfolio log-return is

```
r_p(α_s, α_b; s_t, ξ) = log_R_bill(s_t)
    + α_s · log_x_s(s_t, ξ) + α_b · log_x_b(s_t, ξ)
    + 0.5 · (α_s · σ²_xr + α_b · σ²_xb)
    − 0.5 · (α_s² · σ²_xr + 2 α_s α_b · σ_xrxb + α_b² · σ²_xb)
```

with `R_p = exp(r_p)` (verified in
[solver.py:829-839](lifecycle/solver.py#L829)). The two `0.5·α·σ²` terms
are the Jensen corrections converting log-excess returns to expected gross
returns; the quadratic form is the Itô vol-drag from log-aggregating a
portfolio. `log_R_bill` is **deterministic given current state** post
real-yields pivot ([solver.py:912-916](lifecycle/solver.py#L912)).

End-of-period wealth on the alive branch:
`W_{t+1} = s · R_p + Y_{t+1}`. Bequest realisation if dead:
`B = s · R_p` (bequest happens one period after the last choice, hence the
β multiplier on b in the terminal Bellman).

### 2.1 Working age (`age < retire_age − 1`)

```
V_t(z_t, s_t, W_t) = max_{c, α_s, α_b} {
    u(c) + β · ψ_{z_t} · E_t[ V_{t+1}(z_{t+1}, s_{t+1}, W_{t+1}) ]
         + β · (1 − ψ_{z_t}) · E_t[ b(s · R_p, A(s_t)) ]
}
```

with `z_{t+1} = ρ z_t + η_{t+1}` (Catherine 2025 mixture), `Y_{t+1} =
exp(f(age) + z_{t+1} + ε_{t+1})` (after-tax via `disposable_income_working`,
see [model.py:421](lifecycle/model.py#L421)). The expectation factors as
`E_v ⊗ E_ξ ⊗ E_η ⊗ E_ε` (conditional independence of the four shock
families given `s_t, z_t`); this is exactly the four-axis broadcast in
`working_foc_jac_ccv` ([solver.py:1190](lifecycle/solver.py#L1190),
1221-1225).

### 2.2 Boundary (`age == retire_age − 1`)

Same Bellman as working, except `Y_{t+1} = pension(z_{t+1})` (no eps
shock at retirement; pension is a deterministic function of the realised
persistent state). Implementation: working FOC reused, with
`income_table[k_eta, i_e] = pension(z_{t+1}[k_eta])` broadcast across the
eps axis ([solver.py:2493-2497](lifecycle/solver.py#L2493) and
[solver.py:2606-2610](lifecycle/solver.py#L2606)). The eps axis ends up as a
trivial sum-to-one weighting — mathematically equivalent to dropping eps,
but kept for kernel-shape uniformity.

### 2.3 Retirement (`retire_age ≤ age < terminal_age`)

`z` is frozen (ρ-decay would still apply at the model level but `z` no
longer feeds production income; the AR(1) ε/η shocks are gone).
`Y_{t+1} = pension(z_t)` (deterministic given current `z`).

```
V_t(z, s_t, W_t) = max_{c, α_s, α_b} {
    u(c) + β · ψ_z · E_t[ V_{t+1}(z, s_{t+1}, s · R_p + pension(z)) ]
         + β · (1 − ψ_z) · E_t[ b(s · R_p, A(s_t)) ]
}
```

Code: `retirement_foc_jac_ccv` ([solver.py:1030](lifecycle/solver.py#L1030)).

### 2.4 Terminal (`age == terminal_age`)

Pure bequest, no continuation:

```
V_T(z, s_T, W_T) = max_{c, α_s, α_b} { u(c) + β · E_T[ b(s · R_p, A(s_T)) ] }
```

The β multiplier is correct because the bequest accrues one period after
the choice (period-end-death convention). z is inert at terminal — policy
broadcast across z by the orchestrator
([solver.py:2680-2700](lifecycle/solver.py#L2680)). Code:
`terminal_foc_jac_ccv` ([solver.py:850](lifecycle/solver.py#L850)).

---

## §3. CRRA + bequest derivatives

### 3.1 CRRA utility (γ ≠ 1)

```
u(c) = c^{1-γ} / (1-γ)
u'(c) = c^{-γ}
u''(c) = -γ · c^{-γ-1} = -γ · u'(c) / c
(u')^{-1}(μ) = μ^{-1/γ}
```

Code: [model.py:273-284](lifecycle/model.py#L273) (NumPy reference) and
inline in solver kernels (e.g. `mu_alive = c_at_xn ** (-gamma)` at
[solver.py:1108](lifecycle/solver.py#L1108)). Inverted Euler uses
`c_egm = (β V_dot)^{-1/γ}` at [solver.py:1322-1323](lifecycle/solver.py#L1322).

`mup_alive = -γ · μ / c · mpc_at_xn` at
[solver.py:1109](lifecycle/solver.py#L1109): this is
`d/dα_s [u'(c_{t+1}(W))] = u''(c) · dc/dW · dW/dα_s = (-γ μ/c) · mpc · s · dR_p/dα_s`.
The `mpc · s · dR_p/dα_s` factors are added downstream (the `mup` here is
`d u'(c)/dW`, not `d u'(c)/dα`).

### 3.2 Luxury bequest (De Nardi 2004 / Catherine 2025)

```
C_bar(W; A) = W/A + δ
b(W; A) = b_bar · C_bar^{1-γ} / (1-γ)
b'(W) = b_bar · C_bar^{-γ} / A
b''(W) = -γ · b_bar · C_bar^{-γ-1} / A² = -γ · b'(W) / (A · C_bar)
```

Code: [model.py:366-414](lifecycle/model.py#L366) (NumPy reference);
inline `bequest_mu_and_mup` at [solver.py:404-414](lifecycle/solver.py#L404):

```python
C_bar = W / A + delta
mu = b_bar * C_bar ** (-gamma) / A
mup = -gamma * mu / (A * C_bar)
```

Verified by direct substitution: `mup = -γ · b_bar · C_bar^{-γ}/A · 1/(A · C_bar)
= -γ · b_bar · C_bar^{-γ-1} / A²` = `b''(W)` ✓.

The annuity factor `A(s_t) = Σ_{k=1}^{b_bar} (1+y(k))^{-k}` with
`y(k) = y_1 + spr · (k-1)/(b_bar-1)` (linear interp on the term structure)
is per [model.py:319-351](lifecycle/model.py#L319). Discrete compounding
matches the comment in [model.py:336-340](lifecycle/model.py#L336)
(continuous compounding would mismatch by ~12 bp/yr).

---

## §4. CCV log-wealth dynamics: R_p, gradient, Hessian

### 4.1 Level

`r_p` is verified in §2 above and at
[solver.py:829-839](lifecycle/solver.py#L829). Two correction terms
work in opposite directions:
- `+0.5 · (α_s · σ²_xr + α_b · σ²_xb)`: Jensen correction for log-mean of
  log-normal excess returns (raises the level).
- `−0.5 · (α_s² · σ²_xr + 2 α_s α_b · σ_xrxb + α_b² · σ²_xb)`: Itô
  vol-drag from log-portfolio aggregation (lowers the level for any
  non-degenerate covariance).

### 4.2 Gradient `dr/dα`

Differentiating r_p w.r.t. α_s:

```
dr/dα_s = log_x_s + 0.5 · σ²_xr − (α_s · σ²_xr + α_b · σ_xrxb)
        = log_x_s + σ²_xr · (0.5 − α_s) − α_b · σ_xrxb
```

Code: [solver.py:841](lifecycle/solver.py#L841). Symmetric expression for
α_b at [solver.py:842](lifecycle/solver.py#L842). **Verified analytically
and by JAX autodiff** (test 1 below; rel diff = 7e-18, i.e. fp64 noise).

### 4.3 Hessian `d²r/dα²`

```
d²r/dα_s² = -σ²_xr
d²r/dα_b² = -σ²_xb
d²r/dα_s dα_b = -σ_xrxb
```

These show up in the FOC Jacobian as the **`extra_*` terms**:

```
∂(R_p · dr/dα_s)/∂α_s = R_p · (dr/dα_s)² + R_p · d²r/dα_s²
                     = R_p · ((dr/dα_s)² − σ²_xr)
```

Code: [solver.py:876](lifecycle/solver.py#L876) (terminal),
[solver.py:1126-1128](lifecycle/solver.py#L1126) (retirement),
[solver.py:1181-1183](lifecycle/solver.py#L1181) (working bequest piece),
[solver.py:1239-1241](lifecycle/solver.py#L1239) (working alive piece).
**Verified analytically and by autodiff** (test 1b below; diff = 0).

The signs of σ² in the `extra_*` term correctly carry the **negative**
second derivatives of r_p. A reviewer cross-checking sign would be
forgiven for being suspicious: convexity correction terms in option-
pricing-style derivations often *add* a positive σ². Here the sign comes
from the strict identity `d²r/dα² = -σ²` (the Itô vol-drag's
quadratic form has negative definite Hessian).

---

## §5. Working-age FOC: derivation, transcription, term-by-term check

### 5.1 Derivation by hand

Maximand at given `s` (consumption-portfolio separation under EGM):

```
J(α_s, α_b) = β · ψ · E[ V_{t+1}(z', s', s · R_p + Y') ]
            + β · (1 − ψ) · E[ b(s · R_p; A(s_t)) ]
```

Stock FOC: `dJ/dα_s = 0`. Differentiate inside the expectation:

```
d/dα_s [V_{t+1}(W_{t+1})] = V'_{t+1}(W_{t+1}) · s · dR_p/dα_s
                          = u'(c_{t+1}(W_{t+1})) · s · R_p · (dr/dα_s)   [envelope]
d/dα_s [b(sR_p)]         = b'(sR_p) · s · R_p · (dr/dα_s)
```

Both pieces share the common factor `s · R_p · (dr/dα_s)`. Dividing the
FOC equation by `s` (a positive scalar), the root-defining residual is

```
foc_s := E[ ψ · u'(c_{t+1}(W_{t+1})) · R_p · (dr/dα_s)
          + (1−ψ) · b'(sR_p) · R_p · (dr/dα_s) ]
       = E[ μ_comb · R_p · (dr/dα_s) ]
       = E[ μ_comb · dRp/dα_s ]
```

where `μ_comb = ψ · u'(c_{t+1}) + (1−ψ) · b'(sR_p)` and we have used
`dRp/dα_s = R_p · dr/dα_s`. Same structure for α_b.

### 5.2 Transcription

[solver.py:1140-1254](lifecycle/solver.py#L1140) computes `foc_s_bq +
foc_s_al` and `foc_b_bq + foc_b_al`:

```python
mu_bq, mup_bq = bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)   # 1171
prob_death = 1.0 - psi_z
bequest_factor = weight_kv_kr * prob_death
dRp_das = R_p * dr_da_s
foc_s_bq = jnp.sum(bequest_factor * mu_bq * dRp_das)                  # 1177

# alive piece
mu_alive = c_at_xn ** (-gamma)                                         # 1218
alive_factor = weight_full * psi_z                                     # 1226
foc_s_al = jnp.sum(alive_factor * mu_alive * dRp_das[:, :, None, None])# 1228

return (foc_s_bq + foc_s_al, ...)                                      # 1247-1248
```

This **matches §5.1 exactly**: `μ_comb · dRp/dα_s` with the alive vs.
bequest pieces split via separate weight tensors (because the alive
piece has additional `(eta, eps)` axes).

### 5.3 Jacobian

Differentiate the FOC residual w.r.t. α_s. Two contributions:
1. `(d/dα_s μ_comb) · dRp/dα_s`. For alive: `μ_comb' = ψ · u''(c_{t+1}) ·
   dc/dW · s · dR_p/dα_s = ψ · (-γ μ_alive/c) · mpc · s · dR_p/dα_s`. For
   bequest: `(1-ψ) · b''(sR_p) · s · dR_p/dα_s`. Code: `mup_alive` at
   [solver.py:1109](lifecycle/solver.py#L1109) and `mup_bq` from
   `bequest_mu_and_mup`. Combined into `wmup` then multiplied by `s_val`
   into `jac_lin` ([solver.py:1232](lifecycle/solver.py#L1232)).
2. `μ_comb · d/dα_s (R_p · dr/dα_s) = μ_comb · R_p · ((dr/dα_s)² + d²r/dα_s²)
   = μ_comb · R_p · ((dr/dα_s)² − σ²_xr)`. Code: `extra_ss_al` at
   [solver.py:1239](lifecycle/solver.py#L1239) (and bq counterpart at
   1181).

Total Jacobian J_ss = `jac_lin · dRp_das² + extra_ss` summed over
quadrature axes ([solver.py:1243](lifecycle/solver.py#L1243)). **Identical
algebra for J_bb and J_sb**. Confirmed by JAX autodiff (test 3 below;
rel diff ~ 1e-16).

### 5.4 Term-by-term verdict

| Term | Math                                  | Code                                                        | OK |
|------|---------------------------------------|-------------------------------------------------------------|----|
| Bequest FOC `foc_s_bq` | `(1-ψ) E[b'(sR_p)·R_p·dr/dα_s]` | [solver.py:1177](lifecycle/solver.py#L1177) | ✓ |
| Alive FOC `foc_s_al` | `ψ E[u'(c_{t+1})·R_p·dr/dα_s]` | [solver.py:1228](lifecycle/solver.py#L1228) | ✓ (envelope) |
| Bequest J_ss linear | `(1-ψ) E[b''·s·(R_p·dr/dα_s)²]` | [solver.py:1180,1184](lifecycle/solver.py#L1180) | ✓ |
| Alive J_ss linear | `ψ E[(-γ μ_a/c)·mpc·s·(R_p·dr/dα_s)²]` | [solver.py:1232,1243](lifecycle/solver.py#L1232) | ✓ |
| `extra_ss` convex | `μ_comb · R_p · ((dr/dα_s)² − σ²_xr)` | [solver.py:1181,1239](lifecycle/solver.py#L1181) | ✓ |
| `J_sb` (off-diag) | `μ_comb · R_p · (dr/dα_s · dr/dα_b − σ_xrxb)` | [solver.py:1183,1241](lifecycle/solver.py#L1183) | ✓ |
| Symmetric `J_sb = J_bs` | by Schwarz | not asserted in code | ✓ structurally |
| `e_sum = V_dot` | `E[μ_comb · R_p]` (Euler RHS) | [solver.py:1230,1253](lifecycle/solver.py#L1230) | ✓ |

**Note on s_val absorption.** The FOC equation `foc_s = 0` is unchanged
under multiplicative rescaling, so dividing through by `s` is harmless
**at the root**. But the Jacobian must be the derivative of the *same
residual being zeroed*, so the `s` multiplier appears explicitly via
`jac_lin = wmup * s_val` ([solver.py:1232](lifecycle/solver.py#L1232)).
Verified internally consistent by autodiff (test 3).

---

## §6. Retirement FOC

### 6.1 Derivation

Retirement Bellman has no eta/eps axes; income next is the deterministic
pension `pension(z)`. Otherwise identical to working:

```
foc_s := E_{v,ξ}[ ψ · u'(c_{t+1}(s · R_p + pension)) · R_p · dr/dα_s
                + (1−ψ) · b'(s · R_p) · R_p · dr/dα_s ]
```

### 6.2 Transcription

[solver.py:1030-1133](lifecycle/solver.py#L1030):

```python
R_p, dr_da_s, dr_da_b = _ccv_log_return_and_grad(...)
sR_p = s_val * R_p
x_next = sR_p + pension_next_z
mu_bq, mup_bq = bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)
# c_at_xn / mpc_at_xn from per_kv_kr inline interp ...
mu_alive = c_at_xn ** (-gamma)
mup_alive = -gamma * mu_alive / c_at_xn * mpc_at_xn
mu_comb = psi_z * mu_alive + prob_death * mu_bq
mup_comb = psi_z * mup_alive + prob_death * mup_bq
wmu = weight_kv_kr * mu_comb
wmup = weight_kv_kr * mup_comb
foc_s = jnp.sum(wmu * dRp_das)
J_ss = jnp.sum(jac_lin * dRp_das * dRp_das + extra_ss)
```

### 6.3 Term-by-term

| Term | Math | Code | OK |
|------|------|------|----|
| `mu_comb` | `ψ u'(c_{t+1}) + (1-ψ) b'(sR_p)` | [solver.py:1112](lifecycle/solver.py#L1112) | ✓ |
| `mup_comb` | `ψ u''(c)·mpc + (1-ψ) b''(sR_p)` | [solver.py:1109,1113](lifecycle/solver.py#L1109) | ✓ |
| FOC sum | `E[μ_comb · dRp/dα]` | [solver.py:1121-1122](lifecycle/solver.py#L1121) | ✓ |
| J extra | `μ_comb · R_p · (dr·dr − σ²)` | [solver.py:1126-1128](lifecycle/solver.py#L1126) | ✓ |
| V_dot | `E[μ_comb · R_p]` | [solver.py:1123](lifecycle/solver.py#L1123) | ✓ |

Same Jacobian-vs-autodiff identity (test 3) applies and passes to fp64
noise.

---

## §7. Boundary FOC and terminal FOC

### 7.1 Boundary

The boundary at `age == retire_age − 1` reuses
`working_foc_jac_ccv` with `income_table[k_eta, i_e] = pension(z_{t+1}[k_eta])`
broadcast across the eps axis (see
[solver.py:2493-2497](lifecycle/solver.py#L2493) for the pmap path and
[solver.py:2606-2610](lifecycle/solver.py#L2606) for the vmap-only path).
The same `(iz_lo, frac_z)` that brackets z_{t+1} for the c_next gather is
used for the pension interpolation, so the agent's age-66 saving correctly
anticipates age-67 retirement income at the realised z′. The eps axis sums
to one (eps_weights), so the broadcast is mathematically a no-op — the
implementation just keeps the kernel signature uniform with working.

**Verdict:** mathematically the same FOC as working with degenerate eps;
code path is faithful.

### 7.2 Terminal

```
V_T(z, s_T, W_T) = max_{c, α_s, α_b} { u(c) + β · E_T[ b(s · R_p, A) ] }
```

FOC w.r.t. (α_s, α_b) given s:

```
foc_s := E_{v,ξ}[ b'(s · R_p) · R_p · dr/dα_s ]
foc_b := analogous
```

Code: `terminal_foc_jac_ccv` at [solver.py:850-883](lifecycle/solver.py#L850).
No alive piece (no continuation); only the bequest contribution. Term-by-term:

| Term | Math | Code | OK |
|------|------|------|----|
| `μ` | `b'(sR_p)` | `bequest_mu_and_mup(sR_p, A_is, ...)` | ✓ |
| `μ'` | `b''(sR_p)` | `mup` returned from same call | ✓ |
| FOC `foc_s` | `E[μ · R_p · dr/dα_s]` | [solver.py:871](lifecycle/solver.py#L871) | ✓ |
| `J_ss` linear | `E[μ' · s · (R_p · dr/dα_s)²]` | [solver.py:880](lifecycle/solver.py#L880) | ✓ |
| `J_ss` extra | `E[μ · R_p · ((dr/dα_s)² − σ²_xr)]` | [solver.py:876,880](lifecycle/solver.py#L876) | ✓ |
| V_dot | `E[μ · R_p]` (used by EGM) | [solver.py:873](lifecycle/solver.py#L873) | ✓ |

Test 2 below: terminal autodiff of `E[b(sR_p)]` gives gradient matching
`foc_s · s_val` to fp64 noise (rel diff 0); Hessian matches `J_** · s_val`
to fp64 noise. **Strongest possible identity check.**

---

## §8. Variant FOC `working_foc_jac_ccv_pi_z`

[solver_pi_z_variant.py:125-247](lifecycle/solver_pi_z_variant.py#L125)
replaces the integration over the persistent-income innovation `η`
(quadrature with weights `eta_weights` and broken-out `(iz_lo, frac_z)`
brackets at off-grid `z_{t+1} = ρ z + η`) with a discrete sum over the
**z-grid** weighted by Tauchen-style transition probabilities `Pi_z[z, z']`.

Math:

```
foc_s := E_{v,ξ}[ E_{z',ε}[ ψ u'(c_{t+1}(s R_p + Y(z', ε))) · R_p · dr/dα_s ] ]
       + (bequest part as before)
       = sum_{kv, kr} weight_kv_kr · sum_{z', ε} Pi_z[z, z'] · eps_weight[ε] · ...
```

Transcription:
- `z_probs = Pi_z[z_idx]` ([solver_pi_z_variant.py:603](lifecycle/solver_pi_z_variant.py#L603))
- `weight_full = weight_kv_kr ⊗ z_transition_probs ⊗ eps_weights`
  ([solver_pi_z_variant.py:208-212](lifecycle/solver_pi_z_variant.py#L208))
- `c_at_xn` interp uses `_interp_c_and_mpc_at_zgrid_cell` instead of bracketed
  z-interp (the z_next state IS a grid node, so `iz_lo = z'` and no fractional
  weighting is needed).

The bequest piece is identical to production
([solver_pi_z_variant.py:163-178](lifecycle/solver_pi_z_variant.py#L163)).
The alive piece swaps the (eta, eps) axes for (z', eps) but is otherwise
arithmetically equivalent. The `extra_ss/bb/sb` Jacobian convexity
corrections appear at lines 226-234 in the same form as production. **No
mathematical drift from the production kernel** beyond the choice of
discretisation for the persistent-income dynamics.

**Verified the Jacobian internal consistency via autodiff** (same test
pattern as §5/§6, omitted for brevity — passes).

---

## §9. EGM mechanics

### 9.1 The Euler equation being inverted

Stochastic Euler equation for an interior consumer with mortality and
luxury bequest:

```
u'(c_t) = β · E_t[ ψ · u'(c_{t+1}) · R_p + (1 − ψ) · b'(s · R_p) · R_p ]
       = β · E_t[ μ_comb · R_p ]
       = β · V_dot
```

Inverting under CRRA:

```
c_t = (β · V_dot)^{-1/γ}
```

Code: [solver.py:1322-1323](lifecycle/solver.py#L1322):

```python
beta_e = jnp.maximum(beta * V_dot, euler_inv_floor)
c_opt = jnp.maximum(beta_e ** (-1.0 / gamma), min_consumption)
```

`V_dot = e_sum` is the sixth return of every FOC kernel
(terminal/retirement/working). `e_sum` is computed as `E[μ_comb · R_p]`
exactly as required. **The EGM inversion is mathematically correct.**

The `euler_inv_floor = 1e-20` clamp is on the **right side** (`β · V_dot`,
inside the inverse): it prevents `beta * V_dot` from being zero or negative
before the `^(-1/γ)` power. Since both terms in `μ_comb` are nonnegative
(`u' = c^{-γ} ≥ 0`, `b' = b_bar (W/A + δ)^{-γ}/A ≥ 0`) and `R_p > 0`, the
expectation is non-negative; the clamp only fires under fp64 underflow at
extremely high savings (`c^{-γ}` collapse). **No bias direction:** the
clamp returns a finite `c_opt` instead of `inf`; downstream `min_consumption`
floor on `c_opt` is the second guard.

### 9.2 EGM scan

[solver.py:1261-1346](lifecycle/solver.py#L1261). For each `s ∈ s_grid`
in parallel under `vmap`:
1. Build `foc_factory(s_val)` returning a closure of the appropriate
   `*_foc_jac_ccv`.
2. Scale-normalise: `e0 = foc_fn(0,0)[5]`, `inv_foc_scale = 1/max(|e0|, 1e-30)`.
   FOC and Jacobian are divided by this scalar; **the root is unchanged**
   (a positive multiplicative scaling of f doesn't move where f = 0); the
   Newton step `J^{-1} f` is exactly invariant under uniform scaling of
   `(f, J)`. V_dot/e is **NOT** scaled (so the inversion uses the unscaled
   continuation expectation).
3. Run `newton_2d_with_line_search` to get `(α_s*, α_b*)` and `V_dot`.
4. `c_opt = (β·V_dot)^{-1/γ}` (above).
5. If `s ≤ tiny_savings = 1e-6`: override to `c = min_consumption`,
   `α = init_α` (cold-start fallback). The Newton call still runs but its
   outputs are discarded — wasteful but not incorrect.
6. `x_egm = c_opt + s` (implied wealth).

### 9.3 `egm_anchor`

The anchor `(x = egm_anchor, c = egm_anchor, α = 0)` is prepended at
[solver.py:1336-1344](lifecycle/solver.py#L1336). With
`egm_anchor = 1e-10`, this is far below any production wealth_min (≥ 0.05),
so it serves as a **left-end sentinel** for the linear interp lift, not a
genuine policy point. After the Path-B clamp (§9.5) it is masked out for
all real wealth-grid points.

### 9.4 Lift to wealth grid

[solver.py:1349-1416](lifecycle/solver.py#L1349). After sorting `x_egm`,
`jnp.interp(wealth_grid, x_sorted, c_sorted)` is the standard EGM lift.
Mathematically: given the endogenous tuples `(x_i, c_i, α_i)` from EGM,
recover the policy on the fixed wealth grid by piecewise linear interp.

### 9.5 Path-B clamp (constrained corner)

[solver.py:1409-1415](lifecycle/solver.py#L1409):

```python
is_real = c_sorted > 2.0 * min_consumption
first_real_idx = jnp.argmax(is_real)
W_min_real = x_sorted[first_real_idx]
constrained = wealth_grid < W_min_real
c_w = jnp.where(constrained, wealth_grid, c_w)
a_s_w = jnp.where(constrained, jnp.zeros_like(a_s_w), a_s_w)
a_b_w = jnp.where(constrained, jnp.zeros_like(a_b_w), a_b_w)
```

This clamps wealth-grid points below `W_min_real` to the genuine
borrowing-constrained corner `c = W, α_s = α_b = 0`. The threshold
`c > 2 · min_consumption` cleanly separates the egm_anchor (1e-10) and
tiny_savings fallback (1e-10) sentinels from real interior solves.

**Math check:** at the constrained corner, `s = W − c = 0`, so the
portfolio-weights are mathematically irrelevant. The convention `α = 0`
(invest nothing because nothing is invested) is standard. The Euler
equation does NOT bind at the corner; the FOC residual reported by
downstream tests (e.g. `verify/ee_residuals.py`) does not have an
interpretation at constrained cells — this is correct and noted in the
code docstring [solver.py:1373-1377](lifecycle/solver.py#L1373).

**Important update vs. prior audit.** The earlier review
[MODEL_REVIEW_BELLMAN_FOC_2026-05-09.md §3, §6](docs/scans/MODEL_REVIEW_BELLMAN_FOC_2026-05-09.md)
flagged the absence of a constrained-corner branch as a RED FLAG. The
Path-B clamp added at [solver.py:1393-1415](lifecycle/solver.py#L1393)
**resolves that issue** for wealth-grid points below the smallest real
EGM-solved x. The earlier review's remaining concern about wealth-grid
points *between* `W_min_real` and `s_grid[1]·R_p_min + Y_min` (the small
gap in coverage where the policy is determined by the linear bridge from
`W_min_real`) is unchanged.

### 9.6 Bit-identity caveats

- The line-search and fori_loop architectures are documented as
  "bit-identical math" in the code comments, modulo XLA reduction-order
  effects.
- f32 gather precision option produces ~1e-5 relative drift vs f64 (real
  arithmetic noise from the gather pipeline; see
  [SolverConfig.gather_precision](lifecycle/model.py#L251) doc).
  The cast back to fp64 happens **before** any CRRA / FOC arithmetic, so
  no fp32 quantity flows into Newton state. Mathematically the FOC and
  Jacobian formulas are unchanged; the input `c_next, mpc_next` carries
  ~1e-7 relative noise.

---

## §10. Fixed-point structure (infinite-horizon)

[inf_horizon_solver.py](lifecycle/inf_horizon_solver.py): the retirement
kernel from production is iterated as a Bellman operator
`T : C → C_new` defined by

```
T(C)(z, s, W) := arg max_c { u(c) + β · ψ · E[ V(C; z, s', s · R_p + pension(z)) ]
                                  + β · (1 − ψ) · E[ b(s · R_p) ] }
```

where `V(C; ...)` reconstructs the next-period value via envelope (using
`u'(C(W)) = u'(c_{t+1}(W))`). For `b_bar = 0` (the benchmark) and ψ = 1
(no mortality), this is the standard Carroll buffer-stock-with-portfolio
operator. Contraction proxy: `Z(α) = β · E[exp((1-γ) r_p)]`, computed
in `_compute_stability_proxy`
([inf_horizon_solver.py:319-374](lifecycle/inf_horizon_solver.py#L319)).
Strict contraction iff `Z < 1`. The diagnostic reports `max_proxy` over
state cells; the fixed-point iteration cannot guarantee global contraction
without that bound, but the Catherine/CCV calibration sits well below 1.

Convergence criterion ([inf_horizon_solver.py:649-655 region](lifecycle/inf_horizon_solver.py#L649)):
sup-norm on `xi = c/W` and on portfolio shares `(α_s, α_b)`. **Mathematically
sound** as a Banach contraction stopping rule (sup-norm convergence on
the policy is the appropriate criterion for the policy operator since
`xi` is bounded and `α` is finite-dimensional in this benchmark).

Damping `λ ∈ (0, 1]` is a standard fixed-point relaxation: `C_{n+1} = λ
T(C_n) + (1-λ) C_n`. With `λ = 1` (default) it's plain Picard iteration.

**Verdict:** mathematically sound iteration on the right operator with
appropriate stopping criterion.

---

## §11. Boundary case handling

### 11.1 Tiny savings clamp

`s ≤ tiny_savings = 1e-6` triggers `c = min_consumption, α = init_α`
([solver.py:1325-1328](lifecycle/solver.py#L1325)). The Newton call
**still runs** at the tiny-s point — the result is overwritten, not
prevented. This is **wasteful but mathematically harmless**: the EGM
output point `(x = c + s ≈ s)` lies in the constrained region, and the
Path-B clamp will overwrite it. The α fallback to `init_α` only matters
for the **warm-start** that gets propagated to the next age (Variant B);
under `failure_seed_from_neighbor=True`, neighbor seeding handles this.

**Math correctness:** OK — the tiny-savings cell would never be picked up
by the wealth-grid lift anyway (tiny-s implies tiny-x, which is in the
clamped region).

### 11.2 Singular Jacobian fallback

[solver.py:526-535](lifecycle/solver.py#L526) (while-loop) and
[solver.py:743-752](lifecycle/solver.py#L743) (fori-loop). When
`|det J| < singular_det = 1e-15`:

```python
step_s_grad = -grad_step_size * fs / (err + grad_denom_eps)
step_b_grad = -grad_step_size * fb / (err + grad_denom_eps)
```

This is `−η · f / ‖f‖` — a **descent step in the residual direction**.
Mathematically: for the merit function `Φ = ½‖f‖²`, the steepest descent
direction is `−∇Φ = −J^T f`. When J is near-singular, `J^T f → 0` and the
gradient direction degenerates. The ad-hoc `−η · f / ‖f‖` is **not** the
true gradient direction (it drops the J^T factor), but it is a unit-norm
descent direction in the residual space and the line search guards
against ascent.

**Earlier audit ([MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09.md §6](docs/scans/MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09.md))**
flagged this branch as missing the leading minus and producing anti-progress.
The current code **has** the leading minus (`step_*_grad = -grad_step_size *
... / grad_norm` at lines 527-528 and 744-745). The fix has been applied;
the comment at lines 519-525 / 736-742 explicitly notes "the leading minus
is load-bearing".

**Math correctness:** OK with the current minus sign. A more principled
fix would be Levenberg–Marquardt damping `(J + λI) s = -f` with λ
proportional to `singular_det − |det|`, but the current heuristic with
line-search guard is mathematically defensible (descent toward zero of f).

### 11.3 Line-search exhaustion

If no halving improves `‖f‖`, `EC_NEWTON_FAIL` is recorded
([solver.py:778, 813](lifecycle/solver.py#L778)). The cell's last-good
α is held; the wealth-grid policy at that cell is the previous Newton
state (which may be far from a true root). Downstream:
- The FOC residual is recorded in `age_max_foc` for diagnostics.
- The cell is counted in `age_newton_fail`.
- The α-grid output (warm-start source) is replaced by a converged
  neighbor's α via `_fixup_failed_cells` if
  `failure_seed_from_neighbor=True`
  ([solver.py:3104-3107](lifecycle/solver.py#L3104)).

**Math correctness:** the failure semantics is honest — failed cells
report a non-converged FOC and are not pretended to be optimal. The
downstream wealth-grid policy at a failed cell is the partially-converged
α; this is a numerical-stability issue (the α may be slightly off the
root), not a math correctness issue.

### 11.4 Step cap

`line_search_max_step = 2.0` caps the raw Newton step length per iter
([solver.py:537-540, 754-757](lifecycle/solver.py#L537)). This is a
trust-region heuristic that **does not change the root** but prevents
the Newton step from overshooting wildly when the Jacobian is ill-
conditioned (but not yet singular). Mathematically benign.

---

## §12. Numerical identity verifications

All checks run with `JAX_ENABLE_X64=1` on the CPU backend. Each test is
on a synthetic problem with random but reproducible (seed-fixed) data.

### Test 1 — CCV `dr/dα` matches autodiff

```python
log_R_bill, log_x_s, log_x_b = 0.02, 0.05, 0.01
σ²_xr, σ²_xb, σ_xrxb = 0.025, 0.005, 0.001
α_s, α_b = 0.6, 0.3

# Analytic from _ccv_log_return_and_grad (solver.py:841-842):
dr/dα_s = 0.04720000  ← matches jax.grad(r_p, argnums=0)  diff = 6.94e-18
dr/dα_b = 0.01040000  ← matches jax.grad(r_p, argnums=1)  diff = 0
```

### Test 1b — CCV `d²r/dα²` matches autodiff

```python
d²r/dα_s² (analytic) = -σ²_xr = -0.025  ← jax.hessian gives -0.025  diff = 0
d²r/dα_b² (analytic) = -σ²_xb = -0.005  ← jax.hessian gives -0.005  diff = 0
d²r/dα_sα_b           = -σ_xrxb = -0.001 ← jax.hessian gives -0.001 diff = 0
```

### Test 2 — Terminal FOC matches `dE[b(sR_p)]/dα`

Built `n_sq=3, n_rq=4` synthetic shock tensors with random log_x and
weights summing to 1; `s_val = 2`, `α = (0.5, 0.4)`, γ=5, b_bar=10,
δ=0.005, A=4. Compared `terminal_foc_jac_ccv` outputs against
`jax.grad(E[b(sR_p)])`:

```
AUTODIFF dE/dα_s     = 2.9551060004
CODE     foc_s · s   = 2.9551060004   diff = 0
AUTODIFF dE/dα_b     = -0.4633571354
CODE     foc_b · s   = -0.4633571354  diff = 0
AUTODIFF H[0,0]      = -7.0879808600
CODE     J_ss · s    = -7.0879808600  diff = 0
AUTODIFF H[1,1]      = -1.3290760409
CODE     J_bb · s    = -1.3290760409  diff = 0
AUTODIFF H[0,1]      = -0.1508752259
CODE     J_sb · s    = -0.1508752259  diff = 0
```

**Strongest possible identity:** terminal has no envelope ambiguity
(no continuation value), and the code's `(foc, J) · s_val` exactly
reproduces `(grad, hess) E[b(sR_p)]` to fp64 noise.

### Test 3 — Retirement / Working FOC Jacobian = autodiff Jacobian

Built `n_sq=2, n_rq=3, n_state=2, n_z=4, n_eta=2, n_eps=2` synthetic
problem with arbitrary linear `c_corners` rule (deliberately not the
optimal continuation policy). Computed `(J_ss, J_bb, J_sb)` analytically
in code and compared to `jax.jacfwd(foc_only)`:

```
WORKING:
  dfs/dα_s : autodiff = -0.1982433022, code J_ss = -0.1982433022, rel = 1e-16
  dfs/dα_b : autodiff = -0.0096466351, code J_sb = -0.0096466351, rel = 4e-16
  dfb/dα_s : autodiff = -0.0096466351, code J_sb = -0.0096466351, rel = 2e-16  ← symmetry
  dfb/dα_b : autodiff = -0.0268371969, code J_bb = -0.0268371969, rel = 1e-16

RETIREMENT:
  dfs/dα_s : autodiff = -0.1998509962, code J_ss = -0.1998509962, rel = 0
  dfs/dα_b : autodiff = -0.0097269907, code J_sb = -0.0097269907, rel = 5e-16
  dfb/dα_s : autodiff = -0.0097269907, code J_sb = -0.0097269907, rel = 5e-16  ← symmetry
  dfb/dα_b : autodiff = -0.0270553600, code J_bb = -0.0270553600, rel = 1e-16
```

**The analytic Jacobians for both working and retirement match autodiff
of the analytic FOC residual to fp64 noise.** This is the cleanest
correctness check for the Jacobian: regardless of envelope semantics or
the choice of `c_corners` rule, the second derivative is consistent with
the first derivative as code computes them.

### Test 4 — `c_egm = (β·V_dot)^{-1/γ}` corresponds to the Euler

By construction at the converged `(α_s*, α_b*)`, `V_dot = E[μ_comb · R_p]`
and `c_egm = (β · V_dot)^{-1/γ}` ⇒ `u'(c_egm) = β · V_dot = β · E[μ_comb ·
R_p]`. This is the inverted Euler identity. Mechanical check: substitute
`u' = c^{-γ}` ⇒ `c_egm^{-γ} = β V_dot` ⇒ `c_egm = (β V_dot)^{-1/γ}`. ✓

---

## §13. Findings

| # | Severity | Location | Issue | Fix sketch |
|---|----------|----------|-------|------------|
| F1 | **NONE** (CONFIRMED CLEAR) | `terminal_foc_jac_ccv`, `retirement_foc_jac_ccv`, `working_foc_jac_ccv`, `working_foc_jac_ccv_pi_z` | All four FOC functions and their Jacobians are mathematically correct and match autodiff to fp64 noise. The convexity correction terms (`extra_*`) carry the right sign of `−σ²` from `d²r/dα² = −σ²`. The mortality combine `μ_comb = ψ μ_alive + (1−ψ) μ_bq` is correct with `prob_death = 1 − ψ_z`. | None needed. |
| F2 | LOW (DOC) | `_egm_scan_cell` Euler-inversion pipeline | The "extra `s_val` factor" question: at the FOC root the multiplicative `s` is irrelevant, but the Jacobian's `wmup * s_val` term ([solver.py:1125](lifecycle/solver.py#L1125), 1180, 1232) is load-bearing for the off-root linearisation. This is correct in code but is not documented anywhere; a reader cross-checking `J_ss = sum(jac_lin · dRp_das²)` against `d²/dα² E[V_{t+1}]` could be tripped up because the latter has an explicit `s²` factor whereas code shows only `s`. The reason: the FOC residual is divided by `s` (relative to the textbook gradient), so its derivative carries one factor of `s`, not two. | Add a 2-line comment near [solver.py:1125](lifecycle/solver.py#L1125) explaining "The FOC residual is `dJ/dα / s_val`; its Jacobian therefore carries one factor of s_val (not s²)." |
| F3 | LOW (DOC) | `_lift_to_wealth_grid` Path-B clamp | Path-B clamp resolves the prior RED FLAG from [MODEL_REVIEW_BELLMAN_FOC_2026-05-09.md §3](docs/scans/MODEL_REVIEW_BELLMAN_FOC_2026-05-09.md) about the unsolved constrained corner. But the prior audit's note still stands as documentation: callers reading the code without the new clamp's history may not realise the borrowing constraint is now enforced explicitly at low-W cells. | Cross-link the Path-B clamp at [solver.py:1393-1415](lifecycle/solver.py#L1393) from `docs/STATE_SPACE.md §4` and from the prior MODEL_REVIEW_BELLMAN_FOC scan. |
| F4 | LOW (NUMERICAL) | `singular_det = 1e-15` fallback at [solver.py:526-528, 743-745](lifecycle/solver.py#L526) | The `−η · f / ‖f‖` ad-hoc descent direction is not the true `−∇Φ = −J^T f`; it drops the `J^T` factor. Mathematically still a descent direction in residual norm (line search backstops), but ad-hoc relative to a Levenberg-Marquardt damping `(J + λI) s = −f` which would respect the local geometry. Per the prior audit, empirical failure rates do not show a singular-driven mode, so the practical impact is minimal. | Optional: replace with Levenberg–Marquardt `(J + λI) s = −f` with `λ = max(0, singular_det − |det|)`. Math becomes more principled at near-singular cells. Not blocking. |
| F5 | LOW (DOC) | `_egm_scan_cell` `tiny_savings = 1e-6` branch | The Newton call at tiny-s still runs (and is wasted), then its outputs are overwritten with `c = min_consumption, α = init_α`. Mathematically harmless because Path-B clamp covers low-W, but unnecessary compute. The "tiny" threshold `1e-6` is somewhat arbitrary and never documented relative to `s_grid[0] = 1e-8`. With the canonical s_grid, only the smallest savings point trips this branch. | Consider `if jnp.any(s_grid <= tiny_savings)` mask outside the per-savings vmap to skip the Newton call entirely. Minor compute saving; no math change. |
| F6 | LOW (DOC) | `working_foc_jac_ccv_pi_z` variant module docstring | The Pi_z variant is mathematically equivalent to the production kernel with a different discretisation of the persistent-income transition (Tauchen-style discrete sum vs. quadrature). The variant module docstring at [solver_pi_z_variant.py:1-17](lifecycle/solver_pi_z_variant.py#L1) calls out the reducibility caveat at canonical (rho=0.991), but does not cite the math equivalence proof. | Add a one-line note: "Mathematically: `sum_{z'} Pi_z[z, z'] f(z') = E[f(z')|z]` exactly; quadrature variant approximates same integral via Gauss-Hermite. Both share the bequest piece byte-identically." |
| F7 | INFO | Boundary FOC reuses `working_foc_jac_ccv` with degenerate eps axis | Mathematically sound — pension is deterministic given z′, so the eps integration over a constant collapses to the constant. But this means the eps sub-axis is **wasted compute** at the boundary age (one extra outer-product dim with each entry the same scalar). | Optional: a dedicated `boundary_foc_jac_ccv` without the eps axis would be ~`n_eps`× faster at one age. Not blocking. |
| F8 | MEDIUM (DOC) | Inf-horizon contraction proxy reports `max_proxy` per cell | `_compute_stability_proxy` ([inf_horizon_solver.py:319-374](lifecycle/inf_horizon_solver.py#L319)) computes `Z(α) = β · E[exp((1-γ) r_p)]` cell-by-cell at one mid-wealth, mid-z point. Reports the max across i_s. **Math gap:** Z must be < 1 **at every cell visited by the iteration**, not just the probe cell. The mid-z, mid-w probe is a representative-agent proxy; if a high-σ² cell at the boundary has Z ≥ 1, the iteration may still converge in practice but the contraction guarantee is not established. | Run the proxy across all (i_z, i_s, i_w_trim) cells, not just one (i_z, i_w). Report 95th percentile and max. Or document the probe-only nature explicitly. |

**No HIGH-severity findings.** The math is consistent end-to-end; the
analytic Jacobians match autodiff in every test. Prior audits' RED FLAGS
(constrained corner, singular fallback sign) have been resolved in the
current code.

---

## §14. Verdict

**PASS-WITH-CAVEATS.**

The four FOC kernels (`terminal`, `retirement`, `working`, `boundary`-via-
working, `working_pi_z` variant) are **mathematically correct**. Every term
in the FOC residual matches the analytic Lagrangian for the corresponding
Bellman equation; every term in the analytic Jacobian matches JAX autodiff
of the FOC residual to fp64 noise; the second-derivative `extra_*`
convexity corrections carry the correct sign of `−σ²` from `d²r/dα² = −σ²`.
The terminal FOC additionally matches `(grad, hess) E[b(sR_p)]` directly
(no envelope ambiguity), to fp64 noise. The EGM Euler inversion
`c_egm = (β · V_dot)^{-1/γ}` correctly inverts `u'(c) = β · E[μ_comb · R_p]`
under CRRA, and the `euler_inv_floor` clamp is on the correct side and
unbiased. The Path-B clamp at `_lift_to_wealth_grid` correctly enforces
the borrowing-constrained corner `c = W, α = 0` for wealth-grid points
below the smallest interior EGM solve.

**Caveats** are all LOW severity (documentation polish, optional numerical
refinements — see §13). The single MEDIUM finding is the inf-horizon
contraction-proxy probing only one cell (F8), which is a documentation
clarity issue rather than a math error.

The math is ready for production. The remaining items in §13 are
follow-ups, not blockers.
