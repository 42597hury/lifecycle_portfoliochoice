# HANDOFF — Theory review: portfolio FOC pathology near `sR_p = 0`

## Scope

You are a theory reviewer. **You are not asked to write code.** I want a careful read of the model's first-order conditions in retirement (and survival continuation in working age), a verification of where the suspected pathology lives, and a clear answer to the questions in §6 below.

The empirical context: a CRRA-with-luxury-bequest lifecycle model with three risky assets (bill / stock / bond). Quadrature-based solver delivers Euler residuals that are 2 orders of magnitude above publication grade in retirement. The residual is concentrated at leveraged cells (|α_b| ≥ 1 OR |α_bill| ≥ 1). My current hypothesis is that **the pathology is in the portfolio FOC, not in the consumption FOC**, because the consumption choice doesn't see the kink directly while the portfolio choice does — but I want this verified, refined, or refuted.

## 1. Model summary

CRRA period utility, β-discounted, with survival probability ψ_t and a luxury (De Nardi 2004) shifted bequest:

```
u(c)   = c^(1−γ)/(1−γ)
b(W)   = b̄ · (max(W,0)/A + δ)^(1−γ) / (1−γ),   δ = 0.005 (numerical regularizer)
```

State: persistent income index z, three-dim macro state s (cy spread, term spread, 1y yield), wealth-on-hand x. Decision: consumption c, savings s = x − c, and portfolio shares (α_s, α_b, α_bill = 1 − α_s − α_b). `α ∈ [−6, +6]^2` (leverage cap).

Returns: `R_p = α_s R_s + α_b R_b + α_bill R_bill`. R_s, R_b, R_bill are jointly lognormal with conditional means driven by a VAR in s. State innovation v^s also enters return means via a coupling matrix M (`μ_r = const + A_r · s + M · v^s`).

Bequest at death is **realised wealth, not chosen**: heir inherits `max(s · R_p, 0)`. Below zero the heir gets nothing — the agent is bankrupt at death.

## 2. The Bellman recursion in retirement

For t ∈ {retire_age, ..., terminal_age − 1}, with continuation value V_{t+1}:

```
V_t(x, s, z) = max_{c, α_s, α_b}  u(c)
              + β · ψ_t · E[V_{t+1}(x', s', z) | s, α]
              + β · (1 − ψ_t) · E[b(s · R_p) | s, α]
```

with `s = x − c`, `x' = max(s · R_p, 0) + π_{t+1}(z)`, and `s'` evolving via VAR. The hard floor `max(·, 0)` on `x'` is the bankruptcy clamp ("Path B" in the codebase).

In the survivor branch, **for `s · R_p ≤ 0` the agent reaches retirement with only pension `π_{t+1}(z)`** — the levered portfolio wiped out savings. The bequest `b(·)` evaluates to zero at the boundary because `max(s · R_p, 0) = 0` and `b(0) = b̄ · δ^(1−γ)/(1−γ)` is a finite constant; its **marginal** w.r.t. agent's choice is zero on the bankrupt branch (since the heir gets nothing regardless of how negative s·R_p went). Verify this is the right interpretation.

## 3. The two FOCs the solver implements

The numba kernel at [lifecycle/solver.py:826-975](../../lifecycle/solver.py#L826) computes simultaneously:

### Consumption FOC (EGM-side)

```
u'(c_t) = β · E[ ψ_t · u'(c_{t+1}(x', s', z)) · R_p
              + (1 − ψ_t) · b'(s · R_p) · R_p ]
        ≡ β · euler_sum
```

`euler_sum` in the kernel ([solver.py:966](../../lifecycle/solver.py#L966)) accumulates `weight · μ_comb · R_p`, where:

```
μ_comb = ψ_t · u'(c_{t+1}) + (1 − ψ_t) · b'(s · R_p)
```

EGM inverts this to back out `c_t` from `s = x − c_t`. The Euler residual reported by the diagnostic is exactly `1 − c_implied / c_policy` where `c_implied` is from this inversion.

### Portfolio FOC (Newton-side)

For each share α_j ∈ {α_s, α_b}:

```
∂V/∂α_j = s · β · E[ μ_comb · (R_j − R_bill) ] = 0
```

In code ([solver.py:967-968](../../lifecycle/solver.py#L967-L968)):

```python
foc_s += weight * μ_comb * Rex_s   # Rex_s = R_s − R_bill
foc_b += weight * μ_comb * Rex_b   # Rex_b = R_b − R_bill
```

Newton drives `(foc_s, foc_b) = (0, 0)` jointly to find optimal shares.

### Where the bankruptcy clamp enters both

[solver.py:923-961](../../lifecycle/solver.py#L923-L961) shows that **at every quadrature node** the integrand has a branch:

```python
sR_p = s_val * R_p
if sR_p > 0.0:
    x_next = sR_p + pension_next_scalar
    mu_bequest, mup_bequest = _shifted_bequest_mu_and_mup(sR_p, ...)
else:
    x_next = pension_next_scalar           # bankruptcy: agent only has pension
    mu_bequest = 0.0
    mup_bequest = 0.0
```

The same `μ_comb` then enters BOTH `euler_sum` (consumption Euler) AND `foc_s, foc_b` (portfolio FOCs).

## 4. The empirical pathology

After δ-shift fixed the unbounded-bequest blowup, residuals at leveraged cells are still bad:

- v4_lobatto retirement simpath EE: mean log10|EE| = −2.57, max −0.08 (target: −4.5 mean / −3 max).
- 167 cells with |EE| > 3% concentrate at i₂=6 corners with α_b ≈ +2 (long-bond leverage).
- Worst leveraged cell: log10|EE| = −0.08 (≈83% rel error). Worst unleveraged cell: −0.59 (≈25%).
- Failure rate scales monotonically with |α|: cells with |α_b| ∈ [2.0, 2.5) have 11× the unleveraged failure rate.
- Lobatto explicitly samples ±7σ tail nodes on the bond-return residual and on state innovations (M-coupled to bond returns). At leveraged α these tail nodes can drive `sR_p` near zero or negative — even though no realised return path in the simulator reaches there.

User's intuition (worth honouring): **historically, no realised return path makes a 100%-equity-or-bond portfolio bankrupt.** The kink fires only because (i) the agent can lever up to ±6, (ii) the conditional return distribution is Gaussian-tailed (unbounded), (iii) Lobatto puts explicit nodes at ±7σ. So `sR_p < 0` is a *quadrature-cell event*, not a *realised event*.

## 5. The current claim, and what I want you to verify

**Claim (to verify or refute):** the dominant pathology is in the **portfolio FOC**, not in the consumption FOC.

Sketch of the argument I want you to walk through carefully:

(a) The consumption FOC integrand is `μ_comb · R_p`. At a quadrature node where `sR_p` flips from positive to negative, R_p flips from positive to negative too. μ_comb also has a step (from `ψ · u'(c_{t+1}) + (1−ψ) · b'(sR_p)` to `ψ · u'(c_{t+1}|x_next=π) + 0`). Both jump sign/magnitude together — the Euler sum is sensitive to the kink but only at cells whose quadrature cloud straddles `sR_p = 0`.

(b) The portfolio FOC integrand is `μ_comb · Rex`, where Rex = R_j − R_bill. **Rex does not flip at the bankruptcy boundary** (it's a difference of two independent gross returns). So the portfolio FOC integrand has a step in μ_comb but is multiplied by a smoothly-varying excess return. The step is, if anything, *more* pronounced relative to the local average of the integrand, because there's no R_p amplitude masking it.

(c) The optimizer first converges the portfolio FOC at the cell (Newton) and *then* the consumption FOC is computed via EGM at the converged α. So if the portfolio FOC has a discontinuous integrand, Newton's converged α is wrong, and the consumption FOC inherits that error rather than producing it.

If (a)-(c) are correct, the fix priority should be on the portfolio FOC's integrand, and the headline EE diagnostic (which measures only the consumption FOC residual) is **a downstream symptom**, not the seat of the pathology.

## 6. Questions for you

Please answer each of the following with explicit reference to the math in §2-§3 and the numerics in §4. State assumptions where you make them.

1. **Is the FOC structure in §3 correctly transcribed?** I derived it from the Bellman in §2 but I want you to verify the envelope conditions, in particular the `β · ψ` and `β · (1 − ψ)` coefficients on the survivor and bequest branches in both consumption and portfolio FOCs. Is the standard "expected continuation marginal value times R_p / Rex" form the right one given the *realised* (not chosen) bequest specification?

2. **Where does the kink actually live in each FOC?** Walk through the bankruptcy boundary at `sR_p = 0` for each FOC separately. State precisely:
   - Which terms in `μ_comb` jump at `sR_p = 0⁺ → 0⁻`.
   - Which terms multiplying μ_comb (`R_p` or `Rex_j`) are continuous and which jump.
   - The size of the jump in the integrand at a single boundary node.
   - For the consumption FOC: confirm or refute that R_p and μ_comb jump together such that the *product* may be smaller than the individual jumps.

3. **Is my (a)-(c) argument right that the portfolio FOC is the more-singular integrand?** Walk through. If wrong, where? If right, sharpen it.

4. **Does the consumption FOC's Euler residual `1 − c_implied/c_policy` correctly diagnose the pathology, given that the pathology is upstream in the portfolio choice?** I'm asking: even if the portfolio FOC is the problem, does the consumption residual capture it because the wrong α propagates into the consumption inversion? Or does the consumption residual *underreport* the actual policy error because c_implied and c_policy share the bad α?

5. **Is the EE diagnostic the right gate at all, given the realised-bequest specification?** The standard Euler equation `u'(c) = β E[u'(c') R_p]` assumes a smooth integrand. With the hard clamp, the equation should formally include the boundary contribution `(1−ψ) · ∫_{R_p < 0} b'(sR_p) · R_p · dF` which is **identically zero** (b' = 0 on the bankrupt branch), but the *quadrature approximation* of that integral has a step error. Is there a more honest diagnostic — e.g. measuring the portfolio FOC residual `(foc_s, foc_b)` directly, or measuring the value-function residual via a separate forward-evaluation — that would surface the pathology more cleanly than the consumption Euler residual does?

6. **The leverage cap as a fix.** The cap ±6 currently allows α positions where `R_p < 0` is reachable in the conditional cloud. If we tighten the cap such that for every (state, α) cell visited by the policy, `R_p > 0` at every quadrature node — does the model still admit the original Merton-optimal interior solution at the cells where the cap doesn't bind? In particular: at cells where the unconstrained Merton α has |α| > cap, is the constrained optimum (α at cap with KKT multiplier μ > 0) the *correct* optimum, or does the binding cap distort the policy at neighbouring (interior) cells through the value-function continuation? Sketch the argument both ways.

7. **Truncated returns as a fix.** Replace the Gaussian-tail Lobatto rule with a quadrature on a truncated return distribution (e.g. truncate at ±4σ on each axis). Economically defensible? In particular, does the agent's optimal α under the truncated rule match the Merton α from the untruncated rule in the limit of large truncation σ? At what truncation σ does the truncated and untruncated solution diverge meaningfully?

8. **Smoothing the clamp as a fix.** This is the proposal in [HANDOFF_BANKRUPTCY_CLAMP_SMOOTHING.md](HANDOFF_BANKRUPTCY_CLAMP_SMOOTHING.md). The smoothing replaces the hard `if sR_p > 0` with a C^∞ blend over a window ε around `sR_p = 0`. The agent acts as if a small fraction of bankrupt outcomes still produce some bequest. Does this smoothing preserve the Bellman's interpretation of bequest as *realised* `max(s · R_p, 0)`, or does it implicitly move the model from a realised-bequest spec to a chosen-bequest spec (with a small "default insurance" term)? If it does shift the spec, how do we defend it?

9. **The right thing to diagnose first.** Given your answers above, what is the smallest, cleanest test that would discriminate between the candidates in (6)–(8)? I want a single test that answers: "is the pathology in the portfolio FOC integrand, or downstream of it?" — so we know whether smoothing the clamp / capping leverage / truncating returns is even targeting the right object.

## 7. Reading order

1. **The Bellman and bequest spec**: [docs/DESIGN.md](../DESIGN.md) (full model writeup), [docs/UTILITY.md](../UTILITY.md) (CRRA + luxury-bequest math), [HANDOFF_LUXURY_BEQUEST_IMPLEMENTATION.md](HANDOFF_LUXURY_BEQUEST_IMPLEMENTATION.md) (the δ-shift implementation and its theoretical justification — De Nardi 2004 framing).

2. **The retirement FOC kernel**: [lifecycle/solver.py:826-975](../../lifecycle/solver.py#L826-L975). Bequest helper `_shifted_bequest_mu_and_mup` near top of solver.py — read it for the analytic form of `μ_bequest` and its derivative on the solvent branch.

3. **The working-age FOC kernel**: [lifecycle/solver.py:984-1060](../../lifecycle/solver.py#L984-L1060). Same structure as retirement plus a working-survival-and-promotion term.

4. **The current EE diagnostic kernel**: [scripts/diagnostics/_diag_euler_errors.py:706-738](../../scripts/diagnostics/_diag_euler_errors.py#L706-L738) for retirement; the working-age block analogously. Confirms the diagnostic measures the consumption Euler, not the portfolio FOC.

5. **The smoothing proposal**: [HANDOFF_BANKRUPTCY_CLAMP_SMOOTHING.md](HANDOFF_BANKRUPTCY_CLAMP_SMOOTHING.md). Read for the empirical decomposition of the residual into mode (1) low-x and mode (2) leveraged-kink — the latter is what we're targeting.

6. **The Lobatto config tracker**: [docs/notes/LOBATTO_CONFIG_TRACKER.md](../notes/LOBATTO_CONFIG_TRACKER.md). For empirical context on which knobs have been tested and what works at the wide vs narrow envelope. §3.4 in particular: "wider grid is harder regardless of Lobatto config" — the residual ceiling at ~−2.27 mean log10|EE| at n_stds=2.93 across four configs is the empirical signature of the structural problem.

7. **The diagnostic-filter gap**: the current `cell_set=unconstrained` filter strips savings≈0 cells but does NOT strip cap-bound cells. Means any cap-tightening experiment will report KKT slack as Euler residual unless the diagnostic is fixed. Code reference: [_diag_euler_errors.py:1007](../../scripts/diagnostics/_diag_euler_errors.py#L1007) sets `is_constrained` only on `s/x < kink_tol`.

## 8. Deliverable

A single markdown response, ~1500–2500 words, that:

1. Confirms or corrects the FOC transcription in §3.
2. Answers questions 1–9 in §6 directly, in order.
3. **States whether the portfolio FOC or consumption FOC is the seat of the pathology**, with the reasoning that justifies the call.
4. Recommends ONE diagnostic to run next that would discriminate between the candidate fixes (clamp smoothing / leverage cap / truncated returns) on a theoretically-clean basis.
5. Flags any specification incoherence I haven't surfaced — e.g. is the realised-bequest semantics really consistent with how the FOC integrates over the bankrupt branch? Are we double-counting the death-state bequest at `sR_p = 0` (where `b(0) = b̄ · δ^(1−γ)/(1−γ) > 0` evaluated at the boundary, not 0)?

Length cap: 2500 words. Don't restate the model — analyse it.
