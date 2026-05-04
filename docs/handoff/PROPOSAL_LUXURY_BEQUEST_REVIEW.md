# PROPOSAL — Luxury-bequest shift to fix bankruptcy-boundary quadrature spike

## What you are asked to do

Review whether the proposed change to the bequest specification is (a) **economically defensible** as a structural feature of the model and (b) **likely to solve** the numerical pathology documented in §2. Specifically:

1. Is the proposed bequest formula a sensible model of post-mortem preferences? Can we cite De Nardi (2004) and ship?
2. Does it correctly bound the FOC integrand the way we claim?
3. Are there economic distortions or second-order effects we haven't considered (e.g., effects on the consumption-savings trade-off, optimal portfolio composition near retirement, or wealth-at-death distribution)?
4. Is the recommended calibration band for δ defensible, and how should we choose the final value?
5. Anything else that should make us pause before implementing.

Length cap: under 1000 words. We need a verdict, not a treatise.

## 1. The model and the current bequest specification

A retired household with savings `s_t`, portfolio shares `(α_s, α_b, α_bill)`, and realised portfolio gross return `R_p` faces the wealth transition

`W_{t+1} = s_t · R_p + income_{t+1}`.

If the household dies at the end of period `t` with positive estate, heirs receive a CRRA-style bequest utility (Catherine 2025 specification):

```
b(W, A) = b_bar · (W/A)^(1-γ) / (1-γ)        (current model)
```

where `W = s_t · R_p` is the estate, `A` is the annuity factor at the household's nominal-yield state, `b_bar` is the bequest weight, and `γ = 5` is the CRRA curvature. The model imposes a **bankruptcy clamp** ("Path B"): if `W ≤ 0`, the estate is set to zero and the bequest contributes nothing.

The marginal utility of estate that enters the household's FOC is

```
μ_bequest(W) = b_bar · (W/A)^(-γ) / A · 1{W > 0}.
```

For γ = 5, `(W/A)^(-γ)` has an **Inada singularity** at `W → 0+`: the integrand diverges as `W^(-γ)` just inside the solvent boundary.

## 2. The numerical pathology

Empirical evidence from [scripts/diagnostics/_diag_split_rule_sanity.py](../../scripts/diagnostics/_diag_split_rule_sanity.py) on the production v4_lobatto bundle. At the worst-leveraged retirement-age sim-path cell (age 88, `α = (+0.55, −1.21, +1.66)`, `s_val = 9.67`), one Lobatto quadrature node lands at `s·R_p = 7.7 × 10⁻⁶`, where:

| | value |
|---|---:|
| node weight | `1.12 × 10⁻¹¹` |
| `s·R_p` at node | `+7.7 × 10⁻⁶` |
| `μ_bequest` at node | `+1.20 × 10³⁰` |
| contribution to `E[μ·R_p]` | **`+8.81 × 10¹¹`** |

That single node dominates the integral by 13 orders of magnitude. A reference rule (dense Gauss-Hermite, K=(5,5,101) returns × K=(7,7,7) state) gives `E[μ·R_p] = +0.66`, of which the bequest piece is `+0.023`. Implied Euler-equation residual: `EE = +0.996` under Lobatto vs `+0.0014` under dense GH. Re-solving optimal `α` under the dense rule moves leverage materially: `α_b: −1.21 → −2.35`, `α_bill: +1.66 → +2.64`.

Diagnosis: it isn't only the bankruptcy *kink* that's biting (that's the C⁰ discontinuity of `1{W>0}`). The dominant problem is the **Inada blow-up** of `μ_bequest(W) = b_bar · (W/A)^(-γ)` at `W → 0+`. With γ = 5 the integrand peak grows as `W^(-5)`, and Lobatto's tail-coverage design lands a node arbitrarily close to W = 0.

## 3. The proposal

Replace the bequest specification with **De Nardi (2004)'s luxury-bequest shift**:

```
b(W, A) = b_bar · (max(W, 0)/A + δ)^(1-γ) / (1-γ)
μ_bequest(W) = b_bar · (max(W, 0)/A + δ)^(-γ) / A · 1{W > 0}
```

with **δ ≈ 0.005** (annuity-normalised; equivalent to a few percent of one annuity unit).

Properties:

- **Bounded marginal utility**: `μ_bequest ≤ b_bar · δ^(-γ) / A` everywhere. For γ=5, δ=0.005, A≈4: cap ≈ 8 × 10¹¹. The Lobatto spike of 10³⁰ is impossible.
- **Continuous** at `W = 0`: both branches of the formula equal `b_bar · δ^(1-γ)/(1-γ)`. The kink in `μ_bequest` (derivative discontinuity) survives, but its size is bounded.
- **Asymptotically identical** to the original CRRA: for `W/A >> δ`, `(W/A + δ)^(-γ) ≈ (W/A)^(-γ)` to within `(δA/W)·γ` percent. With δ=0.005, γ=5, A=4: agreement is <0.5% above `W = 5`, <0.1% above `W = 10`.
- **The hard bankruptcy clamp survives** via `max(W, 0)`. Bankrupt households still leave zero estate; debt does not pass to heirs.

Implementation: three call sites — `compute_foc_jac_retirement_quad` and `compute_foc_jac_working_quad` in [lifecycle/solver.py](../../lifecycle/solver.py), and `_compute_euler_sum_retirement_continuous` / `_compute_euler_sum_working_continuous` in [scripts/diagnostics/_diag_euler_errors.py](../../scripts/diagnostics/_diag_euler_errors.py). Single shared helper. Simulator stays hard-clamped (single realisation, no integral to spike). Lobatto tail-rule stays (it's needed for tail-coverage / arbitrage prevention; nothing here changes that).

## 4. Economic interpretation

This is the **luxury-bequest** specification of De Nardi (2004), adopted by De Nardi, French, Jones (2010), Lockwood (2018), and others. Standard reading: bequests are a luxury good — δ controls *how much wealth* triggers the bequest motive materially. δ small → households care about bequests at all wealth levels (close to original CRRA). δ large → only wealthy households care about bequests in any operative way.

Empirical context: De Nardi calibrates `θ_2` (her shifter) to match the right tail of the wealth-at-death distribution. Plausible values in dollars-of-consumption terms put `δ` on the order of one to two annuity units when normalised to her parameters. Our proposed δ = 0.005 is **at the very small end** of that band — close to original CRRA, large enough to bound the integrand.

## 5. What we are NOT changing

- The hard bankruptcy clamp `max(W, 0)` on the estate (agent's `x_next` still equals `income` when bankrupt; debt doesn't carry).
- The simulator's realised bequest (single realisation, no quadrature, hard clamp stays).
- The Lobatto tail rule (`Z = 7` on stock and bond axes; needed for tail coverage).
- The CRRA curvature γ = 5 or the bequest weight `b_bar = 10`.
- Any other aspect of the household problem (consumption-savings, portfolio choice mechanics, state dynamics).

## 6. Questions for review

A. **Economic defensibility**: Is the De Nardi luxury-bequest shift a defensible structural choice for this paper, given the model class (retirement-phase life-cycle with bequest motive, CRRA preferences, three-asset portfolio)? Does it interact problematically with anything in the household problem we should know about?

B. **Calibration**: Is δ = 0.005 (annuity-normalised) defensible as a "small regularization, close to CRRA"? Or should we calibrate δ to a moment (wealth-at-death distribution, share of households with positive estate, etc.) and report sensitivity bands? What's the smallest defensible δ given that we want to bound `μ_bequest` to a finite value?

C. **Numerical claim**: Is the bound `μ_bequest ≤ b_bar · δ^(-γ)/A` correct, and does that bound suffice to suppress the Lobatto spike to a level the rule's polynomial-exactness can absorb? We claim yes (residual error becomes O(1/K²) from the surviving derivative kink). Do you agree, or is there a subtler interaction we've missed?

D. **Policy implications**: With the spike removed, optimal leverage at the worst cell moves from `α_b = −1.21` to `α_b = −2.35`. Is this *more aggressive* leverage economically reasonable (Lobatto was over-cautious because of the phantom spike), or does it indicate a deeper specification issue?

E. **Anything we should pause on** before we implement.

## 7. Proposed validation if you green-light

1. Apply the change to the three FOC kernels with δ=0.005 via shared helper.
2. Re-run [scripts/diagnostics/_diag_split_rule_sanity.py](../../scripts/diagnostics/_diag_split_rule_sanity.py); confirm worst-cell per-node bequest contribution drops from 8.8 × 10¹¹ to <0.1.
3. Re-solve the canonical bundle. Run full EE diagnostic battery; expect `mean log10|EE| < −4.5` and `max < −3.0` at retirement (publication gates).
4. Sensitivity: re-solve at δ ∈ {0.001, 0.005, 0.01, 0.02}. Report worst-cell α shifts. If policy is stable to within ~5% across the band, ship at δ=0.005.

## References

- De Nardi, M. (2004). "Wealth Inequality and Intergenerational Links." *Review of Economic Studies* 71(3), 743–768.
- De Nardi, M., French, E., Jones, J. (2010). "Why Do the Elderly Save? The Role of Medical Expenses." *Journal of Political Economy* 118(1), 39–75.
- Lockwood, L. (2018). "Incidental Bequests and the Choice to Self-Insure Late-Life Risks." *American Economic Review* 108(9), 2513–2550.
