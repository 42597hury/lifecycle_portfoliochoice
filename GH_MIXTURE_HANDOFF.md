# HANDOFF: Mixture-Normal Gauss–Hermite Quadrature — Mathematical Audit

**To:** math-focused validation agent
**From:** labour-income validation session (2026-04-16)
**Scope:** Verify, using mathematics, that the two-component GH quadrature
treatment of the persistent income innovation η (and the analogous transitory
ε) is correct. Do NOT modify code. Do NOT run MC simulations as a substitute
for the mathematical check — numerical agreement is not what's in question.

---

## 0. Context (one paragraph)

The persistent income state follows `z_{t+1} = ρ·z_t + η`, where η is a
two-component normal mixture (Guvenen et al. 2021 / Catherine 2025). The
solver evaluates the continuation expectation `E[V(ρ·z_t + η)]` by Gauss–
Hermite quadrature. The code discretizes the mixture by applying GH
*separately to each component*, then concatenating nodes and weights with the
component mixture probabilities. A key subtlety: the code **overrides** the
model's `mu_eta2` parameter inside the quadrature construction, substituting
a value that algebraically enforces `E[η]=0` for any `μ_1, p, σ_1, σ_2`. Your
job is to verify that this discretization (including the override) correctly
approximates the theoretical mixture integral.

---

## 1. The theoretical object

```
E[V(z_{t+1}) | z_t = z_i]  =  ∫ V(ρ·z_i + η) · f_η(η) dη
```

where

```
f_η(η)  =  p · φ(η; μ_1, σ_1)  +  (1-p) · φ(η; μ_2, σ_2)
```

and `φ(η; μ, σ)` is the normal density with mean μ and std σ.

Model parameters (from [saved_runs/.../metadata.json](saved_runs/constrained_grid7x7x7_nz11/metadata.json)):

```
ρ         = 0.991
p  = pz   = 0.176
μ_1       = -0.524
σ_1       =  0.113
μ_2       =  0.11192233...   (≈ -(p/(1-p))·μ_1 by construction in config)
σ_2       =  0.046
```

The same structure applies to the transitory shock ε with its own
`(pe, mu_eps1, sigma_eps1, mu_eps2, sigma_eps2)`.

---

## 2. The implementation to audit

[discretization.py:188-213](discretization.py#L188-L213) — `get_eta_quadrature_mixture(model, n_nodes)`:

```python
nodes, weights = roots_hermite(n_nodes)     # raw Hermite nodes/weights
weights = weights / np.sqrt(np.pi)
nodes   = nodes   * np.sqrt(2.0)

# Component 1: uses model's μ_1, σ_1 unchanged
e1 = nodes * model.sigma_eta1 + model.mu_eta1
w1 = weights * model.pz

# Component 2: OVERRIDES μ_2 with an algebraic zero-mean enforcement
mu_eta2_eff = -(model.pz / (1.0 - model.pz)) * model.mu_eta1
e2 = nodes * model.sigma_eta2 + mu_eta2_eff
w2 = weights * (1.0 - model.pz)

eta_nodes   = np.concatenate([e1, e2])       # shape (2·n_nodes,)
eta_weights = np.concatenate([w1, w2])       # shape (2·n_nodes,)
```

The exact same pattern is used for ε in
[get_eps_quadrature_corrected()](discretization.py#L159).

---

## 3. What we claim is true (what you need to verify)

Please produce a written confirmation or rejection of each claim, with the
explicit derivation. If a claim is *approximately* true (e.g. exact only up
to some polynomial degree), state the exactness order.

### Claim A — Linearity split

```
∫ V(ρz + η) f_η(η) dη
  =  p ·∫ V(ρz + η) φ(η; μ_1, σ_1) dη
   + (1-p)·∫ V(ρz + η) φ(η; μ_2, σ_2) dη
```

Trivial but state it for completeness.

### Claim B — Single-component GH reduction

For one Gaussian component with mean μ and std σ, the change of variable
`η = μ + √2·σ·x` produces

```
∫ V(ρz + η) φ(η; μ, σ) dη
  =  (1/√π) ∫ V(ρz + μ + √2·σ·x) e^{-x²} dx
  ≈  Σ_k (h_k/√π) · V(ρz + μ + √2·σ·x_k)
  =  Σ_k w_k · V(ρz + η_k)
```

with `η_k = μ + √2·σ·x_k`, `w_k = h_k/√π`, summing to 1. Verify the
Jacobian, the weight scaling, and the node scaling match the implementation
(`nodes *= √2`, `weights /= √π`, then `nodes·σ + μ`).

### Claim C — Mixture recombination

Concatenating the two single-component rules with mixture weights `p` and
`1-p` approximates the mixture integral, and the concatenated weights sum
to 1 exactly:

```
Σ_j W_j = p · Σ_k w_k + (1-p) · Σ_k w_k = p·1 + (1-p)·1 = 1
```

Confirm. No approximation here — this is exact arithmetic.

### Claim D — Exactness order for each component

GH with K nodes is exact for polynomial integrands of degree ≤ 2K-1 against
the weight `e^{-x²}`. Translated back through the change of variable, each
component's quadrature is exact whenever `V(ρz + μ + √2·σ·x)` is polynomial
in `x` of degree ≤ 2K-1. With `K=3` (config default), degree ≤ 5; with
`K=5`, degree ≤ 9. Verify and state whether this exactness carries to the
moments of η itself:
- Is `∫ η^m φ(η; μ, σ) dη` (the m-th moment of a single Gaussian) reproduced
  exactly by the GH rule for `m ≤ 2K-1`? **This is the key claim for
  variance reproduction.** Derive.

### Claim E — Mixture moments reproduced

Given that each single-component GH rule with K nodes reproduces that
component's moments up to degree 2K-1 exactly, the mixture moments are also
reproduced exactly up to 2K-1:

```
E_discrete[η^m]  =  p · E_GH,comp1[η^m]  +  (1-p) · E_GH,comp2[η^m]
                 =  p · E_true,comp1[η^m] + (1-p) · E_true,comp2[η^m]    (if m ≤ 2K-1)
                 =  E_true,mixture[η^m]
```

Confirm this equality holds component-by-component and therefore for the
mixture. Flag any subtlety (e.g. if `V` is not polynomial, only smooth, what
is the error structure for the mixture).

### Claim F — Zero-mean override: validity

This is the claim we are least confident about. The code computes

```
μ_2_eff  =  -(p / (1-p)) · μ_1
```

and uses `μ_2_eff` in place of `model.mu_eta2`. Observations:

1. Algebraically, `p·μ_1 + (1-p)·μ_2_eff = 0` for any choice of p, μ_1, σ_1,
   σ_2 — the override enforces zero mean exactly at the rule level, before
   any numerical error.

2. The model-configured `mu_eta2` (in `metadata.json`: `0.1119223...`) is
   numerically equal to `-(pz/(1-pz))·mu_eta1 = -(0.176/0.824)·(-0.524) =
   0.1119223...`. So in the current calibration, `μ_2_eff == model.mu_eta2`
   to machine precision — the override is a no-op.

3. If a user supplied a `mu_eta2` that did *not* satisfy
   `p·μ_1 + (1-p)·μ_2 = 0`, the override would silently change the
   distribution being discretized — the discretization would correspond to a
   different mixture than the one the model configuration specifies.

Questions for you to answer:

- **F1.** Is the override mathematically equivalent to "first enforce
  `E[η]=0` on the theoretical mixture, then quadrature it"? That is, the
  code silently redefines the input distribution. Is this the intent, and is
  it documented elsewhere we should find?

- **F2.** Under what circumstance (if any) would the override produce a
  discretized distribution whose variance differs non-trivially from the
  model's intended `Var(η)`? Derive the relationship between the "intended"
  variance (using model's `μ_2`) and the "effective" variance (using
  `μ_2_eff`).

- **F3.** Is there a cleaner mathematically-equivalent form — e.g., rescale
  both components' means so the whole mixture is zero-mean while preserving
  the input `μ_2` — that would avoid silently discarding the model's
  `mu_eta2` parameter?

- **F4.** Verify numerically that
  `Σ_j W_j · H_j = 0` exactly (up to roundoff) under the override — the
  print warning at [discretization.py:209-211](discretization.py#L209-L211)
  suggests this was a concern during development.

### Claim G — Variance reproduction under the override

Using the override `μ_2_eff`, the discretized rule reproduces a variance of

```
Var_eff(η)  =  p · (σ_1² + μ_1²)  +  (1-p) · (σ_2² + μ_2_eff²)
            =  p·σ_1² + (1-p)·σ_2² + p·μ_1² + (1-p)·(p/(1-p))²·μ_1²
            =  p·σ_1² + (1-p)·σ_2² + p·μ_1²·(1 + p/(1-p))
            =  p·σ_1² + (1-p)·σ_2² + p·μ_1² / (1-p)
```

(since `Σ w_k·μ = 0`, this equals `E[η²]` exactly for each component, i.e.
`Var_eff = E[η²]` given zero mean). Verify this derivation, and compare
against the `var_eta` formula used in the z-grid construction at
[discretization.py:131](discretization.py#L131):

```python
var_eta = p*(σ_1² + (μ_1 - μ_η)²) + (1-p)*(σ_2² + (μ_2 - μ_η)²)
```

Note that the grid construction uses the model's `μ_2`, while the quadrature
uses `μ_2_eff`. If the user-supplied `μ_2 ≠ μ_2_eff`, the grid σ_z and the
quadrature σ_η will correspond to slightly different mixtures. Is this
internally inconsistent? In the current calibration both agree (see F2).

### Claim H — ε (transitory) quadrature is analogous

Same analysis applies to
[get_eps_quadrature_corrected()](discretization.py#L159), with all `η`
replaced by `ε` and `(pz, μ_η1, ...)` replaced by `(pe, μ_ε1, ...)`. Confirm
the analogous claims or flag any structural difference.

---

## 4. What NOT to do

- Do NOT re-derive GH quadrature from scratch unless needed to settle a
  specific question. Cite standard references (Stroud & Secrest 1966;
  Abramowitz & Stegun).
- Do NOT propose code changes. Your output is a math audit, not a patch.
- Do NOT validate the z-grid itself or Tauchen — that is already closed.
  See [LABOUR.md Section 5](LABOUR.md#L336).
- Do NOT run Monte Carlo to "check" the math. MC could show numerical
  agreement yet miss a theoretical inconsistency. The point of this audit
  is the theoretical side.
- Do NOT touch the η ↔ return covariance question — that is separately
  deferred. See [INCOME_RETURN_COV_HANDOFF.md](INCOME_RETURN_COV_HANDOFF.md).

A short numerical sanity check at the end (reproducing zero mean and
variance on the specific calibrated parameters) is fine and useful, but
should not replace the derivations above.

---

## 5. Output format expected

A written memo (can be added to LABOUR.md Section 5 or returned as a
standalone note) containing:

1. A one-line verdict per claim (A–H): **confirmed / confirmed with caveat /
   incorrect**.
2. For each "confirmed with caveat" or "incorrect", the derivation showing
   where the claim breaks or what assumption is needed.
3. A direct answer to F1–F4 about the zero-mean override, including whether
   the current behaviour is acceptable or whether LABOUR.md should document
   the override as an approximation (it silently redefines the input
   distribution).
4. Closing verdict: is the mixture-GH treatment as implemented
   mathematically correct for the intended use
   (approximating `E[V(ρz+η)]` in the Bellman equation)?

---

## 6. References that may be useful

- Stroud & Secrest (1966) — standard GH weights/nodes and error bounds
- Abramowitz & Stegun (1964), §25 — orthogonal polynomials
- Tauchen (1986), Tauchen & Hussey (1991) — discretization of AR(1)
  (context, not central)
- Guvenen et al. (2021, "What Do Data on Millions of U.S. Workers Reveal
  About Lifecycle Earnings Dynamics?") — the source mixture specification
- Catherine (2025, §3) — the lifecycle portfolio application

## 7. User preferences (relayed)

- Terse; no trailing summaries
- Markdown code references as `[file.py:42](file.py#L42)`
- Do not dump validation TODOs without approval; work incrementally
- Thesis deadline 2026-05-18; this is an audit, not exploratory research
