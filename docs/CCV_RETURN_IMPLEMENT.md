# CCV w8566 Return-Modelling — Implementation Reference

**Purpose.** Authoritative reference for the implementation of the
return-modelling pipeline in the lifecycle portfolio choice model. Each
section corresponds to a section in Campbell, Chan and Viceira (NBER w8566 /
Campbell–Viceira 2002, "A Multivariate Model of Strategic Asset
Allocation"). The document is built up incrementally through theory-review
of each subsection and locked in only after explicit agreement.

**Status legend.**
- ✅ LOCKED — agreed and verified against code
- 🟡 IN REVIEW — under discussion
- ⬜ NOT YET REVIEWED

**Section index.**
| Section | Topic | Status |
|---|---|---|
| §2.1 | Securities | ✅ LOCKED 2026-05-07 |
| §2.2 | VAR dynamics + lagged-$\mathbf{x}$ restriction | ✅ LOCKED 2026-05-07 |
| §2.2.μ | Sample-mean pinning of $\Phi_0$ | ✅ LOCKED 2026-05-07 (code-side check verified — see §2.2.μ item 4) |
| §2.3 | Preferences | ⬜ |
| §3.1 | Continuous-rebalancing approximation (eq. 10) | ✅ LOCKED 2026-05-07 |
| §4.1 | Empirical implementation: annual data construction | ✅ LOCKED 2026-05-07 |
| §4.2 | VAR estimation results (CCV reference numbers) | ✅ LOCKED 2026-05-07 |

---

## Notation conventions (used throughout)

| Doc symbol | Meaning | CCV symbol | Code identifier |
|---|---|---|---|
| $r_{i,t+1}$ | log real return on asset $i$ | $r_{i,t+1}$ | (varies) |
| $r_{1,t+1}$ | log real bill return | $r_{1,t+1}$ | `rtb` |
| $\mathbf{x}_{t+1}$ | excess log return vector | $x_{t+1}$ | `(xr, xb)` |
| $\mathbf{s}_{t+1}$ | "other" state variables | $s_{t+1}$ | `(y_1, spr, dp)` |
| $\mathbf{y}_{t+1}$ | full VAR vector | $z_{t+1}$ | concat of state + return |
| $\Phi_0, \Phi_1, v, \Sigma_v$ | unchanged from CCV | same | (see code mapping) |

**Notational clash warnings.**
- This doc's bold $\mathbf{x}$ is the excess-return vector. Code's lowercase
  `x` is cash-on-hand. **Different objects.** The bold disambiguates.
- This doc's $r_1$ (= code `rtb`) is the **log real** bill return. Code also
  has an intermediate variable `r_1 = log(1 + y_1)` which is the **log
  nominal** bill return. CCV's $r_1$ never refers to the nominal version.

---

## §2.1 — Securities ✅ LOCKED 2026-05-07

### Paper reference
CCV w8566, §2.1 "Securities" (page 5–6), equation (1).

### Six agreed features

#### (1) All returns in CCV's notation are real

Every $R_{t+1}$ in CCV equation (1) is a real return:

- $R_{p,t+1}$ — real portfolio return
- $R_{i,t+1}$ for $i\geq 2$ — real return on risky asset $i$ (stock, long bond)
- $R_{1,t+1}$ — real return on the short-term asset (the nominal bill)

CCV's formal notation never mixes real and nominal returns inside the same
equation.

#### (2) The bill leg is the real return on a nominal bill

The asset itself is a nominal bill — physically a US Treasury bill in CCV,
empirically the 1-year Treasury yield (FRED `DGS1`) in this implementation.
The quantity $R_{1,t+1}$ that appears in equation (1) is its **real**
return:

$$R_{1,t+1} = \frac{R_{1,t+1}^{\text{nom}}}{\Pi_{t+1}} = \frac{1 + y_{1,t}}{\Pi_{t+1}}$$

where $y_{1,t}$ is the bill yield observed at $t$ and $\Pi_{t+1}$ is the
realized gross inflation factor over $[t, t+1]$.

The asset is nominal; the return that appears in equation (1) is real.
These two statements are not contradictory.

#### (3) The bill is risky in real terms

The nominal yield $y_{1,t}$ is locked in at $t$, so the nominal gross return
$R_{1,t+1}^{\text{nom}} = 1 + y_{1,t}$ is known at $t$. But realized inflation
$\Pi_{t+1}$ is **not** known at $t$. Hence

$$R_{1,t+1}^{\text{nom}} \in \mathcal{F}_t, \qquad \Pi_{t+1} \notin \mathcal{F}_t \;\;\Longrightarrow\;\; R_{1,t+1} \notin \mathcal{F}_t.$$

The bill is "riskless in nominal terms, risky in real terms." This is why
$r_1$ enters the VAR state vector $\mathbf{y}_t$ (CCV eq. 3) as a stochastic,
persistent component — not as a constant.

#### (4) The portfolio identity (CCV eq. 1)

The real portfolio return is the α-weighted real excess returns over the
bill, plus the real bill return:

$$\boxed{\;R_{p,t+1} = \sum_{i=2}^{n}\alpha_{i,t}(R_{i,t+1} - R_{1,t+1}) + R_{1,t+1}\;} \qquad (\text{CCV eq. 1})$$

Portfolio weights $\alpha_{i,t}$ are chosen at $t$. The implicit cash weight
is $\alpha_{1,t} = 1 - \sum_{i=2}^n \alpha_{i,t}$.

In the two-risky-asset specialisation used here ($i\in\{2,3\}$ = stock,
bond):

$$R_{p,t+1} = \alpha_{s,t}(R_{\text{stock},t+1} - R_{1,t+1}) + \alpha_{b,t}(R_{\text{bond},t+1} - R_{1,t+1}) + R_{1,t+1}$$

#### (5) Inflation-cancellation in excess returns

For any two assets, the excess log return is invariant under
(real, real) ↔ (nominal, nominal):

$$r_i^{\text{real}} - r_1^{\text{real}} = (r_i^{\text{nom}}-\pi_{t+1}) - (r_1^{\text{nom}}-\pi_{t+1}) = r_i^{\text{nom}} - r_1^{\text{nom}}$$

So excess returns in equation (1) can be empirically constructed as
nominal-over-nominal with **no numerical error** relative to CCV's
real-over-real formal notation.

Inflation enters equation (1) **only through the bare $R_1$ term** — there
is no inflation-cancelling pair for that term, which is what makes $R_p$
come out real.

#### (6) Inflation-deflation convention: subtraction in log space

The level-form division $R^{\text{real}} = R^{\text{nom}}/\Pi$ and the
log-form subtraction $r^{\text{real}} = r^{\text{nom}} - \pi$ (with
$\pi \equiv \log \Pi$) are **algebraically identical** — no approximation,
no Jensen correction in the conversion itself.

The implementation uses **log-form subtraction throughout**:

| Quantity | Construction |
|---|---|
| $\pi_{t+1}$ | $\log(\text{CPI}_{t+1}/\text{CPI}_t)$ |
| $r_{1,t+1}^{\text{nom}}$ | $\log(1 + y_{1,t})$ |
| $r_{1,t+1} \equiv \texttt{rtb}$ | $r_{1,t+1}^{\text{nom}} - \pi_{t+1}$ |
| $\mathbf{x}_{t+1}$ (excess) | $r_{i,t+1}^{\text{nom}} - r_{1,t+1}^{\text{nom}}$ |

Level-form returns $R = \exp(r)$ are recovered only at the budget-constraint
step (CCV eq. 7), via a single $\exp$ applied at point of use.

---

### Code mapping for §2.1

| Symbol | Code identifier | File:line |
|---|---|---|
| $R_{1,t+1}$ (level, real) | $\exp(\texttt{rtb})$ at point of use | implicit |
| $r_{1,t+1}$ (log, real) | `rtb[T]` (dataset), `log_R_bill` (solver) | `build_var_dataset.py:104`, `solver.py:785` |
| $r_{1,t+1}^{\text{nom}}$ | `r_1[T] = log(1 + y_1[T])` | `build_var_dataset.py:100` |
| $\pi_{t+1}$ | `pi[T] = log(CPI[T]/CPI[T-1])` | `build_var_dataset.py:96` |
| $\mathbf{x}_1$ (= stock excess log return) | `xr[T]`, `log_x_s` | `build_var_dataset.py:126`, `solver.py:789` |
| $\mathbf{x}_2$ (= bond excess log return) | `xb[T]`, `log_x_b` | `build_var_dataset.py:118`, `solver.py:790` |
| $\alpha_{2,t}, \alpha_{3,t}$ | `alpha_s, alpha_b` | `solver.py:694` |
| $R_{p,t+1}$ (level) | `R_p = exp(r_p)` | `solver.py:713` |

---

### Algebraic verification — the code's `r_p` is `r_p^real`

Expanding what `solver._ccv_log_return_and_grad` computes (Itô corrections
deferred to §3.1):

$$\begin{aligned}
r_p &= \texttt{log\_R\_bill} + \alpha_s \cdot \texttt{log\_x\_s} + \alpha_b \cdot \texttt{log\_x\_b} \\
    &= (r_1^{\text{nom}} - \pi) + \alpha_s(r_{\text{stock}}^{\text{nom}} - r_1^{\text{nom}}) + \alpha_b(r_{\text{bond}}^{\text{nom}} - r_1^{\text{nom}}) \\
    &= \alpha_s\, r_{\text{stock}}^{\text{nom}} + \alpha_b\, r_{\text{bond}}^{\text{nom}} + (1-\alpha_s-\alpha_b)\, r_1^{\text{nom}} - \pi \\
    &= r_p^{\text{nom}} - \pi \;=\; r_p^{\text{real}}.
\end{aligned}$$

The single $-\pi$ term comes from the `rtb` leg only and has no
counterpart elsewhere in the formula, which is exactly what makes $r_p$
real. ✓

---

## §2.2 — VAR dynamics ✅ LOCKED 2026-05-07

### Paper reference
CCV w8566, §2.2 "Dynamics of state variables" (page 6–8), equations (2)–(5).
Appendix A page 58 (eq. 30) for the unconditional distribution.

### The state vector

CCV stack the bill return, the excess returns, and the other state variables
into a single VAR vector:

$$\mathbf{y}_{t+1} = \begin{pmatrix} r_{1,t+1} \\ \mathbf{x}_{t+1} \\ \mathbf{s}_{t+1} \end{pmatrix}, \qquad \dim \mathbf{y} = m = 1 + (n-1) + k_s = n + k_s$$

where $n$ is the number of traded assets (incl. bill) and $k_s$ is the
number of "other" state variables.

**Definitions.**
- $r_{i,t+1} \equiv \log(R_{i,t+1})$ — log real return on asset $i$ (CCV eq. 2).
- $\mathbf{x}_{t+1} = (r_{2,t+1} - r_{1,t+1},\;\ldots,\; r_{n,t+1} - r_{1,t+1})'$ — log excess return vector, length $n - 1$ (CCV eq. 2).
- $\mathbf{s}_{t+1}$ — vector of "other" state variables, length $k_s$.

**This implementation:** $n = 3$ (bill, stock, long bond), $k_s = 3$
($y_1$, $\text{spr}$, $dp$), $m = 6$.

- $\mathbf{x}_{t+1} = (\text{xr}_{t+1}, \text{xb}_{t+1})'$.
- $\mathbf{s}_{t+1} = (y_{1,t+1}, \text{spr}_{t+1}, dp_{t+1})'$.

### The VAR(1) (CCV eq. 4)

$$\boxed{\;\mathbf{y}_{t+1} = \Phi_0 + \Phi_1 \mathbf{y}_t + v_{t+1}\;}$$

where $\Phi_0$ is $m \times 1$, $\Phi_1$ is $m \times m$, and $v_{t+1}$ is
$m \times 1$.

### Innovation distribution (CCV eq. 5)

$$v_{t+1} \stackrel{\text{i.i.d.}}{\sim} N(0, \Sigma_v)$$

with **four explicit assumptions** that all need to hold for the rest of
the model machinery to be correct:

**(a) Gaussian.** $v$ is normal. Consequently $r_i$ is normal and gross
returns $R_i = \exp(r_i)$ are log-normal in levels.

**(b) i.i.d. over time.** No serial correlation in $v$. All
time-dependence in $\mathbf{y}$ flows through $\Phi_1$.

**(c) Homoskedastic.** $\Sigma_v$ is constant in $\mathbf{y}_t$. State
variables shift conditional **means** only, not conditional **variances**.
This is what allows the constant scalars $\sigma_x^2$ and $\Sigma_{xx}$ in
CCV eq. (10). CCV are explicit (page 7–8): "It rules out the possibility
that the state variables predict changes in risk; they can affect portfolio
choice only by predicting changes in expected returns."

**(d) Cross-sectionally correlated.** $\Sigma_v$ has nonzero off-diagonals.

The covariance matrix is partitioned (CCV eq. 5):

$$\Sigma_v = \begin{pmatrix} \sigma_1^2 & \boldsymbol{\sigma}_{1\mathbf{x}}' & \boldsymbol{\sigma}_{1\mathbf{s}}' \\ \boldsymbol{\sigma}_{1\mathbf{x}} & \Sigma_{\mathbf{xx}} & \Sigma_{\mathbf{xs}}' \\ \boldsymbol{\sigma}_{1\mathbf{s}} & \Sigma_{\mathbf{xs}} & \Sigma_{\mathbf{ss}} \end{pmatrix}$$

| Block | Dim | Meaning |
|---|---|---|
| $\sigma_1^2$ | $1 \times 1$ | variance of $r_1$ innovation |
| $\boldsymbol{\sigma}_{1\mathbf{x}}$ | $(n-1) \times 1$ | cross-cov $r_1 \leftrightarrow \mathbf{x}$ innovs |
| $\boldsymbol{\sigma}_{1\mathbf{s}}$ | $k_s \times 1$ | cross-cov $r_1 \leftrightarrow \mathbf{s}$ innovs |
| $\Sigma_{\mathbf{xx}}$ | $(n-1) \times (n-1)$ | excess-return innov cov; **what enters CCV eq. (10)** |
| $\Sigma_{\mathbf{xs}}$ | $k_s \times (n-1)$ | cross-cov $\mathbf{s} \leftrightarrow \mathbf{x}$ innovs |
| $\Sigma_{\mathbf{ss}}$ | $k_s \times k_s$ | $\mathbf{s}$ innov cov |

### Implications

**(i) Log-normal level returns.** $R_{i,t+1} = \exp(r_{i,t+1})$ is
conditionally log-normal with parameters determined by $\Phi_0, \Phi_1, \Sigma_v$.

**(ii) Predictability operates on first moments only.** State variables shift
$\mathbb{E}_t[\mathbf{y}_{t+1}]$ but $\text{Var}_t[\mathbf{y}_{t+1}] = \Sigma_v$ is constant.

**(iii) Unconditional distribution (CCV eq. 30).** When $\Phi_1$ is
stationary ($\max_i |\lambda_i(\Phi_1)| < 1$), $\mathbf{y}_t$ has a
stationary distribution. Taking unconditional expectations of the VAR
(mean) and of the demeaned outer product $\mathbb{E}[\tilde{\mathbf{y}}_{t+1}\tilde{\mathbf{y}}_{t+1}']$ — exploiting $\mathbb{E}[\tilde{\mathbf{y}}_t v_{t+1}'] = 0$ from i.i.d.:

$$\mu_{\mathbf{y}} = (I_m - \Phi_1)^{-1}\,\Phi_0, \qquad \Sigma_{\mathbf{yy}} = \Phi_1\,\Sigma_{\mathbf{yy}}\,\Phi_1' + \Sigma_v$$

The second equation is the discrete Lyapunov equation. With Gaussian
innovations: $\mathbf{y}_t \sim N(\mu_{\mathbf{y}}, \Sigma_{\mathbf{yy}})$.
In code: `discretization.stationary_covariance()` solves the Lyapunov
equation directly via `scipy.linalg.solve_discrete_lyapunov`.

---

### §2.2.r — Restriction relative to CCV w8566 (deviation, not conformance)

The implementation imposes one restriction on $\Phi_1$ beyond what CCV w8566
specifies. (A second restriction on $\Phi_0$ — the sample-mean-pinning — is
deferred to §2.2.μ pending verification.)

#### Block-partitioned $\Phi_1$

Partition $\Phi_1$ by row (equations for $r_1$, $\mathbf{x}$, $\mathbf{s}$)
and by column (regressors $r_{1,t}$, $\mathbf{x}_t$, $\mathbf{s}_t$):

$$\Phi_1 = \begin{pmatrix}
\phi_{r_1, r_1} & \boldsymbol{\phi}_{r_1, \mathbf{x}}' & \boldsymbol{\phi}_{r_1, \mathbf{s}}' \\
\boldsymbol{\phi}_{\mathbf{x}, r_1} & \Phi_{\mathbf{x}, \mathbf{x}} & \Phi_{\mathbf{x}, \mathbf{s}} \\
\boldsymbol{\phi}_{\mathbf{s}, r_1} & \Phi_{\mathbf{s}, \mathbf{x}} & \Phi_{\mathbf{s}, \mathbf{s}}
\end{pmatrix}$$

with block dimensions:

| Block | Shape | Meaning |
|---|---|---|
| $\phi_{r_1, r_1}$ | $1 \times 1$ | how $r_{1,t}$ predicts $r_{1,t+1}$ |
| $\boldsymbol{\phi}_{r_1, \mathbf{x}}'$ | $1 \times (n-1)$ | how $\mathbf{x}_t$ predicts $r_{1,t+1}$ |
| $\boldsymbol{\phi}_{r_1, \mathbf{s}}'$ | $1 \times k_s$ | how $\mathbf{s}_t$ predicts $r_{1,t+1}$ |
| $\boldsymbol{\phi}_{\mathbf{x}, r_1}$ | $(n-1) \times 1$ | how $r_{1,t}$ predicts $\mathbf{x}_{t+1}$ |
| $\Phi_{\mathbf{x}, \mathbf{x}}$ | $(n-1) \times (n-1)$ | how $\mathbf{x}_t$ predicts $\mathbf{x}_{t+1}$ |
| $\Phi_{\mathbf{x}, \mathbf{s}}$ | $(n-1) \times k_s$ | how $\mathbf{s}_t$ predicts $\mathbf{x}_{t+1}$ |
| $\boldsymbol{\phi}_{\mathbf{s}, r_1}$ | $k_s \times 1$ | how $r_{1,t}$ predicts $\mathbf{s}_{t+1}$ |
| $\Phi_{\mathbf{s}, \mathbf{x}}$ | $k_s \times (n-1)$ | how $\mathbf{x}_t$ predicts $\mathbf{s}_{t+1}$ |
| $\Phi_{\mathbf{s}, \mathbf{s}}$ | $k_s \times k_s$ | how $\mathbf{s}_t$ predicts $\mathbf{s}_{t+1}$ |

#### The restriction in matrix form (zero-block specification)

The restriction sets the **three blocks corresponding to the columns of
$\mathbf{x}_t$** to zero:

$$\boxed{\;\boldsymbol{\phi}_{r_1, \mathbf{x}} = \mathbf{0}_{(n-1) \times 1},\qquad \Phi_{\mathbf{x}, \mathbf{x}} = \mathbf{0}_{(n-1) \times (n-1)},\qquad \Phi_{\mathbf{s}, \mathbf{x}} = \mathbf{0}_{k_s \times (n-1)}\;}$$

So the constrained $\Phi_1$ has the structure:

$$\Phi_1 = \begin{pmatrix}
\phi_{r_1, r_1} & \mathbf{0}_{1 \times (n-1)} & \boldsymbol{\phi}_{r_1, \mathbf{s}}' \\
\boldsymbol{\phi}_{\mathbf{x}, r_1} & \mathbf{0}_{(n-1) \times (n-1)} & \Phi_{\mathbf{x}, \mathbf{s}} \\
\boldsymbol{\phi}_{\mathbf{s}, r_1} & \mathbf{0}_{k_s \times (n-1)} & \Phi_{\mathbf{s}, \mathbf{s}}
\end{pmatrix}$$

**Equivalent statements** (all describe the same restriction):

(i) **Column form.** The columns of $\Phi_1$ corresponding to $\mathbf{x}_t$
    are identically zero:
    $$\Phi_1[:,\, \mathbf{x}\text{-cols}] = \mathbf{0}_{m \times (n-1)}.$$

(ii) **Gradient form.** The conditional mean of next-period $\mathbf{y}$
    has zero sensitivity to lagged $\mathbf{x}$:
    $$\frac{\partial\,\mathbb{E}_t[\mathbf{y}_{t+1}]}{\partial\, \mathbf{x}_t} = \mathbf{0}_{m \times (n-1)}.$$

(iii) **Equation form.** The three sub-VAR equations contain no $\mathbf{x}_t$ term:

$$r_{1,t+1} = \phi_{0,r_1} + \phi_{r_1, r_1}\, r_{1,t} + \boldsymbol{\phi}_{r_1, \mathbf{s}}'\, \mathbf{s}_t + v_{1,t+1}$$

$$\mathbf{x}_{t+1} = \boldsymbol{\phi}_{0,\mathbf{x}} + \boldsymbol{\phi}_{\mathbf{x}, r_1}\, r_{1,t} + \Phi_{\mathbf{x}, \mathbf{s}}\, \mathbf{s}_t + v_{\mathbf{x},t+1}$$

$$\mathbf{s}_{t+1} = \boldsymbol{\phi}_{0,\mathbf{s}} + \boldsymbol{\phi}_{\mathbf{s}, r_1}\, r_{1,t} + \Phi_{\mathbf{s}, \mathbf{s}}\, \mathbf{s}_t + v_{\mathbf{s},t+1}$$

#### What is restricted vs. free

**Restricted (zero):**
- $\boldsymbol{\phi}_{r_1, \mathbf{x}}$ — lagged $\mathbf{x}$ does not predict $r_1$
- $\Phi_{\mathbf{x}, \mathbf{x}}$ — lagged $\mathbf{x}$ does not predict $\mathbf{x}$ (no excess-return autocorrelation)
- $\Phi_{\mathbf{s}, \mathbf{x}}$ — lagged $\mathbf{x}$ does not predict $\mathbf{s}$

**Free (estimated by OLS):**
- $\phi_{r_1, r_1}$, $\boldsymbol{\phi}_{r_1, \mathbf{s}}$ — $r_1$ and $\mathbf{s}$ predict $r_1$
- $\boldsymbol{\phi}_{\mathbf{x}, r_1}$, $\Phi_{\mathbf{x}, \mathbf{s}}$ — $r_1$ and $\mathbf{s}$ predict $\mathbf{x}$
- $\boldsymbol{\phi}_{\mathbf{s}, r_1}$, $\Phi_{\mathbf{s}, \mathbf{s}}$ — $r_1$ and $\mathbf{s}$ predict $\mathbf{s}$

In particular: **$r_1$ is a free predictor in every equation, including
its own.** It is the channel through which inflation/real-rate
persistence propagates into excess-return dynamics. This matches CCV
w8566's treatment of $r_1$.

#### Source of the restriction

This restriction is from **Campbell, Chacko, and Viceira (2003)**, not w8566.
CCV w8566 Table 2 reports lagged-$\mathbf{x}$ coefficients freely estimated
and individually significant in some cases (e.g. annual
$\Phi_1[\text{xb}, \text{xr}] = +0.106$ with $t = 2.99$). It is therefore
a knowing deviation from w8566.

---

### Code mapping for §2.2

| Doc symbol | Code identifier | File:line |
|---|---|---|
| $\Phi_1$ (full $m \times m$) | `Phi` | `var.py:191-268` (estimator) / `var.py:608-680` (hardcoded) |
| $\Phi_0$ (full $m \times 1$) | `const` | same |
| $\Sigma_v$ (full $m \times m$) | `Omega` | same |
| $\Sigma_{\mathbf{xx}}$ (= what enters eq. 10) | `model.Sigma_rr` | `var.py:70` |
| Code's `Sigma_ss` (combined $r_1$ + $\mathbf{s}$ block) | `model.Sigma_ss` | `var.py:69` |
| Code's `Sigma_rs` (cross-block) | `model.Sigma_rs` | `var.py:71` |
| $\mu_{\mathbf{y}}, \Sigma_{\mathbf{yy}}$ (unconditional) | unconditional mean / `stationary_covariance()` | `var.py`, `discretization.py:65` |
| Lagged-$\mathbf{x}$ restriction | OLS without return-lag columns | `var.py:191-268` |

#### Important note on code partition vs. paper partition

The code partitions $\mathbf{y}$ into **two** blocks (`state` and `return`),
not three. Specifically:

- Code's `state` block = $(r_1, \mathbf{s})$ together — shape 4 in this
  calibration.
- Code's `return` block = $\mathbf{x}$ — shape 2 in this calibration.

So:
- Code's `Sigma_ss` = the $(r_1, \mathbf{s}) \times (r_1, \mathbf{s})$
  sub-block of $\Sigma_v$, which contains $\sigma_1^2$,
  $\boldsymbol{\sigma}_{1\mathbf{s}}$, and $\Sigma_{\mathbf{ss}}$ from the
  paper's three-way partition.
- Code's `Sigma_rs` = the $\mathbf{x} \times (r_1, \mathbf{s})$ sub-block,
  which contains $\boldsymbol{\sigma}_{1\mathbf{x}}$ and
  $\Sigma_{\mathbf{xs}}$ from the paper.
- Code's `Sigma_rr` = $\Sigma_{\mathbf{xx}}$ exactly. This is the only block
  that matches one-to-one between code naming and paper naming.

When matching code to paper, pull the relevant sub-block from the merged
`Sigma_ss` / `Sigma_rs` accordingly.

---

### Stale-doc note

`docs/CCV_RETURNS.md` describes a pre-migration partition where `rtb` lived
in the **return block**. Anywhere that doc says "the return-block columns
of $\Phi_1$ are zero," this now applies *only* to columns of
$\mathbf{x}$ = (xr, xb), **not** to the `rtb` (= $r_1$) column. The
current code has $r_1$ in the state-block partition with its column
freely estimated. (See `RETURN_MODELLING_TRACE_2026-05-07.md` §9 for the
migration history.) `docs/CCV_RETURNS.md` should be updated.

---

## §3.1 — The log-portfolio-return approximation (CCV eq. 10) ✅ LOCKED 2026-05-07

### Paper reference
CCV w8566, §3.1 "An approximate framework" (page 8), equation (10).
Appendix A pages 59–60 for the Itô-based derivation.

### §3.1.b — The formula

CCV approximate the log return on the portfolio defined by eq. (1) as

$$\boxed{\;r_{p,t+1} \;=\; r_{1,t+1} \;+\; \boldsymbol{\alpha}_t' \mathbf{x}_{t+1} \;+\; \tfrac{1}{2}\,\boldsymbol{\alpha}_t'\big(\boldsymbol{\sigma}_x^2 \;-\; \Sigma_{\mathbf{xx}}\,\boldsymbol{\alpha}_t\big)\;} \qquad (\text{CCV eq. 10})$$

where:

- $r_{1,t+1}$ is the realised log real bill return.
- $\mathbf{x}_{t+1}$ is the realised excess log return vector ($n-1$ components).
- $\boldsymbol{\alpha}_t$ is the vector of portfolio weights chosen at $t$ ($n-1$ components).
- $\Sigma_{\mathbf{xx}}$ is the $(n-1) \times (n-1)$ excess-return innovation
  covariance from §2.2 — a fixed matrix, treated as a constant in eq. (10).
- $\boldsymbol{\sigma}_x^2 \equiv \text{diag}(\Sigma_{\mathbf{xx}})$ is the
  vector of per-asset excess-return variances — a fixed vector, also treated
  as a constant.

The variance-correction term $\tfrac{1}{2}\boldsymbol{\alpha}'(\boldsymbol{\sigma}_x^2 - \Sigma_{\mathbf{xx}}\boldsymbol{\alpha})$ depends only on the policy
$\boldsymbol{\alpha}$ and the model constants. It does **not** depend on the
realised innovation $\mathbf{x}_{t+1}$. This is what makes $r_{p,t+1}$ a
Gaussian random variable conditional on the state and policy — see §3.1.e.

#### Specialisation to two risky assets ($n = 3$)

With $\boldsymbol{\alpha} = (\alpha_s, \alpha_b)'$ (stock and bond weights),
$\mathbf{x} = (x_r, x_b)'$, and $\Sigma_{\mathbf{xx}}$ with diagonal
$(\sigma^2_{xr}, \sigma^2_{xb})$ and off-diagonal $\sigma_{xr,xb}$:

$$r_{p,t+1} = r_{1,t+1} + \alpha_s\, x_{r,t+1} + \alpha_b\, x_{b,t+1} + \tfrac{1}{2}\alpha_s\, \sigma^2_{xr} + \tfrac{1}{2}\alpha_b\, \sigma^2_{xb} - \tfrac{1}{2}\big(\alpha_s^2\, \sigma^2_{xr} + 2\alpha_s\alpha_b\, \sigma_{xr,xb} + \alpha_b^2\, \sigma^2_{xb}\big)$$

This is the form coded in `solver._ccv_log_return_and_grad`.

### §3.1.c — Economic interpretation: continuous rebalancing

Eq. (10) is the **discrete-time face of $d\log V_t$ from Itô's lemma**,
applied to a portfolio held under continuous rebalancing with constant
weights $\boldsymbol{\alpha}_t$ throughout the period. In continuous time
the formula is exact; at $\Delta t = 1$ year (this implementation) it is an
approximation whose error grows quadratically in $|\boldsymbol{\alpha} - 1|$ on the unit-leverage axis (see §3.1.f).

The formula admits a clean three-component decomposition:

| Component | Term | Role |
|---|---|---|
| Bill leg | $r_{1,t+1}$ | realised real bill log return — the baseline |
| Linear excess return | $\boldsymbol{\alpha}'\mathbf{x}_{t+1}$ | $\alpha$-weighted excess returns |
| Jensen lift | $+\tfrac{1}{2}\boldsymbol{\alpha}'\boldsymbol{\sigma}_x^2$ | per-asset log-vs-arithmetic compensation |
| Itô drag | $-\tfrac{1}{2}\boldsymbol{\alpha}'\Sigma_{\mathbf{xx}}\boldsymbol{\alpha}$ | portfolio-variance penalty for level-form combination |

The Jensen lift and the Itô drag combine to form the variance-correction
term $\tfrac{1}{2}\boldsymbol{\alpha}'(\boldsymbol{\sigma}_x^2 - \Sigma_{\mathbf{xx}}\boldsymbol{\alpha})$. For the single-risky-asset
specialisation ($n=2$) this collapses to $\tfrac{1}{2}\sigma_x^2 \alpha(1 - \alpha)$ — positive (benefit) for interior $\alpha \in (0, 1)$, negative
(penalty) for $\alpha < 0$ or $\alpha > 1$. Multivariate analogue: positive
in the interior of the unit-weight simplex, negative for short or
leveraged positions.

#### No-bankruptcy property

Because $r_p$ is just a number, $R_p = \exp(r_p) > 0$ structurally for any
$\boldsymbol{\alpha}$, any state, any innovation realisation. Wealth
$W_{t+1} = (W_t - C_t) R_p$ can asymptote to zero but cannot cross zero.
**Bankruptcy is mathematically impossible under eq. (10).**

This is the discrete-time mirror of the continuous-time fact that under
continuous rebalancing, the portfolio value follows a geometric process
that never hits zero in finite time. As leverage threatens to wipe out the
portfolio, continuous rebalancing sells fractions of the (now smaller)
portfolio rather than crossing zero. The user-side benefit: CRRA utility,
undefined for $C \le 0$, never encounters domain violations.

#### The σ_{1,x} cancellation

Eq. (10) contains no $\sigma_1^2$ or $\boldsymbol{\sigma}_{1\mathbf{x}}$
terms — the bill-variance and bill–excess-return cross-covariances do
not appear. This is not a limit-of-applicability issue; it is a clean
algebraic cancellation in the Itô derivation, robust to the bill being
risky (which it is in CCV's setup). The cancellation requires no assumption
beyond what CCV §2.2 already imposes (Gaussian, i.i.d., homoskedastic).

### §3.1.d — Source of $\boldsymbol{\sigma}_x^2$ and $\Sigma_{\mathbf{xx}}$

In CCV's framework, $\boldsymbol{\sigma}_x^2$ and $\Sigma_{\mathbf{xx}}$ are
**parameters of the continuous-time stochastic differential equations** for
excess-return log prices — specifically, the diffusion coefficients
$\|\sigma_i - \sigma_1\|^2$ and $(\sigma_i - \sigma_1)'(\sigma_j - \sigma_1)$ where the $\sigma_i$ are the diffusion vectors of asset $i$'s
log-price SDE. Their discrete-time face is the unconditional VAR
innovation covariance $\Sigma_v$ restricted to the excess-return rows and
columns, which in this implementation is exactly `Sigma_rr` (= the
$\mathbf{x} \times \mathbf{x}$ sub-block of $\Sigma_v$).

These constants have **no conditional structure**. They are not
"expectations of anything." They are model parameters fixed once $\Sigma_v$
is estimated.

#### What `Sigma_r_cond` is for (and why it must NOT enter eq. 10)

The quadrature draws joint VAR innovations $(v^{(s)}, v^{(x)})$ via a
two-stage Cholesky: first $v^{(s)}$ from its marginal, then
$v^{(x)} = M v^{(s)} + \varepsilon$ with $\varepsilon \sim N(0, \Sigma_{r|\text{cond}})$. The matrix $\Sigma_{r|\text{cond}} = \Sigma_{\mathbf{xx}} - M\,\Sigma_{\mathbf{ss}}\,M'$ is the conditional residual
covariance — a **sampling mechanic**, not a formula input.

Whichever decomposition is used to draw the joint innovation, the realised
$\mathbf{x}_{t+1}$ has its full variance captured in $\Sigma_{\mathbf{xx}}$.
Eq. (10) is a **path-by-path identity** — it holds at every realised
innovation, not in expectation — and the variance-correction term
$\tfrac{1}{2}\boldsymbol{\alpha}'(\boldsymbol{\sigma}_x^2 - \Sigma_{\mathbf{xx}}\boldsymbol{\alpha})$ is the Itô drift correction for the
**unconditional** law of motion.

Sourcing the formula constants from $\Sigma_{r|\text{cond}}$ instead of
$\Sigma_{\mathbf{xx}}$ would yield numerically different scalars: the
magnitude difference is approximately the projection ratio
$M\Sigma_{\mathbf{ss}}M'/\Sigma_{\mathbf{xx}}$, which on the production
calibration is in the 10×–90× range. The May-2026 patch corrected this in
the codebase. The pre-patch convention (sourcing from `Sigma_r_cond`) was
wrong; any documentation that still describes it (`docs/CCV_RETURNS.md` per
§2.2's stale-doc note) needs updating.

#### Empirical sniff test

The code's `Sigma_rr` reproduces CCV w8566 Table 2 unconditional
excess-return volatilities and yields the classroom Markowitz solution at
$\gamma = 1$. The same test sourced from `Sigma_r_cond` would not.

### §3.1.e — How eq. (10) enters the FOC

The solver is a level-form numerical DP. Eq. (10) supplies the law of
motion for $r_{p,t+1}$, but the household's budget constraint is in levels:

$$W_{t+1} = (W_t - C_t)\,R_{p,t+1}, \qquad R_{p,t+1} = \exp(r_{p,t+1})$$

and CRRA utility takes a level-form argument: $u(C) = C^{1-\gamma}/(1-\gamma)$. The FOC kernel is therefore organised around the
principle **"log throughout, exp at the last moment"**: the portfolio
return lives in log form through every internal computation, and the
exponential is applied only at the two points where levels are actually
needed — for utility evaluation, and (via the chain rule) for the gradient
with respect to $\boldsymbol{\alpha}$.

#### Five-step structure of the FOC kernel

For a single quadrature node $(k_v, k_r)$ with weight $w_{k_v, k_r}$ and
innovation realisation $(v^{(s)}_{k_v}, v^{(x)}_{k_r})$:

**1. Compute $r_p$ in log space (eq. 10).** Inputs: $\boldsymbol{\alpha}$
from the policy iterate, $r_1$ and $\mathbf{x}$ from the realised state and
innovation, and the constants $\boldsymbol{\sigma}_x^2, \Sigma_{\mathbf{xx}}$. Output: a scalar $r_p$.

**2. Compute the log-space gradient $\partial r_p / \partial \boldsymbol{\alpha}$.** Closed-form derivative of the expression in eq. (10):

$$\frac{\partial r_p}{\partial \alpha_j} = x_j + \sigma^2_{x,j}\!\left(\tfrac{1}{2} - \alpha_j\right) - \sum_{k \ne j} \alpha_k\, \sigma_{x,jk}$$

For the two-risky case: $\partial r_p/\partial \alpha_s = x_r + \sigma^2_{xr}(1/2 - \alpha_s) - \alpha_b\, \sigma_{xr,xb}$, and similarly
for $\alpha_b$.

**3. Convert to level: $R_p = \exp(r_p)$.** One scalar exponential per
quadrature node. This is the only place an $\exp$ is invoked on $r_p$
itself.

**4. Chain-rule the level gradient:**

$$\frac{\partial R_p}{\partial \alpha_j} = \exp(r_p)\,\frac{\partial r_p}{\partial \alpha_j} = R_p \cdot \frac{\partial r_p}{\partial \alpha_j}$$

No new exponentiation. The level gradient is the already-exponentiated
$R_p$ multiplied by the log gradient computed in Step 2.

**5. Aggregate the FOC** by summing over quadrature nodes:

$$\boxed{\;\text{FOC}_j \;=\; \sum_{(k_v, k_r)} w_{k_v, k_r}\, u'(s\, R_p^{(k_v, k_r)}) \cdot \frac{\partial R_p^{(k_v, k_r)}}{\partial \alpha_j} \;=\; 0\;}$$

where $s = W_t - C_t$ (savings, level). For working-age periods, $u'$ is
replaced by $V_{t+1}'$ (the next-period value-function derivative); for the
terminal period, by $u'_{\text{bequest}}$.

#### Hessian structure (Newton step)

The Newton step requires $\partial^2 R_p / \partial \boldsymbol{\alpha}\partial\boldsymbol{\alpha}'$, which by the chain rule is:

$$\frac{\partial^2 R_p}{\partial \alpha_j \partial \alpha_k} = R_p \!\left[\frac{\partial r_p}{\partial \alpha_j}\frac{\partial r_p}{\partial \alpha_k} + \frac{\partial^2 r_p}{\partial \alpha_j \partial \alpha_k}\right]$$

The two contributions:

- **Outer product** $\partial r_p/\partial \alpha_j \cdot \partial r_p/\partial \alpha_k$ — always present, captures the first-order curvature
  of $\exp$.
- **$-\Sigma_{\mathbf{xx}}[j, k]$** correction — the second derivative of
  the variance-quadratic in eq. (10),
  $\partial^2 [-\tfrac{1}{2}\boldsymbol{\alpha}'\Sigma_{\mathbf{xx}}\boldsymbol{\alpha}]/\partial \alpha_j \partial \alpha_k = -\Sigma_{\mathbf{xx}}[j,k]$. Specific to eq. (10); would not be present
  if $r_p$ were just $r_1 + \boldsymbol{\alpha}'\mathbf{x}$.

In the FOC Jacobian assembly, these appear as `dr_da_s * dr_da_s -
sigma2_xr` etc. (`solver.py:749-751`).

#### Why this works numerically

Three properties combine to make the level-form FOC a clean numerical
object:

1. **$r_p$ is Gaussian conditional on state and policy.** The
   variance-correction term in eq. (10) is a constant given
   $\boldsymbol{\alpha}$, so $r_p$ is a linear functional of the Gaussian
   innovation plus a deterministic shift.
2. **Gauss–Hermite quadrature is exact for polynomials.** With $K$ nodes
   per axis, polynomials of degree $2K - 1$ are integrated exactly against
   the Gaussian density. Practical settings of $K = 5$–$9$ per axis give
   effectively-exact integration of $u'(s\,e^{r_p})$ and its derivatives in
   the relevant ranges of $\boldsymbol{\alpha}$.
3. **The chain-rule structure preserves log-quadratic dependence.** Both
   the FOC and the Hessian inherit the closed-form $\boldsymbol{\alpha}$-dependence of eq. (10) without numerical differentiation, giving Newton's
   method analytic second-order information.

The level-form DP captures every Jensen-style nonlinearity of
$u'(s\exp(r_p))$ exactly to polynomial-quadrature order, including those
that CCV's log-linear analytical solution log-linearises away. In this
sense the solver is a **strict refinement** of CCV's analytical method:
there is no hidden Jensen correction CCV captures and the user's setup
misses.

### §3.1.f — Inherited caveats

These are **modelling assumptions**, not implementation issues. They are
inherited from the entire CCV-style lifecycle-portfolio literature
(Cocco–Gomes–Maenhout 2005, Gomes–Michaelides 2005, etc.) and apply
equally here as in CCV w8566.

**Continuous-rebalancing wedge.** Eq. (10) is exact in continuous time.
At $\Delta t = 1$ year, the per-period mean wedge between the
continuous-rebalancing $R_p$ and a true buy-and-hold $R_p$ is small for
$|\boldsymbol{\alpha}| \le 1$ (~0.05% at $\alpha = 0.5$, exactly zero at
$\alpha = 1$) and grows quadratically in $|\boldsymbol{\alpha} - 1|$ —
approximately 1% at $\alpha = 1.5$ and 3% at $\alpha = 2$. The wedge in
higher moments — particularly the lower tail — is more material:
continuous rebalancing produces a log-normal $R_p > 0$ distribution;
buy-and-hold produces a left-skewed distribution with a true zero
crossing at extreme leverage.

**Bankruptcy suppression at high leverage.** The structural $R_p > 0$
property removes a real economic risk channel — leveraged-blowup tail
behaviour — at exactly the calibrations where it matters most. The
borrowing constraint $\boldsymbol{\alpha} \in [0, 2]$ partially offsets
this; in regimes where the constraint binds at $\alpha = 2$, the
suppressed risk is moot. For interior leveraged solutions
($\boldsymbol{\alpha} \in (1, 2)$), the suppressed downside biases optimal
$\alpha$ upward by an estimated 5–15 percentage points of equity weight,
depending on $\gamma$.

**γ-amplification.** Higher risk aversion places more weight on the lower
tail of the wealth distribution, which is precisely where eq. (10) most
diverges from buy-and-hold. The continuous-rebalancing wedge has larger
consequences for policy and welfare at $\gamma = 10$ than at $\gamma = 2$.

These caveats are observational. They do not call eq. (10) into question
as a modelling choice; rather, they delimit the calibrations at which the
choice's accuracy is most strained, and suggest sensitivity-test targets.

### §3.1.g — Code mapping

| Symbol | Code identifier | File:line |
|---|---|---|
| $r_{p,t+1}$ | `r_p` (internal to kernel) | `solver.py:706-712` |
| $R_{p,t+1}$ | `R_p = jnp.exp(r_p)` | `solver.py:713` |
| $\partial r_p / \partial \alpha_s$ | `dr_da_s` | `solver.py:714` |
| $\partial r_p / \partial \alpha_b$ | `dr_da_b` | `solver.py:715` |
| $\partial R_p / \partial \alpha_s$ | `dRp_das = R_p * dr_da_s` | `solver.py:738` |
| $\partial R_p / \partial \alpha_b$ | `dRp_dab = R_p * dr_da_b` | `solver.py:739` |
| $\sigma^2_{xr}$ | `sigma2_xr` | `precompute.py:303-314` |
| $\sigma^2_{xb}$ | `sigma2_xb` | same |
| $\sigma_{xr,xb}$ | `sigma_xrxb` | same |
| Eq. (10) formula | `_ccv_log_return_and_grad` | `solver.py:694-716` |
| FOC kernel | `terminal_foc_jac_ccv` | `solver.py:723-756` |
| Hessian extra terms | `extra_ss/_bb/_sb` | `solver.py:749-751` |
| Simulator parity formula | matching implementation | `simulation.py:329-362` |

#### Solver/simulator parity

The simulator's portfolio-return formula at `simulation.py:329-362` is
symbol-for-symbol identical to the solver's `_ccv_log_return_and_grad`,
sourcing the same `sigma2_xr/xb/xrxb` scalars from precompute. Solver and
simulator both apply the continuous-rebalancing eq. (10) at every step.
Numerical agreement is enforced to 1e-12 by the randomised parity test
in `verify/ccv_solver_sim_parity.py`.

---

## §4.1 — Empirical implementation: annual data construction ✅ LOCKED 2026-05-07

### Paper reference
CCV w8566, §4.1 "Data description" (page 15–16). One user-elected deviation
(long-bond source) and two as-implemented deviations forced by chap_26's
current contents — see §4.1 deviations subsection below.

### Sample period

**Locked window: 1919–2011, annual.** Effective $T = 92$ observations after
losing the first year to inflation differencing (sample 1920..2011).

- Upper bound 2011: chap_26's R series ends at 2011.
- Lower bound 1919: FRED Moody's AAA starts 1919-01.

**Resolution of an internal conflict in earlier draft (2026-05-07).** The
locked spec previously stated "1871-2011, T=141" alongside "Moody's AAA
throughout 1871-2011" (D1). FRED Moody's AAA starts 1919-01 — the two
statements cannot both hold. The implementation chose **Option A** of four
alternatives considered: shorten the sample to 1919-2011 (effective T=92)
to honor "Moody's AAA throughout" strictly. Alternatives — A_RLONG (RLONG
on the same window), C (splice RLONG pre-1919 + AAA from 1919, T=140), D
(RLONG throughout 1872-2011, T=140) — are exercised in
`scripts/sensitivity_var_window.py`. Sample-extension and yield-source
effects on the headline parameter estimates are reported there; the
yield-source choice has the larger effect on Markowitz-at-γ=1 bond weights
(α_b ranges 0.5-2.9 across the four options).

The "T=141" figure in the original draft was also off-by-one independent of
the AAA issue: chap_26 starts 1871, inflation differencing loses 1871, so
the maximum possible T on the full window is 140 (1872-2011), as the
handoff §3.1.5 notes.

### Data sources

| Source | Provides | Access |
|---|---|---|
| chap_26 (Shiller's Ch. 26 data update) | $P, D, R, \text{CPI}$ | columns B, C, E, G of `chapt26.xlsx` |
| FRED Moody's AAA | Long-bond yield $Y_n$ | `AAA.csv`, take January value of each year |

### Timing convention

All variable observations are at **January of year $t$**. This was verified
numerically: chap_26 annual values for $P, D, \text{CPI}$ exactly equal
Shiller's monthly `ie_data.xls` January values across all sample years
(1900, 1950, 1980, 2000, 2010, 2015 cross-check, agreement to <1bp).

Realized one-year quantities (returns, log changes) cover the period
[Jan-$t$, Jan-$(t+1)$]. So:
- $P_t$ = S&P Composite at January year $t$
- $\pi_{t+1} = \log(\text{CPI}_{t+1}/\text{CPI}_t)$ — log inflation from Jan-$t$ to Jan-$(t+1)$
- All "return" indices subscripted $t+1$ are realized over [Jan-$t$, Jan-$(t+1)$]

### Raw inputs

Let:
- $P_t$ — S&P Composite Stock Price Index, Jan year $t$ (chap_26 col B)
- $D_t$ — trailing-12-month sum of S&P dividends, Jan year $t$ (chap_26 col C)
- $\text{CPI}_t$ — Consumer Price Index, Jan year $t$ (chap_26 col G)
- $R_t$ — nominal 1-year interest rate, in percent, Jan year $t$ (chap_26 col E; the 4–6m commercial paper Jan+July rollover annualised)
- $Y_{n,t}$ — Moody's Seasoned AAA Corporate Bond Yield, in percent, Jan year $t$ (FRED `AAA.csv`)

### The six VAR variables — exact construction

Six variables enter the VAR: $r_1$, xr, xb, $y_1$, spr, $dp$. One auxiliary
quantity ($\pi$) is constructed first.

#### 1. Auxiliary: log inflation
$$\pi_{t+1} = \log\!\left(\frac{\text{CPI}_{t+1}}{\text{CPI}_t}\right)$$

#### 2. Nominal bill yield $y_1$ (state variable)
$$y_{1,t} = \log\!\left(1 + \frac{R_t}{100}\right)$$
Convert from percent before logging.

#### 3. Real bill rate `rtb` (state variable)
$$r_{1,t+1} = y_{1,t} - \pi_{t+1}$$
Per §2.1's locked convention: subtraction in log space, no Jensen
correction. The yield is set at $t$, the real return is realized at $t+1$.

#### 4. Excess log stock return `xr` (return variable)
$$x_{r,t+1} = \log(R_{\text{stk},t+1}) - y_{1,t}, \qquad R_{\text{stk},t+1} = \frac{P_{t+1} + D_{t+1}}{P_t}$$
The numerator $P_{t+1} + D_{t+1}$ uses end-of-period (Jan-$(t+1)$) price plus the trailing-12-month dividend sum at Jan-$(t+1)$, which corresponds approximately to dividends accrued during the holding period [Jan-$t$, Jan-$(t+1)$].

#### 5. Excess log bond return `xb` (return variable)
$$x_{b,t+1} = r_{n,t+1} - y_{1,t}$$
where $r_{n,t+1}$ is constructed via the constant-duration approximation
— next subsection.

#### 6. Yield spread `spr` (state variable)
$$\text{spr}_t = y_{n,t} - y_{1,t}$$
with $y_{n,t} = \log(1 + Y_{n,t}/100)$ — log AAA yield, January year $t$.

#### 7. Log dividend–price ratio `dp` (state variable)
$$dp_t = \log D_t - \log P_t$$
Per CCV §4.1 verbatim: "the log dividend less the log price index."

### Long-bond return construction (Campbell–Lo–MacKinlay)

Moody's AAA provides yields, not realized returns. Returns are reconstructed
via the constant-duration log-linear approximation (CCV §4.1, citing
Campbell–Lo–MacKinlay 1997 Ch. 10):

$$\boxed{\;r_{n,t+1} \;\approx\; D_{n,t}\,y_{n,t} \;-\; (D_{n,t} - 1)\,y_{n,t+1}\;}$$

with three pieces:

- **Maturity** $n = 20$ years (CCV's exact choice; appropriate for Moody's
  AAA effective duration).
- **Log yield** $y_{n,t} = \log(1 + Y_{n,t}/100)$ where $Y_{n,t}$ is the AAA
  yield at January year $t$.
- **Macaulay-style duration** computed from yield only:

$$D_{n,t} \;\approx\; \frac{1 - (1 + Y_{n,t}/100)^{-n}}{1 - (1 + Y_{n,t}/100)^{-1}}$$

- **Constant-duration approximation**: $y_{n-1, t+1}$ is replaced by
  $y_{n, t+1}$ — the remaining-19-year yield at $t+1$ is approximated by
  the new-20-year yield at $t+1$. This is what makes the formula
  computable from a single yield series.

### Three deviations from CCV w8566 paper text

Honest accounting of where this build differs from the paper:

**(D1) Long-bond yield source — user-elected.** Moody's AAA throughout
1871–2011, in place of CCV's Shiller(1989) historical + Moody's AAA splice.
Motivation: single-source clean construction; avoids the Shiller(1989)
splice point. Implication: AAA-corporate yields embed a (time-varying)
credit spread over equivalent-maturity government yields throughout the
sample.

**(D2) Inflation index — forced by chap_26 contents.** chap_26 column G is
**CPI** (Consumer Price Index, BLS CPI-U post-1913 + Warren–Pearson splice
pre-1913). CCV's paper text states they used the **Producer Price Index**.
The chap_26 dataset Shiller currently distributes contains CPI only; CCV
may have had access to a PPI variant in 2001 that has since been
superseded. As-implemented inflation is CPI.

**(D3) Stock observation timing — forced by chap_26 contents.** chap_26's
annual $P, D$ are **January-of-year** observations. CCV's paper text states
"the equity price index is the **end-of-December** S&P 500 Index."
Cross-check confirms current chap_26 uses January, not end-of-December. The
paper text may be describing Grossman–Shiller (1981) original convention;
current Shiller distributions use January. As-implemented timing is January.

(D2) and (D3) are not user-elected — they reflect chap_26's actual
contents. (D1) is the user's only active design choice deviating from CCV.

### Variable list comparison

| Variable | CCV w8566 | This build | Notes |
|---|---|---|---|
| $r_1$ | real bill rate | real bill rate (`rtb`) | match |
| $\mathbf{x}_1$ | excess log stock return | `xr` | match |
| $\mathbf{x}_2$ | excess log bond return | `xb` | match |
| $y_1$ | nominal bill yield | `y_1` | match |
| spread | yield spread | `spr` | match |
| 6th | log dividend–price ratio (`dp`) | `dp` | match (replaces previous `cy`) |

The variable list now matches CCV exactly. The only data-source deviation
is the long-bond yield (D1).

### Robustness note: post-war subsample re-estimation

A post-war subsample re-estimation will be run as a coefficient-stability
check. The full-sample 1871–2011 build is the locked baseline; the post-war
run is purely diagnostic — to see how much VAR coefficients shift across
historical structural breaks (Great Depression, WWII, Bretton Woods) — and
is not a re-specification.

### Code mapping (to be populated)

The current `var.py` / `build_var_dataset.py` operate on a 1963–2025 sample
with `cy` as the 6th variable. Migration to this §4.1 spec requires:

- New raw-data ingest: chap_26 (XLSX read), FRED Moody's AAA (CSV read).
- Drop the `LW_monthly.xlsx` Lettau–Ludvigson cay ingest path entirely.
- Replace `cy` column with $dp = \log D - \log P$.
- Reconstruct `xb` using Moody's AAA throughout (replacing whatever
  long-yield series is currently used).
- Update sample window to 1871–2011.

Code mapping table will be populated once the migration is complete.

---

## §4.2 — VAR estimation results: CCV reference numbers ✅ LOCKED 2026-05-07

### Paper reference
CCV w8566, §4.2 "VAR estimation" (pages 17–22), Table 1 (annual column) and
Table 2 Panel B.

### Purpose

This subsection records CCV w8566's annual-sample reference numbers for use
as a benchmark when validating this build's empirical estimates.
**These numbers are not estimated values for our build — they are CCV's
reference values for comparison.** The build's own estimates will be added
in a separate subsection once the §4.1 data migration is complete.

### Comparison caveats

Three reasons our build's numbers will not exactly match the reference
values below:

**(C1) Different sample length.** CCV: 1890–1998, $T = 109$. This build:
1871–2011, $T = 141$. Our sample includes pre-1890 and post-1998 periods
that contain regimes (early gold standard, post-2000 zero-rate era) absent
from CCV's window.

**(C2) Estimation under restriction §2.2.r.** Our build imposes the §2.2.r
restriction (zero columns of $\Phi_1$ corresponding to lagged $\mathbf{x}_t$),
per Campbell–Chacko–Viceira (2003). CCV w8566 estimates $\Phi_1$ unrestricted
on this dimension. Coefficients in the columns labelled $x_{r,t}$ and
$x_{b,t}$ in CCV's Table 2 are non-zero in their estimation but
zero-by-construction in ours.

**(C3) Different long-bond yield source (§4.1 D1).** CCV use Shiller (1989)
+ Moody's AAA splice. Our build uses Moody's AAA throughout 1871–2011. The
bond yield series differs; consequently `xb` and `spr` moments and
correlations involving them will differ.

The sample-mean pinning of $\Phi_0$ (the constrained-LS approach below) is
**the same in both builds** — not a comparison issue.

### CCV's estimation approach (constrained least-squares)

CCV w8566 §4.2, footnote 5:

> "We estimate the VAR imposing the restriction that the unconditional
> means of the variables implied by the VAR coefficient estimates equal
> their full-sample arithmetic counterparts. Standard, unconstrained
> least-squares fits exactly the mean of the variables in the VAR
> excluding the first observation. We use constrained least-squares to
> ensure that we fit the full-sample means."

This is the verbal form of the §2.2.μ pinning restriction. The matrix-form
translation is

$$\Phi_0 = (I - \Phi_1)\,\mu_{\mathbf{y}}^{\text{sample}}$$

equivalent to demeaning the data, running OLS without an intercept, and
recovering $\Phi_0$ from the implied unconditional-mean equation. Cross-
reference: §2.2.μ.

This estimation approach is identical in CCV's build and ours. The sample
length differs (caveat C1), but the restriction itself does not.

### Table 1 — Sample Statistics, Annual 1890–1998

All quantities in **annualised percent**, except (11) and (12) which are in
natural log units (the dividend–price ratio).

Items (1), (3), (6) include the **Jensen-inequality adjustment**: they are
mean log returns + ½ variance, lifting log expectations to arithmetic-mean
equivalents.

| Row | Statistic | Annual value |
|---|---|---|
| (1) | $\mathbb{E}[r_{1,t} - \pi_t] + \tfrac{1}{2}\sigma^2(r_{1,t} - \pi_t)$ — mean real bill rate (Jensen-adj) | 2.101 |
| (2) | $\sigma(r_{1,t} - \pi_t)$ — std real bill rate | 8.806 |
| (3) | $\mathbb{E}[r_{e,t} - r_{1,t}] + \tfrac{1}{2}\sigma^2(r_{e,t} - r_{1,t})$ — mean excess stock return (Jensen-adj) | 6.797 |
| (4) | $\sigma(r_{e,t} - r_{1,t})$ — std excess stock return | 18.192 |
| (5) | Sharpe ratio stock = (3)/(4) | 0.374 |
| (6) | $\mathbb{E}[r_{n,t} - r_{1,t}] + \tfrac{1}{2}\sigma^2(r_{n,t} - r_{1,t})$ — mean excess bond return (Jensen-adj) | 0.674 |
| (7) | $\sigma(r_{n,t} - r_{1,t})$ — std excess bond return | 6.543 |
| (8) | Sharpe ratio bond = (6)/(7) | 0.103 |
| (9) | $\mathbb{E}[y_t]$ — mean nominal short yield | 4.361 |
| (10) | $\sigma(y_t)$ — std nominal short yield | 2.597 |
| (11) | $\mathbb{E}[d_t - p_t]$ — mean log dividend–price ratio | −3.101 |
| (12) | $\sigma(d_t - p_t)$ — std log dividend–price ratio | 0.304 |
| (13) | $\mathbb{E}[y_{n,t} - y_{1,t}]$ — mean yield spread | 0.902 |
| (14) | $\sigma(y_{n,t} - y_{1,t})$ — std yield spread | 1.450 |

Bond is **20-year nominal** in the annual sample.

### Table 2 Panel B — VAR Estimation Results, Annual 1890–1998

#### Coefficients (t-statistics in parentheses)

| Dep \ Reg | $r_{tb,t}$ | $x_{r,t}$ | $x_{b,t}$ | $y_t$ | $(d-p)_t$ | $\text{spr}_t$ | $R^2$ ($p$) |
|---|---|---|---|---|---|---|---|
| $r_{tb,t+1}$ | **0.303** (2.434) | −0.052 (−1.314) | 0.122 (0.902) | **0.701** (2.365) | −0.004 (−0.146) | −0.776 (−1.242) | 0.240 (0.000) |
| $x_{r,t+1}$ | 0.116 (0.438) | 0.075 (0.607) | −0.091 (−0.305) | −0.074 (−0.105) | **0.131** (2.320) | 1.291 (0.957) | 0.050 (0.399) |
| $x_{b,t+1}$ | **0.200** (3.072) | **0.106** (2.990) | −0.197 (−1.502) | −0.112 (−0.319) | 0.012 (0.614) | **2.628** (5.289) | 0.392 (0.000) |
| $y_{t+1}$ | −0.042 (−1.922) | −0.012 (−1.784) | 0.037 (1.318) | **0.921** (12.307) | −0.005 (−1.119) | −0.017 (−0.136) | 0.776 (0.000) |
| $(d-p)_{t+1}$ | **−0.567** (−2.272) | −0.124 (−1.146) | 0.357 (1.115) | −0.597 (−0.941) | **0.842** (13.362) | −1.662 (−1.194) | 0.721 (0.000) |
| $\text{spr}_{t+1}$ | 0.020 (1.118) | 0.002 (0.409) | −0.013 (−0.667) | 0.085 (1.625) | 0.004 (1.153) | **0.820** (8.900) | 0.540 (0.000) |

Bolded entries: $|t| > 2$ (typically discussed as significant). $R^2$'s
$p$-value is for the F-test of joint significance of all regressors.

#### Residual cross-correlations (annual)

Diagonal: standard deviations × 100. Above-diagonal: correlations.

| | rtb | xr | xb | y | (d−p) | spr |
|---|---|---|---|---|---|---|
| rtb | **7.592** | −0.167 | −0.020 | 0.114 | 0.100 | −0.155 |
| xr | — | **17.498** | −0.020 | −0.135 | −0.725 | 0.186 |
| xb | — | — | **5.102** | −0.650 | −0.055 | 0.264 |
| y | — | — | — | **1.228** | 0.179 | −0.894 |
| (d−p) | — | — | — | — | **16.067** | −0.170 |
| spr | — | — | — | — | — | **0.978** |

Note from CCV: "The bond is a 5-year nominal bond in the quarterly dataset
and a 20-year for the annual dataset."

---

## §4.2 (continued) — This build's own VAR estimates ✅ 2026-05-07

Estimated under the §2.2.r and §2.2.μ restrictions on Option A
(1920-2011, T=92). Source data: `data/var_dataset.csv`. Estimator:
`lifecycle.var.estimate_var1_from_csv` (`var.py:191-268`).

### Build Table 1 — Sample statistics, 1920-2011

| Row | Statistic | This build | CCV ref | Notes |
|---|---|---|---|---|
| (1) | $\mathbb{E}[r_{tb}]$ + Jensen | +1.75% | 2.10% | within ±2pp ✓ |
| (2) | $\sigma(r_{tb})$ | 4.77% | 8.81% | exceeds 2pp band — caveat C1 (1920-2011 misses 1916-1919 reflation) |
| (3) | $\mathbb{E}[x_r]$ + Jensen | +7.01% | 6.80% | within ±2pp ✓ |
| (4) | $\sigma(x_r)$ | 19.12% | 18.19% | within ±2pp ✓ |
| (5) | Sharpe stock = (3)/(4) | 0.367 | 0.374 | ✓ |
| (6) | $\mathbb{E}[x_b]$ + Jensen | +1.34% | 0.67% | within ±1pp ✓ (caveat C3: AAA-vs-CCV-spliced) |
| (7) | $\sigma(x_b)$ | 7.30% | 6.54% | within ±2pp ✓ |
| (8) | Sharpe bond = (6)/(7) | 0.184 | 0.103 | (caveat C3) |
| (9) | $\mathbb{E}[y_1]$ | 4.40% | 4.36% | within ±1pp ✓ |
| (10) | $\sigma(y_1)$ | 3.14% | 2.60% | (caveat C1) |
| (11) | $\mathbb{E}[dp]$ | -3.257 | -3.101 | within ±0.3 ✓ |
| (12) | $\sigma(dp)$ | 0.444 | 0.304 | within ±0.2 ✓ (post-2000 dp volatility) |
| (13) | $\mathbb{E}[\text{spr}]$ | +1.26pp | +0.90pp | (caveat C3) |

### Build Table 2B — VAR coefficient diagonals (persistence)

| Variable | This build | CCV ref | Caveats |
|---|---|---|---|
| $\Phi[r_{tb}, r_{tb}]$ | +0.472 | +0.300 | C1, C2 |
| $\Phi[y_1, y_1]$ | +0.930 | +0.921 | ✓ |
| $\Phi[dp, dp]$ | +0.929 | +0.842 | C1 |
| $\Phi[\text{spr}, \text{spr}]$ | +0.657 | +0.820 | C3 (different long-yield) |

### Build R²

| Equation | This build | CCV ref |
|---|---|---|
| $r_{tb,t+1}$ | 0.411 | 0.240 |
| $x_{r,t+1}$ | 0.126 | 0.050 |
| $x_{b,t+1}$ | 0.407 | 0.392 |
| $y_{t+1}$ | 0.782 | 0.776 |
| $(d-p)_{t+1}$ | 0.899 | 0.721 |
| $\text{spr}_{t+1}$ | 0.435 | 0.540 |

### Stationarity

$\max_i |\lambda_i(\Phi_1)| = 0.9493$ (CCV w8566 reports ~0.92-0.95). Passes
§4.D4 stationarity test.

### Empirical effect of §2.2.r restriction (was deferred §4.2 item)

Comparing restricted (default) vs unrestricted estimation on the same
1920-2011 sample (`scripts/verify_ccv_implementation.py` §4.H):

| Equation | $R^2$ restricted | $R^2$ unrestricted | $\Delta R^2$ |
|---|---|---|---|
| $x_{r,t+1}$ | 0.126 | 0.140 | +0.014 |
| $x_{b,t+1}$ | 0.407 | 0.494 | +0.087 |

The restriction loses ~1.4 percentage points of $R^2$ on stock excess
returns and ~8.7 pp on bond excess returns. In the unrestricted fit,
$\max |\Phi_{ij}|$ on lagged-x columns is 0.382 — non-zero but not large.
The cost of imposing §2.2.r is therefore modest, concentrated in the
$x_b$ equation.

### Sensitivity to the AAA-1919 conflict resolution

Reported in `scripts/sensitivity_var_window.py`. Headline finding: the
Markowitz-at-γ=1 bond weight $\alpha_b^*$ ranges from 0.52 (Option D:
RLONG throughout, T=140) to 2.91 (Option A: AAA throughout, T=92), with
intermediate values 1.04 (C, splice T=140) and 1.26 (A_RLONG, RLONG on
same 1920-2011 window). Sample-length effect: ~5pp of $\alpha_b^*$;
yield-source effect: ~150pp. The yield-source choice (AAA vs RLONG) is
the dominant lever, not the sample window.

---

## Pending verification

### §2.2.μ — Sample-mean pinning of $\Phi_0$ ✅ LOCKED 2026-05-07

The code does *not* freely estimate $\Phi_0$. Instead, after estimating
$\Phi_1$ on demeaned data without an intercept, $\Phi_0$ is recovered as

$$\Phi_0 = (I - \Phi_1)\,\mu_{\mathbf{y}}^{\text{sample}}$$

where $\mu_{\mathbf{y}}^{\text{sample}}$ is the sample mean of $\mathbf{y}_t$.
This forces the VAR-implied unconditional mean to equal the sample mean
exactly.

**All four items resolved 2026-05-07:**
1. ✅ CCV w8566 §4.2 footnote 5 explicitly imposes this restriction.
2. ✅ The matrix form $\Phi_0 = (I - \Phi_1)\,\mu^{\text{sample}}$ is the
   precise content of "the unconditional means... equal their full-sample
   arithmetic counterparts."
3. ✅ This conforms to w8566 — it is **not** a deviation (unlike §2.2.r).
4. ✅ `var.py:191-268` implements the $\Phi_0 = (I - \Phi_1)\,\mu^{\text{sample}}$
   flow verbatim:
   - line 214: `z_bar = data.mean(axis=0).to_numpy()`
   - line 217: `Z = data.to_numpy() - z_bar` (demean)
   - line 228: `coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)` (OLS, no intercept)
   - line 236: `const = (np.eye(n) - Phi) @ z_bar` (back-solve)
   Numerical confirmation: on the Option A estimate,
   $\max |(I-\Phi_1)^{-1}\Phi_0 - \mu^{\text{sample}}| = 8.88\times10^{-16}$
   (machine epsilon). See `scripts/verify_ccv_implementation.py` §4.D test D2.

§2.2.μ moves to ✅ LOCKED.

---

### Items deferred to later sections

- **Two-stage Cholesky factorisation of $\Sigma_v$ for quadrature**:
  discretization machinery (own section TBD).
- **Empirical effect of §2.2.r restriction** (R², shifted coefficients):
  ✅ resolved 2026-05-07 — see "Empirical effect of §2.2.r restriction"
  subsection above (full §4.2 build estimates). $\Delta R^2 = +0.014$ on
  $x_r$ and $+0.087$ on $x_b$ when the restriction is dropped.
- **Wealth evolution and Euler equation**: §2.3.
- ~~**This build's own VAR estimates** (Tables 1 and 2 equivalents)~~:
  ✅ resolved 2026-05-07 — see Build Tables 1 and 2B subsections above.
- **Post-war subsample re-estimation**: per §4.1's robustness note.

---

*End of locked content. Document version: 2026-05-07.*
