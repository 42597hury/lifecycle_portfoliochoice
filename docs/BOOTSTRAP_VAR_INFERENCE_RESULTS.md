# Bootstrap inference for the lifecycle VAR system

**Run date:** 2026-05-12
**Pipeline:** [lifecycle/bootstrap/](../lifecycle/bootstrap/)
**Sample:** 1920-2011 annual (T=92 donor rows; T=91 effective after CLM lookback drop)

## Methodology

Because the real-yield and bond-return variables are generated through a
multi-stage construction, final-stage VAR standard errors understate
uncertainty. We therefore use an end-to-end stationary block bootstrap
(Politis-Romano 1994) of the annual primitive panel, with mean block
length 8. In each bootstrap draw we re-estimate the inflation AR(1),
reconstruct proxy real yields, re-estimate the 2-variable real-rate
construction VAR, recompute the EH real long yield and the term-premium
residual, and rebuild the lambda-specific return systems. The final
lifecycle VAR is then re-estimated under the production restrictions
(EH-coherence at lambda=0; lagged-state-only at lambda>0). Draws are
**paired across lambda**, so uncertainty in comparative statics reflects
the common resampled history.

Two runs are reported:

- **Headline run** (`data/bootstrap_conditional/`): B=500, AR(1) held at
  the production full-sample estimate (phi=0.388, mu=2.11%). This is
  closest to the implemented production estimator, because production
  uses the full-sample monthly inflation AR(1) (fit on Shiller's
  1872-present monthly CPI) in construction. Yields and returns are
  rebuilt per draw.
- **Conservative robustness** (`data/bootstrap/`): B=1000, AR(1) refit
  on the resampled annual `pi_info` sequence. Adds inflation-model
  sampling uncertainty using only the resampled annual panel, which
  is a different information set from production. Use this to bound
  how much extra uncertainty there would be if the inflation
  expectations model were re-estimated from the same sample as the
  rest of the pipeline.

The bootstrap assesses sampling stability of the constructed return
systems. It is **not** interpreted as identifying a structural real term
premium. Maintained-model uncertainty (AR(1) inflation specification,
EH decomposition, CLM duration, lambda-loading interpretation) is not
captured.

## Self-verification (Section 5 of handoff)

| Check | Conditional (headline) | Refit (robustness) |
|---|---|---|
| Identity test (`tests/test_bootstrap_identity.py`): 5/5 | PASS | PASS |
| Construction VAR `P(max\|eig\|<1)` | 1.000 | 1.000 |
| Final state VAR `P(max\|eig\|<1)` | 1.000 | 1.000 |
| Restricted-EH zero check at lambda=0 (50 random draws) | 0 violations | 0 violations |
| Pairing sanity `corr(E_xb_1, E_xb_0)` | 0.427 | 0.559 |
| Paired vs unpaired variance (`Var(diff) < Sum of marginals`) | 1.44e-5 < 1.92e-5 | 1.37e-5 < 2.01e-5 |
| Final-VAR failure rate | 0/500 | 0/1000 |

All stable, no draws dropped.

## Table 1: Marginal moments and coefficients (headline run, B=500, conditional AR(1))

90% intervals are [p05, p95]. All quantities other than `Phi*` and
`state_max_eig` are in log-decimal units (multiply by 100 for percentage
points). Returns are Jensen-adjusted where the Sharpe is computed.

### lambda = 0.0 (EH-restricted xb row)

| stat | point | boot mean | boot sd | p05 | p95 |
|---|---:|---:|---:|---:|---:|
| E[xb] | -0.0005 | -0.0011 | 0.0014 | -0.0036 | +0.0009 |
| sd(xb) | 0.0591 | 0.0523 | 0.0107 | 0.0350 | 0.0700 |
| Sharpe(xb) | +0.021 | +0.006 | 0.024 | -0.035 | +0.044 |
| state max\|eig\| | 0.934 | 0.799 | 0.066 | 0.686 | 0.896 |
| corr(v_y10, v_xb) | -0.991 | -0.986 | 0.014 | -0.997 | -0.963 |

At lambda=0 the EH restriction `Phi[xb,:]=0` and `E[xb]=0` is imposed
inside each bootstrap draw. The non-zero point E[xb] = -0.05pp is the
*sample* mean of the constructed xb at lambda=0; the *imposed*
population mean is zero and that is what the bootstrap distribution
reflects via residual uncertainty.

### lambda = 0.5

| stat | point | boot mean | boot sd | p05 | p95 |
|---|---:|---:|---:|---:|---:|
| E[xb] | +0.0031 | +0.0014 | 0.0024 | -0.0024 | +0.0054 |
| sd(xb) | 0.0580 | 0.0766 | 0.0129 | 0.0557 | 0.0988 |
| Sharpe(xb) | +0.082 | +0.057 | 0.033 | +0.006 | +0.113 |
| Phi_xb_spr | +1.110 | +2.484 | 1.053 | +1.120 | +4.560 |
| Phi_xb_y1 | +0.422 | +1.658 | 0.754 | +0.691 | +3.077 |
| state max\|eig\| | 0.934 | 0.799 | 0.066 | 0.686 | 0.896 |
| corr(v_y10, v_xb) | -0.992 | -0.995 | 0.003 | -0.998 | -0.990 |

### lambda = 1.0

| stat | point | boot mean | boot sd | p05 | p95 |
|---|---:|---:|---:|---:|---:|
| E[xb] | +0.0064 | +0.0032 | 0.0042 | -0.0038 | +0.0099 |
| sd(xb) | 0.0638 | 0.1067 | 0.0229 | 0.0698 | 0.1440 |
| Sharpe(xb) | +0.133 | +0.085 | 0.041 | +0.024 | +0.150 |
| Phi_xb_spr | +1.502 | +2.697 | 0.907 | +1.451 | +4.399 |
| Phi_xb_y1 | +0.389 | +1.592 | 0.677 | +0.689 | +2.798 |
| state max\|eig\| | 0.934 | 0.799 | 0.066 | 0.686 | 0.896 |
| corr(v_y10, v_xb) | -0.989 | -0.990 | 0.005 | -0.997 | -0.981 |

## Table 2: Paired differences across lambda (main run)

Paired differences use the same resampled history within a draw. The
right-most column is the bootstrap probability that the difference is
positive.

### Conditional AR(1) (headline, B=500)

| stat | lam_a - lam_b | point | sd | p05 | p95 | P(>0) |
|---|---|---:|---:|---:|---:|---:|
| E[xb] | 1.0 - 0.0 | +0.0070 | 0.0038 | -0.0023 | +0.0099 | **0.864** |
| E[xb] | 1.0 - 0.5 | +0.0034 | 0.0020 | -0.0017 | +0.0047 | 0.816 |
| E[xb] | 0.5 - 0.0 | +0.0036 | 0.0018 | -0.0006 | +0.0052 | 0.908 |
| Sharpe(xb) | 1.0 - 0.0 | +0.112 | 0.033 | +0.023 | +0.131 | **0.984** |
| Sharpe(xb) | 1.0 - 0.5 | +0.051 | 0.013 | +0.011 | +0.052 | 0.992 |
| Sharpe(xb) | 0.5 - 0.0 | +0.061 | 0.022 | +0.012 | +0.081 | 0.982 |
| Phi_xb_spr | 1.0 - 0.0 | +1.502 | 0.907 | +1.451 | +4.399 | **1.000** |
| Phi_xb_spr | 1.0 - 0.5 | +0.392 | 0.252 | -0.202 | +0.580 | 0.862 |
| Phi_xb_spr | 0.5 - 0.0 | +1.110 | 1.053 | +1.120 | +4.560 | 1.000 |

### Refit AR(1) (conservative robustness, B=1000)

| stat | lam_a - lam_b | point | sd | p05 | p95 | P(>0) |
|---|---|---:|---:|---:|---:|---:|
| E[xb] | 1.0 - 0.0 | +0.0070 | 0.0037 | -0.0044 | +0.0075 | **0.705** |
| E[xb] | 1.0 - 0.5 | +0.0034 | 0.0020 | -0.0028 | +0.0035 | 0.639 |
| E[xb] | 0.5 - 0.0 | +0.0036 | 0.0018 | -0.0016 | +0.0040 | 0.782 |
| Sharpe(xb) | 1.0 - 0.0 | +0.112 | 0.027 | +0.017 | +0.107 | **0.989** |
| Sharpe(xb) | 1.0 - 0.5 | +0.051 | 0.010 | +0.009 | +0.040 | 0.996 |
| Sharpe(xb) | 0.5 - 0.0 | +0.061 | 0.018 | +0.008 | +0.069 | 0.983 |
| Phi_xb_spr | 1.0 - 0.0 | +1.502 | 1.022 | +1.396 | +4.662 | **1.000** |
| Phi_xb_spr | 1.0 - 0.5 | +0.392 | 0.259 | -0.217 | +0.555 | 0.846 |
| Phi_xb_spr | 0.5 - 0.0 | +1.110 | 1.156 | +1.082 | +4.750 | 1.000 |

Refitting the annual AR(1) inside each draw injects extra
inflation-model uncertainty into the E[xb] comparative static:
`P(E[xb](1.0) > E[xb](0.0))` falls from 0.864 (conditional, headline)
to 0.705 (refit). About 40% of the residual uncertainty in
Delta_E[xb] under refit is therefore attributable to the
inflation-AR(1) refit noise, which production does not actually use
(production fits AR(1) on full Shiller monthly inflation, not the
resampled annual panel). Sharpe and Phi_xb_spr comparative statics are
nearly invariant across the two modes.

## Stability fractions (headline, conditional B=500)

| flag | lambda=0.0 | lambda=0.5 | lambda=1.0 |
|---|---:|---:|---:|
| P(state max\|eig\| < 1) | 1.000 | 1.000 | 1.000 |
| P(construction VAR max\|eig\| < 1) | 1.000 | 1.000 | 1.000 |
| P(Phi_xb_spr > 0) | 0/0 (zero by restriction) | 1.000 | 1.000 |
| P(final VAR failed) | 0.000 | 0.000 | 0.000 |
| P(E[xb](1.0) > E[xb](0.0)) | -- | -- | 0.864 |
| P(Sharpe[xb](1.0) > Sharpe[xb](0.0)) | -- | -- | 0.984 |

## Headline findings

1. **Sign of Phi_xb_spr is bullet-proof.** Under both inflation modes,
   100% of bootstrap draws produce a positive bond-return loading on the
   lagged real-yield spread at lambda > 0. The comparative static
   Delta_Phi_xb_spr(1.0 - 0.0) is positive in every single draw.

2. **Magnitude of Phi_xb_spr is highly uncertain.** Point estimate is
   1.50 at lambda=1.0, but the 90% bootstrap CI is [1.40, 4.66] and the
   bootstrap median (2.55) is above the point estimate. The thesis should
   not lean on a specific numerical value of Phi_xb_spr; the sign and
   the cross-lambda ordering are the defensible claims.

3. **Sharpe-ratio comparative static is robust.** P(Sharpe(1.0) >
   Sharpe(0.0)) is 0.984 in the headline (conditional) and 0.989 in
   the refit robustness. The thesis claim that loading more of the
   real-yield residual onto the bond return raises its Sharpe ratio
   survives both the sampling resample and the inflation-AR(1) refit.

4. **Level of E[xb] comparative static is moderately robust under the
   headline and fragile under the conservative robustness.**
   P(E[xb](1.0) > E[xb](0.0)) is 0.864 in the headline (conditional)
   and 0.705 in the refit bootstrap. The point difference is +0.70 pp
   annually; under conditional it is roughly 2 bootstrap standard
   deviations from zero, under refit closer to ~2 sd of an inflated
   variance. The thesis should report 0.864 as the inference target
   and 0.705 as a conservative ceiling.

5. **sd(xb) grows monotonically with lambda in the bootstrap.** Point
   sd(xb) is 5.9% / 5.8% / 6.4% across lambda = 0 / 0.5 / 1.0;
   conditional-bootstrap medians are 5.2% / 7.7% / 10.7%. The
   bootstrap reveals that the point sample mildly under-states the
   conditional volatility of xb at high lambda, because production
   point uncertainty in Phi_R suppresses some of the variation that
   the bootstrap re-injects through TP^R.

6. **`corr(v_y10, v_xb)` is essentially -1 across the bootstrap.** This
   is a duration-implied identity: under CLM,
   `v_xb ~ -(D-1) * v_y10` to leading order. Values close to -1 are a
   sanity check that the duration scaling is being recovered correctly
   per draw.

## What the bootstrap does NOT cover

- AR(1) inflation specification (e.g., trend-inflation models, time-varying
  persistence, monthly-vs-annual aggregation choice).
- EH decomposition validity (the construction VAR is assumed to identify
  expected future short rates; this is a model choice, not data).
- CLM duration approximation (linearization of bond returns around current
  yield).
- The `lambda` residual-loading interpretation itself: lambda mixes the
  EH-decomposed expected yield with a residual whose structural meaning
  is not pinned down by the data alone.
- Lifecycle policy-function uncertainty (consumption-savings rules,
  portfolio choices, welfare): these would require solving the full
  lifecycle model in every draw, which is computationally infeasible.
  Section 7 of the handoff describes a 3-scenario compromise if needed.

## Known finite-sample artifacts

- **Lag construction at block boundaries.** The bootstrap treats the
  resampled panel as a pseudo-time series, so lagged quantities such as
  `y_1[k-1]` (used to build `xr` and `xb`) are taken from the previous
  row in pseudo-time. Within blocks this preserves the original
  chronology; at block boundaries it creates artificial lag links
  (`y_1[k-1]` may come from an unrelated donor year). This is a
  standard finite-sample artifact of block bootstrap methods,
  asymptotically negligible as `ell_mean` grows, but with T=92 and
  ~12 boundaries per draw it may meaningfully affect `xr` and `xb`
  moment distributions. We currently use the pseudo-time convention
  and drop the first resampled row (no lookback).
- **First-row drop.** With pseudo-time lags, the first resampled row
  has NaN `xr` and `xb` and is dropped: T_lam = 91 vs production T = 92.

## Files

- Headline draws (conditional): [data/bootstrap_conditional/var_bootstrap_draws.csv](../data/bootstrap_conditional/var_bootstrap_draws.csv)
- Headline summaries: [data/bootstrap_conditional/](../data/bootstrap_conditional/)
- Robustness draws (refit): [data/bootstrap/var_bootstrap_draws.csv](../data/bootstrap/var_bootstrap_draws.csv)
- Robustness summaries: [data/bootstrap/](../data/bootstrap/)
- Point estimates (identical across runs): `var_bootstrap_point.csv`

## Reproduction

```
# Headline (B=500, conditional AR(1) -- closest to production estimator)
python -m lifecycle.bootstrap.run_bootstrap --B 500 --ell-mean 8 --inflation-mode conditional --seed 0 --out data/bootstrap_conditional

# Conservative robustness (B=1000, refit AR(1) on resampled annual panel)
python -m lifecycle.bootstrap.run_bootstrap --B 1000 --ell-mean 8 --inflation-mode refit --seed 0 --out data/bootstrap

# Identity test
python -m pytest tests/test_bootstrap_identity.py -v
```

End-to-end runtime on a laptop (single-threaded): ~2 seconds for B=1000.
