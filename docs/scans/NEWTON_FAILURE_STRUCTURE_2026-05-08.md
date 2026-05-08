# Newton-failure structural scan (2026-05-08)

**Branch:** `jax-rewrite`
**Scope:** post-fix (commit 8bfaec9 + exit-code wiring 5c02d19) lifecycle and
inf-horizon bundles only.
**Status:** read-only structural follow-up to
[`NEWTON_HISTOGRAM_AUDIT_2026-05-07.md`](NEWTON_HISTOGRAM_AUDIT_2026-05-07.md)
(which fixed counting; this report uses the now-reliable histograms).

## TL;DR

* Failure rates show a clear **U-shape over age**: 9–10 % young, trough at
  3–7 % near retirement transition (60–67), climbing back to **14–17 % at
  ages 95–98**. Terminal age 99: ~1 % (separate path).
* Newton p99 saturates at max_iter in every bundle (lifecycle 30, inf-horizon
  100); backtrack p99 sits at 60–80 % of the worst-case bound. **Bumping
  max_iter or max_backtrack will not fix this.**
* Inf-horizon: failures peak in **mid-Bellman-iter** (≈25–31), decay to <1 %
  near convergence — but Newton p99 still saturates at 100.
* Cross-bundle: fail rate **does not** scale with state-space size. It scales
  with **wealth-grid spacing at the high-wealth tail** (log → bakh: 9.76 %
  → 6.07 %).
* Dominant mechanism: **`tol = 1e-7` is unreachable at high-savings cells**.
  At γ=5 with large `s`, FOC scale is below fp64 precision relative to tol.

System II 4×4/5×5/6×6 bundles are **not yet on disk**; the 7×7 S2 bundles
on disk pre-date the histogram fix and are excluded. Conclusions rest on
6 post-fix S1 + 4 post-fix S4 inf-horizon bundles.

## §1. Per-age distribution

`age_newton_fail / (n_z · N_state · n_savings)` — plot:
[`figures/newton_fail_per_age.png`](figures/newton_fail_per_age.png).

| age           | w180_log | w180_bakh | w90_log | w60_log |
|---------------|---------:|----------:|--------:|--------:|
| 22 (start)    |   8.8 %  |    5.6 %  |  9.5 %  |  9.0 %  |
| 45 (peak inc) |  10.4 %  |    7.0 %  | 10.4 %  | 10.5 %  |
| 67 (retire 1) |   6.8 %  |    3.0 %  |  6.7 %  |  7.3 %  |
| 80            |   9.7 %  |    5.3 %  |  9.7 %  |  9.9 %  |
| 90            |  12.4 %  |    7.4 %  | 12.4 %  | 12.5 %  |
| 95            |  13.8 %  |    9.0 %  | 14.1 %  | 13.9 %  |
| 98 (oldest)   |  13.6 %  |   17.2 %  | 13.8 %  | 13.9 %  |
| 99 (terminal) |   1.0 %  |    1.0 %  |  1.0 %  |  1.0 %  |

The U-shape rules out "Newton stuck on the income-shock integral":
retirement at age 95 (no income shock) fails *more* than working at 30. The
bequest-driven FOC near terminal is the harder path.

## §2. Newton-iter histogram

Across all post-fix bundles: median Newton solve converges in 1–2 iters
thanks to backward warm-start; the top 1 % saturates at max_iter regardless
of budget.

| bundle           | cells   | fails   | rate %| p50 | p95 | p99 | max | mxIt |
|------------------|--------:|--------:|------:|----:|----:|----:|----:|-----:|
| S1 w180_log      | 2.43 M  | 236 915 |  9.76 |   2 |  30 |  30 |  30 |   30 |
| S1 w180_bakh     | 2.43 M  | 147 231 |  6.07 |   2 |  30 |  30 |  30 |   30 |
| S1 w120_log      | 1.62 M  | 158 858 |  9.82 |   2 |  30 |  30 |  30 |   30 |
| S1 w90_log       | 1.21 M  | 120 232 |  9.91 |   2 |  30 |  30 |  30 |   30 |
| S1 w90_bakh      | 1.21 M  |  75 271 |  6.20 |   2 |  30 |  30 |  30 |   30 |
| S1 w60_log       | 0.81 M  |  81 221 | 10.04 |   2 |  30 |  30 |  30 |   30 |
| S4 inf 3⁴        | 1.20 M  | 104 162 |  8.71 |   1 | 100 | 100 | 100 |  100 |
| S4 inf 4⁴        | 4.47 M  | 270 835 |  6.06 |   1 | 100 | 100 | 100 |  100 |
| S4 inf 5⁴        |11.25 M  | 568 421 |  5.05 |   1 |  84 | 100 | 100 |  100 |
| S4 inf 5⁴ axis 1 | 6.97 M  | 511 961 |  7.34 |   1 | 100 | 100 | 100 |  100 |

Fail rate ≈ share of histogram pinned at max_iter. At 100 iters, p99 still
saturates → those cells don't converge in <100 iters either, ruling out
"tight max_iter" as the proximate cause. S4-5⁴'s p95 = 84 (only sub-max p95)
shows bigger state grids move bulk cells off saturation but the saturating
tail persists.

## §3. Backtrack histogram

Per-cell **sum** of line-search halvings. Worst-case bound =
`max_iter × max_backtrack_iter`.

| bundle             | bt_p50 | bt_p95 | bt_p99 | bt_max | bound |
|--------------------|-------:|-------:|-------:|-------:|------:|
| S1 (any post-fix)  |      0 |  ≈42   | 89–94  |  ≈121 | **150** |
| S4 inf 3⁴/4⁴/5⁴    |      0 | 19–21  | 96–100 |   401 | **500** |
| S4 inf 5⁴ axis 1   |      0 |    42  |   125  |   500 | **500** |

Lifecycle p99 at ~60 % of bound; inf-horizon p99 at ~20 %. Line search has
slack. **Hypothesis 5 (backtrack saturation) is rejected** — bumping
`max_backtrack_iter` 5 → 10 will not move the fail rate.

## §4. Inf-horizon: failures vs. Bellman iter

Plot:
[`figures/newton_inf_horizon_per_iter.png`](figures/newton_inf_horizon_per_iter.png).

| bundle (n_iter)  | iter 0 | iter 5 | iter 20 |    mid     | last iter |
|------------------|-------:|-------:|--------:|-----------:|----------:|
| inf 3⁴ (82)       |  7.0 % |  6.6 % |  14.2 % | 12.6 % @41 | **0.6 %** |
| inf 4⁴ (97)       |  6.1 % |  4.7 % |  10.4 % |  8.7 % @48 | **0.6 %** |
| inf 5⁴ (100)      |  5.8 % |  3.8 % |   8.4 % |  7.0 % @50 | **0.6 %** |
| inf 5⁴ axis 1 (62)|  6.2 % |  4.0 % |   8.6 % |  9.9 % @31 | 4.3 %     |

Failures peak in the steepest-descent phase (~iter 25–31) and decay sharply
at convergence. Once the Bellman iterate settles, warm start carries 99.4 %
of cells home. The non-fully-converged axisbump_r1 (4.3 % at last iter) is
consistent with not yet reaching the warm-start basin.

## §5. Cross-bundle pattern

Fail rate **correlates with** wealth-grid spacing at the high-wealth tail.
w180_log → 9.76 %; w180_bakh (Bakhvalov v3) → 6.07 %. Same calibration, state
grid, n_z. **37 % reduction** purely from grid layout at large `w` — strong
evidence high-wealth/high-savings cells dominate the failure population.

Fail rate does **not** correlate with state-space size: S4-3⁴ → 8.7 %,
S4-4⁴ → 6.1 %, S4-5⁴ → 5.1 %. **Going from 81 to 625 state cells *decreases*
the rate**, opposite to "extreme state cells fail more". Per-axis bumps push
it back to 7.3 %, but small relative to the wealth-grid effect.

## §6. Likely cause

**Dominant mechanism: `tol = 1e-7` is unreachable at high-savings cells.**
Evidence: (1) p99 saturates regardless of max_iter = 30 or 100 (§2).
(2) Backtrack p99 at 60–80 % of bound — descent direction is fine, the
convergence test is the bottleneck (§3). (3) Wealth-grid sensitivity (§5)
localizes failures to the high-savings tail; with γ = 5 and large `s`, FOC
scale `e0 ~ c^{−γ}` is tiny and `err < tol·scale` falls below fp64
precision. (4) The audit showed top 2–3 savings indices saturate at
max_iter while middle savings finish in 6–10 iters — same FOC, only `s_val`
changes.

**Secondary:** high-leverage flat-FOC basins (γ = 5, unbounded α) explain
why age 22 still fails 8.8 % despite a simple problem.

**Ruled out:** tight `max_backtrack_iter` (§3), ill-posedness at extreme
states (§5), working-FOC harder than retirement (§1).

## §7. Recommendations (ranked by leverage)

| mitigation | expected fail-rate change | wall-time cost |
|---|---|---|
| **Bakhvalov v3 wealth grid (vs log)** | 9.8 % → 6.1 % (–37 %) | **~0 %** |
| **Loosen `tol` 1e-7 → 1e-6**          | likely –50–80 %       | ~–5 % |
| Switch to abs+rel mixed tol            | likely –80 %          | ~0 % |
| Bump `max_iter` 30 → 60                | small (~10 %)         | +5–10 % |
| Bump `max_backtrack_iter` 5 → 10       | negligible            | negligible |
| α-box-cap (\|α_b\| ≤ 4)               | unknown (math change) | bit-identity broken |

**Top recommendation for a 12 h × 8× A100 production run:** Bakhvalov v3
wealth spacing (already in `lifecycle/wealth_grid.py`) + loosen `tol` to
1e-6. Together this realistically targets **<3 % fail rate at zero wall-time
cost**. Bit-identity vs. log-grid baselines is already broken by the spacing
change. If strict bit-identity is required, the cheapest pure knob is
`max_iter 30 → 50` — but it won't clear the tol-unreachable tail.

## Reproducibility

* Summary JSON:
  [`newton_failure_structure_2026-05-08.json`](newton_failure_structure_2026-05-08.json)
* Driver: [`scripts/analysis/newton_failure_structure.py`](../../scripts/analysis/newton_failure_structure.py)
* Per-age: [`scripts/analysis/newton_failure_age_breakdown.py`](../../scripts/analysis/newton_failure_age_breakdown.py)
* Inf per-iter: [`scripts/analysis/newton_failure_inf_breakdown.py`](../../scripts/analysis/newton_failure_inf_breakdown.py)
* Plots in `docs/scans/figures/`.

## Caveats

Per-cell exit codes are summed to per-age scalar `age_newton_fail[t]`
(`solver.py:2689,2781`); per-z/state/savings breakdowns are not persisted.
z/state/savings concentration is inferred via the wealth-grid sensitivity
(§5) and the per-savings drilldown in
[`NEWTON_HISTOGRAM_AUDIT_2026-05-07.md`](NEWTON_HISTOGRAM_AUDIT_2026-05-07.md).
Exposing exit-code arrays to `_build_iter_histograms` is a small follow-up
if z/state/savings heatmaps are needed. The dominant mechanism lives in
`_egm_scan_cell`'s per-savings FOC, shared across systems → should
generalise to S2 once those bundles arrive.
