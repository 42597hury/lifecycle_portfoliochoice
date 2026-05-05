"""
_diag_quadrature_cloud.py — Diagnostic tests on the joint state×return quadrature cloud.

Implements T-Q1 through T-Q7 from docs/handoff/HANDOFF_UNCONSTRAINED_LEVERAGE.md, all of
which run on (model, pc) plus the saved policy bundle — no fresh solves.

The tests are designed to localize the source of the residual unconstrained-
leverage pathology to one of:
  - discretization arbitrage in the joint cloud (T-Q1)
  - no-bankruptcy boundary at modest leverage (T-Q2)
  - cloud asymmetry / lopsided upside mass (T-Q3)
  - off-the-chart conditional Sharpe ratios (T-Q4)
  - high-Sharpe states having meaningful vs negligible stationary probability (T-Q5)
  - per-state Merton vs saved-policy alpha mismatch (T-Q6)
  - quadrature moment recovery against the analytical Σ (T-Q7)

Run from repo root: python -m scripts.diagnostics._diag_quadrature_cloud
"""

from __future__ import annotations

import numpy as np

from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import build_model, Precompute
from lifecycle.model import DiscretizationConfig


# =============================================================================
# Bundle loading utilities
# =============================================================================

def _unpack(x):
    if isinstance(x, dict) and x.get("kind") == "ndarray":
        return np.array(x["values"], dtype=float)
    return np.array(x, dtype=float)


def load_bundle(path: str):
    """Load a saved bundle and rebuild model+pc to match the saved discretization."""
    C, S, B, _diag, meta = load_policy_bundle(path)
    rc = meta["run_config"]
    bc = rc["base_config"]
    vc = rc["var_config"]
    dc_raw = rc["discretization_config"]

    vc["Phi"] = _unpack(vc["Phi"])
    vc["Omega"] = _unpack(vc["Omega"])
    vc["z_bar"] = _unpack(vc["z_bar"])

    n_ret_nodes_1d = dc_raw["n_ret_nodes_1d"]
    if isinstance(n_ret_nodes_1d, list):
        n_ret_nodes_1d = tuple(n_ret_nodes_1d)

    disc = DiscretizationConfig(
        n_wealth=dc_raw["n_wealth"],
        wealth_min=dc_raw["wealth_min"],
        wealth_max=dc_raw["wealth_max"],
        n_savings=dc_raw["n_savings"],
        savings_min=dc_raw["savings_min"],
        savings_max=dc_raw.get("savings_max"),
        state_grid_sizes=tuple(dc_raw["state_grid_sizes"]),
        state_grid_mode=dc_raw.get("state_grid_mode", "naive"),
        state_n_stds=dc_raw.get("state_n_stds", 3.0),
        n_z=dc_raw["n_z"],
        n_stds=dc_raw.get("n_stds", 3.0),
        n_eps_nodes=dc_raw["n_eps_nodes"],
        n_eta_nodes=dc_raw.get("n_eta_nodes", 3),
        n_ret_nodes_1d=n_ret_nodes_1d,
        n_state_quad_nodes=dc_raw.get("n_state_quad_nodes", 3),
    )
    model = build_model(bc, vc, verbose=False)
    pc = Precompute(model, disc, verbose=False)
    return model, pc, C, S, B, dc_raw


# =============================================================================
# Joint cloud construction
# =============================================================================

def build_cloud(model, pc, i_s: int):
    """Build the joint state-innovation × return-residual cloud at state i_s.

    Returns
    -------
    log_r : (n_total, 3)   log returns (rtb, xr, xb) at each node
    gross : (n_total, 3)   gross returns (R_bill, R_stock, R_bond) at each node
    w     : (n_total,)     joint quadrature weights, sum to 1
    """
    s_i = pc.state_grid[i_s]
    base = np.asarray(pc.const_r) + pc.A_r @ s_i  # (3,)

    n_v = len(pc.v_weights)
    n_r = len(pc.ret_weights)
    M = np.asarray(model.M)

    # Pre-compute M @ v_k for each state-innovation node (3-vector each)
    Mv = pc.v_nodes @ M.T  # (n_v, 3)

    # Tensor product: log_r[k_v * n_r + k_r] = base + Mv[k_v] + ret_nodes[k_r]
    log_r = (base[None, None, :] + Mv[:, None, :] + pc.ret_nodes[None, :, :]).reshape(-1, 3)
    w = np.outer(pc.v_weights, pc.ret_weights).ravel()

    R_bill = np.exp(log_r[:, 0])
    R_stock = R_bill * np.exp(log_r[:, 1])
    R_bond = R_bill * np.exp(log_r[:, 2])
    gross = np.column_stack([R_bill, R_stock, R_bond])

    return log_r, gross, w


# =============================================================================
# T-Q1: Convex-hull arbitrage gap (2D)
# =============================================================================

def arbitrage_gap_2d(X: np.ndarray, n_angles: int = 720) -> float:
    """Arbitrage gap = max_{||d||=1} min_n d · X^(n).

    Positive value ⇒ origin is strictly outside the convex hull of {X^(n)} ⇒
    a separating direction exists ⇒ discrete arbitrage. Computed by sweeping
    the unit circle in 2D.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    D = np.column_stack([np.cos(angles), np.sin(angles)])  # (n_angles, 2)
    proj = D @ X.T  # (n_angles, n_nodes)
    min_proj = proj.min(axis=1)
    return float(max(0.0, min_proj.max()))


# =============================================================================
# T-Q2: Worst-case R_port at moderate leverage
# =============================================================================

def worst_case_rport(gross: np.ndarray, alphas: list[tuple[float, float]]) -> list[float]:
    """For each (α_s, α_b), return min_n R_port_n over the cloud."""
    R_bill = gross[:, 0]
    R_stock = gross[:, 1]
    R_bond = gross[:, 2]
    out = []
    for a_s, a_b in alphas:
        a_bill = 1.0 - a_s - a_b
        R_port = a_s * R_stock + a_b * R_bond + a_bill * R_bill
        out.append(float(R_port.min()))
    return out


# =============================================================================
# T-Q3: Cloud asymmetry / lopsidedness
# =============================================================================

def weighted_skewness(values: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    if var <= 0:
        return 0.0
    return float(np.sum(weights * ((values - mean) / np.sqrt(var)) ** 3))


def upside_mass_ratio(values: np.ndarray, weights: np.ndarray) -> float:
    """Mass above mean / mass below mean. Lopsided cloud ⇒ ratio ≫ 1 (or ≪ 1)."""
    mean = float(np.sum(weights * values))
    above = float(np.sum(weights[values > mean]))
    below = float(np.sum(weights[values < mean]))
    if below <= 0:
        return float("inf")
    return above / below


# =============================================================================
# T-Q4: Per-state conditional Sharpe and Merton inputs
# =============================================================================

def excess_moments(gross: np.ndarray, w: np.ndarray):
    """Compute (μ_e, Σ_e) for excess level returns over bills."""
    R_bill = gross[:, 0]
    Xs = gross[:, 1] - R_bill
    Xb = gross[:, 2] - R_bill
    mu_s = float(np.sum(w * Xs))
    mu_b = float(np.sum(w * Xb))
    var_s = float(np.sum(w * (Xs - mu_s) ** 2))
    var_b = float(np.sum(w * (Xb - mu_b) ** 2))
    cov_sb = float(np.sum(w * (Xs - mu_s) * (Xb - mu_b)))
    mu_e = np.array([mu_s, mu_b])
    Sigma_e = np.array([[var_s, cov_sb], [cov_sb, var_b]])
    return mu_e, Sigma_e


def sharpes(mu_e: np.ndarray, Sigma_e: np.ndarray) -> tuple[float, float, float]:
    s_s = mu_e[0] / np.sqrt(Sigma_e[0, 0]) if Sigma_e[0, 0] > 0 else 0.0
    s_b = mu_e[1] / np.sqrt(Sigma_e[1, 1]) if Sigma_e[1, 1] > 0 else 0.0
    try:
        s_j = float(np.sqrt(max(0.0, mu_e @ np.linalg.solve(Sigma_e, mu_e))))
    except np.linalg.LinAlgError:
        s_j = 0.0
    return float(s_s), float(s_b), s_j


# =============================================================================
# T-Q6: Closed-form Merton on the cloud's empirical moments
# =============================================================================

def merton_alpha(mu_e: np.ndarray, Sigma_e: np.ndarray, gamma: float) -> np.ndarray:
    """Two-asset Merton: α* = (1/γ) Σ_e^{-1} μ_e. Returns (α_stock, α_bond)."""
    return (1.0 / gamma) * np.linalg.solve(Sigma_e, mu_e)


# =============================================================================
# T-Q7: Cloud moment recovery vs analytical
# =============================================================================

def moment_recovery(model, pc, i_s: int):
    log_r, _, w = build_cloud(model, pc, i_s)
    mean_emp = (w[:, None] * log_r).sum(axis=0)
    diff = log_r - mean_emp
    cov_emp = (diff.T * w) @ diff

    s_i = pc.state_grid[i_s]
    mean_anal = np.asarray(pc.const_r) + pc.A_r @ s_i
    M = np.asarray(model.M)
    cov_anal = M @ np.asarray(model.Sigma_ss) @ M.T + np.asarray(model.Sigma_r_cond)
    return mean_emp, mean_anal, cov_emp, cov_anal


# =============================================================================
# Main driver
# =============================================================================

def main():
    bundle_path = "saved_runs/unconstrained_principal_grid5x5x5_nz9"
    model, pc, C, S, B, dc_raw = load_bundle(bundle_path)
    gamma = float(model.gamma)
    n_states = pc.N_state
    stat_probs = np.asarray(pc.state_stationary_probs)

    print(f"Bundle: {bundle_path}")
    print(f"  γ = {gamma},  N_state = {n_states},  n_z = {pc.n_z},  n_w = {pc.n_w}")
    print(
        f"  state_grid_mode = {pc.state_grid_mode},  state_n_stds = "
        f"{dc_raw.get('state_n_stds', 3.0)}"
    )
    print(
        f"  K_state = {dc_raw.get('n_state_quad_nodes', 3)} → "
        f"{pc.n_state_quad} joint state nodes"
    )
    print(
        f"  K_ret per dim = {pc.n_ret_nodes_1d} → "
        f"{pc.n_ret_quad} joint return nodes"
    )
    print(f"  Joint cloud per state: {pc.n_state_quad * pc.n_ret_quad} nodes")

    # ----- T-Q7 first (cheap moment check) -----
    print("\n" + "=" * 78)
    print("T-Q7  Cloud moment recovery (empirical vs analytical, log returns)")
    print("=" * 78)
    mean_errs = np.zeros(n_states)
    cov_errs = np.zeros(n_states)
    for i_s in range(n_states):
        m_emp, m_anal, c_emp, c_anal = moment_recovery(model, pc, i_s)
        mean_errs[i_s] = float(np.max(np.abs(m_emp - m_anal)))
        cov_errs[i_s] = float(np.max(np.abs(c_emp - c_anal)))
    verdict = "PASS" if (mean_errs.max() < 1e-10 and cov_errs.max() < 1e-9) else "FAIL"
    print(f"  Across {n_states} states:")
    print(f"    Max mean error:  {mean_errs.max():.2e}")
    print(f"    Max cov error:   {cov_errs.max():.2e}")
    print(f"    {verdict}: tensor-product GH should match low-order log-moments")
    print(f"          to machine precision; failure ⇒ rule construction bug.")

    # ----- Per-state cloud diagnostics (T-Q1..T-Q4, T-Q6) -----
    arb = np.zeros(n_states)
    minR_2_1 = np.zeros(n_states)
    minR_5_0 = np.zeros(n_states)
    minR_3_2 = np.zeros(n_states)
    skew_s = np.zeros(n_states)
    skew_b = np.zeros(n_states)
    upside_s = np.zeros(n_states)
    sh_s = np.zeros(n_states)
    sh_b = np.zeros(n_states)
    sh_j = np.zeros(n_states)
    a_m = np.zeros((n_states, 2))  # Merton (stock, bond)

    # Saved-policy α at age 22, median z, wealth nearest 0.25 (typical age-22 x)
    iz_med = pc.n_z // 2
    iw_typical = int(np.argmin(np.abs(pc.wealth_grid - 0.25)))
    a_solver = np.zeros((n_states, 2))

    for i_s in range(n_states):
        log_r, gross, w = build_cloud(model, pc, i_s)
        Xs = gross[:, 1] - gross[:, 0]
        Xb = gross[:, 2] - gross[:, 0]
        X2d = np.column_stack([Xs, Xb])

        arb[i_s] = arbitrage_gap_2d(X2d)
        mr = worst_case_rport(gross, [(2.0, 1.0), (5.0, 0.0), (3.0, 2.0)])
        minR_2_1[i_s], minR_5_0[i_s], minR_3_2[i_s] = mr

        skew_s[i_s] = weighted_skewness(Xs, w)
        skew_b[i_s] = weighted_skewness(Xb, w)
        upside_s[i_s] = upside_mass_ratio(Xs, w)

        mu_e, Sigma_e = excess_moments(gross, w)
        sh_s[i_s], sh_b[i_s], sh_j[i_s] = sharpes(mu_e, Sigma_e)
        a_m[i_s] = merton_alpha(mu_e, Sigma_e, gamma)

        a_solver[i_s, 0] = S[0, iz_med, i_s, iw_typical]
        a_solver[i_s, 1] = B[0, iz_med, i_s, iw_typical]

    # ----- T-Q1: Arbitrage gap -----
    print("\n" + "=" * 78)
    print("T-Q1  Convex-hull arbitrage gap")
    print("=" * 78)
    n_arb = int((arb > 1e-10).sum())
    print(f"  States with gap > 1e-10:  {n_arb}/{n_states}")
    print(f"  Max gap:                  {arb.max():.2e}")
    if n_arb == 0:
        print("  CLEAN — no discrete free lunch on the unit-circle support direction.")
    else:
        print("  ARBITRAGE PRESENT — solver is being handed a discrete free lunch.")
        worst = np.argsort(-arb)[:5]
        for i_s in worst:
            print(
                f"    i_s={i_s:>4}  s={np.round(pc.state_grid[i_s], 3)}  "
                f"gap={arb[i_s]:.3e}  stat_p={stat_probs[i_s]*100:.3f}%"
            )

    # ----- T-Q2: Worst-case R_port -----
    print("\n" + "=" * 78)
    print("T-Q2  Worst-case R_port at moderate leverage (over the joint cloud)")
    print("=" * 78)
    for label, arr in [("(α_s=2, α_b=1)", minR_2_1), ("(α_s=3, α_b=2)", minR_3_2), ("(α_s=5, α_b=0)", minR_5_0)]:
        print(f"  {label}:")
        print(f"    min over states: {arr.min():.4f}  (worst = {np.argmin(arr)})")
        print(f"    median:          {np.median(arr):.4f}")
        print(f"    states with min R_port < 0:    {int((arr < 0).sum())}/{n_states}")
        print(f"    states with min R_port < 0.05: {int((arr < 0.05).sum())}/{n_states}")
        print(f"    states with min R_port < 0.50: {int((arr < 0.50).sum())}/{n_states}")
    print(
        "  H1b (no-bankruptcy boundary) hypothesis: if min R_port → 0 at modest\n"
        "  leverage, unconstrained CRRA can lever up until the worst quadrature\n"
        "  node clamps to the wealth-grid floor, irrespective of true tail risk."
    )

    # ----- T-Q3: Cloud asymmetry -----
    print("\n" + "=" * 78)
    print("T-Q3  Cloud asymmetry — weighted skewness and upside-mass ratio")
    print("=" * 78)
    print("  Excess stock returns:")
    print(f"    skew  min={skew_s.min():.3f}  p50={np.median(skew_s):.3f}  max={skew_s.max():.3f}")
    print(f"    up/dn min={upside_s[np.isfinite(upside_s)].min():.3f}  "
          f"p50={np.median(upside_s[np.isfinite(upside_s)]):.3f}  "
          f"max={upside_s[np.isfinite(upside_s)].max():.3f}")
    print("  Excess bond returns:")
    print(f"    skew  min={skew_b.min():.3f}  p50={np.median(skew_b):.3f}  max={skew_b.max():.3f}")
    print(
        "  Large positive skew or up/dn ≫ 1 means most quadrature mass sits on\n"
        "  the upside; thin downside under-disciplines leverage even without\n"
        "  strict arbitrage."
    )

    # ----- T-Q4: Sharpe ratios -----
    print("\n" + "=" * 78)
    print("T-Q4  Per-state conditional Sharpe ratios (level excess returns)")
    print("=" * 78)
    print(f"  Stock Sharpe: p50={np.median(sh_s):.3f}, max={sh_s.max():.3f}, "
          f"states>2: {int((sh_s > 2).sum())}, states>5: {int((sh_s > 5).sum())}")
    print(f"  Bond Sharpe:  p50={np.median(sh_b):.3f}, max={sh_b.max():.3f}, "
          f"states>2: {int((sh_b > 2).sum())}, states>5: {int((sh_b > 5).sum())}")
    print(f"  Joint Sharpe: p50={np.median(sh_j):.3f}, max={sh_j.max():.3f}, "
          f"states>2: {int((sh_j > 2).sum())}, states>5: {int((sh_j > 5).sum())}")
    print(
        "  Reference: empirical US joint Sharpe ≈ 0.4–0.6/yr at the unconditional\n"
        "  moments. >2 is a red flag, >5 is essentially-arbitrage."
    )

    # ----- T-Q5: Sharpe vs stationary probability -----
    print("\n" + "=" * 78)
    print("T-Q5  Joint Sharpe ranked vs stationary probability of state")
    print("=" * 78)
    order = np.argsort(-sh_j)
    cum_p = np.cumsum(stat_probs[order])
    print(f"  Top-10 states by joint Sharpe:")
    _hdr = "state " + str(tuple(model.state_names))
    print(f"    {'rank':>4} {'i_s':>4}  {_hdr:>26}  {'Sharpe':>7}  "
          f"{'stat_p%':>9}  {'cum_p%':>9}")
    for r in range(min(10, n_states)):
        i_s = int(order[r])
        s_vec = np.round(pc.state_grid[i_s], 3)
        print(
            f"    {r+1:>4} {i_s:>4}  {str(tuple(s_vec)):>26}  {sh_j[i_s]:>7.3f}  "
            f"{stat_probs[i_s]*100:>8.4f}%  {cum_p[r]*100:>8.4f}%"
        )
    print(f"\n  Cumulative stationary mass of states with joint Sharpe > τ:")
    for tau in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        mask = sh_j > tau
        n = int(mask.sum())
        m = float(stat_probs[mask].sum())
        print(f"    τ={tau:>5.1f}: {n:>4} states, {m*100:>6.3f}% stationary mass")
    print(
        "  If the high-Sharpe states have negligible stationary mass (<1%), the\n"
        "  pathology lives in rarely-visited corners — cure = state-grid pruning\n"
        "  or local quadrature refinement at those corners. If it has meaningful\n"
        "  mass, the cure is global K_xb / K_state refinement."
    )

    # ----- T-Q6: Per-state Merton vs saved-policy α -----
    print("\n" + "=" * 78)
    print(
        "T-Q6  Per-state Merton α vs saved-policy α  (age 22, z=median, "
        f"x ≈ {pc.wealth_grid[iw_typical]:.3f})"
    )
    print("=" * 78)
    eps = 1e-6
    ratio_s = np.where(np.abs(a_m[:, 0]) > eps, a_solver[:, 0] / a_m[:, 0], np.nan)
    ratio_b = np.where(np.abs(a_m[:, 1]) > eps, a_solver[:, 1] / a_m[:, 1], np.nan)
    print(f"  Stock:  α_solver  p25/p50/p75 = "
          f"({np.percentile(a_solver[:, 0], 25):.2f}, "
          f"{np.percentile(a_solver[:, 0], 50):.2f}, "
          f"{np.percentile(a_solver[:, 0], 75):.2f}),  "
          f"α_merton p25/p50/p75 = ("
          f"{np.percentile(a_m[:, 0], 25):.2f}, "
          f"{np.percentile(a_m[:, 0], 50):.2f}, "
          f"{np.percentile(a_m[:, 0], 75):.2f})")
    print(f"  Bond :  α_solver  p25/p50/p75 = "
          f"({np.percentile(a_solver[:, 1], 25):.2f}, "
          f"{np.percentile(a_solver[:, 1], 50):.2f}, "
          f"{np.percentile(a_solver[:, 1], 75):.2f}),  "
          f"α_merton p25/p50/p75 = ("
          f"{np.percentile(a_m[:, 1], 25):.2f}, "
          f"{np.percentile(a_m[:, 1], 50):.2f}, "
          f"{np.percentile(a_m[:, 1], 75):.2f})")
    print(f"  ratio (stock): p50={np.nanmedian(ratio_s):.2f}, "
          f"max={np.nanmax(ratio_s):.2f}, "
          f"|ratio|>3: {int(np.sum(np.abs(ratio_s) > 3))} states")
    print(f"  ratio (bond):  p50={np.nanmedian(ratio_b):.2f}, "
          f"max={np.nanmax(ratio_b):.2f}, "
          f"|ratio|>3: {int(np.sum(np.abs(ratio_b) > 3))} states")
    print(
        "  Hedging-demand tolerance is ~50%. |ratio|>3 means the saved policy\n"
        "  disagrees with the per-period optimum on its own cloud — flags\n"
        "  multi-period EGM amplification on top of cloud geometry."
    )

    # ----- Cross-cutting report card: top-15 worst states -----
    print("\n" + "=" * 78)
    print("Cross-cut: top-15 states ranked by α_solver discrepancy (|stock| + |bond|)")
    print("=" * 78)
    abs_disagree = np.abs(np.where(np.isfinite(ratio_s), ratio_s, 0)) + np.abs(
        np.where(np.isfinite(ratio_b), ratio_b, 0)
    )
    bad = np.argsort(-abs_disagree)[:15]
    print(
        f"  {'rank':>4} {'i_s':>4}  {'state':>22}  {'stat_p%':>8}  {'arb':>9}  "
        f"{'minR(2,1)':>10}  {'sh_j':>6}  {'α_M_s':>7}  {'α_S_s':>7}  {'α_M_b':>7}  {'α_S_b':>7}"
    )
    for r, i_s in enumerate(bad):
        s_vec = np.round(pc.state_grid[int(i_s)], 3)
        print(
            f"  {r+1:>4} {int(i_s):>4}  {str(tuple(s_vec)):>22}  "
            f"{stat_probs[i_s]*100:>7.3f}%  {arb[i_s]:>9.2e}  {minR_2_1[i_s]:>10.4f}  "
            f"{sh_j[i_s]:>6.2f}  {a_m[i_s, 0]:>7.2f}  {a_solver[i_s, 0]:>7.2f}  "
            f"{a_m[i_s, 1]:>7.2f}  {a_solver[i_s, 1]:>7.2f}"
        )

    # ----- Solution implications -----
    print("\n" + "=" * 78)
    print("Solution implications (conditional on what the tests above showed)")
    print("=" * 78)
    bullets = []

    # T-Q7
    if mean_errs.max() < 1e-10 and cov_errs.max() < 1e-9:
        bullets.append(
            "T-Q7 PASS: log-moment recovery is exact ⇒ the rule itself is well-formed.\n"
            "          Pathology is not a quadrature-construction bug."
        )
    else:
        bullets.append(
            "T-Q7 FAIL: low-order moments don't match. Fix the rule construction\n"
            "          BEFORE attempting any solver fix — every other test depends on it."
        )

    # T-Q1
    if (arb > 1e-10).sum() == 0:
        bullets.append(
            "T-Q1 CLEAN: no states have strict convex-hull arbitrage. H1a remains\n"
            "          ruled out; cure does not require eliminating arbitrage by node\n"
            "          count alone."
        )
    else:
        bullets.append(
            f"T-Q1 ARB at {int((arb>1e-10).sum())} states. Refine K_xr (cholesky axis,\n"
            f"          per RETURNS.md §6.12) until n_arb_states = 0."
        )

    # T-Q2
    bad_2_1 = int((minR_2_1 < 0.05).sum())
    bad_3_2 = int((minR_3_2 < 0.05).sum())
    if bad_2_1 > 0 or bad_3_2 > 0:
        bullets.append(
            f"T-Q2 H1b CONFIRMED: at modest leverage (2,1) {bad_2_1} states / (3,2) "
            f"{bad_3_2} states\n"
            f"          have min R_port < 0.05. Cure: bump K_xb (and possibly K_state)\n"
            f"          to thicken the worst-case bond-loss tail. Bond residual std is\n"
            f"          only 2.26%, so K_xb=3 gives a coarse 3-point representation that\n"
            f"          truncates ~0.27% of analytical Gaussian mass on each side."
        )
    else:
        bullets.append(
            "T-Q2 OK: min R_port stays comfortably positive at moderate leverage.\n"
            "          The no-bankruptcy boundary doesn't bind in normal states."
        )

    # T-Q3
    if np.median(skew_s) > 0.5 or upside_s[np.isfinite(upside_s)].max() > 3:
        bullets.append(
            "T-Q3 LOPSIDED: the cloud is positively skewed in the stock dimension\n"
            "          across many states. Cure same as T-Q2 — more mass on the\n"
            "          downside via cholesky-axis K refinement."
        )

    # T-Q4 & T-Q5
    high_sharpe_states = int((sh_j > 2).sum())
    high_sharpe_mass = float(stat_probs[sh_j > 2].sum())
    if high_sharpe_states > 0:
        bullets.append(
            f"T-Q4/T-Q5: {high_sharpe_states} states have joint Sharpe > 2 with "
            f"cumulative stationary mass {high_sharpe_mass*100:.2f}%."
        )
        if high_sharpe_mass < 0.01:
            bullets.append(
                "          High-Sharpe states are CORNERS (cum mass < 1%). Cure:\n"
                "          state-grid pruning at those corners (drop them or apply\n"
                "          local-only quadrature refinement). Refining globally would\n"
                "          waste compute on states the agent rarely visits."
            )
        elif high_sharpe_mass > 0.05:
            bullets.append(
                "          High-Sharpe states have MEANINGFUL probability (>5%).\n"
                "          Cure is global refinement — likely K_xb (which directly\n"
                "          controls the bond-residual tail that drives the joint\n"
                "          Sharpe up via volatility under-statement)."
            )
        else:
            bullets.append(
                "          High-Sharpe states have INTERMEDIATE probability (1–5%).\n"
                "          Either approach can work; default to global K_xb refinement\n"
                "          because it doesn't require state-by-state surgery."
            )

    # T-Q6
    big_disagree_s = int(np.sum(np.abs(ratio_s) > 3))
    big_disagree_b = int(np.sum(np.abs(ratio_b) > 3))
    if big_disagree_s + big_disagree_b > 0:
        bullets.append(
            f"T-Q6 MULTI-PERIOD AMPLIFICATION: {big_disagree_s} states with stock\n"
            f"          α_solver/α_merton > 3, {big_disagree_b} states with bond ratio > 3.\n"
            f"          The cure for these states is upstream of the per-period cloud —\n"
            f"          either tighter EGM step control or a value-function regularizer."
        )
    else:
        bullets.append(
            "T-Q6 OK: saved policy α tracks per-period Merton on the cloud's empirical\n"
            "          moments within hedging-demand tolerance. The cure is fully at\n"
            "          the cloud level — no multi-period EGM fix needed."
        )

    for b in bullets:
        print("  • " + b.replace("\n", "\n    "))

    print()


if __name__ == "__main__":
    main()
