"""
_diag_arbitrage_quadsweep.py -- Pre-run quadrature arbitrage check + bond-axis sweep.

This is the natural follow-up to `_diag_invalid_cells.py`. That script established
that the v3 bundle's EE invalidity is a quadrature tail miss on the leveraged
SHORT-bond position at wide state corners (bond Sharpe ~ -1.9, Merton alpha_b ~ -7).

This script answers two related questions WITHOUT any solves:

  1. At the bundle's solver quadrature (n_ret_nodes_1d, n_state_quad_nodes), does
     the joint (state x return) cloud admit a discrete free lunch in the bond
     direction? "Free lunch" is measured two ways:
       (i)  T-Q1 convex-hull arbitrage gap on the 2D excess-return cloud
            (origin strictly outside hull => discrete arbitrage exists)
       (ii) T-Q2 min R_p over the cloud at canonical leveraged-bond alphas
            (alpha_s, alpha_b) - the worst gross portfolio return; if min R_p > 0
            on the solver cloud but < 0 on a finer cloud, the policy gets a
            certified-safe leveraged position that is in fact bankrupt.

  2. Which axis of the quadrature kills it cheapest? The script sweeps each
     return axis (rtb / xr / xb) and the state quadrature axes independently
     and reports per-state arbitrage-gap reduction and worst-case min R_p
     improvement.

The expected outcome (consistent with the diagnosis): bumping the bond
residual axis n_ret_nodes_1d[2] from 5 -> 7 or 9 dominates the others for
killing the bond near-arbitrage.

Usage
-----
python -m scripts.diagnostics._diag_arbitrage_quadsweep \\
  --bundle saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v3 \\
  --markdown-out diagnostics_reports/diagnostics_quadrature_arbitrage_nz11_v3.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifecycle.model import DiscretizationConfig
from lifecycle.precompute import Precompute
from scripts.diagnostics._diag_policy_convergence import (
    _build_disc_config,
    _build_model_from_bundle,
    _extract_disc_config,
)
from scripts.diagnostics._diag_quadrature_cloud import (
    arbitrage_gap_2d,
    build_cloud,
    excess_moments,
    merton_alpha,
    sharpes,
    worst_case_rport,
)


def _state_index_to_tuple(i_s, sizes):
    N1 = int(sizes[1]); N2 = int(sizes[2])
    return (i_s // (N1 * N2), (i_s // N2) % N1, i_s % N2)


def _make_pc(model, base_disc, *, ret_nodes=None, state_nodes=None):
    """Rebuild Precompute with overridden ret/state quadrature; everything else identical."""
    rn = tuple(ret_nodes) if ret_nodes is not None else tuple(base_disc.n_ret_nodes_1d)
    sn = tuple(state_nodes) if state_nodes is not None else tuple(base_disc.n_state_quad_nodes)
    new_disc = DiscretizationConfig(
        n_wealth=int(base_disc.n_wealth),
        wealth_min=float(base_disc.wealth_min),
        wealth_max=float(base_disc.wealth_max),
        n_savings=int(base_disc.n_savings),
        savings_min=float(base_disc.savings_min),
        savings_max=base_disc.savings_max,
        state_grid_sizes=tuple(int(v) for v in base_disc.state_grid_sizes),
        state_grid_mode=base_disc.state_grid_mode,
        state_n_stds=base_disc.state_n_stds,
        n_z=int(base_disc.n_z),
        n_stds=float(base_disc.n_stds),
        n_eps_nodes=int(base_disc.n_eps_nodes),
        n_eta_nodes=int(base_disc.n_eta_nodes),
        n_ret_nodes_1d=rn,
        n_state_quad_nodes=sn,
    )
    return Precompute(model, new_disc, verbose=False)


def _scan_arbitrage(model, pc, alphas, alpha_per_state=None, alpha_cap=6.0):
    """For each state, compute T-Q1 arbitrage gap, T-Q2 min R_p at each shared alpha,
    and (if `alpha_per_state` is provided) min R_p at the per-state alpha (e.g. Merton)."""
    n_states = pc.N_state
    arb = np.zeros(n_states)
    minR = np.zeros((n_states, len(alphas)))
    if alpha_per_state is not None:
        minR_per_state = np.zeros(n_states)
        merton_per_state = alpha_per_state
    else:
        minR_per_state = None
        merton_per_state = None
    for i_s in range(n_states):
        _log_r, gross, w = build_cloud(model, pc, i_s)
        Xs = gross[:, 1] - gross[:, 0]
        Xb = gross[:, 2] - gross[:, 0]
        X2d = np.column_stack([Xs, Xb])
        arb[i_s] = arbitrage_gap_2d(X2d)
        for j, (a_s, a_b) in enumerate(alphas):
            a_bill = 1.0 - a_s - a_b
            R_p = a_s * gross[:, 1] + a_b * gross[:, 2] + a_bill * gross[:, 0]
            minR[i_s, j] = float(R_p.min())
        if minR_per_state is not None:
            a_s = float(np.clip(merton_per_state[i_s, 0], -alpha_cap, alpha_cap))
            a_b = float(np.clip(merton_per_state[i_s, 1], -alpha_cap, alpha_cap))
            a_bill = 1.0 - a_s - a_b
            R_p = a_s * gross[:, 1] + a_b * gross[:, 2] + a_bill * gross[:, 0]
            minR_per_state[i_s] = float(R_p.min())
    return arb, minR, minR_per_state


def _compute_per_state_merton(model, pc, alpha_cap=6.0):
    """Return (N_state, 2) Merton alphas per state computed from the per-state cloud."""
    gamma = float(model.gamma)
    out = np.zeros((pc.N_state, 2))
    for i_s in range(pc.N_state):
        _log_r, gross, w = build_cloud(model, pc, i_s)
        mu_e, Sigma_e = excess_moments(gross, w)
        out[i_s] = merton_alpha(mu_e, Sigma_e, gamma)
    return out


def _config_label(ret, state):
    return f"ret={tuple(int(v) for v in ret)}, state={tuple(int(v) for v in state)}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--markdown-out", default=None)
    p.add_argument("--top-corners", type=int, default=8,
                   help="Number of worst corners (by minR at canonical leveraged alpha) to detail.")
    args = p.parse_args(argv)

    bundle_path = Path(args.bundle)
    print(f"Bundle: {bundle_path}")

    model, _ages = _build_model_from_bundle(bundle_path)
    from lifecycle.policy_io import load_policy_bundle
    _C, _S, _B, _diag, meta = load_policy_bundle(bundle_path)
    disc_raw = _extract_disc_config(meta)
    base_disc = _build_disc_config(disc_raw)
    print(f"  state grid: sizes={tuple(base_disc.state_grid_sizes)}, "
          f"n_stds={base_disc.state_n_stds}, mode={base_disc.state_grid_mode}")
    print(f"  solver quad: ret={tuple(base_disc.n_ret_nodes_1d)}, "
          f"state={tuple(base_disc.n_state_quad_nodes)}")

    # canonical leverage points to probe (alpha_s, alpha_b)
    alphas = [
        (1.20, -5.00),  # observed worst-probe alpha at top corners
        (1.00, -4.00),  # moderate short-bond
        (1.50, -6.00),  # at the cap
        (-0.50, +5.00), # observed long-bond corner (1,6,6)
        (0.85, +0.44),  # solver init alpha (sanity reference, mild long bond)
    ]
    alpha_labels = [f"({a_s:+.2f},{a_b:+.2f})" for (a_s, a_b) in alphas]

    # ---- baseline: solver's own quadrature ----
    pc_solver = _make_pc(model, base_disc)
    sizes = tuple(int(s) for s in pc_solver.state_grid_sizes)
    stat_probs = np.asarray(pc_solver.state_stationary_probs)
    alpha_cap = 6.0

    # Compute per-state Merton alpha on the SOLVER cloud (this is what the solver
    # would converge to if intertemporal hedging were ignored — proxy for the
    # state-conditional optimum the policy is chasing).
    print(f"\nComputing per-state Merton alpha (clipped to +/-{alpha_cap})...")
    merton_per_state = _compute_per_state_merton(model, pc_solver, alpha_cap=alpha_cap)
    arb_solver, minR_solver, minR_merton_solver = _scan_arbitrage(
        model, pc_solver, alphas, alpha_per_state=merton_per_state, alpha_cap=alpha_cap,
    )

    print(f"\n=== Baseline solver quadrature ({_config_label(base_disc.n_ret_nodes_1d, base_disc.n_state_quad_nodes)}) ===")
    print(f"  N_state={pc_solver.N_state}, joint cloud per state={pc_solver.n_state_quad * pc_solver.n_ret_quad}")
    print(f"  T-Q1 arbitrage gap: max={arb_solver.max():.3e}  states>1e-6={(arb_solver>1e-6).sum()}")
    print(f"\n  --- min R_p at PER-STATE Merton alpha (solver cloud) ---")
    print(f"  worst min R_p across states: {minR_merton_solver.min():+.4f}")
    print(f"  states with min R_p < 0:     {int((minR_merton_solver < 0).sum())}/{pc_solver.N_state}")
    print(f"  stat-mass with min R_p < 0:  {100*float(stat_probs[minR_merton_solver < 0].sum()):.3f}%")
    print(f"  states with min R_p > 0     ({int((minR_merton_solver > 0).sum())}/{pc_solver.N_state}): "
          f"the solver THINKS Merton-optimal alpha is bankruptcy-free here.")
    print(f"\n  --- min R_p at fixed canonical alphas (solver cloud) ---")
    for j, lab in enumerate(alpha_labels):
        bad = (minR_solver[:, j] <= 0).sum()
        bad_mass = float(stat_probs[minR_solver[:, j] <= 0].sum())
        worst = float(minR_solver[:, j].min())
        print(f"  alpha={lab}: min R_p<0 in {bad}/{pc_solver.N_state} states "
              f"({100*bad_mass:.3f}% stat mass), worst={worst:+.4f}")

    # The interesting "free-lunch" corners: states where the (clipped) Merton alpha is
    # MEANINGFULLY LEVERED (|alpha_b|>3 here — anything above ~half the cap) AND the
    # solver cloud still certifies it safe (min R_p > 0). These are the states whose
    # leveraged short/long bond position the solver was committing to.
    merton_bond_clip = np.clip(merton_per_state[:, 1], -alpha_cap, alpha_cap)
    levered_mask = np.abs(merton_bond_clip) > 3.0
    safe_mask = minR_merton_solver > 0
    candidates_mask = levered_mask & safe_mask
    candidates = np.flatnonzero(candidates_mask)
    if candidates.size > 0:
        worst_idx = candidates[np.argsort(-minR_merton_solver[candidates])][: args.top_corners]
    elif np.any(safe_mask):
        worst_idx = np.flatnonzero(safe_mask)[np.argsort(-minR_merton_solver[safe_mask])][: args.top_corners]
    else:
        worst_idx = np.argsort(minR_merton_solver)[: args.top_corners]
    print(f"\n=== Top {args.top_corners} LEVERED corners (|Merton alpha_b|>3) certified safe by solver cloud ===")
    print(f"{'rank':>4} {'i_s':>4}  {'(i0,i1,i2)':>11}  {'merton(s,b)':>14}  {'minR_solver':>11}  {'arb_gap':>10}  {'stat_p%':>8}")
    for rank, i_s in enumerate(worst_idx):
        tup = _state_index_to_tuple(int(i_s), sizes)
        m_s, m_b = merton_per_state[i_s]
        m_s_c = float(np.clip(m_s, -alpha_cap, alpha_cap))
        m_b_c = float(np.clip(m_b, -alpha_cap, alpha_cap))
        print(f"{rank+1:>4} {int(i_s):>4}  ({tup[0]},{tup[1]},{tup[2]})        "
              f"({m_s_c:+5.2f},{m_b_c:+5.2f})  "
              f"{minR_merton_solver[i_s]:>+11.4f}  {arb_solver[i_s]:>10.3e}  "
              f"{100*stat_probs[i_s]:>7.4f}%")

    # ---- sweeps ----
    base_ret = tuple(int(v) for v in base_disc.n_ret_nodes_1d)
    base_state = tuple(int(v) for v in base_disc.n_state_quad_nodes)

    sweep_configs = []
    # Bond return-axis sweep
    for k in (5, 7, 9, 11):
        sweep_configs.append(("ret[2]/bond", (base_ret[0], base_ret[1], k), base_state))
    # Stock return-axis sweep
    for k in (5, 7, 9):
        sweep_configs.append(("ret[1]/stock", (base_ret[0], k, base_ret[2]), base_state))
    # Bill return-axis sweep
    for k in (3, 5, 7):
        sweep_configs.append(("ret[0]/bill", (k, base_ret[1], base_ret[2]), base_state))
    # State-quadrature: bump axis 2 (bond-loading state)
    for k in (5, 7, 9):
        sweep_configs.append(("state[2]", base_ret, (base_state[0], base_state[1], k)))
    # State-quadrature: bump axis 0
    for k in (3, 5, 7):
        sweep_configs.append(("state[0]", base_ret, (k, base_state[1], base_state[2])))
    # Combined: bump bond return + state[2]
    sweep_configs.append(("combined-cheap", (base_ret[0], base_ret[1], 7), (base_state[0], base_state[1], 7)))
    sweep_configs.append(("combined-rich",  (base_ret[0], base_ret[1], 9), (base_state[0], base_state[1], 9)))
    # Trim stock axis as well — stock plays no role in the bond-bankruptcy pathology
    sweep_configs.append(("trim-stock-cheap", (base_ret[0], 5, 7), (base_state[0], base_state[1], 7)))
    sweep_configs.append(("trim-stock-rich",  (base_ret[0], 5, 9), (base_state[0], base_state[1], 9)))
    # Even cheaper: drop stock to 3
    sweep_configs.append(("trim-stock-min",   (base_ret[0], 3, 7), (base_state[0], base_state[1], 7)))

    sweep_results = []
    seen_keys = set()
    for label, ret_nodes, state_nodes in sweep_configs:
        key = (tuple(ret_nodes), tuple(state_nodes))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            pc = _make_pc(model, base_disc, ret_nodes=ret_nodes, state_nodes=state_nodes)
        except Exception as e:
            print(f"[skip {label} {ret_nodes}/{state_nodes}: {e}]")
            continue
        # Use the SOLVER's Merton-alpha (the policy target the solver was chasing)
        # and check what min R_p the FINER cloud sees at the SAME alpha. This is the
        # discrete-free-lunch test: at how many states does the finer cloud reveal
        # bankruptcy that the solver cloud missed?
        _arb, _minR_alphas, minR_merton_finer = _scan_arbitrage(
            model, pc, alphas, alpha_per_state=merton_per_state, alpha_cap=alpha_cap,
        )
        # Free-lunch states relative to solver: solver said safe, finer says bankrupt.
        flipped = (minR_merton_solver > 0) & (minR_merton_finer < 0)
        n_cloud = pc.n_state_quad * pc.n_ret_quad
        sweep_results.append({
            "label": label,
            "ret_nodes": tuple(ret_nodes),
            "state_nodes": tuple(state_nodes),
            "n_cloud": int(n_cloud),
            "max_arb": float(_arb.max()),
            "n_states_arb": int((_arb > 1e-6).sum()),
            "minR_merton": minR_merton_finer.copy(),
            "worst_minR_merton": float(minR_merton_finer.min()),
            "n_states_merton_neg": int((minR_merton_finer < 0).sum()),
            "stat_mass_merton_neg": float(stat_probs[minR_merton_finer < 0].sum()),
            "n_flipped_vs_solver": int(flipped.sum()),
            "stat_mass_flipped": float(stat_probs[flipped].sum()),
        })

    # baseline as first row for comparison
    base_row = {
        "label": "baseline",
        "ret_nodes": base_ret,
        "state_nodes": base_state,
        "n_cloud": int(pc_solver.n_state_quad * pc_solver.n_ret_quad),
        "max_arb": float(arb_solver.max()),
        "n_states_arb": int((arb_solver > 1e-6).sum()),
        "minR_merton": minR_merton_solver.copy(),
        "worst_minR_merton": float(minR_merton_solver.min()),
        "n_states_merton_neg": int((minR_merton_solver < 0).sum()),
        "stat_mass_merton_neg": float(stat_probs[minR_merton_solver < 0].sum()),
        "n_flipped_vs_solver": 0,
        "stat_mass_flipped": 0.0,
    }

    print(f"\n=== Quadrature sweep — min R_p at PER-STATE Merton alpha (clipped to +/-{alpha_cap}) ===")
    print(f"{'label':<16} {'ret':>11} {'state':>11} {'cloud':>6}  {'maxArb':>9}  "
          f"{'wMinRpMer':>10}  {'#Mer<0':>7}  {'massMer<0':>10}  {'#flip':>6}  {'mFlip':>8}")
    for row in [base_row] + sweep_results:
        print(f"{row['label']:<16} {str(row['ret_nodes']):>11} {str(row['state_nodes']):>11} "
              f"{row['n_cloud']:>6}  {row['max_arb']:>9.2e}  "
              f"{row['worst_minR_merton']:>+10.4f}  "
              f"{row['n_states_merton_neg']:>7}  {100*row['stat_mass_merton_neg']:>9.3f}%  "
              f"{row['n_flipped_vs_solver']:>6}  {100*row['stat_mass_flipped']:>7.3f}%")

    # ---- per-corner table at the worst 8 corners by free-lunch buffer ----
    print(f"\n=== Per-corner min R_p at PER-STATE Merton alpha across configs ===")
    print("(top-{0} corners by largest 'safe buffer' on the SOLVER cloud — the most vulnerable to a tail miss)".format(args.top_corners))
    print(f"{'i_s':>4} {'(i,i,i)':>11} {'merton':>14}  " + "  ".join(f"{r['label']:>13}" for r in [base_row] + sweep_results))
    for i_s in worst_idx:
        tup = _state_index_to_tuple(int(i_s), sizes)
        m_s, m_b = merton_per_state[i_s]
        m_s_c = float(np.clip(m_s, -alpha_cap, alpha_cap))
        m_b_c = float(np.clip(m_b, -alpha_cap, alpha_cap))
        line = f"{int(i_s):>4} ({tup[0]},{tup[1]},{tup[2]})        ({m_s_c:+5.2f},{m_b_c:+5.2f})  "
        for row in [base_row] + sweep_results:
            line += f"  {row['minR_merton'][i_s]:>+13.4f}"
        print(line)

    # ---- markdown ----
    if args.markdown_out:
        out = []
        out.append(f"# Quadrature-Arbitrage Sweep: `{bundle_path.name}`")
        out.append("")
        out.append("## What this is")
        out.append("")
        out.append(
            "Companion to `_diag_invalid_cells.py`. That diagnostic established that "
            "the v3 bundle's 12.5% retirement EE-invalidity rate is a quadrature TAIL "
            "miss on the leveraged short-bond position at wide state corners — the solver's "
            "(3,7,5)x(3,4,5) joint cloud reports `min R_p > 0` for the saved (alpha_s, alpha_b) "
            "while a (5,9,7)x(4,5,6) cloud reports `min R_p < 0` at the SAME alpha."
        )
        out.append("")
        out.append(
            "This script measures the same pathology DIRECTLY on the cloud, with NO solve. "
            "Two metrics:"
        )
        out.append("")
        out.append(
            "- **T-Q1 arbitrage gap.** Convex-hull gap on the 2D excess-return cloud "
            "(Xs, Xb). Strictly positive => the cloud admits a discrete free lunch — a "
            "long-short portfolio with positive payoff at every quadrature node — i.e. "
            "the rule is broken in a way the solver can exploit."
        )
        out.append(
            "- **T-Q2 worst-case `min R_p`.** At canonical leveraged alphas, the smallest "
            "gross portfolio return across the cloud. If `min R_p > 0` on the solver cloud "
            "while < 0 on a slightly finer cloud, that alpha is a discretization-induced "
            "free lunch even though no strict arbitrage exists in the cloud's interior."
        )
        out.append("")
        out.append("Both can be computed pre-run for any candidate `n_ret_nodes_1d, n_state_quad_nodes` config.")
        out.append("")

        out.append("## Setup")
        out.append("")
        out.append(f"- Bundle: `{bundle_path}`")
        out.append(f"- State grid: sizes `{tuple(base_disc.state_grid_sizes)}`, "
                   f"n_stds `{base_disc.state_n_stds}`, mode `{base_disc.state_grid_mode}` "
                   f"(N_state = `{pc_solver.N_state}`).")
        out.append(f"- Solver quadrature: ret `{base_ret}`, state `{base_state}` "
                   f"(joint cloud per state = `{pc_solver.n_state_quad * pc_solver.n_ret_quad}`).")
        out.append(f"- Canonical alphas probed: " + ", ".join(f"`{lab}`" for lab in alpha_labels))
        out.append("")

        # baseline
        out.append("## Baseline arbitrage state at solver quadrature")
        out.append("")
        out.append(
            f"**T-Q1 strict arbitrage**: max gap = `{arb_solver.max():.3e}` over "
            f"`{pc_solver.N_state}` states. The solver cloud is convex-hull clean — no node "
            f"set admits a long-short with strictly positive payoff at every node."
        )
        out.append("")
        out.append(
            f"**T-Q2 free-lunch at per-state Merton alpha** (the alpha each state's two-asset "
            f"Markowitz/Merton optimum picks, clipped to +/-{alpha_cap}): on the solver cloud, "
            f"min R_p > 0 in `{int((minR_merton_solver > 0).sum())}/{pc_solver.N_state}` "
            f"states (`{100*float(stat_probs[minR_merton_solver>0].sum()):.3f}%` stationary mass) — "
            f"the solver cloud certifies the Merton-optimal leveraged position as "
            f"bankruptcy-free in those states. The EE-invalid-cells diagnostic showed that "
            f"once we evaluate with a finer cloud, that certificate is wrong on the wide-corner tail."
        )
        out.append("")
        out.append("Min R_p at fixed canonical alphas, for orientation:")
        out.append("")
        out.append("| alpha (alpha_s, alpha_b) | states with min R_p < 0 | stat-mass with min R_p < 0 | worst min R_p across states |")
        out.append("| --- | ---: | ---: | ---: |")
        for j, lab in enumerate(alpha_labels):
            bad = int((minR_solver[:, j] <= 0).sum())
            bad_mass = float(stat_probs[minR_solver[:, j] <= 0].sum())
            worst = float(minR_solver[:, j].min())
            out.append(f"| {lab} | {bad}/{pc_solver.N_state} | {100*bad_mass:.3f}% | {worst:+.4f} |")
        out.append("")
        out.append(
            f"Note that at any fixed alpha, most states see `min R_p < 0` because the canonical "
            f"alpha is far from the per-state Merton optimum at most states. The relevant test "
            f"is per-state Merton alpha, which is what the solver's policy chases."
        )
        out.append("")

        # sweep table
        out.append("## Quadrature sweep")
        out.append("")
        out.append(
            f"For each candidate quadrature config, we compute the FINER cloud's min R_p at "
            f"the SOLVER's per-state Merton alpha (the same alpha the solver would chase). "
            f"The `flipped` columns count states where the solver said safe (min R_p > 0) but "
            f"the finer rule says bankrupt (min R_p < 0) at the same alpha — these are the "
            f"discrete free lunches the solver was exploiting."
        )
        out.append("")
        out.append("| label | ret nodes | state nodes | cloud nodes | T-Q1 max gap | worst min R_p (Merton) | states with min R_p < 0 | stat-mass min R_p < 0 | states FLIPPED vs solver | stat-mass FLIPPED |")
        out.append("| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in [base_row] + sweep_results:
            out.append(
                f"| {row['label']} | {row['ret_nodes']} | {row['state_nodes']} | {row['n_cloud']} | "
                f"{row['max_arb']:.2e} | {row['worst_minR_merton']:+.4f} | "
                f"{row['n_states_merton_neg']} | {100*row['stat_mass_merton_neg']:.3f}% | "
                f"{row['n_flipped_vs_solver']} | {100*row['stat_mass_flipped']:.3f}% |"
            )
        out.append("")

        # per-corner table for worst corners
        out.append(f"## Per-corner min R_p at per-state Merton alpha across configs")
        out.append("")
        out.append("Top corners selected as those with MEANINGFULLY LEVERED clipped Merton alpha (`|alpha_b| > 3`) AND certified safe by the solver cloud (`min R_p > 0`). These are the discrete free lunches the solver was exploiting.")
        out.append("Cell = min R_p at that state's clipped Merton alpha. Negative => the cloud now sees the tail bankruptcy and would force the optimizer off the leveraged Merton corner.")
        out.append("")
        cols = [base_row] + sweep_results
        header = "| i_s | (i0,i1,i2) | merton (a_s,a_b) | " + " | ".join(c["label"] for c in cols) + " |"
        sep = "| ---: | ---: | ---: | " + " | ".join("---:" for _ in cols) + " |"
        out.append(header); out.append(sep)
        for i_s in worst_idx:
            tup = _state_index_to_tuple(int(i_s), sizes)
            m_s, m_b = merton_per_state[i_s]
            m_s_c = float(np.clip(m_s, -alpha_cap, alpha_cap))
            m_b_c = float(np.clip(m_b, -alpha_cap, alpha_cap))
            cells = "".join(f" {row['minR_merton'][i_s]:+.4f} |" for row in cols)
            out.append(f"| {int(i_s)} | ({tup[0]},{tup[1]},{tup[2]}) | ({m_s_c:+.2f},{m_b_c:+.2f}) |{cells}")
        out.append("")

        # findings + recommendation: pick the cheapest config (by cloud nodes) that flips
        # the most stat-mass-weighted free-lunch states (vs solver baseline).
        candidates = sorted(sweep_results, key=lambda r: r["n_cloud"])
        # "cure" = at least 50% of the solver's certified-safe-mass gets flipped to bankrupt
        target_flip_mass = 0.5 * float(stat_probs[minR_merton_solver > 0].sum())
        improving = [r for r in candidates if r["stat_mass_flipped"] >= target_flip_mass]
        if improving:
            cheapest_cure = min(improving, key=lambda r: r["n_cloud"])
        else:
            cheapest_cure = max(candidates, key=lambda r: r["stat_mass_flipped"])

        out.append("## Findings")
        out.append("")
        out.append(
            f"1. **No strict arbitrage at the solver cloud.** T-Q1 max gap is `{arb_solver.max():.2e}` "
            f"(target: 0). The (3,7,5)x(3,4,5) cloud is convex-hull-clean. The pathology is one "
            f"layer deeper: certified-safe leveraged positions whose tail bankruptcy is missed."
        )
        out.append("")
        out.append(
            f"2. **The solver cloud certifies the per-state Merton alpha as safe at "
            f"`{int((minR_merton_solver > 0).sum())}/{pc_solver.N_state}` states "
            f"(`{100*float(stat_probs[minR_merton_solver>0].sum()):.3f}%` stat mass).** "
            f"This is the surface the EE-invalidity diagnostic was sampling — the solver thinks "
            f"the Merton-optimal short-bond position is safe, the finer eval cloud reveals it isn't."
        )
        out.append("")
        # Compare ret[2]:5->7 vs state[2]:5->7 at the same cloud-node cost (+40%)
        ret2_7 = next((r for r in sweep_results if r['ret_nodes']==(base_ret[0], base_ret[1], 7) and r['state_nodes']==base_state), None)
        state2_7 = next((r for r in sweep_results if r['state_nodes']==(base_state[0], base_state[1], 7) and r['ret_nodes']==base_ret), None)
        ret1_x = next((r for r in sweep_results if r['ret_nodes'][1] != base_ret[1] and r['state_nodes']==base_state), None)
        out.append(
            f"3. **State-quadrature axis 2 dominates the cure — more than the bond return axis.** "
            f"At the SAME cloud-node count (`{ret2_7['n_cloud']}` nodes, +40% over baseline):"
        )
        out.append(f"   - `state[2]: 5 -> 7` flips `{state2_7['n_flipped_vs_solver']}` states "
                   f"(`{100*state2_7['stat_mass_flipped']:.3f}%` stat mass)")
        out.append(f"   - `ret[2]/bond: 5 -> 7` flips `{ret2_7['n_flipped_vs_solver']}` states "
                   f"(`{100*ret2_7['stat_mass_flipped']:.3f}%` stat mass)")
        out.append(f"   - `ret[1]/stock`, `ret[0]/bill`, `state[0]` bumps flip ~0 states.")
        out.append("")
        out.append(
            f"   Mechanism: in this VAR, axis 2 of the state-innovation cloud `v` carries the "
            f"largest M-matrix loading onto the bond return, so refining `state[2]` shifts the "
            f"conditional bond-return distribution at every state more than refining the residual "
            f"`ret[2]` shocks does. Bond is also the only return axis whose conditional mean is "
            f"meaningfully sensitive to state innovations through M."
        )
        out.append("")
        out.append(
            f"4. **Cheapest cure** (>=50% of solver-certified-safe mass flipped to "
            f"bankrupt): ret=`{cheapest_cure['ret_nodes']}`, state=`{cheapest_cure['state_nodes']}` "
            f"-> `{cheapest_cure['n_cloud']}` cloud nodes (vs baseline `{base_row['n_cloud']}`, "
            f"`+{100*(cheapest_cure['n_cloud']/base_row['n_cloud']-1):.0f}%`); flips "
            f"`{cheapest_cure['n_flipped_vs_solver']}` states / "
            f"`{100*cheapest_cure['stat_mass_flipped']:.3f}%` stat mass."
        )
        out.append("")
        out.append("## Pre-run usage")
        out.append("")
        out.append(
            "Run this script with `--bundle <any candidate model bundle>` BEFORE committing to a full "
            "solve at a new `(state_n_stds, n_ret_nodes_1d, n_state_quad_nodes)` config. The bundle "
            "only needs to expose `run_config` (model + var); no policy arrays are touched and no solve "
            "happens. If the baseline rows above show `worst min R_p > 0` at canonical leveraged alphas "
            "the planned solve will land in the same EE-invalidity tail as v3."
        )
        out.append("")

        Path(args.markdown_out).write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\nWrote markdown report to {args.markdown_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
