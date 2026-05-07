"""verify_arbitrage.py -- Discrete-cloud arbitrage check (pre-solve diagnostic).

Spurious arbitrage is a discretization artifact: at some state corner i_s, the
discrete (state x return) quadrature cloud admits a feasible portfolio that
DOMINATES another asset on every quadrature node — i.e. the worst node of one
asset still beats the best node of another, OR more generally a separating
direction exists between the origin and the convex hull of excess returns.
The continuous lognormal distribution does NOT admit such a dominance, so any
"arbitrage" the integrator finds is fake; if the solver exploits it (Lobatto
tails or finer GH would erase it), policies tilt toward unhedged-bankruptcy
positions that look risk-free under the cloud but blow up out-of-sample.

This is a pre-solve check. It depends ONLY on (model + discretization), not
on solved policies. Run it before paying for a solve; if it fails, re-config
``ret_lobatto_Z`` / ``state_lobatto_Z`` and re-test.

What's tested
-------------
For each state corner i_s in the joint state grid, build the gross-return
cloud (R_bill, R_stock, R_bond) at every (k_v, k_r) quadrature node and
evaluate two gaps:

    1. Axis-aligned dominance gap (the user-facing definition):
           gap_dom = max over ordered (i, j) pairs of  min_n R_i  -  max_n R_j
       gap_dom > 0 ⇒ asset i's worst quadrature node still strictly beats
       asset j's best — a discrete dominance arbitrage.

    2. Convex-hull arbitrage gap (the more general check):
           gap_hull = max over unit directions d in R^2 of
                      min_n d · (X_s, X_b)_n
       where X_s = R_stock - R_bill, X_b = R_bond - R_bill. gap_hull > 0 ⇒
       the origin is strictly outside the convex hull of the excess-return
       cloud ⇒ ∃ (alpha_s, alpha_b) such that R_p(alpha) > R_bill on every
       node, even if no single axis-aligned pair dominates.

Both gaps are zero (within float roundoff) on a well-resolved cloud and
strictly positive when the discretization mis-resolves the lognormal tails.

Usage
-----
    # Bundle path (uses bundle metadata.run_config to rebuild precompute):
    python verify_arbitrage.py <bundle-name-or-path>

    # Config module path (.py): import as module, read base_config +
    # disc_config_template; this is the pre-launch entry point that does
    # NOT need a saved bundle:
    python verify_arbitrage.py configs/<some>.py

Output
------
    <bundle>/arbitrage.json   when invoked on a bundle directory
    ./arbitrage.json           when invoked on a config (cwd output)

Pass criteria
-------------
- PASS:        max(gap_dom, gap_hull) < 1e-6      (within float roundoff)
- CONCERNING:  max in [1e-6, 1e-4]                (consider Lobatto)
- FAIL:        max > 1e-4                          (Lobatto required)

The thresholds match the post-rtb-as-state Sigma_r_cond rank-drift gate
(``1e-5``) order of magnitude — anything above ~1e-6 means the cloud is
materially mis-resolving the tails relative to where solver convergence
checks operate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

import numpy as np

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import build_model, build_precompute
from lifecycle.var import build_nominal_system1_var_config_hardcoded


# =============================================================================
# Input resolution: bundle directory OR config .py
# =============================================================================

def _resolve_input(arg: str) -> tuple[str, Path]:
    """Return ('bundle', dir_path) or ('config', file_path)."""
    p = Path(arg)
    if p.is_file() and p.suffix == ".py":
        return "config", p
    if p.is_dir():
        return "bundle", p
    p2 = Path("saved_runs") / arg
    if p2.is_dir():
        return "bundle", p2
    raise FileNotFoundError(
        f"Could not resolve {arg!r} as a bundle directory or a .py config. "
        f"Tried: {p}, {p2}."
    )


def _list_to_tuple_recursive(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_list_to_tuple_recursive(x) for x in v)
    return v


def _rehydrate_disc_config(d: dict) -> DiscretizationConfig:
    tuple_fields = {
        "state_grid_sizes",
        "n_state_quad_nodes",
        "n_ret_nodes_1d",
        "state_n_stds",
        "ret_lobatto_Z",
        "state_lobatto_Z",
    }
    valid = set(DiscretizationConfig._fields)
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k not in valid:
            continue
        if k in tuple_fields:
            kwargs[k] = _list_to_tuple_recursive(v)
        else:
            kwargs[k] = v
    return DiscretizationConfig(**kwargs)


def _build_pc_from_bundle(bundle_path: Path, verbose: bool):
    C, _S, _B, _diag, metadata = load_policy_bundle(bundle_path)
    run_config = metadata.get("run_config")
    if run_config is None:
        raise ValueError(
            "Bundle metadata has no 'run_config'; cannot rebuild the precompute."
        )
    base_config = run_config.get("base_config")
    disc_config_dict = run_config.get("discretization_config")
    if base_config is None or disc_config_dict is None:
        raise ValueError(
            "Bundle run_config missing base_config or discretization_config."
        )
    disc_config = _rehydrate_disc_config(disc_config_dict)
    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(base_config, var_config, verbose=verbose)
    pc = build_precompute(model, disc_config, verbose=verbose)
    return model, pc, disc_config, metadata


_CONFIG_NAMES = {
    "base_config", "BASE_CONFIG",
    "disc_config", "disc_config_template", "DISC_CONFIG",
}


def _is_config_assign(stmt) -> bool:
    """Whitelist: keep only assignments whose target is one of the known
    config names. Skips ``model = build_model(...)``, ``pc = build_precompute(...)``,
    and especially ``C, S, B, diag = run_lifecycle_solver(...)`` — none of
    which are config definitions."""
    import ast
    if isinstance(stmt, ast.Assign):
        names = []
        for tgt in stmt.targets:
            if isinstance(tgt, ast.Name):
                names.append(tgt.id)
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                for elt in tgt.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
        return any(n in _CONFIG_NAMES for n in names)
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id in _CONFIG_NAMES
    return False


def _safe_load_config_module(config_path: Path) -> dict:
    """Parse a .py file with AST and execute ONLY:
      - top-level imports, AND
      - top-level assignments whose target is in ``_CONFIG_NAMES``.

    Function/class definitions, expression statements, control flow blocks,
    and any other assignment (e.g. ``model =``, ``pc =``,
    ``C, S, B, diag = run_lifecycle_solver(...)``) are skipped. This lets us
    extract ``disc_config`` from a runner script
    (e.g. ``verify_benchmark_bundle_6666.py``) without kicking off the
    solve, save, or S3 upload that the script does at module top level.

    Returns the resulting namespace dict.
    """
    import ast
    source = config_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(config_path))

    safe_body = [
        stmt for stmt in tree.body
        if isinstance(stmt, (ast.Import, ast.ImportFrom)) or _is_config_assign(stmt)
    ]

    safe_module = ast.Module(body=safe_body, type_ignores=[])
    code = compile(safe_module, filename=str(config_path), mode="exec")
    namespace: dict[str, Any] = {"__name__": "__cfg__", "__file__": str(config_path)}
    exec(code, namespace)
    return namespace


def _build_pc_from_config(config_path: Path, verbose: bool):
    namespace = _safe_load_config_module(config_path)

    base_config = (
        namespace.get("base_config")
        or namespace.get("BASE_CONFIG")
    )
    disc_config = (
        namespace.get("disc_config")
        or namespace.get("disc_config_template")
        or namespace.get("DISC_CONFIG")
    )
    if base_config is None or disc_config is None:
        raise ValueError(
            f"Config {config_path} must define one of "
            f"(base_config / BASE_CONFIG) AND one of "
            f"(disc_config / disc_config_template / DISC_CONFIG)."
        )

    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(base_config, var_config, verbose=verbose)
    pc = build_precompute(model, disc_config, verbose=verbose)
    return model, pc, disc_config


# =============================================================================
# Per-state-corner gross-return cloud
# =============================================================================

def _build_gross_cloud_per_state(model, pc) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the gross-return cloud (R_bill, R_stock, R_bond, weights) at every
    state corner i_s.

    Returns
    -------
    R_bill  : (N_state, n_state_quad, n_ret_quad)
    R_stock : (N_state, n_state_quad, n_ret_quad)
    R_bond  : (N_state, n_state_quad, n_ret_quad)
    weights : (n_state_quad, n_ret_quad)  joint quadrature weights, sum to 1

    Identical math to the solver's ``_build_step_log_returns`` (rtb-as-state):
      - R_bill[k_v, k_r] = exp(s_next[k_v, rtb_idx])  (constant across k_r)
      - R_stock = R_bill * exp(log_x_s)
      - R_bond  = R_bill * exp(log_x_b)
    where log_x_s, log_x_b are the (xr, xb) log-excess returns at each node.
    NumPy-only — no JAX needed; the cloud is small (N_state * n_state_quad *
    n_ret_quad ~ a few million float64 entries even for 9^4 grids).
    """
    state_grid = np.asarray(pc.state_grid, dtype=np.float64)         # (N_state, n_state)
    const_r = np.asarray(pc.const_r, dtype=np.float64)               # (n_ret,)
    A_r = np.asarray(pc.A_r, dtype=np.float64)                        # (n_ret, n_state)
    M_v_nodes = np.asarray(pc.M_v_nodes, dtype=np.float64)           # (n_state_quad, n_ret)
    ret_nodes = np.asarray(pc.ret_nodes, dtype=np.float64)           # (n_ret_quad, n_ret)
    Phi_0 = np.asarray(model.Phi_0_state, dtype=np.float64)          # (n_state,)
    Phi_11 = np.asarray(model.Phi_11, dtype=np.float64)              # (n_state, n_state)
    v_nodes = np.asarray(pc.v_nodes, dtype=np.float64)               # (n_state_quad, n_state)

    rtb_idx = int(model.rtb_index_in_state)
    ret_names = list(model.ret_names)
    xr_pos = ret_names.index("xr")
    xb_pos = ret_names.index("xb")

    # Mean log-excess at each (i_s, k_v): same as _build_step_log_returns vectorised over i_s
    base_mu_r = const_r[None, :] + state_grid @ A_r.T                # (N_state, n_ret)
    mu_r_per = base_mu_r[:, None, :] + M_v_nodes[None, :, :]         # (N_state, n_state_quad, n_ret)

    # s_next[i_s, k_v, :] = Phi_0 + Phi_11 @ s_t + v[k_v]
    s_next = (
        Phi_0[None, None, :]
        + state_grid[:, None, :] @ Phi_11.T[None, :, :]
        + v_nodes[None, :, :]
    )                                                                # (N_state, n_state_quad, n_state)
    log_R_bill_kv = s_next[:, :, rtb_idx]                            # (N_state, n_state_quad)

    # Broadcast across k_r — bill is realised at the state node, identical for all return nodes.
    R_bill = np.exp(log_R_bill_kv)[:, :, None]                       # (N_state, n_state_quad, 1)

    # log_x_s[i_s, k_v, k_r] = mu_r_per[i_s, k_v, xr_pos] + ret_nodes[k_r, xr_pos]
    res_xs = ret_nodes[:, xr_pos]                                    # (n_ret_quad,)
    res_xb = ret_nodes[:, xb_pos]
    log_x_s = mu_r_per[:, :, xr_pos:xr_pos + 1] + res_xs[None, None, :]
    log_x_b = mu_r_per[:, :, xb_pos:xb_pos + 1] + res_xb[None, None, :]

    R_stock = R_bill * np.exp(log_x_s)
    R_bond = R_bill * np.exp(log_x_b)

    # Broadcast bill to full shape so callers don't need to think about it.
    R_bill_full = np.broadcast_to(
        R_bill, (state_grid.shape[0], M_v_nodes.shape[0], ret_nodes.shape[0])
    ).copy()

    v_weights = np.asarray(pc.v_weights, dtype=np.float64)
    ret_weights = np.asarray(pc.ret_weights, dtype=np.float64)
    weights = np.outer(v_weights, ret_weights)                       # (n_state_quad, n_ret_quad)

    return R_bill_full, R_stock, R_bond, weights


# =============================================================================
# Per-state-corner gaps
# =============================================================================

ASSET_NAMES = ("bill", "stock", "bond")


def _axis_aligned_dominance_gaps(
    R_bill: np.ndarray, R_stock: np.ndarray, R_bond: np.ndarray,
) -> dict:
    """Per-state-corner axis-aligned dominance: for each ordered pair (i, j)
    of distinct assets, gap_ij[i_s] = min_node R_i[i_s] - max_node R_j[i_s].

    gap_ij > 0 ⇒ asset i strictly dominates asset j on every quadrature node
    at state corner i_s — a spurious dominance arbitrage in the discrete
    cloud (the user's definition: "the worst return of asset i is still
    better than the best return of asset j").
    """
    flat = lambda R: R.reshape(R.shape[0], -1)                       # (N_state, n_nodes)
    R_flat = {"bill": flat(R_bill), "stock": flat(R_stock), "bond": flat(R_bond)}
    R_min = {a: R_flat[a].min(axis=1) for a in ASSET_NAMES}
    R_max = {a: R_flat[a].max(axis=1) for a in ASSET_NAMES}

    pairs: dict[str, np.ndarray] = {}
    for i in ASSET_NAMES:
        for j in ASSET_NAMES:
            if i == j:
                continue
            pairs[f"{i}_dominates_{j}"] = R_min[i] - R_max[j]
    return pairs


def _convex_hull_arbitrage_gap(
    R_bill: np.ndarray, R_stock: np.ndarray, R_bond: np.ndarray,
    n_angles: int = 720,
) -> np.ndarray:
    """Per-state-corner 2D convex-hull arbitrage gap on the excess-return cloud.

        gap_hull[i_s] = max_{||d||=1, d in R^2}  min_n d . (X_s, X_b)_n

    where X_s = R_stock - R_bill and X_b = R_bond - R_bill. gap_hull > 0 ⇒
    a separating direction d exists s.t. d . X > 0 at every node ⇒ the
    portfolio with weights ∝ d achieves strictly positive excess returns on
    every quadrature node. This is the most general discrete-arbitrage test
    (axis-aligned dominance is the special case d = e_s or e_b).
    """
    Xs = (R_stock - R_bill).reshape(R_stock.shape[0], -1)            # (N_state, n_nodes)
    Xb = (R_bond - R_bill).reshape(R_bond.shape[0], -1)

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    cos = np.cos(angles)                                              # (n_angles,)
    sin = np.sin(angles)
    # proj[i_s, n, a] = cos[a] * Xs[i_s, n] + sin[a] * Xb[i_s, n]
    # min over nodes per (i_s, angle), then max over angles per i_s.
    # Memory for the full (N_state, n_nodes, n_angles) tensor can be large at
    # 9^4 x 36 x 25 x 720 ~ 8 GB in float64; chunk over angles to bound peak.
    N_state = Xs.shape[0]
    gap = np.full(N_state, -np.inf, dtype=np.float64)
    chunk = 64
    for a0 in range(0, n_angles, chunk):
        a1 = min(a0 + chunk, n_angles)
        c = cos[a0:a1][None, None, :]                                 # (1, 1, n_a)
        s = sin[a0:a1][None, None, :]
        # (N_state, n_nodes, n_a)
        proj = Xs[:, :, None] * c + Xb[:, :, None] * s
        # min over nodes, then max over angles in this chunk
        chunk_min = proj.min(axis=1)                                  # (N_state, n_a)
        chunk_max = chunk_min.max(axis=1)                             # (N_state,)
        gap = np.maximum(gap, chunk_max)
    # Clip at zero per main's definition: "gap" is positive iff origin
    # strictly outside hull; a small negative value just means origin is
    # comfortably inside (no arbitrage).
    return np.maximum(gap, 0.0)


# =============================================================================
# Aggregation + reporting
# =============================================================================

def _summarise_dominance(pair_gaps: dict[str, np.ndarray]) -> dict:
    out: dict[str, Any] = {}
    worst_overall = -np.inf
    worst_overall_pair = None
    worst_overall_idx = None
    for pair, g in pair_gaps.items():
        max_idx = int(np.argmax(g))
        max_val = float(g[max_idx])
        n_pos = int(np.sum(g > 0.0))
        out[pair] = {
            "max_gap": max_val,
            "argmax_state_idx": max_idx,
            "n_state_corners_dominant": n_pos,
            "n_state_corners_total": int(g.size),
        }
        if max_val > worst_overall:
            worst_overall = max_val
            worst_overall_pair = pair
            worst_overall_idx = max_idx
    out["_worst"] = {
        "pair": worst_overall_pair,
        "max_gap": float(worst_overall) if np.isfinite(worst_overall) else None,
        "argmax_state_idx": worst_overall_idx,
    }
    return out


def _summarise_hull(gap_hull: np.ndarray) -> dict:
    max_idx = int(np.argmax(gap_hull))
    return {
        "max_gap": float(gap_hull[max_idx]),
        "argmax_state_idx": max_idx,
        "n_state_corners_with_arb": int(np.sum(gap_hull > 0.0)),
        "n_state_corners_total": int(gap_hull.size),
        "median_gap": float(np.median(gap_hull)),
        "p99_gap": float(np.percentile(gap_hull, 99.0)),
    }


def _state_index_to_tuple(i_s: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Decode a flat row-major index into per-axis indices for the state grid."""
    out = []
    rem = int(i_s)
    for n in reversed(sizes):
        out.append(rem % int(n))
        rem //= int(n)
    return tuple(reversed(out))


def _format_worst_corner(idx: int, pc) -> str:
    sizes = tuple(int(s) for s in pc.state_grid_sizes)
    tup = _state_index_to_tuple(idx, sizes)
    coords = np.asarray(pc.state_grid[idx], dtype=np.float64)
    coord_str = ", ".join(f"{c:+.4f}" for c in coords)
    return f"i_s={idx}  axes={tup}  state=({coord_str})"


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "target",
        help="Bundle directory, bare bundle name (looked up under saved_runs/), "
             "or a .py config module that defines base_config + disc_config_template.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing arbitrage.json.",
    )
    parser.add_argument(
        "--n-angles",
        type=int,
        default=720,
        help="Direction count for the convex-hull sweep (default 720).",
    )
    parser.add_argument(
        "--top-corners",
        type=int,
        default=8,
        help="Detail this many worst state corners in the JSON output.",
    )
    args = parser.parse_args()

    kind, in_path = _resolve_input(args.target)
    print(f"Input ({kind}): {in_path}", flush=True)

    print("Building model + precompute...", flush=True)
    t0 = time.time()
    metadata: dict[str, Any] | None = None
    if kind == "bundle":
        model, pc, disc_config, metadata = _build_pc_from_bundle(in_path, verbose=False)
    else:
        model, pc, disc_config = _build_pc_from_config(in_path, verbose=False)
    print(
        f"  Setup wall: {time.time() - t0:.1f}s; "
        f"N_state={pc.N_state}, n_state_quad={pc.n_state_quad}, "
        f"n_ret_quad={pc.n_ret_quad}",
        flush=True,
    )

    print("Building per-state-corner gross-return cloud...", flush=True)
    t1 = time.time()
    R_bill, R_stock, R_bond, weights = _build_gross_cloud_per_state(model, pc)
    print(
        f"  Cloud shapes: bill/stock/bond = {R_bill.shape}; "
        f"weights = {weights.shape}; sum(weights) = {weights.sum():.6f} "
        f"(should be ~1.0); wall: {time.time() - t1:.2f}s",
        flush=True,
    )

    print("\nComputing axis-aligned dominance gaps...", flush=True)
    t2 = time.time()
    pair_gaps = _axis_aligned_dominance_gaps(R_bill, R_stock, R_bond)
    dom_summary = _summarise_dominance(pair_gaps)
    print(f"  Dominance wall: {time.time() - t2:.2f}s", flush=True)

    print(f"\nComputing convex-hull arbitrage gap (n_angles={args.n_angles})...", flush=True)
    t3 = time.time()
    gap_hull = _convex_hull_arbitrage_gap(R_bill, R_stock, R_bond, n_angles=args.n_angles)
    hull_summary = _summarise_hull(gap_hull)
    print(f"  Hull wall: {time.time() - t3:.2f}s", flush=True)

    # Per-pair report.
    print("\n" + "-" * 70, flush=True)
    print("Axis-aligned dominance (one asset min > another asset max)", flush=True)
    print("-" * 70, flush=True)
    for pair in pair_gaps:
        s = dom_summary[pair]
        flag = "  ARB" if s["max_gap"] > 0.0 else ""
        print(
            f"  {pair:>22}: "
            f"max={s['max_gap']:+.3e}  "
            f"corners_dominant={s['n_state_corners_dominant']}/{s['n_state_corners_total']}  "
            f"argmax_i_s={s['argmax_state_idx']}{flag}",
            flush=True,
        )
    worst_dom = dom_summary["_worst"]
    print(f"\n  Worst dominance: {worst_dom['pair']}  gap={worst_dom['max_gap']:+.3e}  "
          f"at i_s={worst_dom['argmax_state_idx']}", flush=True)

    print("\n" + "-" * 70, flush=True)
    print(f"Convex-hull arbitrage (separating direction in (X_s, X_b))", flush=True)
    print("-" * 70, flush=True)
    print(
        f"  max_gap={hull_summary['max_gap']:+.3e}  "
        f"p99={hull_summary['p99_gap']:+.3e}  "
        f"median={hull_summary['median_gap']:+.3e}",
        flush=True,
    )
    print(
        f"  Corners with hull arb: {hull_summary['n_state_corners_with_arb']}/"
        f"{hull_summary['n_state_corners_total']}  "
        f"argmax_i_s={hull_summary['argmax_state_idx']}",
        flush=True,
    )

    # Top-K worst corners by hull gap.
    top_k = min(args.top_corners, gap_hull.size)
    worst_idx = np.argsort(-gap_hull)[:top_k]
    worst_corners = []
    print(f"\n  Top {top_k} worst corners (by hull gap):", flush=True)
    for rank, idx in enumerate(worst_idx, 1):
        idx_int = int(idx)
        # Largest pairwise dominance at this corner across all 6 ordered pairs.
        per_corner_dom = max(
            float(g[idx_int]) for g in pair_gaps.values()
        )
        worst_corners.append({
            "rank": rank,
            "i_s": idx_int,
            "axis_indices": list(_state_index_to_tuple(
                idx_int, tuple(int(s) for s in pc.state_grid_sizes),
            )),
            "state_coords": list(map(float, np.asarray(pc.state_grid[idx_int]))),
            "hull_gap": float(gap_hull[idx_int]),
            "max_dominance_gap": per_corner_dom,
        })
        print(
            f"    {rank:>2}.  hull={float(gap_hull[idx_int]):+.3e}  "
            f"max_dom={per_corner_dom:+.3e}  "
            f"{_format_worst_corner(idx_int, pc)}",
            flush=True,
        )

    # Verdict: max of dominance and hull gaps.
    max_dom = float(worst_dom["max_gap"]) if worst_dom["max_gap"] is not None else 0.0
    max_hull = float(hull_summary["max_gap"])
    overall_max = max(max_dom, max_hull)

    if overall_max > 1e-4:
        verdict = "FAIL  (max gap > 1e-4 — Lobatto required)"
    elif overall_max > 1e-6:
        verdict = "CONCERNING  (max gap in [1e-6, 1e-4] — consider Lobatto)"
    else:
        verdict = "PASS  (max gap < 1e-6 — discrete cloud admits no arbitrage)"

    summary = {
        "input_kind": kind,
        "input_path": str(in_path),
        "n_state_corners": int(R_bill.shape[0]),
        "n_quad_nodes_per_corner": int(R_bill.shape[1] * R_bill.shape[2]),
        "n_state_quad": int(R_bill.shape[1]),
        "n_ret_quad": int(R_bill.shape[2]),
        "ret_lobatto_Z": _list_to_tuple_recursive(disc_config.ret_lobatto_Z),
        "state_lobatto_Z": _list_to_tuple_recursive(disc_config.state_lobatto_Z),
        "n_angles": int(args.n_angles),
        "axis_aligned_dominance": dom_summary,
        "convex_hull_arbitrage": hull_summary,
        "worst_corners": worst_corners,
        "max_dominance_gap": max_dom,
        "max_hull_gap": max_hull,
        "overall_max_gap": overall_max,
        "verdict": verdict.split()[0],
    }

    print("\n" + "=" * 70, flush=True)
    print("Arbitrage summary", flush=True)
    print("=" * 70, flush=True)
    print(f"  Input            : {in_path}", flush=True)
    print(f"  ret_lobatto_Z    : {summary['ret_lobatto_Z']}", flush=True)
    print(f"  state_lobatto_Z  : {summary['state_lobatto_Z']}", flush=True)
    print(f"  N_state corners  : {summary['n_state_corners']}", flush=True)
    print(f"  Nodes per corner : {summary['n_quad_nodes_per_corner']}", flush=True)
    print(f"  Max dominance    : {max_dom:+.3e}", flush=True)
    print(f"  Max hull arb     : {max_hull:+.3e}", flush=True)
    print(f"  Overall max gap  : {overall_max:+.3e}", flush=True)
    print(f"  Verdict          : {verdict}", flush=True)
    print("=" * 70, flush=True)

    if not args.no_save:
        out_path = (in_path / "arbitrage.json") if kind == "bundle" else Path("arbitrage.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_path}", flush=True)
    else:
        print("\n(--no-save: skipping JSON write)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
