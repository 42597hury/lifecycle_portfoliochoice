"""
Simulate a saved lifecycle policy and report the wealth distribution per age
in AWI units (1 unit = $54.1k of 2019 USD).

Despite the legacy filename, this script now dispatches the VAR builder via
the post-pivot `build_bundle_var_config` helper, so it works on System 1 / 2 /
Full bundles as soon as they are solved on the new code path. Pre-pivot
4-axis "system_i_*"/"system_iv_*" bundles will be rejected at reload time
with a pointer to the pivot doc; use the pre-pivot revision of this repo
to analyse legacy bundles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lifecycle.policy_io import load_policy_bundle  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.simulation import simulate_lifecycle  # noqa: E402
from verify._diag_helpers import build_bundle_var_config  # noqa: E402
from lifecycle.model import DiscretizationConfig  # noqa: E402
from lifecycle.wealth_grid import disc_config_with_bundle_wealth_grid  # noqa: E402

AWI_2019_USD = 54_099.99


def _list_to_tuple(v):
    if isinstance(v, list):
        return tuple(_list_to_tuple(x) for x in v)
    return v


def _rehydrate_disc_config(d: dict) -> DiscretizationConfig:
    tuple_fields = {"state_grid_sizes", "n_state_quad_nodes", "n_ret_nodes_1d",
                    "state_n_stds", "ret_lobatto_Z", "state_lobatto_Z"}
    valid = set(DiscretizationConfig._fields)
    kwargs = {}
    for k, v in d.items():
        if k not in valid:
            continue
        kwargs[k] = _list_to_tuple(v) if k in tuple_fields else v
    return DiscretizationConfig(**kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=REPO / "saved_runs" / "ablations" / "system_1_grid7_nz70_calib1")
    parser.add_argument("--n-simulations", type=int, default=10_000)
    parser.add_argument("--initial-wealth", type=float, default=0.1,
                        help="initial wealth in AWI units (default 0.1 ≈ $5.4k)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=REPO / "docs" / "scans" / "sim_wealth_system_i_nz70.json")
    args = parser.parse_args(argv)

    print(f"Loading bundle {args.bundle.name}...", flush=True)
    C, S, B, _, metadata = load_policy_bundle(args.bundle)
    rc = metadata["run_config"]
    base_config = rc["base_config"]
    disc_solver = _rehydrate_disc_config(rc["discretization_config"])
    disc_solver = disc_config_with_bundle_wealth_grid(
        disc_solver, args.bundle, metadata
    )
    var_config = build_bundle_var_config(metadata, args.bundle)
    model = build_model(base_config, var_config, verbose=False)
    pc = build_precompute(model, disc_solver, verbose=False)

    print(f"Simulating {args.n_simulations} households (initial_w={args.initial_wealth} AWI = "
          f"${args.initial_wealth * AWI_2019_USD/1000:.1f}k)...", flush=True)
    sim = simulate_lifecycle(
        C, S, B, pc, model,
        n_simulations=args.n_simulations,
        initial_wealth=args.initial_wealth,
        initial_z="stationary",
        initial_state="median",
        seed=args.seed,
        verbose=False,
    )

    x = np.asarray(sim["x"])           # (n_sim, n_age) — wealth at start of period
    alive = np.asarray(sim["alive"])    # (n_sim, n_age) bool
    ages = np.asarray(sim["ages"])      # (n_age,)
    print(f"  Sim panel: x.shape = {x.shape}, ages = {ages[0]}..{ages[-1]}", flush=True)

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print(f"\n{'age':>4} {'n_alive':>7} {'mean':>10} {'p5':>8} {'p25':>8} {'p50':>8} "
          f"{'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8}  (in AWI units)", flush=True)
    rows = []
    for t, a in enumerate(ages):
        m = alive[:, t]
        if not m.any():
            continue
        wealth = x[m, t]
        qs = np.quantile(wealth, quantiles)
        mean = float(wealth.mean())
        rows.append({
            "age": int(a),
            "n_alive": int(m.sum()),
            "mean": mean,
            "p5": float(qs[0]),
            "p25": float(qs[1]),
            "p50": float(qs[2]),
            "p75": float(qs[3]),
            "p90": float(qs[4]),
            "p95": float(qs[5]),
            "p99": float(qs[6]),
            "max": float(wealth.max()),
        })
        if a in (22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 67, 70, 75, 80, 85, 90, 95, 99):
            print(f"{a:>4} {m.sum():>7} {mean:>10.3f} {qs[0]:>8.3f} {qs[1]:>8.3f} "
                  f"{qs[2]:>8.3f} {qs[3]:>8.3f} {qs[4]:>8.3f} {qs[5]:>8.3f} {qs[6]:>8.3f}",
                  flush=True)

    print(f"\nIn USD (AWI 2019 = ${AWI_2019_USD:,.0f}):")
    print(f"{'age':>4} {'n_alive':>7} {'mean $':>12} {'p50 $':>12} {'p90 $':>14} {'p99 $':>14}",
          flush=True)
    for r in rows:
        if r['age'] in (22, 30, 40, 50, 60, 67, 75, 85, 95):
            print(f"{r['age']:>4} {r['n_alive']:>7} "
                  f"${r['mean']*AWI_2019_USD/1000:>10.1f}k "
                  f"${r['p50']*AWI_2019_USD/1000:>10.1f}k "
                  f"${r['p90']*AWI_2019_USD/1000:>12.1f}k "
                  f"${r['p99']*AWI_2019_USD/1000:>12.1f}k",
                  flush=True)

    # Wealth-grid coverage check
    w_max_grid = float(rc["discretization_config"]["wealth_max"])
    overall_max = max(r["max"] for r in rows)
    print(f"\nMax simulated wealth (any age): {overall_max:.2f} AWI = "
          f"${overall_max * AWI_2019_USD/1_000_000:.2f}M; grid max = {w_max_grid:.2f} AWI")
    print(f"Fraction of households at any age with W >= 50 AWI ($2.7M): "
          f"{float(np.mean([np.mean(x[alive[:,t], t] >= 50) for t in range(len(ages)) if alive[:,t].any()])):.4f}")
    print(f"Fraction with W >= 100 AWI ($5.4M): "
          f"{float(np.mean([np.mean(x[alive[:,t], t] >= 100) for t in range(len(ages)) if alive[:,t].any()])):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle": str(args.bundle),
        "n_simulations": args.n_simulations,
        "initial_wealth": args.initial_wealth,
        "seed": args.seed,
        "AWI_2019_USD": AWI_2019_USD,
        "wealth_grid_max": w_max_grid,
        "max_simulated_wealth_AWI": overall_max,
        "per_age": rows,
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
