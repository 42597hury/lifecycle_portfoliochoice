"""Wall-time + cost estimator for the lifecycle solver.

Calibrated against the empirical anchors collected through 2026-05-08:

  - GH200 single-device 5x5x5x5 retirement-only sq=(2,3,2,3) rq=(5,5) n_z=11
    mi=100 -> 273 s/age (handoff anchor; COMPLEXITY_WALL_TIME_2026-05-06.md)
  - 1x A100 SXM4 inf-horizon g{3,4,5} sq=(3,3,3,4) rq=(4,4) n_z=1 mi=100
    -> 11.95, 37.0, 89.84 s/iter (saved_runs/inf_horizon/...g{3,4,5}/metadata.json)
  - 1x GH200 inf-horizon axis-bump run{1..6} 5x5x5x5 n_z=1 mi=100 tol=1e-4
    -> 16.8, 28.0, 28.0, 28.0, 27.7, 27.7 s/iter
    (saved_runs/inf_horizon/system_iv_inf_axisbump_run*/metadata.json)

The closed form below scales the GH200 retire-only anchor by per-axis ratios.
The hardware multiplier (GH200 -> A100 = 1.88) is back-solved from the
inf-horizon g5 cross-check (predicted 107.7 s/iter on A100, empirical 89.84 ->
ratio 1.20 over the naive 2.26 fp64-peak ratio, leaving 1.88).

Wall projections are stated to +/-30% per the handoff scope. The estimator
should reproduce every anchor in `_anchors()` within +/-30%; the
`--calibrate` mode prints the residual table.

Usage:
  python scripts/analysis/compute_budget_estimator.py
  python scripts/analysis/compute_budget_estimator.py --calibrate
  python scripts/analysis/compute_budget_estimator.py --table
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import prod
from typing import Optional


# ---------------------------------------------------------------------------
# Constants

# Anchor: GH200 single-device, 5^4 retirement-only, sq=(2,3,2,3) rq=(5,5),
# n_z=11, n_w=180, mi=100, max_backtrack=10. From COMPLEXITY_WALL_TIME_2026-05-06.md
ANCHOR_W_PER_AGE_RETIRE_GH200 = 273.0  # seconds/age, warm JIT

# GH200 -> A100 SXM4 ratio for the lifecycle FOC kernel. Calibrated from
# inf-horizon g5 cross-check (see module docstring). Plain fp64-peak ratio
# would predict 2.26x; XLA gather-fusion + L2 caching narrow it to 1.88x.
GH200_TO_A100_RATIO = 1.88

# Multi-device scaling efficiency for the pmap path. From MULTI_GPU_AUDIT
# (2026-05-07): "the multi-GPU ratio at canonical scale will be closer to
# ~0.8x of n_dev". Per-device cell counts in [800, 1500] at canonical 5^4
# stay well above the 50-cell underutilization floor.
SCALING_EFFICIENCY = {1: 1.0, 2: 0.80, 4: 0.78, 8: 0.75}

# Working-age multiplier vs retirement. COMPLEXITY scan estimates 16-20x
# (n_eta x n_eps = 12 from the alive branch + extra bilinear-z corner work).
# Back-solved against the System II finite-horizon anchor (594 s on 2x H100,
# sq3x3 rq3x3 7x7 state n_z=15, mi=100): with eff=0.8 the multiplier lands
# at 20.9 -- consistent with the COMPLEXITY scan's upper-bound estimate.
WORKING_AGE_MULTIPLIER = 21.0

# Lifecycle structure: start_age=22, retire_age=67, terminal_age=99.
N_RETIRE_AGES = 33   # 67..99 inclusive less the boundary
N_WORK_AGES = 46     # 22..66 inclusive
N_BOUNDARY_AGES = 1  # the work->retire transition; charged at working cost

# Default canonical-baseline knobs (used to compute scaling ratios)
ANCHOR_N_STATE = 625        # 5^4
ANCHOR_K_V = 36             # prod((2,3,2,3))
ANCHOR_K_R = 25             # prod((5,5))
ANCHOR_K_CORNERS = 16       # 2^4
ANCHOR_N_Z = 11
ANCHOR_N_W = 180
ANCHOR_MI = 100
ANCHOR_MAX_BACKTRACK = 10
ANCHOR_FOC_CALLS = 1 + ANCHOR_MI * (1 + ANCHOR_MAX_BACKTRACK)  # 1101

# Lambda Cloud rates. Used for $-cost projection only.
USD_PER_HOUR = {
    "1xA100_SXM4": 1.29,
    "8xA100_SXM4": 10.32,        # 8 x 1.29
    "1xGH200": 1.99,
    "1xH100_SXM5": 3.29,
    "2xH100_SXM5": 6.58,
}


# ---------------------------------------------------------------------------
# Core formula

@dataclass(frozen=True)
class Config:
    state_grid_sizes: tuple
    n_state_quad_nodes: tuple
    n_ret_nodes_1d: tuple
    n_z: int
    n_w: int = 180
    max_iter: int = 100
    max_backtrack_iter: int = 10
    horizon: str = "lifecycle"   # 'lifecycle' or 'inf_horizon'
    n_outer_iters: int = 60      # only used if horizon == 'inf_horizon'

    @property
    def N_state(self):
        return prod(self.state_grid_sizes)

    @property
    def K_v(self):
        return prod(self.n_state_quad_nodes)

    @property
    def K_r(self):
        return prod(self.n_ret_nodes_1d)

    @property
    def K_corners(self):
        return 2 ** len(self.state_grid_sizes)

    @property
    def foc_calls(self):
        return 1 + self.max_iter * (1 + self.max_backtrack_iter)


def _hw_factor(hw: str, n_dev: int) -> float:
    """Hardware factor relative to a single GH200 unit of compute.

    Single GH200 = 1.0. Single A100 = 1.88x slower. Single H100 SXM5 ~ 1x
    (roughly equivalent to GH200 on this kernel). Multi-device scales
    near-linearly with the SCALING_EFFICIENCY dict.
    """
    if hw == "GH200":
        per_dev = 1.0
    elif hw in ("H100_SXM5", "H200"):
        per_dev = 1.0          # treat hopper-class identically
    elif hw == "A100_SXM4":
        per_dev = GH200_TO_A100_RATIO
    elif hw == "B200":
        per_dev = 0.7          # ~30% faster than hopper for fp64-heavy kernels
    else:
        raise ValueError(f"unknown hw {hw}")
    eff = SCALING_EFFICIENCY.get(n_dev, max(0.7, 1.0 - 0.04 * (n_dev - 1)))
    return per_dev / (n_dev * eff)


def wall_per_age_retire_s(cfg: Config, hw: str = "A100_SXM4", n_dev: int = 8) -> float:
    """Predicted retirement-age wall on `hw` x `n_dev` devices."""
    # foc_FLOPs(K_corners) = K_v * K_r * (K_corners*12 + 40); the corner
    # term is sub-linear (40-floor) so System II (4 corners) is not 4x cheaper
    # than System IV (16 corners) per FOC eval -- it is 232/88 = 2.64x cheaper.
    anchor_foc_flops = ANCHOR_K_V * ANCHOR_K_R * (ANCHOR_K_CORNERS * 12 + 40)
    cfg_foc_flops = cfg.K_v * cfg.K_r * (cfg.K_corners * 12 + 40)
    ratio = (
        (cfg.N_state / ANCHOR_N_STATE)
        * (cfg_foc_flops / anchor_foc_flops)
        * (cfg.foc_calls / ANCHOR_FOC_CALLS)
        * (cfg.n_z / ANCHOR_N_Z)
        * (cfg.n_w / ANCHOR_N_W)
        # Lobatto vs Gauss is treated as zero-cost: same K, just node placement.
    )
    return ANCHOR_W_PER_AGE_RETIRE_GH200 * ratio * _hw_factor(hw, n_dev)


def wall_per_age_work_s(cfg: Config, hw: str = "A100_SXM4", n_dev: int = 8) -> float:
    return WORKING_AGE_MULTIPLIER * wall_per_age_retire_s(cfg, hw, n_dev)


def wall_total_s(cfg: Config, hw: str = "A100_SXM4", n_dev: int = 8) -> float:
    """Total wall (seconds)."""
    if cfg.horizon == "inf_horizon":
        # n_z is structurally 1; per-iter cost ~ per-age-retire cost.
        per_iter = wall_per_age_retire_s(cfg, hw, n_dev)
        return cfg.n_outer_iters * per_iter
    # lifecycle: 33 retire + 46 work + 1 boundary (charged as work)
    Wr = wall_per_age_retire_s(cfg, hw, n_dev)
    Ww = wall_per_age_work_s(cfg, hw, n_dev)
    return N_RETIRE_AGES * Wr + (N_WORK_AGES + N_BOUNDARY_AGES) * Ww


def cost_usd(cfg: Config, hw: str, n_dev: int) -> Optional[float]:
    key = f"{n_dev}x{hw}" if n_dev > 1 else f"1x{hw}"
    if key not in USD_PER_HOUR:
        return None
    hours = wall_total_s(cfg, hw, n_dev) / 3600
    return hours * USD_PER_HOUR[key]


# ---------------------------------------------------------------------------
# Calibration / anchors

def _anchors():
    """Return (label, predicted, empirical) tuples for calibration."""
    out = []

    # Anchor 1: GH200 5^4 retire-only sq=(2,3,2,3) rq=(5,5) n_z=11 mi=100
    cfg = Config(
        state_grid_sizes=(5,5,5,5),
        n_state_quad_nodes=(2,3,2,3),
        n_ret_nodes_1d=(5,5),
        n_z=11,
    )
    pred = wall_per_age_retire_s(cfg, "GH200", 1)
    out.append(("GH200 anchor (5^4 retire sq2323 rq55)", pred, 273.0))

    # Inf-horizon anchors on 1x A100
    for grid_g, empirical_total, n_iter in [
        (3, 980.0, 82), (4, 3590.0, 97), (5, 8984.0, 100),
    ]:
        cfg = Config(
            state_grid_sizes=(grid_g,)*4,
            n_state_quad_nodes=(3,3,3,4),
            n_ret_nodes_1d=(4,4),
            n_z=1,
            horizon="inf_horizon",
            n_outer_iters=n_iter,
        )
        pred = wall_total_s(cfg, "A100_SXM4", 1)
        out.append((f"A100 inf-horizon g{grid_g} (sq3334 rq44)", pred, empirical_total))

    # System II finite-horizon (lifecycle) anchor on 2x H100 SXM5
    # 7x7 state (2D!), n_z=15, sq=(3,3), rq=(3,3), mi=100
    cfg = Config(
        state_grid_sizes=(7,7),
        n_state_quad_nodes=(3,3),
        n_ret_nodes_1d=(3,3),
        n_z=15,
    )
    pred = wall_total_s(cfg, "H100_SXM5", 2)
    out.append(("2xH100 System II lifecycle (7^2 sq33 rq33 nz15)", pred, 594.2))

    # Inf-horizon axis-bump anchors on 1x GH200, mi=100, n_z=1
    bump_cases = [
        ("axis-bump run1 (5^4 sq3333 rq33)",  (3,3,3,3), (3,3), 62, 1040.1),
        ("axis-bump run2 (5^4 sq3335 rq33)",  (3,3,3,5), (3,3), 64, 1791.5),
        ("axis-bump run3 (5^4 sq5333 rq33)",  (5,3,3,3), (3,3), 63, 1763.2),
        ("axis-bump run4 (5^4 sq3533 rq33)",  (3,5,3,3), (3,3), 65, 1819.1),
        ("axis-bump run5 (5^4 sq3333 rq35)",  (3,3,3,3), (3,5), 64, 1773.8),
        ("axis-bump run6 (5^4 sq3333 rq53)",  (3,3,3,3), (5,3), 62, 1717.6),
    ]
    for label, sq, rq, n_iter, empirical in bump_cases:
        cfg = Config(
            state_grid_sizes=(5,5,5,5),
            n_state_quad_nodes=sq,
            n_ret_nodes_1d=rq,
            n_z=1,
            horizon="inf_horizon",
            n_outer_iters=n_iter,
        )
        pred = wall_total_s(cfg, "GH200", 1)
        out.append((label, pred, empirical))

    return out


def calibrate():
    print(f"{'anchor':<55} {'predicted_s':>12} {'empirical_s':>12} {'ratio':>7}")
    print("-" * 90)
    worst = 0.0
    for label, pred, emp in _anchors():
        ratio = pred / emp
        worst = max(worst, abs(ratio - 1.0))
        print(f"{label:<55} {pred:>12.1f} {emp:>12.1f} {ratio:>7.2f}")
    print("-" * 90)
    print(f"worst |ratio - 1| = {worst:.2%}")
    print(f"target: |ratio - 1| < 30%   ({'PASS' if worst < 0.30 else 'FAIL'})")


# ---------------------------------------------------------------------------
# Candidate ranking table

def _candidates():
    """Configs the synthesis report ranks. Format: (label, Config)."""
    # Note: max_iter and gather_precision affect wall but gather precision is
    # not a wall-formula knob (it's bandwidth, not FLOPs) -- the formula
    # treats fp32 gather as +0% wall. Empirically fp32 saves 10-20% on the
    # gather-bound retire path. Treat as headroom.
    return [
        ("[A] 5^4, sq3333, rq33, n_z=11, mi=100 (cheap baseline)",
            Config((5,5,5,5), (3,3,3,3), (3,3), 11, max_iter=100)),
        ("[B] 5^4, sq3335, rq33, n_z=11, mi=100 (y_1 bump, no margin)",
            Config((5,5,5,5), (3,3,3,5), (3,3), 11, max_iter=100)),
        ("[C] 5^4, sq3335, rq33, n_z=11, mi=80 (y_1 bump, mi cap reduced)",
            Config((5,5,5,5), (3,3,3,5), (3,3), 11, max_iter=80)),
        ("[D] 5^4, sq3335 Lob_y1, rq33, n_z=11, mi=80 (RECOMMENDED)",
            Config((5,5,5,5), (3,3,3,5), (3,3), 11, max_iter=80)),
        ("[E] 5^4, sq3335, rq44, n_z=11, mi=50 (heavy ret-quad)",
            Config((5,5,5,5), (3,3,3,5), (4,4), 11, max_iter=50)),
        ("[F] 5^4, sq3333, rq33, n_z=15, mi=80 (no bump, more z)",
            Config((5,5,5,5), (3,3,3,3), (3,3), 15, max_iter=80)),
        ("[G] 5^4, sq3335, rq33, n_z=15, mi=50 (bump+z, low mi)",
            Config((5,5,5,5), (3,3,3,5), (3,3), 15, max_iter=50)),
        ("[H] 6^4, sq3335, rq33, n_z=11, mi=100 (g6, over budget)",
            Config((6,6,6,6), (3,3,3,5), (3,3), 11, max_iter=100)),
        ("[I] 7^4, sq3535 Lob, rq55 Lob, n_z=11, mi=100 (current canonical)",
            Config((7,7,7,7), (3,5,3,5), (5,5), 11, max_iter=100)),
        ("[J] 5^4, sq3333, rq44, n_z=17, mi=20 (user hypothesis)",
            Config((5,5,5,5), (3,3,3,3), (4,4), 17, max_iter=20)),
    ]


def table():
    print(f"{'config':<60} {'wall_8xA100_h':>14} {'cost_$':>10} {'fits_12h':>10}")
    print("-" * 100)
    for label, cfg in _candidates():
        wall_h = wall_total_s(cfg, "A100_SXM4", 8) / 3600
        cost = cost_usd(cfg, "A100_SXM4", 8)
        fits = "YES" if wall_h <= 12 else "NO"
        print(f"{label:<60} {wall_h:>14.2f} {(cost or 0):>10.2f} {fits:>10}")


# ---------------------------------------------------------------------------
# CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true", help="Print anchor calibration table")
    p.add_argument("--table", action="store_true", help="Print candidate-config ranking on 8x A100")
    args = p.parse_args()
    if not args.calibrate and not args.table:
        args.calibrate = True
        args.table = True
    if args.calibrate:
        print("# Calibration vs empirical anchors")
        calibrate()
        print()
    if args.table:
        print("# Candidate configs on 8x A100 SXM4 80GB (12h budget)")
        table()


if __name__ == "__main__":
    main()
