"""CLI driver for the lifecycle VAR bootstrap.

Usage:
  python -m lifecycle.bootstrap.run_bootstrap \
      --B 1000 --ell-mean 8 --inflation-mode refit --seed 0 \
      --out data/bootstrap

Writes:
  {out}/var_bootstrap_draws.csv      (one row per (draw_id, lambda))
  {out}/var_bootstrap_point.csv      (production point estimates, same columns)
  {out}/summary.csv                  (per-statistic marginal summary)
  {out}/summary_paired.csv           (paired-lambda differences)
  {out}/summary_stable.csv           (marginal, stable-only)
  {out}/summary_paired_stable.csv    (paired, stable-only)
  {out}/stability.csv                (stability fraction flags)
  {out}/log.txt                      (run log + self-verification block)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .donor_panel import build_donor_panel
from .pipeline import (
    iteration_to_row,
    run_iteration,
)
from .stationary_block import identity_indices, stationary_block_indices
from .summarize import write_summaries

DEFAULT_LAMBDAS = (0.0, 0.5, 1.0)


def _run_one(args) -> dict:
    draw_id, idx, donor_arrays, lambdas, inflation_mode, production_ar1, ell_mean = args
    result = run_iteration(
        donor_arrays=donor_arrays,
        idx=idx,
        lambdas=lambdas,
        inflation_mode=inflation_mode,
        production_ar1=production_ar1,
        draw_id=draw_id,
        block_length_mean=ell_mean,
    )
    rows = [iteration_to_row(result, lam) for lam in lambdas]
    return rows


def run_bootstrap(
    B: int,
    ell_mean: float,
    inflation_mode: str,
    seed: int,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    out_dir: Path | None = None,
    quiet: bool = False,
) -> dict[str, Path]:
    out_dir = Path(out_dir) if out_dir else Path("data/bootstrap")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        if not quiet:
            print(msg, flush=True)

    log("=" * 72)
    log(f"BOOTSTRAP VAR INFERENCE  (B={B}, ell_mean={ell_mean}, mode={inflation_mode}, seed={seed})")
    log("=" * 72)

    t0 = time.time()
    donor = build_donor_panel()
    log(f"Donor panel: {donor.sample_start}-{donor.sample_end}, T={donor.T}")
    log(f"Production AR(1): phi={donor.production_ar1['phi']:.4f}, mu={donor.production_ar1['mu'] * 100:.4f}%")
    donor_arrays = donor.as_arrays()

    log(f"\nComputing production point estimates (identity resample, conditional AR(1))")
    id_idx = identity_indices(donor.T)[0]
    point_result = run_iteration(
        donor_arrays=donor_arrays,
        idx=id_idx,
        lambdas=lambdas,
        inflation_mode="conditional",
        production_ar1=donor.production_ar1,
        draw_id=-1,
        block_length_mean=ell_mean,
    )
    point_rows = [iteration_to_row(point_result, lam) for lam in lambdas]
    point_df = pd.DataFrame(point_rows)

    if not quiet:
        for lam in lambdas:
            fr = point_result.final_results[lam]
            if fr is None:
                log(f"  lambda={lam:.2f}: POINT ESTIMATE FAILED")
                continue
            log(
                f"  lambda={lam:.2f}: T={fr['T_lam']}, "
                f"E[xb]={fr['E_xb'] * 100:+.3f}pp, "
                f"sd(xb)={fr['sd_xb'] * 100:.3f}pp, "
                f"Sh(xb)={fr['Sharpe_xb']:+.3f}, "
                f"Phi_xb,spr={fr['Phi_xb_spr']:+.3f}, "
                f"state_max_eig={fr['state_max_eig']:.4f}"
            )

    log(f"\nDrawing {B} stationary-block index sets")
    indices = stationary_block_indices(T=donor.T, ell_mean=ell_mean, n_draws=B, seed=seed)

    log(f"\nRunning {B} iterations serially")
    t_iter = time.time()
    all_rows: list[dict] = []
    progress_pts = max(1, B // 10)
    for b in range(B):
        result = run_iteration(
            donor_arrays=donor_arrays,
            idx=indices[b],
            lambdas=lambdas,
            inflation_mode=inflation_mode,
            production_ar1=donor.production_ar1,
            draw_id=b,
            block_length_mean=ell_mean,
        )
        for lam in lambdas:
            all_rows.append(iteration_to_row(result, lam))
        if (b + 1) % progress_pts == 0:
            elapsed = time.time() - t_iter
            rate = (b + 1) / elapsed
            eta = (B - (b + 1)) / rate if rate > 0 else float("inf")
            log(f"  {b + 1}/{B}  ({rate:.0f} draws/s, ETA {eta:.0f}s)")

    draws_df = pd.DataFrame(all_rows)
    elapsed = time.time() - t_iter
    log(f"  iteration loop done in {elapsed:.1f}s ({B / elapsed:.0f} draws/s)")

    draws_path = out_dir / "var_bootstrap_draws.csv"
    point_path = out_dir / "var_bootstrap_point.csv"
    draws_df.to_csv(draws_path, index=False)
    point_df.to_csv(point_path, index=False)
    log(f"\nWrote draws:  {draws_path}")
    log(f"Wrote point:  {point_path}")

    log(f"\nComputing summaries")
    paired_pairs = [(1.0, 0.0), (1.0, 0.5), (0.5, 0.0)]
    summary_paths = write_summaries(
        draws_path=draws_path,
        point_path=point_path,
        out_dir=out_dir,
        paired_pairs=paired_pairs,
    )
    for name, p in summary_paths.items():
        log(f"  {name}: {p}")

    log("\nSELF-VERIFICATION CHECKS")
    log("-" * 72)
    _run_self_checks(draws_df, point_df, lambdas, log)

    log_path = out_dir / "log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nLog written to: {log_path}")

    log(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    return {**summary_paths, "draws": draws_path, "point": point_path, "log": log_path}


def _run_self_checks(
    draws: pd.DataFrame,
    point: pd.DataFrame,
    lambdas: tuple[float, ...],
    log,
) -> None:
    """Section 5 of the handoff doc: construction VAR stability, EH zero check,
    lambda monotonicity, paired vs unpaired variance.

    The identity test (check 1) lives in tests/test_bootstrap_identity.py
    and is run separately via pytest.
    """
    n_total = len(draws) // len(lambdas)

    sub_ok = draws[draws["final_var_failed"] == 0]
    if len(sub_ok) == 0:
        log("  [FAIL] all draws have failed final VAR estimation")
        return

    p_construction_stable = float((sub_ok["construction_max_eig"] < 1.0).mean())
    flag = "PASS" if p_construction_stable >= 0.95 else "WARN"
    log(f"  [{flag}] construction VAR P(max|eig|<1) = {p_construction_stable:.3f} (target >=0.95)")

    if 0.0 in lambdas:
        sub_lam0 = draws[(draws["lambda_val"] == 0.0) & (draws["final_var_failed"] == 0)]
        n_violations = 0
        sample = sub_lam0.head(50)
        for _, row in sample.iterrows():
            if abs(row["Phi_xb_cape"]) > 1e-10 or abs(row["Phi_xb_spr"]) > 1e-10 or abs(row["Phi_xb_y1"]) > 1e-10:
                n_violations += 1
        flag = "PASS" if n_violations == 0 else "FAIL"
        log(f"  [{flag}] restricted_eh zero check at lambda=0: {n_violations}/{len(sample)} violations")

    if 1.0 in lambdas and 0.0 in lambdas:
        piv = draws.pivot_table(index="draw_id", columns="lambda_val", values="E_xb", aggfunc="first")
        ok = piv.dropna()
        if len(ok) >= 10:
            corr = float(ok[1.0].corr(ok[0.0]))
            flag = "PASS" if corr > 0.0 else "FAIL"
            log(f"  [{flag}] pairing sanity: corr(E_xb_1, E_xb_0) across draws = {corr:.3f} (target >0)")

            piv_full = ok
            diff_var = float(np.var(piv_full[1.0] - piv_full[0.0], ddof=1))
            marg_var_sum = float(np.var(piv_full[1.0], ddof=1) + np.var(piv_full[0.0], ddof=1))
            flag = "PASS" if diff_var < marg_var_sum else "FAIL"
            log(
                f"  [{flag}] paired vs unpaired variance for E_xb(1)-E_xb(0): "
                f"diff_var={diff_var:.6e}, marg_var_sum={marg_var_sum:.6e}"
            )

    p_failed = float((draws["final_var_failed"] == 1).mean())
    flag = "PASS" if p_failed < 0.02 else "WARN"
    log(f"  [{flag}] final VAR failure rate = {p_failed:.3f} (target <0.02)")

    log(f"  n_draws total = {n_total}, n_failed (any lambda) = {int(p_failed * len(draws))}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Lifecycle VAR bootstrap")
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--ell-mean", type=float, default=8.0)
    p.add_argument(
        "--inflation-mode",
        choices=("refit", "conditional"),
        default="refit",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    p.add_argument("--out", type=Path, default=Path("data/bootstrap"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    run_bootstrap(
        B=args.B,
        ell_mean=args.ell_mean,
        inflation_mode=args.inflation_mode,
        seed=args.seed,
        lambdas=tuple(args.lambdas),
        out_dir=args.out,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
