# HANDOFF — Auto-run EE diagnostics in AWS solve job

## Scope

Currently `scripts/run_solve.py` solves a bundle, saves it, and uploads it to S3. The user then has to run the EE diagnostic battery locally on every bundle they want to evaluate. This handoff makes the solver run the diagnostics itself before uploading, so every bundle ships with its own evaluation report.

Add a post-solve diagnostic step that:

1. Runs the standard EE battery against the freshly-saved bundle.
2. Writes per-diagnostic markdown reports into `<bundle_dir>/diagnostics_reports/`.
3. Compiles a JSON manifest of headline numbers at `<bundle_dir>/diagnostics_summary.json`.
4. Uploads the augmented bundle to S3 along with the original artifacts.
5. Provides a `--skip-diagnostics` flag for cases where the user just wants a fast solve (smoke tests, mid-development iteration).

The diagnostic battery to auto-run is exactly the four "headline" diagnostics from `docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md` §3:

- `_diag_arbitrage_quadsweep` — sanity check on quadrature wiring (~30s)
- `_diag_invalid_cells --eval-mode next_finer` — per-cell breakdown if invalidity exists (~5 min)
- `_diag_gridpoint_ee --eval-mode next_finer` — gridpoint EE (~3 min)
- `_diag_euler_errors --eval-mode next_finer` (sim-path) — the headline publication-grade test (~5 min)

NOT in the auto-run (these are reserved for problem investigation, invoked manually after a bundle has been flagged as anomalous):

- `_diag_simpath_worst_cells.py`
- `_diag_per_axis_tail.py`
- `_diag_state_tail_node.py`
- `_diag_tail_node_position.py`
- `_diag_simulator_validation.py`

Total marginal AWS wall-time: ~13–22 minutes on a 1.5–2.5 hour solve. Acceptable.

## Background context (what this fixes)

- The diagnostic API has shifted twice in the past week (`_evaluate_age_errors` signature change + Lobatto eval-rule propagation in `_build_eval_disc`). Bundles solved before either change can fail the local diagnostic. Auto-running diagnostics inside the same Python process that solved the bundle eliminates this version-skew failure mode.
- AWS sweeps (10-config or larger) currently produce 10 bundles that all need diagnostics run separately — easy to lose track of which has been graded. A standardized JSON manifest in every bundle directory lets `jq` grep across the S3 prefix to find "all bundles where retire publication gate passes" in one query.
- Local laptop is currently tied up for ~20 min after every AWS bundle download. AWS minutes are cheaper and parallelisable.

## What you need to read

1. **`scripts/run_solve.py`** — the launcher. Read top-to-bottom (~230 LOC). The post-solve hook goes between `save_policy_bundle(...)` (line 211) and the S3 upload (line 220). Note `--no-upload` is already a CLI flag; add `--skip-diagnostics` analogously.

2. **`scripts/diagnostics/_diag_arbitrage_quadsweep.py`** — the simplest diagnostic; understand its CLI surface and the markdown output (`--markdown-out`). Look at how `_make_pc` and `_load_bundle_context` are imported to confirm the diagnostic only needs the bundle path + run_config in metadata.

3. **`scripts/diagnostics/_diag_euler_errors.py`** — the most complex. Note the `--eval-mode`, `--n-simulations`, `--eval-households-per-age`, `--initial-z`, `--initial-state` flags. Default sim-path config we want: `--n-simulations 5000 --eval-households-per-age 256 --initial-z stationary --initial-state stationary`.

4. **`scripts/diagnostics/_diag_invalid_cells.py`** and **`scripts/diagnostics/_diag_gridpoint_ee.py`** — same pattern. Both accept `--bundle` (or positional bundle path), `--model-bundle`, `--eval-mode`, `--markdown-out`. Default eval mode for both: `next_finer`. Default wealth indices on `_diag_invalid_cells` is `[0, 15, 75, 134, 149]`; keep that default.

5. **`docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md`** §3 (the investigative process) and §8 (quick-reference command sequence) — these document the exact CLI flags and the publication gates the manifest should report against.

6. **`lifecycle/policy_io.py`** — confirm where the bundle directory is and what files live in it. The auto-diagnostic step must not collide with `save_policy_bundle`'s own output.

7. **`scripts/_pull_bundle.py`** and any S3-upload helpers in `scripts/run_solve.py` — confirm the upload step picks up everything in `<bundle_dir>` (so `diagnostics_reports/` and `diagnostics_summary.json` will be uploaded automatically without changes to the upload path).

## Edge cases to think through

- **Diagnostic failure must not abort the upload.** If `_diag_euler_errors` raises, the bundle (which is the expensive thing to recompute) should still upload. Catch the exception, log it, write a stub manifest noting the failure, and continue. The user can re-run the diagnostic locally if needed.

- **Partial bundles** (where `solve_control.youngest_age_to_solve` skipped working ages, like `run_ret_v*.py` configs): diagnostics on a partial bundle return `nan` for the working-age phase. The manifest should reflect this honestly: `gates.working_publication_pass = null` (not `false`) when no working-age data exists. Match the existing `_diag_euler_errors` semantics for nan handling.

- **Bundles solved with `--no-checkpoint`** vs checkpointed bundles — the bundle directory layout is the same, so no difference for diagnostics.

- **`--n-simulations 5000` is hardcoded in the workflow**, but for very small smoke tests (`run_v3_smoketest.py`-style 3x3x3 bundles) we may want to reduce. Recommendation: keep 5000 fixed in the auto-runner; users running tiny smoke tests pass `--skip-diagnostics`.

- **Initial conditions for sim-path** (`--initial-z`, `--initial-state`, `--initial-x`) affect numbers materially. For consistency across runs, the auto-runner should use a documented default. Recommendation: `initial_z=stationary, initial_state=stationary, initial_x=None` (lets the simulator use its natural distribution). Document this choice in the manifest.

- **Markdown output paths.** Auto-runner should write into `<bundle_dir>/diagnostics_reports/` (subdir created if needed). Filenames should follow the convention from §6 of the workflow doc: `diagnostics_<diag_name>_<bundle_label>.md`.

- **The arbitrage sweep currently has a known issue** (`_make_pc` strips Lobatto fields when constructing override DiscretizationConfigs in `_diag_arbitrage_quadsweep.py:70-91`). It should still produce its baseline row correctly for any bundle (Lobatto or GH); only the per-config sweep rows are misleading on Lobatto bundles. Auto-runner can use it as-is for the baseline check; the misleading sweep rows are diagnostically interesting only when invalidity is non-zero, in which case the user is already in problem-investigation mode.

- **The diagnostics share imports** (`_load_bundle_context`, `Precompute`). On a fresh Python interpreter each diagnostic re-builds the model+precompute. Calling them as 4 separate subprocess invocations adds ~30s × 4 of import/build overhead. Recommendation: invoke them as Python function calls inside `run_solve.py` rather than `subprocess` calls. Each diagnostic exposes a `run(args)` or `main(argv)` entry point — use those.

## JSON manifest schema

`<bundle_dir>/diagnostics_summary.json`:

```json
{
  "schema_version": 1,
  "bundle_label": "<from cfg.BUNDLE_SUFFIX>",
  "bundle_dir": "<absolute path>",
  "config_label": "<one-line summary: ret(...) Lob(...) | state(...) Lob(...) | n_stds(...)>",
  "solve_label": "constrained" | "unconstrained",
  "solved_age_window": [67, 99],
  "diagnostic_runtime_seconds": 1234,
  "diagnostic_eval_mode": "next_finer",
  "diagnostic_eval_quadrature": {"ret": [5,7,7], "state": [4,6,7], "ret_lobatto_Z": [null, 7.0, 7.0], "state_lobatto_Z": [null, null, 7.0]},
  "solver_health": {
    "worst_foc_resid": 0.21,
    "total_newton_failures": 1206079,
    "total_calls": 21732480,
    "avg_newton_iter": 1.35
  },
  "arbitrage_sweep": {
    "max_arbitrage_gap": 0.0,
    "worst_min_R_p_at_merton_solver": -0.354,
    "states_with_min_R_p_negative_at_merton": 67,
    "stat_mass_with_min_R_p_negative": 0.046
  },
  "gridpoint_ee_next_finer": {
    "n_total": 12960,
    "n_valid": 12860,
    "invalid_share": 0.0077,
    "mean_log10_EE": -2.57,
    "p95_log10_EE": -1.48,
    "max_log10_EE": -0.069
  },
  "simpath_ee_next_finer": {
    "n_simulations": 5000,
    "eval_households_per_age": 256,
    "retirement": {"mean_log10_EE": -2.22, "max_log10_EE": -0.054, "median_log10_EE": -2.18},
    "working":    {"mean_log10_EE": null, "max_log10_EE": null, "median_log10_EE": null}
  },
  "gates": {
    "retirement_publication_pass": false,
    "retirement_welfare_pass": false,
    "working_publication_pass": null,
    "working_welfare_pass": null
  },
  "diagnostic_status": "complete" | "partial" | "failed",
  "diagnostic_errors": [],
  "diagnostic_reports": {
    "arbitrage_sweep": "diagnostics_reports/diagnostics_arbitrage_<label>.md",
    "invalid_cells": "diagnostics_reports/diagnostics_invalid_cells_<label>.md",
    "gridpoint_ee": "diagnostics_reports/diagnostics_gridpoint_ee_<label>_nextfiner.md",
    "simpath_ee": "diagnostics_reports/diagnostics_simpath_ee_<label>_nextfiner.md"
  }
}
```

The exact field set should match what each diagnostic's existing markdown reporters extract — write a small helper that parses each markdown for the headline numbers (or, better, modify each diagnostic to also emit a JSON sidecar that the manifest collator reads). The JSON sidecar approach is cleaner.

## Implementation-plan deliverable

Before writing code, produce a written implementation plan covering:

1. **Architecture choice.** Two options:
   - (A) Auto-runner calls each diagnostic's `run(args)` / `main(argv)` entry point in-process from `run_solve.py`. Pros: no import re-overhead, direct error handling. Cons: need to plumb argument structures from CLI to in-process call.
   - (B) Auto-runner shells out to `python -m scripts.diagnostics._diag_*` subprocesses. Pros: matches existing CLI exactly; if a diagnostic crashes it doesn't take the runner with it. Cons: ~2 minutes of redundant import/JIT warmup.

   Recommendation: (A) for production cost-savings; (B) is simpler if you're tight on time. Pick one and justify.

2. **Where each diagnostic's headline numbers come from.** Three options:
   - Parse the markdown (fragile to format changes)
   - Add a `--json-out` flag to each diagnostic that writes a JSON sidecar (cleanest; one-line addition to each diagnostic's reporter)
   - Refactor diagnostics to return their summary dict from `run()` and parse markdown only as a fallback

   Recommendation: middle option (`--json-out` flag). Lowest code churn per diagnostic.

3. **Failure-isolation strategy.** What does `diagnostic_status` mean in the manifest? When does the bundle still upload?

4. **CLI surface for `run_solve.py`.** New flags: `--skip-diagnostics`, optional `--diagnostics-eval-mode` (default `next_finer`), `--diagnostics-only` (run diagnostics on an existing bundle without re-solving).

5. **Test plan.** Smoke-test against:
   - The existing v9 bundle in `saved_runs/` (Lobatto + wider grid)
   - A small smoke test (3x3x3) that should run end-to-end in <5 minutes
   - A bundle the user manually deletes the `diagnostics_reports/` from to confirm regen

6. **Backwards compatibility.** Existing bundles in S3 don't have the manifest. Should the auto-runner add a separate `make_diagnostics_for_existing_bundle` script for retrofitting? (Recommendation: yes, easy 20 LOC.)

Submit the implementation plan; only proceed with code once it's reviewed.

## Reference files

- `scripts/run_solve.py` — launcher target
- `scripts/diagnostics/_diag_*.py` — the diagnostics
- `docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md` — workflow + gates
- `lifecycle/policy_io.py` — bundle layout (`save_policy_bundle`, `load_policy_bundle`)
- `docs/handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md` — recently-completed example handoff in the same repo, similar shape

## Out of scope

- Auto-running deep-dive diagnostics (`_diag_simpath_worst_cells.py`, etc.) on the AWS side. Those stay manual; user invokes them locally when the auto-run flags a problem.
- Cross-bundle comparison automation (the JSON manifest enables it but doesn't perform it).
- Re-running diagnostics on already-uploaded bundles in S3 (separate retrofit script, see plan deliverable item 6).
- Fixing the `_diag_arbitrage_quadsweep._make_pc` Lobatto-stripping bug. Tracked separately; out of scope here.
