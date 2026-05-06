# Lifecycle Portfolio Choice Model

Annual-frequency life-cycle portfolio model with three assets (T-bills, stocks,
20-year nominal AAA bonds), VAR(1) return dynamics, mixture-normal labour
income, earnings-dependent mortality, and a Catherine (2025) bequest motive.
The agent lives from age 22 to 99, retires at 67, and chooses
`(c_t, α_s, α_b)` each year. Calibration follows Catherine (2025) on income
and bequests; the VAR is estimated from 1963–2025 annual data.

This repo is the code companion to a master's thesis in economics.

## Quick start

```python
from lifecycle.precompute import build_model, Precompute
from lifecycle.solver import run_lifecycle_solver
from lifecycle.simulation import simulate_lifecycle
from lifecycle.policy_io import save_policy_bundle, load_policy_bundle
from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.predictability_ablation import prepare_predictability_system

# 1. Estimate the VAR (or pick an ablation: "I", "II", "III", "IV")
system = prepare_predictability_system("IV", csv_path="data/var_dataset.csv",
                                        disc_config_template=CANONICAL_DISC)

# 2. Build the model and precompute grids/quadrature/lookup tables
model = build_model(BASE_CONFIG, system["var_config"], verbose=True)
pc    = Precompute(model, system["disc_config"])

# 3. Solve (EGM + 2D Newton, Numba-jitted; ~minutes for 7×7×7)
C_mat, S_mat, B_mat, diagnostics = run_lifecycle_solver(model, pc,
                                       solver_config=CANONICAL_SOLVER)

# 4. Save the bundle for downstream simulation / figure scripts
save_policy_bundle("saved_runs/my_run", C_mat, S_mat, B_mat,
                   diagnostics=diagnostics, run_config=...)

# 5. Simulate
sim = simulate_lifecycle(C_mat, S_mat, B_mat, pc, model,
                          n_simulations=10_000, seed=42)
```

The orchestration notebook is [main.ipynb](main.ipynb).

## The four aspects

### 1. Solver factories

The "factory" idiom is `build_model` → `Precompute` → `run_lifecycle_solver`.
Each step returns an immutable artefact consumed by the next.

| Step | Function | Module |
|---|---|---|
| Economic spec | `build_model(base_config, var_config)` | [precompute.py](precompute.py) |
| Discretization | `Precompute(model, disc_config)` | [precompute.py](precompute.py) |
| Backward induction | `run_lifecycle_solver(model, pc, solver_config)` | [solver.py](solver.py) |
| Infinite-horizon benchmark | `run_infinite_horizon_solver(...)` | [inf_horizon_solver.py](inf_horizon_solver.py) |

Configurations live in `configs/`; `configs/_canonical.py` is the single source
of truth and `configs/sweep_main/*.py` cells override only the fields they
intentionally vary. See [docs/CONFIG.md](docs/CONFIG.md) for field-by-field
rationale.

### 2. Diagnostics factories

Two flavours — pre-built reports for solved bundles, and one-off probe scripts.

| Flavour | Where | When |
|---|---|---|
| Pre/post-solve reports | [diagnostics.py](diagnostics.py) (`diagnose_*_pre`, `diagnose_*_post`, `diagnose_terminal_portfolio_states`, ...) | Called from the notebook after `Precompute` and after `simulate_lifecycle` |
| One-off probes | [scripts/diagnostics/](scripts/diagnostics/) (Euler errors, gridpoint EE, state-grid coverage, etc.) | Run from the command line on a saved bundle |

The Numba-counter-based solver diagnostics (`diagnostics` dict returned by
`run_lifecycle_solver`) hold convergence rates, FOC residuals, and per-age
exit-code counts — see the `DI_*` / `DF_*` / `EC_*` constants in
[solver.py](solver.py).

### 3. Policy bundle storage

A "bundle" is a directory under `saved_runs/` containing the policy arrays,
the solver diagnostics, and a metadata snapshot of the run config. Bundles are
the canonical way to share solve outputs across the analysis layer.

```
saved_runs/<bundle_name>/
├── policy_arrays.npz     C_mat, S_mat, B_mat (compressed)
├── diagnostics.pkl       solver-side counters (pickled)
├── metadata.json         human-readable: shape, dtype, full run_config
└── sims/                 named simulations
    ├── <label>.npz       arrays from simulate_lifecycle
    └── <label>_meta.json
```

`saved_runs/` is gitignored; bundles are reproducible from
`metadata["run_config"]` plus the matching code revision.

API: `save_policy_bundle`, `load_policy_bundle`, `save_sim_data`,
`load_sim_data`, `list_sims` — all in [policy_io.py](policy_io.py).

### 4. AWS workflow

Per-config solves are run on EC2; bundles upload to S3 and `aws s3 sync`
back to the workstation. The pipeline is hardened for unattended overnight
sweeps (per-age S3 checkpointing, 24h self-destruct timer, $-cap budgeting).

| Script | Role |
|---|---|
| [scripts/launch_run.py](scripts/launch_run.py) | Single-config launcher |
| [scripts/launch_sweep.py](scripts/launch_sweep.py) | Parallel sweep launcher (one EC2 per config) |
| [scripts/launch_serial.py](scripts/launch_serial.py) | Serial sweep on one EC2 |
| [scripts/launch_queue.py](scripts/launch_queue.py) | Queue manager |
| [scripts/preflight_sweep.py](scripts/preflight_sweep.py) | 10-minute pre-launch sanity gate |
| [scripts/setup_budget.py](scripts/setup_budget.py) | Cost ceiling helper |
| [scripts/ec2_userdata.sh](scripts/ec2_userdata.sh) | EC2 user-data template (single solve) |
| [scripts/ec2_userdata_serial.sh](scripts/ec2_userdata_serial.sh) | EC2 user-data template (serial) |
| [scripts/run_solve.py](scripts/run_solve.py) | EC2-side solve runner |

End-to-end runbook: [docs/agents/AWS_WORKFLOW.md](docs/agents/AWS_WORKFLOW.md).

## Repository layout

```
.
├── README.md                  this file
├── main.ipynb                 orchestration notebook
├── requirements.txt           numpy, scipy, numba, pandas
│
├── model.py                   LifecyclePortfolioModel + utility/tax helpers
├── var.py                     VAR estimation + state/return partition
├── discretization.py          Rouwenhorst + Judd quadrature + state grids
├── mortality.py               earnings-dependent survival calibration
├── precompute.py              build_model() + Precompute()
├── solver.py                  backward induction (EGM + 2D Newton)
├── inf_horizon_solver.py      CCV-style infinite-horizon benchmark
├── simulation.py              forward simulation
├── diagnostics.py             pre/post-solve calibration + Newton reports
├── numerics.py                shared PCHIP + bin-prob helpers
├── plots.py                   pre/post-solve figures
├── policy_io.py               bundle save/load
├── predictability_ablation.py systems I-IV
│
├── configs/                   canonical config + sweep cells
│   ├── _canonical.py          single source of truth
│   ├── smoke_test.py
│   └── sweep_main/            01_base, 02_state33, 03_grid9_base, ...
│
├── scripts/                   workflow scripts
│   ├── launch_*.py            AWS launchers
│   ├── run_solve.py           EC2-side runner
│   ├── ec2_userdata*.sh       user-data templates
│   ├── preflight_sweep.py
│   ├── setup_budget.py
│   ├── diagnostics/           one-off bundle probes
│   ├── validation/            heavier correctness audits
│   ├── benchmarks/            timing scripts
│   ├── smoke/                 fast sanity probes
│   └── investigation/         retired investigation scripts
│
├── tests/                     pytest suite
├── data/                      var_dataset.csv + raw FRED/Shiller inputs
├── saved_runs/                solve outputs (gitignored)
│
└── docs/                      everything markdown
    ├── DESIGN.md              full architectural spec
    ├── CONVENTIONS.md         timing, indexing, units, signs
    ├── CONFIG.md              field-by-field rationale for _canonical.py
    ├── RETURNS.md             VAR + return-quadrature details
    ├── LABOUR.md              income calibration
    ├── UTILITY.md             CRRA + bequest spec
    ├── STATE_SPACE.md         financial state grid
    ├── RESULTS.md             figure / table inventory
    ├── TODO.md                living task list
    ├── agents/                agent runbooks (AWS, EE diagnostic, etc.)
    ├── handoff/               active sprint tickets
    ├── notes/                 chronological dev trackers
    └── archive/               retired handoffs
```

## Where to start reading

| Goal | Start here |
|---|---|
| Run a solve end-to-end | [main.ipynb](main.ipynb) |
| Understand the model | [docs/DESIGN.md](docs/DESIGN.md), then [docs/CONVENTIONS.md](docs/CONVENTIONS.md) |
| Tune a sweep config | [docs/CONFIG.md](docs/CONFIG.md), then `configs/_canonical.py` |
| Run on AWS | [docs/agents/AWS_WORKFLOW.md](docs/agents/AWS_WORKFLOW.md) |
| Add a new diagnostic probe | [scripts/diagnostics/](scripts/diagnostics/) — copy an existing `_diag_*.py` |
| Reproduce a saved bundle | `metadata.json` in the bundle directory |

## Dependencies

Python 3.11+. Hot loops require `numba`. `pandas` is used only for VAR data
prep; the solver and simulator are pure NumPy + Numba.

```
pip install -r requirements.txt
```
