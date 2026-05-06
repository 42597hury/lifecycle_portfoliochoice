# AWS Workflow — Lifecycle Solver Runs

> **NUMBA / `main` BRANCH REFERENCE — JAX agents do not run this workflow.**
> The sweep machinery this doc describes (`scripts/launch_run.py`,
> `scripts/run_solve.py`, `scripts/preflight_sweep.py`, etc.) was deleted in
> handoff 1 of the JAX rewrite and has not been re-ported. JAX cloud-runner
> agents follow [AWS_TRIAL_JAX.md](AWS_TRIAL_JAX.md) instead.
>
> **Why it's still here:** §6.A documents lessons (resume-checkpoint validates
> array *shape* only, never value semantics — silently corrupts on canonical
> changes that don't change shape) that the JAX cloud-runner needs to know
> about when checkpoint resume is wired up on this branch.
>
> **Agent role (Numba sweep agent only):** this is your end-to-end runbook —
> launcher choice, vCPU quota math, budget alarms, parallel vs serial mode.
> Treat §6.A as a BLOCKER; everything else is procedural.

Run model solves on EC2 from your laptop. Bundles land in `saved_runs/` just like local runs.

**Prereqs (already done — see Section 4 if you ever need to redo them):**
- AWS CLI v2 installed; `aws sts get-caller-identity` works
- IAM user `hugo-cli` has policies: EC2FullAccess, S3FullAccess, IAMFullAccess, SSMReadOnlyAccess
- IAM role `thesis-ec2-runner` exists with `AmazonS3FullAccess` (used as EC2 instance profile)
- EC2 key pair `hugo-thesis` exists in `eu-north-1`
- S3 bucket `hugo-thesis-runs` exists in `eu-north-1`

All commands below are for **Windows PowerShell**. Region is always `eu-north-1`.

---

## 1. Daily workflow

### A. Make a new config

Copy an existing config and edit it:
```
cp configs/system_iv_5x5x5.py configs/my_new_run.py
```

Open `configs/my_new_run.py` and edit. Common knobs:

| Where | Knob | What it controls |
|---|---|---|
| `disc_config_template` | `state_grid_sizes` | (5,5,5), (7,7,7), etc. — biggest cost driver |
| `disc_config_template` | `n_z` | Persistent-income grid (must be odd) |
| `disc_config_template` | `n_wealth`, `n_savings` | Wealth / savings grid resolution |
| `disc_config_template` | `n_state_quad_nodes` | Quadrature nodes per state axis |
| `solver_config` | `tol` | Newton tolerance |
| `solver_config` | `max_iter_unconstrained` | Newton max iterations |
| `base_config` | `gamma`, `beta`, `constrained` | Preferences + leverage flag |
| top of file | `BUNDLE_SUFFIX` | String appended to bundle dir name |

Bundle name is auto-generated as:
`<system_label>_<constrained|unconstrained>_<grid_mode>_grid<NxNxN>_nz<n><BUNDLE_SUFFIX>`

### B. Pick an instance type

| Use case | Instance | $/hr | Approx solve time |
|---|---|---|---|
| Smoke test (3×3×3, tiny grids) | `c6i.2xlarge` | $0.34 | ~5–10 min |
| Default (5×5×5) | `c6i.4xlarge` | $0.68 | ~20–40 min |
| Larger (7×7×7) | `c7i.16xlarge` | $3.05 | ~1–2 hr |
| Production HPC | `hpc8a.96xlarge` | ~$8 | needs old account |

Cost = hourly rate × wall-clock time. A 30-min run on c6i.4xlarge ≈ $0.35.

### C. Launch

```
python scripts/launch_run.py configs/my_new_run.py --instance-type c6i.4xlarge --key-name hugo-thesis
```

The launcher prints a **launch_id** like `2026-05-02T16-58-54Z_my_new_run`. **Save it** (or scroll up later) — you need it to check progress.

### D. Watch progress

Substitute your launch_id:
```
aws s3 cp s3://hugo-thesis-runs/launches/<launch-id>/userdata.log - | Select-Object -Last 50
```

Re-run every minute or so. Log refreshes every 30 sec. What to look for:

| In log | Means |
|---|---|
| `[run_solve] solving...` + age messages | Working normally; just wait |
| `[run_solve] solve done in Xs` + `[5/5] solve complete` | Done. Instance is auto-terminating. |
| `!!! ERROR at line N` | Broken; see Section 3. Instance stays up; terminate manually. |
| `404 NoSuchKey` | Bootstrap hasn't reached first log upload yet. Wait. |

### E. Pull the bundle

When the log shows `solve complete`:
```
aws s3 sync s3://hugo-thesis-runs/saved_runs/ ./saved_runs/
```

Bundle appears at `saved_runs/<bundle-name>/` with `policy_arrays.npz`, `diagnostics.pkl`, `metadata.json`. Identical format to local solves — load it with `load_policy_bundle()` as usual.

### F. Running a sweep (many configs at once)

One folder per sweep, one config file per variant. Each variant launches its own EC2 instance in parallel.

**1. Make a sweep folder and populate it:**
```
mkdir configs/sweep_<name>
cp configs/system_iv_5x5x5.py configs/sweep_<name>/var_a.py
cp configs/system_iv_5x5x5.py configs/sweep_<name>/var_b.py
# ...
```

**2. In each file, edit the swept parameter AND set a distinct `BUNDLE_SUFFIX`:**
```python
gamma = 3.0                         # whatever you're varying
BUNDLE_SUFFIX = "_sweep_gamma3"     # MUST differ across files
```
Without distinct suffixes, configs with the same grid/constraint produce identical bundle names and overwrite each other in S3.

**3. Preview (no launch, no money):**
```
python scripts/launch_sweep.py configs/sweep_<name>/ --dry-run
```

**4. Fire the sweep:**
```
python scripts/launch_sweep.py configs/sweep_<name>/ --instance-type c6i.4xlarge --key-name hugo-thesis
```
Launches one EC2 instance per config in parallel, each independent. A failed launch doesn't affect the others.

**5. Watch + pull as in single runs.** When all running thesis instances are gone:
```
aws s3 sync s3://hugo-thesis-runs/saved_runs/ ./saved_runs/
```
You get one bundle per config, labelled by its `BUNDLE_SUFFIX`.

**Quota note:** each running instance consumes vCPU. On the new account the default cap is 32–64 vCPU total. 5 × c6i.4xlarge (16 vCPU each) = 80 vCPU, which will exceed the cap and fail some launches. Either request a quota increase, use smaller instances per launch, or sweep in batches.

---

## 2. Cheat sheet

### List things

```
# All thesis instances + their state
aws ec2 describe-instances --filters "Name=tag:Name,Values=thesis-*" --query 'Reservations[].Instances[].[InstanceId,State.Name,Tags[?Key==`Name`].Value|[0]]' --output table --region eu-north-1
```
```
# All previous launches in S3
aws s3 ls s3://hugo-thesis-runs/launches/
```
```
# All bundles in S3
aws s3 ls s3://hugo-thesis-runs/saved_runs/
```

### Kill things

```
# Terminate one instance (replace <id> with the instance ID)
aws ec2 terminate-instances --instance-ids <id> --region eu-north-1
```
```
# Terminate ALL running thesis instances at once
$running = (aws ec2 describe-instances --filters "Name=tag:Name,Values=thesis-*" "Name=instance-state-name,Values=running" --query 'Reservations[].Instances[].InstanceId' --output text --region eu-north-1)
if ($running) { aws ec2 terminate-instances --instance-ids $running.Split() --region eu-north-1 }
```

### Investigate

```
# Tail a specific launch's log (substitute <launch-id>)
aws s3 cp s3://hugo-thesis-runs/launches/<launch-id>/userdata.log - | Select-Object -Last 50
```
```
# Serial console output for an instance (bypasses PowerShell encoding bug)
python -c "import subprocess, json; r = subprocess.run(['aws','ec2','get-console-output','--instance-id','<instance-id>','--region','eu-north-1','--output','json'], capture_output=True); print(json.loads(r.stdout.decode('utf-8'))['Output'][-5000:])"
```

### Costs

```
# Open AWS Billing dashboard in browser
start https://console.aws.amazon.com/billing/home
```
```
# Delete old launch artifacts (does NOT touch bundles in saved_runs/)
aws s3 rm s3://hugo-thesis-runs/launches/ --recursive
```

### Compare EC2 bundle vs local bundle

```python
import numpy as np
from lifecycle.policy_io import load_policy_bundle
C_l, S_l, B_l, _, _ = load_policy_bundle('saved_runs/<local-bundle>')
C_e, S_e, B_e, _, _ = load_policy_bundle('saved_runs/<ec2-bundle>')
print('C match:', np.allclose(C_l, C_e, rtol=1e-5))
print('S match:', np.allclose(S_l, S_e, rtol=1e-5))
print('B match:', np.allclose(B_l, B_e, rtol=1e-5))
```

---

## 3. Troubleshooting

### `'charmap' codec can't encode characters in position N`
PowerShell + Python encoding mismatch on commands that print Unicode (esp. `aws ec2 get-console-output`). Use the python one-liner in Section 2 → Investigate. `chcp 65001` sometimes helps but isn't reliable.

### `An error occurred (PendingVerification) when calling the RunInstances operation`
New-account verification gate, one-time per region. Wait for the AWS approval email (minutes to ~4 hours). Re-run the launch command after.

### `User: ... is not authorized to perform: iam:...`
IAM user missing IAM permissions. Console → IAM → Users → `hugo-cli` → Add permissions → Attach `IAMFullAccess`.

### `User: ... is not authorized to perform: ssm:GetParameter`
Add `AmazonSSMReadOnlyAccess` to `hugo-cli`. The launcher uses SSM to look up the latest Amazon Linux AMI.

### `InvalidParameterCombination ... not eligible for Free Tier`
Account on AWS Free Plan. Console → Billing → Plans → "Upgrade to Paid Plan". No charge for upgrading.

### Instance launched but won't terminate (racking up cost)
The bootstrap auto-terminates on success but leaves the instance up on failure (intentional, for debugging). To check for runaways:
```
aws ec2 describe-instances --filters "Name=tag:Name,Values=thesis-*" "Name=instance-state-name,Values=running" --query 'Reservations[].Instances[].[InstanceId,Tags[?Key==`Name`].Value|[0]]' --output table --region eu-north-1
```
Then terminate. See Section 2 → Kill things for the bulk version.

### `404 NoSuchKey` on userdata.log
Bootstrap hasn't reached its first log upload yet (every 30 sec). Wait a minute. If the instance is `running` for 5+ min with still no log, it's stuck — terminate and re-launch.

### Bundle didn't appear in S3 even though instance terminated
The solve crashed before the upload step. Pull the userdata.log from the launch's S3 prefix and look for the error.

### `aws: command not found` (in Git Bash)
Bash session predates AWS CLI install. Restart bash, or:
```bash
export PATH="$PATH:/c/Program Files/Amazon/AWSCLIV2"
```

### Tarball over 50 MB / launch upload takes forever
Project has accumulated junk. Edit `TARBALL_EXCLUDES` in `scripts/launch_run.py` to skip large files. Common culprits: cached `.npz` files at project root, archive folders.

### Solve produces lots of Newton failures / max-iter messages
Likely under-resolution (too-coarse grid). Compare your config's `state_grid_sizes`, `n_wealth`, `n_state_quad_nodes` against `configs/system_iv_5x5x5.py` (which is known to converge cleanly). The smoke config is intentionally too coarse.

---

## 4. One-time setup (only if starting fresh on a new account or machine)

1. **Install AWS CLI**: `winget install Amazon.AWSCLIV2` → `aws configure` (region `eu-north-1`).
2. **IAM user `hugo-cli`**: attach `AmazonEC2FullAccess`, `AmazonS3FullAccess`, `IAMFullAccess`, `AmazonSSMReadOnlyAccess`. Generate an access key for CLI use.
3. **IAM role `thesis-ec2-runner`**: trusted entity = EC2, attached policy = `AmazonS3FullAccess`. Used as instance profile by the launcher.
4. **EC2 key pair `hugo-thesis`** in `eu-north-1`: save the `.pem` file (used for SSH if a run ever hangs and you want to debug live).
5. **Verify**:
   ```
   aws sts get-caller-identity
   aws ec2 describe-key-pairs --region eu-north-1
   aws iam get-instance-profile --instance-profile-name thesis-ec2-runner
   ```
   All three should return without error.

The S3 bucket `hugo-thesis-runs` is auto-created on first launch.

---

## 5. What each script does

Five Python scripts + two bash bootstraps. They chain together. Single-cell parallel:

```
launch_sweep.py  ─┐
                  ├─> launch_run.py  ─> ec2_userdata.sh         ─> run_solve.py
launch_run.py  ───┘   (your laptop)      (on EC2, single cell)     (on EC2)
```

Multi-cell serial (one instance runs N cells back-to-back):

```
launch_serial.py ─> ec2_userdata_serial.sh  ─> run_solve.py (×N, in series)
(your laptop)       (on EC2, loops cells)        (on EC2, each cell resumes from S3 ckpt)
```

### `scripts/_gen_sweep_main.py` — the sweep matrix

Reads the `SWEEP` matrix at the top of the file and emits one config per cell into `configs/sweep_main/`. Each emitted config imports from [configs/_canonical.py](configs/_canonical.py) and applies only the cell-specific overrides. Re-run after editing `SWEEP`.

### `configs/_canonical.py` — single source of truth

`PREDICTABILITY_SYSTEM`, `BASE_CONFIG`, `CANONICAL_DISC`, `CANONICAL_SOLVER`. Every sweep cell and the smoketest import these. Dialing here propagates everywhere on the next regen.

### `scripts/run_solve.py` — the actual solve

Runs ONE solve and saves the bundle. Reads a config `.py` file, builds model+precompute, calls `run_lifecycle_solver` with a `SolveControl` (per-age S3-checkpoint resume + writeback), calls `save_policy_bundle`.

- **Where it runs**: anywhere — your laptop OR the EC2 instance
- **Input**: a config `.py` file (positional arg)
- **Output**: bundle written to `./saved_runs/<bundle-name>/`; optionally uploaded to S3 if `--bucket` or `$S3_BUCKET` is set
- **Useful flags**: `--no-upload`, `--bundle-suffix`, `--output-root`, `--no-checkpoint`
- **Resume behavior**: when `--bucket` is set, on startup it checks `s3://<bucket>/checkpoints/<bundle-name>/`; if present, syncs locally and the solver pre-fills C/S/B from it (only ages with `solved_age_mask=True` are trusted; loop solves the remainder). Shape mismatch refuses to resume; **value-only canonical changes (state_n_stds, wealth_min, alpha cap) silently load garbage** — purge S3 checkpoints first when changing those (see Section 6.A).

You can run this directly without AWS: `python scripts/run_solve.py configs/foo.py --no-upload` — same as the notebook's solve cell.

### `scripts/launch_run.py` — single-cell EC2 launcher

Spins up ONE EC2 instance to run ONE cell. Uses `scripts/ec2_userdata.sh` as bootstrap.

- **Where it runs**: your laptop
- **What it does**:
  1. Tarballs the project (excluding heavy/junk dirs — see `TARBALL_EXCLUDES` in the file)
  2. Uploads tarball + config to `s3://<bucket>/launches/<launch-id>/`
  3. Looks up the latest Amazon Linux 2023 AMI
  4. Calls `aws ec2 run-instances` with `ec2_userdata.sh` as user-data
  5. Prints the instance ID + follow-up commands
- **Useful flags**: `--dry-run` (no AWS calls, just preview), `--instance-type`, `--key-name`

### `scripts/launch_sweep.py` — parallel sweep launcher

Fires `launch_run.py` in parallel, once per config in a directory. N cells → N instances, all running at once.

- **Where it runs**: your laptop
- **What it does**:
  1. Discovers all `*.py` files in the given directory (skips `__init__.py` and `_*.py`)
  2. Calls `launch_run.py` for each one in a thread pool (default max 10 concurrent)
  3. Collects exit codes; prints per-config success/failure
- **Output**: N independent EC2 instances, one per cell

### `scripts/launch_serial.py` — serial multi-cell launcher

Spins up ONE EC2 instance and runs N cells back-to-back, with each cell resuming from its own S3 checkpoint if any. Uses `scripts/ec2_userdata_serial.sh`. Ideal when you want a few big instances each saturating their CPUs on one cell at a time.

- **Where it runs**: your laptop
- **What it does**:
  1. Tarballs the project once
  2. Uploads tarball + each cell's config to `s3://<bucket>/launches/<launch-id>/configs/`
  3. Renders `ec2_userdata_serial.sh` with the ordered list of cell filenames
  4. Calls `aws ec2 run-instances`
  5. Prints instance ID + follow-up
- **Useful flags**: `--configs <a.py> <b.py> <c.py> …` (in execution order), `--instance-type c6i.24xlarge`, `--label lineA`

### `scripts/preflight_sweep.py` — end-to-end smoketest

Launches one cheap instance with [configs/sweep_main_smoketest.py](configs/sweep_main_smoketest.py), watches `userdata.log`, prints PASS/FAIL. Smoketest config inherits from `_canonical.py` so it exercises the exact wealth_min, state_n_stds, alpha cap, solver tuning, and pchip path that production uses.

### `scripts/setup_budget.py` — one-time budget alert

Creates / updates / deletes the `thesis-ec2-monthly-cap` AWS Budget. Sends email at 80% and 100% of cap. See Section 6.B.

### `scripts/ec2_userdata.sh` — single-cell bootstrap (runs ON the instance)

Injected by `launch_run.py` as user-data.

- **In order**: arm 24h self-destruct timer, start log uploader (every 30s) + checkpoint uploader (every 60s) sidecars, install Python 3.11+pip, download project + config, pip install, run `run_solve.py run_config.py --bucket <bucket>`, on success `shutdown -h now`, on error leave the instance up for SSH.
- **Visibility**: log at `s3://<bucket>/launches/<launch-id>/userdata.log`, refreshed every 30s.

### `scripts/ec2_userdata_serial.sh` — multi-cell serial bootstrap

Injected by `launch_serial.py`. Same sidecars as `ec2_userdata.sh`, but instead of one solve it loops through the list of cell configs:

- One cell's failure logs loudly but does NOT abort the line — the loop continues to the next cell.
- After all cells attempted, prints `successes:` / `failures:` summary then `shutdown -h now`.
- 24h self-destruct fires regardless of progress.

### Flow recap (serial mode)

```
laptop:  launch_serial.py --configs A.py B.py C.py
            ↓ tarball + upload to S3 (configs/A.py, B.py, C.py)
            ↓ aws ec2 run-instances
EC2:     ec2_userdata_serial.sh
            ↓ download project + all configs from S3
            ↓ pip install
            ↓ for cell in [A, B, C]:
            ↓   run_solve.py cell --bucket <bucket>
            ↓     ↓ checks S3 checkpoints/, resumes if present
            ↓     ↓ saves bundle to s3://bucket/saved_runs/<name>/
            ↓ shutdown -h now (auto-terminate)
laptop:  aws s3 sync s3://bucket/saved_runs/  ./saved_runs/
```

---

## 6. Production sweeps with safety

This section covers the canonical-config workflow for production sweeps: a pre-launch config check, a smoketest, S3-checkpointed crash recovery, budget + 24h-timer caps, and two launch modes (parallel single-cell and serial multi-cell).

### 6.A. Pre-launch config check (do this every time before regenerating)

**Why it matters:** [solver.py](solver.py)'s resume-from-S3-checkpoint hook only validates array **shape**. Several canonical fields can change without changing array shape:

| Field | Changes shape? | Silent-corruption risk if you forget? |
|---|---|---|
| `state_grid_sizes` | Yes (N_state) | resume refuses, safe |
| `n_z`, `n_wealth`, `n_savings` | Yes | resume refuses, safe |
| `state_n_stds` | **No** (just relocates cholesky-mode grid) | **YES — silently loads incompatible policies** |
| `wealth_min`, `wealth_max` | No (only edits grid values) | **YES** |
| `alpha_min`, `alpha_max` | No | **YES** |
| All solver tuning (`max_iter_unconstrained`, `init_alpha_*`, `use_line_search`, etc.) | No | **YES** |

If you change any "no shape" canonical value, **purge S3 checkpoints before re-launching**, otherwise the next solve resumes onto policies that are inconsistent with the new config and produces silent garbage.

**Checklist before any regen + launch:**

1. Open [configs/_canonical.py](configs/_canonical.py) and read every value. Confirm:
   - Discretization: `wealth_min`, `state_n_stds`, `n_z`, `state_grid_sizes`
   - Solver: `alpha_min`, `alpha_max`, `max_iter_unconstrained`, `tol`, `use_line_search`
   - Economics: `gamma`, `beta`, `b_bar`, all VAR/income parameters
2. Open [scripts/_gen_sweep_main.py](scripts/_gen_sweep_main.py) `SWEEP` matrix. Confirm cells override only what they intentionally vary.
3. If anything changed in canonical or in the SWEEP since the last sweep:
   ```
   # archive old bundles to v1_archive/ (preserves them in S3 with a non-conflicting prefix)
   for name in $(aws s3 ls s3://hugo-thesis-runs/saved_runs/ --region eu-north-1 | awk '{print $2}' | tr -d '/' | grep -v v1_archive); do
       aws s3 mv --recursive "s3://hugo-thesis-runs/saved_runs/${name}/" \
           "s3://hugo-thesis-runs/saved_runs/v1_archive/${name}/" --region eu-north-1 --quiet
   done

   # purge S3 checkpoints (forces fresh starts under the new canonical)
   aws s3 rm --recursive s3://hugo-thesis-runs/checkpoints/ --region eu-north-1 --quiet
   ```
4. Regenerate cell configs:
   ```
   python scripts/_gen_sweep_main.py
   ```
5. Sanity-check at least one rendered cell:
   ```
   python -c "
   import importlib.util, sys
   sys.path.insert(0, '.')
   spec = importlib.util.spec_from_file_location('cfg', 'configs/sweep_main/01_base.py')
   m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
   print(m.disc_config_template); print(m.solver_config)
   "
   ```

**Skipping any step in this checklist is how you spend a night solving the wrong problem.**

### 6.B. One-time setup: budget alert

Run once per account. Sends email warnings at 80% and 100% of a monthly spend cap.

```
python scripts/setup_budget.py --email hugo@rybergs.net --cap-usd 200
```

AWS sends a confirmation email — accept it, otherwise alerts won't fire. Update or delete:

```
python scripts/setup_budget.py --email hugo@rybergs.net --cap-usd 300
python scripts/setup_budget.py --delete
```

**Hard cap layer:** every EC2 instance launched by `launch_run.py` / `launch_serial.py` self-destructs after **24 hours** regardless of solve state (baked into `scripts/ec2_userdata.sh` and `scripts/ec2_userdata_serial.sh`). With 2 c6i.24xlarge running their 24h timer worst-case: `2 × 24 × $4.08 = $196`. With 10 c6i.4xlarge worst-case: `10 × 24 × $0.68 = $163`. Pick `--cap-usd` accordingly.

`hugo-cli` needs the inline `thesis-budgets-access` policy (granting `budgets:*` and `ce:GetCostAndUsage`); attach it once via:

```
aws iam put-user-policy --user-name hugo-cli --policy-name thesis-budgets-access \
    --policy-document file://path/to/budgets-policy.json
```

Where `budgets-policy.json` allows `budgets:*` and `ce:GetCostAndUsage` on `*`.

### 6.C. Pre-flight smoketest

Verifies the full pipeline end-to-end on a tiny config in ~5-7 min, ~$0.05. Smoketest config inherits from `configs/_canonical.py` so it exercises the **exact same** wealth_min, state_n_stds, alpha cap, solver tuning, and pchip path that production cells use.

```
python scripts/preflight_sweep.py --key-name hugo-thesis
```

Launches one `c6i.2xlarge` with [configs/sweep_main_smoketest.py](configs/sweep_main_smoketest.py) (3×3×3 state grid, n_w=40, n_z=5, otherwise canonical), polls the instance's `userdata.log`, and prints **PASS** when it sees `[run_solve] saved bundle:` or **FAIL** on `!!! ERROR`. Exit code 0 = PASS.

On FAIL, the instance is left running for SSH debug. On timeout, it auto-terminates.

### 6.D. Launch — pick a mode

Two modes for the production sweep, depending on instance count vs vCPU quota:

**Mode 1 — Parallel single-cell** ([scripts/launch_sweep.py](scripts/launch_sweep.py))

One EC2 instance per cell, launched in parallel. Best when cell count fits comfortably under the vCPU quota and you want max wall-clock parallelism.

```
python scripts/launch_sweep.py configs/sweep_main/ \
    --instance-type c6i.4xlarge --key-name hugo-thesis
```

5 cells × 16 vCPU = 80 vCPU. Wall-clock = longest single cell.

**Mode 2 — Serial multi-cell** ([scripts/launch_serial.py](scripts/launch_serial.py))

One EC2 instance runs N cells back-to-back (each at full processor saturation), then self-terminates. Best when you want each solve to use a big instance but your vCPU quota only fits a couple of them. Splits cells into "lines" — typical use is 2 instances, each running 2-4 cells.

Allocation rule we use: order cells by ascending complexity, then alternate-pick (positions 1,3,5 → line A; 2,4 → line B). This puts cheap bundles first on each line so usable output drops fast.

```
# Line A (3 cells, c6i.24xlarge, 96 vCPU)
python scripts/launch_serial.py \
    --configs configs/sweep_main/01_base.py configs/sweep_main/03_grid9_base.py configs/sweep_main/05_state44.py \
    --instance-type c6i.24xlarge --key-name hugo-thesis --label lineA

# Line B (2 cells, c6i.24xlarge, 96 vCPU)
python scripts/launch_serial.py \
    --configs configs/sweep_main/02_state33.py configs/sweep_main/04_inc55.py \
    --instance-type c6i.24xlarge --key-name hugo-thesis --label lineB
```

2 × 96 = 192 vCPU = exactly at the standard On-Demand quota. Each line's serial bash loop is in [scripts/ec2_userdata_serial.sh](scripts/ec2_userdata_serial.sh): one cell's failure logs loudly but does not block the rest of the line.

### 6.E. Crash recovery (per-age checkpoints in S3)

Every solve writes a checkpoint bundle to `./saved_runs/checkpoints/<bundle-name>/` after each solved age. A sidecar in both `ec2_userdata.sh` and `ec2_userdata_serial.sh` syncs that directory to `s3://hugo-thesis-runs/checkpoints/<bundle-name>/` every 60 seconds.

If an instance dies mid-run, re-launch the same cell — [scripts/run_solve.py](scripts/run_solve.py) checks S3 for an existing checkpoint at startup, downloads it, validates that the grid/quadrature **shape** matches the current config (refuses loudly on mismatch), and the solver pre-fills C/S/B arrays so the loop only solves the still-unsolved ages.

Resume one cell:

```
python scripts/launch_run.py configs/sweep_main/05_state44.py \
    --instance-type c6i.24xlarge --key-name hugo-thesis
```

Worst-case lost work per crash: one age.

**Reminder:** resume validates SHAPE only, not value semantics. If you changed `state_n_stds`/`wealth_min`/`alpha_*`/solver tuning since the checkpoint was written, you MUST purge S3 checkpoints first (see 6.A step 3).

### 6.F. Status checks while running

Per-line status table (works for both `launch_sweep.py` parallel and `launch_serial.py` serial):

```
# Anything still running, including instance type:
aws ec2 describe-instances --filters "Name=tag:Name,Values=thesis-*" \
    "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[].Instances[].[Tags[?Key==`Name`].Value|[0],InstanceId,InstanceType,State.Name,LaunchTime]' \
    --output table --region eu-north-1
```

Tail one line's bootstrap log (charmap-safe via Python):

```
python -X utf8 -c "
import subprocess
r = subprocess.run(['aws','s3','cp','s3://hugo-thesis-runs/launches/<LAUNCH_ID>/userdata.log','-','--region','eu-north-1'], capture_output=True)
print(r.stdout.decode('utf-8','replace')[-4000:])
"
```

(Direct `aws s3 cp ... | tail` from PowerShell can crash on the solver's box-drawing chars in cp1252; the Python wrapper avoids it.)

### 6.G. Morning / post-sweep

```
# Pull all bundles to local
aws s3 sync s3://hugo-thesis-runs/saved_runs/ ./saved_runs/

# Cost spent this month
aws ce get-cost-and-usage --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
    --granularity MONTHLY --metrics UnblendedCost \
    --query 'ResultsByTime[0].Total.UnblendedCost' --output text
```

If a cell finished, its bundle is at `s3://hugo-thesis-runs/saved_runs/<bundle-name>/`. If a cell didn't finish, its partial work is at `s3://hugo-thesis-runs/checkpoints/<bundle-name>/` — re-launch that cell to resume.

### 6.H. Budget alarm fired

Email subject `AWS Budgets: thesis-ec2-monthly-cap` arriving at ≥80% of cap:

1. List running thesis instances + ages:
   ```
   aws ec2 describe-instances --filters "Name=tag:Name,Values=thesis-*" \
       "Name=instance-state-name,Values=running" \
       --query 'Reservations[].Instances[].[InstanceId,LaunchTime,Tags[?Key==`Name`].Value|[0]]' \
       --output table --region eu-north-1
   ```
2. If anything is stuck (running long past its expected wall, or no log progress), terminate it:
   ```
   aws ec2 terminate-instances --instance-ids <id> --region eu-north-1
   ```
3. If all running instances look healthy, the alert is informational — there is no automatic kill action wired up. The 24h self-destruct timer is the hard cap.

**Bulk terminate (panic button):**

```
$running = (aws ec2 describe-instances --filters "Name=tag:Name,Values=thesis-*" \
    "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[].Instances[].InstanceId' --output text --region eu-north-1)
if ($running) { aws ec2 terminate-instances --instance-ids $running.Split() --region eu-north-1 }
```
