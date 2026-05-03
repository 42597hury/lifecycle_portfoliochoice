# HANDOFF: AWS Parallel Sweep with Safety Guarantees

## TL;DR

User wants to fire 10 EC2 instances in parallel tonight, run the lifecycle solver on 10 different configs, and download bundles tomorrow. The basic pipeline already works and is validated. **Your job is to harden it for unattended overnight execution**: add checkpointing so interrupted runs don't lose hours of work, enforce a $100 spend cap, and design a 10-minute pre-flight that proves the run is bug-free before the user goes to bed.

Do NOT redesign the AWS infrastructure — it works. Augment it.

---

## 1. Goal / success criteria

User goes to bed at time T, wakes at T+8h, and:

1. All 10 bundles are sitting in `s3://hugo-thesis-runs/saved_runs/` ready to `aws s3 sync` locally.
2. Total spend ≤ $100 (estimate is $43 — buffer is for runaway-instance protection).
3. If any single instance died mid-run (network blip, spot reclamation, OOM), the user can re-launch it and it resumes from the last age-checkpoint instead of starting from age 99 again.
4. Before going to bed, the user has positive evidence the sweep is actually running cleanly (not silently failing on age 1).

These are AND conditions, not "nice to have".

---

## 2. Current state — what already exists and works

### Validated infrastructure

- AWS account `345692660704`, region `eu-north-1`, bucket `hugo-thesis-runs`
- IAM role `thesis-ec2-runner` (instance profile, S3 write access)
- EC2 key pair `hugo-thesis`
- Standard On-Demand vCPU quota: **192** (just approved). HPC quota: pending, not relevant for this sweep.
- A single EC2 5×5×5 solve was run today and **bit-exact-matched** a fresh local resolve (max abs diff ~1e-14). EC2 numerics are validated.

### Existing scripts (DO NOT rewrite — augment)

| File | Role |
|---|---|
| [scripts/run_solve.py](../scripts/run_solve.py) | Reads a config `.py`, runs `run_lifecycle_solver`, calls `save_policy_bundle`, optionally uploads bundle to S3. Runs on Windows OR EC2. |
| [scripts/launch_run.py](../scripts/launch_run.py) | Local launcher: tarballs the project, uploads tarball + config to S3, calls `aws ec2 run-instances` with user-data. |
| [scripts/launch_sweep.py](../scripts/launch_sweep.py) | Calls `launch_run.py` once per config in a folder, in parallel via threadpool. |
| [scripts/launch_queue.py](../scripts/launch_queue.py) | Same as `launch_sweep.py` but with bounded parallelism — dispatches new instances as old ones finish. |
| [scripts/ec2_userdata.sh](../scripts/ec2_userdata.sh) | Bash bootstrap that runs as root on the EC2 instance: installs deps, downloads project + config, runs `run_solve.py`, shutdown -h on success. |
| [scripts/_gen_sweep_main.py](../scripts/_gen_sweep_main.py) | Generator for the 10 sweep configs. |

### The 10 configs (already generated)

[configs/sweep_main/](../configs/sweep_main/) contains `01_base.py` through `10_tight_cap.py`. Each varies the discretization (state grid, quadrature nodes, alpha cap) of the same underlying base config (gamma=5, beta=0.96, unconstrained). Estimated compute:

| File | Grid | Est. hr |
|---|---|---|
| 01_base | 7×7×7 | 2 |
| 02_cap_only | 7×7×7 | 2 |
| 03_state33 | 7×7×7 | 5 |
| 04_state33_cap | 7×7×7 | 5 |
| 05_state44_cap | 7×7×7 | 8 |
| 06_inc55_cap | 7×7×7 | 7 |
| 07_mid_rich_cap | 7×7×7 | **12** |
| 08_grid9_base | 9×9×9 | 5 |
| 09_grid9_state33_cap | 9×9×9 | **12** |
| 10_tight_cap | 7×7×7 | 5 |

Total compute ~63 hr, longest single ~12 hr, parallel wall clock ~12 hr.

Cost on `c6i.4xlarge` ($0.68/hr): 63 × $0.68 = **~$43 expected**, $100 cap leaves buffer.

### User-facing docs already in place

- [AWS_WORKFLOW.md](../AWS_WORKFLOW.md) — daily-workflow runbook the user should keep using day-to-day. Describes the existing scripts.
- [policy_io.py](../policy_io.py) — `save_policy_bundle()` and `load_policy_bundle()` are the canonical bundle format. Don't break it.

### Solver capability that matters for checkpointing

**Already in the codebase**: [model.py:175](../model.py#L175) defines `SolveControl(NamedTuple)` with these fields relevant to crash safety:

```python
class SolveControl(NamedTuple):
    youngest_age_to_solve: int | None = None
    checkpoint_path: str | None = None         # ← USE THIS
    checkpoint_every_n_ages: int | None = None # ← USE THIS
    save_on_interrupt: bool = False
    return_partial_on_interrupt: bool = False
    progress_wealth_source: str = "scf_median"
```

[solver.py:2747](../solver.py#L2747) — `run_lifecycle_solver(model, pc, solver_config=None, n_s_points=None, verbose=1, solve_control=None)`. Pass a `SolveControl` to enable checkpointing. Read [solver.py:2551](../solver.py#L2551) onward (`_normalize_solve_control`) for the full contract — there's existing logic for resuming from a checkpoint file.

This is the foundation for safety property #3. Your job is to wire it through `run_solve.py` and the EC2 bootstrap so that:
- Each instance writes per-age checkpoints to S3 (or to its EBS volume which then syncs to S3)
- A re-launched instance with the same config picks up where the dead one left off

---

## 3. The ask — what to design and build

### A. Per-age checkpoint to S3

Modify `run_solve.py` (or wrap it) so that when run on EC2, the solver:

1. Writes a checkpoint file every N ages (suggest N=1, since each age takes ~10 min on 7×7×7 — small overhead, max 10 min lost work)
2. Periodically syncs the checkpoint file to `s3://hugo-thesis-runs/checkpoints/<bundle-name>/`
3. On startup, if a matching checkpoint exists in S3, downloads it and resumes

The checkpoint mechanism in `solver.py` already serializes partial state. **Don't reinvent it** — just use `SolveControl(checkpoint_path=..., checkpoint_every_n_ages=1)` and add an S3 sync sidecar.

Decision point: instance-side periodic upload (e.g., every 5 min, similar to the existing `userdata.log` uploader at [ec2_userdata.sh](../scripts/ec2_userdata.sh)) vs. solver-callback upload. Pick whichever is simpler.

### B. Cost cap enforcement

Two layers:

1. **AWS Budget alert at $80** (warning) and **$100 hard cap** (action: trigger SNS → Lambda → terminate all running thesis instances). The hard cap is the real safety net.
2. **Per-instance max runtime**: kill any instance running > 16 hours (4-hour buffer above the longest config's 12-hour estimate). Already has a hook in [scripts/launch_queue.py](../scripts/launch_queue.py) (`--max-runtime-min`).

For Layer 1, design needs to decide:
- AWS Budgets API call from `launch_sweep.py` to set up the budget on first run, OR
- Manual one-time setup documented in [AWS_WORKFLOW.md](../AWS_WORKFLOW.md)

The user has IAM permissions for both Budgets and Lambda; whichever is simpler for the agent to implement is fine.

### C. Pre-flight verification before sleep

User explicitly does NOT want to go to bed and find out at 7am that all 10 instances crashed at age 99 due to a typo. Design a pre-flight that:

1. Launches ONE small EC2 instance with the smallest config (probably `01_base.py` with a forced-tiny override, or write a dedicated `configs/sweep_main_smoketest.py`)
2. Waits up to 10 minutes
3. Verifies: instance booted, pip install succeeded, solver started, at least 1–2 ages were solved successfully (look for the per-age log line in `userdata.log`)
4. Reports PASS/FAIL clearly
5. On PASS: prints "OK to launch the full 10-config sweep with launch_sweep.py"
6. On FAIL: prints the relevant log excerpt and exits non-zero

This should be a script the user runs before launching the real sweep. Suggest filename `scripts/preflight_sweep.py`.

DO NOT make this a full smoke test that runs the actual config end-to-end — it should be the cheapest possible check that the bootstrap + numpy/scipy/numba install + first age of solving all work. ~$0.05 budget.

### D. Update AWS_WORKFLOW.md

Add a "Section 6: Overnight sweeps with safety" or similar, documenting:
- How to run the pre-flight
- How the checkpointing works (where checkpoints live, how resume works)
- How to interpret the budget alarm if it fires
- Recovery procedure: which command resumes a dead instance

---

## 4. Safety requirements (explicitly user-stated)

In the user's words:
> Needs to be safe for interrupted runs, so saves need to happen at every age?

Yes. Per-age checkpoint to S3. Re-launch resumes from latest checkpoint.

> Needs to be safe money wise, i dont want to spend more than 100 dollars for this.

Hard cap via AWS Budgets + per-instance max runtime. Both layers required.

> Need to know that the code runs before i go to bed and there wasnt any bugs

Pre-flight script that proves end-to-end pipeline works for at least one config before the big launch.

---

## 5. Relevant resources

### AWS account specifics (DO NOT change these)

- Account: `345692660704`
- Region: `eu-north-1`
- Bucket: `hugo-thesis-runs`
- IAM role for instances: `thesis-ec2-runner`
- Key pair: `hugo-thesis`
- Local CLI user: `hugo-cli` with EC2/S3/IAM/SSM full access

### Code paths to read first

1. [scripts/ec2_userdata.sh](../scripts/ec2_userdata.sh) — understand the existing bootstrap, including the periodic log uploader pattern (good template for periodic checkpoint uploads)
2. [solver.py:2551-2630](../solver.py#L2551) — `_normalize_solve_control` and the surrounding logic show exactly what `SolveControl` does and how checkpoints are loaded/saved
3. [scripts/run_solve.py](../scripts/run_solve.py) — understand how a config is loaded and `run_lifecycle_solver` is called
4. [policy_io.py](../policy_io.py) — bundle format. Checkpoint files are a different file, but reuse this when saving the final bundle
5. [AWS_WORKFLOW.md](../AWS_WORKFLOW.md) Section 5 — "What each script does" — pipeline overview

### Existing similar handoffs

The folder `handoff/` contains prior implementation handoffs. Match the style: clear scoping, code references with file:line, design decisions called out.

---

## 6. Open design questions

The agent should resolve these and document the choices:

1. **Checkpoint storage path** — `s3://hugo-thesis-runs/checkpoints/<bundle-name>/checkpoint.npz`? Or include the launch_id? Trade-off: `<bundle-name>` lets a re-launch find the checkpoint automatically; `<launch-id>` avoids two concurrent runs of the same config stomping each other.

2. **Resume detection logic** — `run_solve.py` should check if a checkpoint exists at startup. If yes, validate it matches the current config (same grid/quadrature) before resuming. If config changed since the checkpoint was written, refuse to resume — fail loudly.

3. **Cost-cap implementation** — AWS Budgets API + Lambda terminate vs. simpler "all instances tagged and a separate watchdog script the user runs in another terminal". Latter is dumb but might be more reliable for a one-night sweep.

4. **Pre-flight scope** — does it need to fully validate ONE 7×7×7 config, or is solving the first 2 ages of a TINY config sufficient? Recommend the second; arguments for either welcome.

5. **What if a checkpoint is corrupt?** — fail loudly, don't auto-skip. User needs to know.

6. **Budget alert noisy?** — If alerts fire on $80 (during a $43-expected sweep), that's nuisance. Design alert thresholds so they only trigger on actual runaway, not normal completion. Suggest: $80 warning, $100 hard.

---

## 7. Out of scope

- Don't add Spot instances. User's solver doesn't have native checkpointing into the solver loop (we're adding it externally), and Spot would compound that risk for a 12-hour run.
- Don't switch instance types per-config to optimize. c6i.4xlarge is the chosen sweet spot per prior conversation.
- Don't refactor `policy_io.py` or `solver.py` significantly. The existing checkpoint hook points are sufficient.
- Don't build a web dashboard or notification system beyond the AWS Budget alert.

---

## 8. Definition of done

A user can:

```
# 1. Run pre-flight (~10 min, ~$0.05)
python scripts/preflight_sweep.py
# -> PASS

# 2. Launch the real sweep
python scripts/launch_sweep.py configs/sweep_main/ --instance-type c6i.4xlarge --key-name hugo-thesis
# -> 10 instances launched, total spend will be ~$43 if all goes well

# 3. Sleep

# 4. Wake up, pull bundles
aws s3 sync s3://hugo-thesis-runs/saved_runs/ ./saved_runs/
# -> 10 bundles ready

# In the worst case (instance N died at hour 6 of 12):
# - Total spend was bounded by AWS Budget alarm at $100
# - Re-launching that one config picks up from where it died
python scripts/launch_run.py configs/sweep_main/07_mid_rich_cap.py --instance-type c6i.4xlarge --key-name hugo-thesis
# -> finds checkpoint in S3, resumes from last age, completes
```

When all four steps work as described, the handoff is complete.
