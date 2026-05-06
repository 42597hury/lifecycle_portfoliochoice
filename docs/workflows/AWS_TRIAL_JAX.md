# AWS Trial Run — JAX Rewrite Branch

**Scope:** verify the JAX-rewrite code runs end-to-end on AWS. **Not** the full sweep workflow yet — that machinery (`scripts/launch_run.py`, `scripts/run_solve.py`, S3 checkpoint sync, etc.) was deleted in handoff 1 and will need re-porting before production sweeps work on this branch. See [AWS_WORKFLOW.md](AWS_WORKFLOW.md) for the main-branch sweep flow.

For now: launch one cheap instance, run `verify_smoke.py`, confirm pass.

---

## What this trial validates

- The package installs cleanly on Linux from `requirements.txt` (we've only tested on Windows).
- All four JAX kernels (terminal, retirement, work-to-retirement boundary, working) compile and execute on AWS hardware.
- The virtual-CPU pmap pattern (auto-enabled at `lifecycle/__init__.py` import time) discovers `nproc` and uses all cores.
- The 6-age tiny-config smoke produces the expected policies.

## What this trial does NOT validate

- GPU code path (need an actual A100/H100 — `p4d.24xlarge` or `p5.48xlarge`).
- Performance scaling to canonical (`n_state=343, n_w=180, n_s=180, n_z=11`). Not feasible on the trial instance type.
- S3 bundle upload, checkpoint resume, parallel sweep — none of that automation exists on `jax-rewrite` yet.

---

## 1. Branch + instance

| | Value |
|---|---|
| Repo | `https://github.com/42597hury/lifecycle_portfoliochoice` |
| Branch | `jax-rewrite` (HEAD `7f62f79` or later) |
| Primary instance (smoke + benchmark) | `hpc8a.64xlarge` — 64 vCPUs (AMD Genoa), HPC-class, matches the Numba reference's likely hardware tier |
| Fallback if HPC quota unavailable | `c8a.4xlarge` (16 vCPU, 32 GB AMD Genoa) — same arch, smaller, slower |
| Smoke-only fallback | `c8a.xlarge` (4 vCPU, 8 GB) — only OK for the 6-age smoke, will OOM on the benchmark |
| OS | Amazon Linux 2023 (or Ubuntu 24.04 — both work) |
| Region | `eu-north-1` (per existing AWS_WORKFLOW.md prereqs) |

## 2. Bootstrap script (userdata)

Drop this in as EC2 user-data. On `hpc8a.64xlarge` the smoke is bottle-necked by the JAX JIT compile, not core count — expect ~3–5 min wall regardless of instance size for the 6-age smoke. The benchmark in section 5 is where the 64 vCPUs actually pay off.

```bash
#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/jax-trial.log) 2>&1

echo "=== JAX trial bootstrap starting at $(date -Is) ==="

# 1. Install Python 3.11 + git on Amazon Linux 2023
dnf install -y python3.11 python3.11-pip git || true   # AL2023
# (For Ubuntu: apt-get update && apt-get install -y python3.11 python3.11-venv git)

# 2. Clone JAX branch
cd /home/ec2-user
sudo -u ec2-user git clone -b jax-rewrite \
    https://github.com/42597hury/lifecycle_portfoliochoice.git
cd lifecycle_portfoliochoice

# 3. Venv + deps
sudo -u ec2-user python3.11 -m venv venv
sudo -u ec2-user ./venv/bin/pip install --upgrade pip
sudo -u ec2-user ./venv/bin/pip install -r requirements.txt

# 4. Run smoke
echo "=== running verify_smoke.py at $(date -Is) ==="
sudo -u ec2-user ./venv/bin/python verify_smoke.py
SMOKE_EXIT=$?

echo "=== verify_smoke.py exited with $SMOKE_EXIT at $(date -Is) ==="

# 5. (Optional) upload log to S3 if you have a bucket configured
# aws s3 cp /var/log/jax-trial.log s3://hugo-thesis-runs/jax-trial/$(date +%Y%m%dT%H%M%S).log

# 6. Auto-terminate on success (comment out if you want to SSH in afterwards)
if [ "$SMOKE_EXIT" -eq 0 ]; then
    echo "=== smoke PASSED, shutting down ==="
    shutdown -h now
fi
```

**For the agent that handles AWS:** if you don't use userdata, the equivalent is to SSH in and run the bash blocks 1–4 manually.

## 3. Success criteria

In `/var/log/jax-trial.log` (or stdout if running interactively), look for these exact lines near the end:

```
  Status: complete  (6/6 ages solved)
  Policy sanity: PASS  (no NaN/Inf in solved ages)
  alpha_s range: [-1.038, 3.082]
  alpha_b range: [-8.996, 9.718]
```

The alpha ranges are bit-identical (to ~1e-12) to a Windows-laptop run on the same branch HEAD — same JAX kernels, same float64, same RNG path.

**If you see:**
- `Status: complete` + `PASS` + the ranges above → ✅ trial passes, terminate the instance
- `NaN`/`Inf` reported, or alpha ranges materially different → real problem, capture full log + report
- ImportError, ModuleNotFoundError, traceback → setup issue (Python/JAX wheel resolution); capture log + report

## 4. Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `pip` resolves to JAX 0.4.x with PCIe warnings | older JAX wheel for older Python | use Python 3.11+; we developed on 3.13 |
| `ModuleNotFoundError: jax` | JAX missing from `requirements.txt` | already fixed in commit `8ed9e8b`; check branch is up to date |
| `aarch64` wheel issues | running on Graviton (ARM) | switch to AMD64 instance (c8a is AMD64; c8g is Graviton — avoid) |
| Smoke runs but takes >15 min | instance has < 4 cores | bump to `c8a.4xlarge` or `hpc8a.64xlarge` |
| `OSError: [Errno 13] Permission denied: '/home/ec2-user/.cache'` | cache dir creation failed | `chown ec2-user:ec2-user /home/ec2-user/.cache` |

## 5. Benchmark mode — match a specific Numba bundle

To benchmark JAX wall time against a Numba reference run, use `verify_benchmark_bundle.py` instead of `verify_smoke.py`. It mirrors `configs/run_ccv_wide9_gh_k4.py` (the most recent retirement-only bundle on `main`):

- `state_grid_sizes=(9,9,9)`, `n_z=11`, `n_w=180`, `n_s=180`
- `n_ret_nodes_1d=(3,5,5)`, `n_state_quad_nodes=(3,4,4)`
- `youngest_age_to_solve=67` (retirement-only, 33 ages)
- **Numba reference: 1341.9s = 22.4 min** (likely on a 16-vCPU AMD/Intel)

**Recommended instance for benchmark:** `hpc8a.64xlarge` (64 vCPU AMD Genoa, HPC-class). The Numba reference 22.4 min was almost certainly on a similar tier; running JAX on the same tier gives a fair head-to-head. Fallback: `c8a.4xlarge` (16 vCPU) if HPC quota isn't approved — JAX will be slower than on hpc8a in proportion to core count, roughly 4× longer wall.

Do NOT run the benchmark on `c8a.xlarge` — the precompute + per-cell c_corners gather will OOM at 8 GB.

**Replace step 4 of the bootstrap script** with:

```bash
echo "=== running verify_benchmark_bundle.py at $(date -Is) ==="
sudo -u ec2-user ./venv/bin/python verify_benchmark_bundle.py
```

**Expected wall on `hpc8a.64xlarge`:**

- ~5–10 min JIT compile cost (one-time per kernel; persistent cache is empty on first run).
- ~10–60 min compute for 33 retirement ages, depending on how well XLA saturates 64 vCPUs.
- **Total: 15 min – 1 hour.** The smoke showed JAX is ~7–40× slower than Numba on small configs (FOC dispatch overhead dominates), but at 9×9×9 dispatch amortizes and the per-cell compute is the bottleneck — should narrow toward 1–2× Numba speed at this config size. With 64 vCPUs vs the Numba reference's likely 16 vCPUs, JAX could end up *faster* despite per-vCPU overhead.

**Hard timeout:** if the run exceeds **2 hours**, terminate the instance and report. That signals JAX-on-CPU at this size is uneconomical and motivates the GPU port.

**On `c8a.4xlarge` fallback (16 vCPUs):** scale the expected wall up by ~4×, hard timeout 4 hours.

The script prints clearly:

```
=== BENCHMARK RESULTS ===
  Wall time      : XXXX.Xs = XX.XX min
  Numba reference: 1341.9s = 22.36 min
  JAX is X.XXx [FASTER|SLOWER] than Numba
  NaN check (solved ages only): C=0  S=0  B=0
```

Capture that block + the per-age progress lines for the report.

## 6. After the trial / benchmark passes

Two paths forward:

**A) Larger CPU run, same branch.** Adapt `verify_smoke.py` to a bigger config (e.g., `verify_canonical_small.py` already in the repo at `n_w=40, n_s=40, n_z=5, state=(3,3,3)`). Expect ~10–20 min wall on `hpc8a.64xlarge`, ~30–50 min on `c8a.4xlarge`. Still no S3 upload — pull from instance via SCP if you want bundles.

**B) GPU run on `p4d.24xlarge` (A100) or `p5.48xlarge` (H100).** Same bootstrap recipe, but:
1. Set `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` (or `JAX_PLATFORMS=cuda`) before running Python.
2. Install JAX with CUDA support: `pip install -U "jax[cuda12]"` (instead of plain `jax`).
3. Verify `python -c "import jax; print(jax.devices())"` shows `[CudaDevice(id=0)]`.

Avoid consumer-grade GPUs (`g4dn`, `g5`) — their fp64 throughput is 1:32 of fp32, which makes our float64 solver unusable. Stick to A100/H100/V100.

## 7. Production sweeps

Not yet supported on `jax-rewrite`. To get the existing AWS_WORKFLOW.md sweep machinery working on JAX, the following needs porting (not done):

- `scripts/launch_run.py` — single-cell EC2 launcher
- `scripts/run_solve.py` — the actual solve harness with S3 bundle upload + checkpoint resume
- `scripts/launch_sweep.py` — parallel sweep orchestrator
- `scripts/preflight_sweep.py` — smoketest harness
- `scripts/_gen_sweep_main.py` — sweep matrix generator
- `scripts/ec2_userdata*.sh` — EC2 bootstrap templates

The Numba versions on `main` are good references; main differences for the JAX port:
- `pip install -r requirements.txt` already pulls JAX (after commit `8ed9e8b`).
- For GPU instances, set `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` in user-data.
- Use `pip install -U "jax[cuda12]"` if running on GPU.
- The solve loop itself (`run_lifecycle_solver`) has the same API, so `run_solve.py` mostly translates 1:1.

Estimated porting effort: 1–2 days. Until then, do single trial runs as in this doc.
