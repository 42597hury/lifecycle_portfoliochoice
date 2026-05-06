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
| Branch | `jax-rewrite` (HEAD `8ed9e8b` or later) |
| Instance type (trial) | `c8a.xlarge` (4 vCPU, 8 GB) — $0.05/hr, sufficient for smoke |
| Instance type (smoke + larger config later) | `c8a.4xlarge` (16 vCPU, 32 GB) — ~$0.20/hr |
| OS | Amazon Linux 2023 (or Ubuntu 24.04 — both work) |
| Region | `eu-north-1` (per existing AWS_WORKFLOW.md prereqs) |

## 2. Bootstrap script (userdata)

Drop this in as EC2 user-data. ~30 seconds setup + 5–10 min smoke wall-time on `c8a.xlarge`.

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
| Smoke runs but takes >15 min | instance has < 4 cores | bump to `c8a.4xlarge` |
| `OSError: [Errno 13] Permission denied: '/home/ec2-user/.cache'` | cache dir creation failed | `chown ec2-user:ec2-user /home/ec2-user/.cache` |

## 5. After the trial passes

Two paths forward:

**A) Larger CPU run, same branch.** Adapt `verify_smoke.py` to a bigger config (e.g., `verify_canonical_small.py` already in the repo at `n_w=40, n_s=40, n_z=5, state=(3,3,3)`). Expect ~30–50 min wall on `c8a.4xlarge`. Still no S3 upload — pull from instance via SCP if you want bundles.

**B) GPU run on `p4d.24xlarge` (A100) or `p5.48xlarge` (H100).** Same bootstrap recipe, but:
1. Set `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` (or `JAX_PLATFORMS=cuda`) before running Python.
2. Install JAX with CUDA support: `pip install -U "jax[cuda12]"` (instead of plain `jax`).
3. Verify `python -c "import jax; print(jax.devices())"` shows `[CudaDevice(id=0)]`.

Avoid consumer-grade GPUs (`g4dn`, `g5`) — their fp64 throughput is 1:32 of fp32, which makes our float64 solver unusable. Stick to A100/H100/V100.

## 6. Production sweeps

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
