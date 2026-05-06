# AWS Trial Run — JAX Rewrite Branch

**Scope:** verify the JAX-rewrite code runs end-to-end on AWS. **Not** the full sweep workflow yet — that machinery (`scripts/launch_run.py`, `scripts/run_solve.py`, S3 checkpoint sync, etc.) was deleted in handoff 1 and will need re-porting before production sweeps work on this branch. See [AWS_WORKFLOW.md](AWS_WORKFLOW.md) for the main-branch sweep flow.

For now: launch one cheap instance, run `verify_smoke.py`, confirm pass.

---

## What this trial validates

- The package installs cleanly on Linux from `requirements.txt` (we've only tested on Windows).
- All four JAX kernels (terminal, retirement, work-to-retirement boundary, working) compile and execute on AWS hardware.
- The virtual-CPU pmap pattern (auto-enabled at `lifecycle/__init__.py` import time) discovers `nproc` and uses all cores.
- The 6-age tiny-config smoke produces the expected policies.

The JAX persistent compilation cache is enabled by default (status line printed at import time). With the S3 sync step in §2 below, second-and-later launches skip the cold JIT compile (~5-15 min on hpc8a) and start solving in seconds.

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

# 3b. Configure JAX compilation cache (cross-launch reuse via S3)
# Hardware-keyed prefix: keep CPU and GPU caches separate (JAX hashes the
# trace key with hardware identity — a hpc8a CPU cache will NOT hit on
# p4d/p5 GPUs). Bump the v1 → v2 segment after a JAX upgrade.
export LIFECYCLE_JAX_CACHE_DIR=/home/ec2-user/.cache/jax_lifecycle
export S3_CACHE_BUCKET=hugo-thesis-runs       # set to "" to disable cross-launch reuse
export S3_CACHE_PREFIX=jax-cache/v1/cpu-hpc8a  # e.g. cpu-hpc8a, gpu-a100, gpu-h100

if [ -n "${S3_CACHE_BUCKET:-}" ]; then
    sudo -u ec2-user mkdir -p "$LIFECYCLE_JAX_CACHE_DIR"
    sudo -u ec2-user aws s3 sync \
        "s3://${S3_CACHE_BUCKET}/${S3_CACHE_PREFIX}/" \
        "$LIFECYCLE_JAX_CACHE_DIR" \
        --region eu-north-1 || echo "cache pull failed; continuing"
fi

# 4. Run smoke
echo "=== running verify_smoke.py at $(date -Is) ==="
sudo -u ec2-user --preserve-env=LIFECYCLE_JAX_CACHE_DIR \
    ./venv/bin/python verify_smoke.py
SMOKE_EXIT=$?

echo "=== verify_smoke.py exited with $SMOKE_EXIT at $(date -Is) ==="

# 5. (Optional) upload log to S3 if you have a bucket configured
# aws s3 cp /var/log/jax-trial.log s3://hugo-thesis-runs/jax-trial/$(date +%Y%m%dT%H%M%S).log

# 6. Push cache back to S3 then auto-terminate
if [ "$SMOKE_EXIT" -eq 0 ] && [ -n "${S3_CACHE_BUCKET:-}" ]; then
    sudo -u ec2-user aws s3 sync \
        "$LIFECYCLE_JAX_CACHE_DIR" \
        "s3://${S3_CACHE_BUCKET}/${S3_CACHE_PREFIX}/" \
        --size-only --region eu-north-1 || echo "cache push failed"
fi

if [ "$SMOKE_EXIT" -eq 0 ]; then
    echo "=== smoke PASSED, shutting down ==="
    shutdown -h now
fi
```

**Cache env vars** (read by `lifecycle/__init__.py`):

| Var | Default | Meaning |
|---|---|---|
| `LIFECYCLE_JAX_CACHE_DIR` | `~/.cache/jax_lifecycle` | Cache directory. Set to `""` to disable. |
| `LIFECYCLE_JAX_CACHE_MIN_COMPILE_SECS` | `1.0` | Only persist traces with compile time ≥ this. |
| `LIFECYCLE_JAX_CACHE_MAX_SIZE_BYTES` | `10737418240` (10 GB) | Total cache size cap. `-1` = unlimited. |

**Hardware-keyed S3 prefix**: JAX namespaces cache entries by hardware identity, so a CPU cache will not hit on GPU and vice versa. Use distinct `S3_CACHE_PREFIX` values per hardware tier (`cpu-hpc8a`, `gpu-a100`, `gpu-h100`) to avoid mixing them. Bump the version segment (`v1` → `v2`) on JAX upgrades to discard stale entries.

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

### 6.A) Same branch, larger CPU run

Adapt `verify_smoke.py` to a bigger config or use `verify_canonical_small.py`
(already in repo at `n_w=40, n_s=40, n_z=5, state=(3,3,3)`). Same `c8a.4xlarge`
or `hpc8a.64xlarge` instance, no recipe change needed. Expect ~10–20 min wall
on `hpc8a.64xlarge`, ~30–50 min on `c8a.4xlarge`. Still no S3 upload — pull
from instance via SCP if you want bundles.

### 6.B) GPU run on `p4d.24xlarge` (A100) or `p5.48xlarge` (H100)

**Instance:** start with `p4d.24xlarge` (8× A100 40 GB). Single-GPU operation
initially — the codebase doesn't yet shard across multiple GPUs, so the other
seven A100s sit idle. (Multi-GPU is the work tracked in
[HANDOFF_PMAP_TO_VMAP.md](../handoff/HANDOFF_PMAP_TO_VMAP.md).)

Avoid consumer-grade GPUs (`g4dn`, `g5`) — their fp64 throughput is 1:32 of
fp32, which makes our float64 solver unusable. Stick to A100/H100/V100. The
one exception: `g6.xlarge` (1× L4) is fp64-acceptable for a smoke bring-up
only (NOT canonical); fp64 on L4 is 1:64.

**AMI:** prefer **AWS Deep Learning AMI GPU PyTorch 2.x (Amazon Linux 2023)**.
It ships with the NVIDIA driver, CUDA 12, and cuDNN already configured.
A bare AL2023 AMI works but adds 20–30 min of driver install — skip unless
you have a reason.

**Userdata:**

```bash
#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/jax-gpu-trial.log) 2>&1

echo "=== JAX GPU trial bootstrap starting at $(date -Is) ==="

# 0. Confirm GPU is visible to the OS
nvidia-smi || { echo "FATAL: nvidia-smi failed; AMI lacks NVIDIA driver"; exit 1; }

# 1. Python 3.11 + git
dnf install -y python3.11 python3.11-pip git || true

# 2. Clone branch
cd /home/ec2-user
sudo -u ec2-user git clone -b jax-rewrite \
    https://github.com/42597hury/lifecycle_portfoliochoice.git
cd lifecycle_portfoliochoice

# 3. Venv + GPU deps. Note: requirements-gpu.txt pulls jax[cuda12].
sudo -u ec2-user python3.11 -m venv venv
sudo -u ec2-user ./venv/bin/pip install --upgrade pip
sudo -u ec2-user ./venv/bin/pip install -r requirements-gpu.txt

# 4. CRITICAL env vars — disable virtual-CPU XLA flag, force CUDA platform
export LIFECYCLE_DISABLE_VIRTUAL_CPUS=1
export JAX_PLATFORMS=cuda
# Persist into the ec2-user env so the venv python sees them
sudo -u ec2-user bash -c 'cat >> ~/.bashrc <<EOF
export LIFECYCLE_DISABLE_VIRTUAL_CPUS=1
export JAX_PLATFORMS=cuda
EOF'

# 5. Sanity check: confirm JAX sees the GPU before the heavy work
sudo -u ec2-user -E ./venv/bin/python -c "
import jax
devs = jax.devices()
plats = sorted({d.platform for d in devs})
print('Devices:', devs)
print('Platforms:', plats)
assert plats == ['cuda'], f'Expected CUDA only, got {plats}'
"

# 6. Run smoke
echo "=== running verify_smoke.py at $(date -Is) ==="
sudo -u ec2-user -E ./venv/bin/python verify_smoke.py
SMOKE_EXIT=$?

echo "=== verify_smoke.py exited with $SMOKE_EXIT at $(date -Is) ==="

# 7. Auto-terminate on success
if [ "$SMOKE_EXIT" -eq 0 ]; then
    echo "=== smoke PASSED, shutting down ==="
    shutdown -h now
fi
```

**Success criteria** (in `/var/log/jax-gpu-trial.log`):

```
[lifecycle] JAX runtime: 1 device(s), platform(s)=['cuda']
...
Devices: [CudaDevice(id=0)]
Platforms: ['cuda']
...
  Status: complete  (6/6 ages solved)
  Policy sanity: PASS  (no NaN/Inf in solved ages)
```

The CUDA-platform line is the new gate — if it says `['cpu']` despite the env
vars, the wheel install fell through to CPU. Investigate `pip list | grep jax`
on the instance: should show `jax-cuda12-pjrt` and `jax-cuda12-plugin`
alongside plain `jax`.

**Common GPU-specific failure modes:**

| Symptom | Cause | Fix |
|---|---|---|
| `[lifecycle] WARNING: GPU env hints set ... but JAX reports CPU only` | Wheel install fell through to CPU | Confirm `requirements-gpu.txt` was used, not `requirements.txt` |
| `nvidia-smi: command not found` | AMI without NVIDIA driver | Use Deep Learning AMI, or add CUDA toolkit install step |
| `RuntimeError: CUDA runtime version mismatch` | jax[cuda12] expects CUDA 12, AMI has CUDA 11 | Use AMI shipped with CUDA 12; do NOT downgrade JAX to a CUDA 11 build |
| OOM on solve at 9×9×9 | Single A100's 40 GB HBM | Switch to A100 80 GB (`p4de.24xlarge`) or H100 (`p5.48xlarge`); profile memory before bumping config |

**Hard timeout:** if the smoke exceeds 30 min on a GPU instance, terminate
and report — JIT pathology, not normal. (CPU smoke takes ~20 min and is
JIT-bound; GPU JIT should be faster, not slower.)

### 6.C) Production sweeps

Not yet supported on `jax-rewrite`. See §7.

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
