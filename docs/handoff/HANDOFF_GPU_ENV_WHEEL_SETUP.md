# Handoff: GPU Environment & Wheel Setup

**Branch:** `jax-rewrite`
**Status when this doc was written:** the codebase auto-configures a virtual-CPU-device XLA backend at import time ([lifecycle/__init__.py:31-67](../../lifecycle/__init__.py#L31-L67)). This is correct for CPU runs and self-disables when `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` or `JAX_PLATFORMS` excludes `cpu`. The package itself is GPU-ready — the work in this handoff is **deployment plumbing**: get a GPU EC2 instance to actually use its GPU, and add a runtime self-check that fails loud if it doesn't.

**Target deployment:** AWS p4d.24xlarge (8× A100 40GB) or p5.48xlarge (8× H100 80GB), single-GPU operation initially. Multi-GPU sharding is a follow-up.

**Effort:** ~30-45 min including verification on a real GPU instance.

---

## 1. Goal

Make GPU launches "just work" with three small, durable pieces:

1. **A pinned GPU wheel** in a separate requirements file or install step. The current `requirements.txt` has plain `jax>=0.4.30`, which on Linux installs the CPU-only XLA backend.
2. **Documented env vars** that EC2 user-data must set before any Python invocation. Already half-documented in [AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md); needs to be authoritative and version-pinned.
3. **A startup self-check** in [lifecycle/__init__.py](../../lifecycle/__init__.py) that prints the active platform and warns loudly if a CPU device is reported on a host where the user clearly intended GPU. Catches the silent-fallback failure mode where a GPU instance ends up running on CPU because of a missed env var or wheel.

---

## 2. Scope and non-goals

### In scope

- Add `requirements-gpu.txt` (a thin overlay that pins `jax[cuda12]>=0.4.30` and pulls in CUDA plugins).
- Add a `_check_runtime_platform()` function in [lifecycle/__init__.py](../../lifecycle/__init__.py) that prints platform + device count once at import.
- Update [AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md) §6 ("After the trial passes" → "B) GPU run") with an end-to-end userdata script for p4d/p5 that includes the right env vars.

### Out of scope

- Multi-GPU sharding (`jax.sharding`, `pmap` replacement) — covered by [HANDOFF_PMAP_TO_VMAP.md](HANDOFF_PMAP_TO_VMAP.md).
- TPU support. The current code targets CPU and CUDA only.
- AMD ROCm support. JAX's ROCm wheels exist but the AWS use case is NVIDIA.
- Container images (Docker/EFA/etc). Bare-metal pip install on Amazon Linux 2023 is the path.
- Driver / CUDA runtime install on the host. Modern AWS Deep Learning AMIs ship with NVIDIA drivers + CUDA 12 already; if the agent is launching a fresh AMI without DLAMI, escalate to the user before adding driver-install steps.

---

## 3. Implementation

### 3.1 `requirements-gpu.txt`

Create at repo root:

```
# requirements-gpu.txt
# Overlay for GPU runs. Use INSTEAD of requirements.txt's plain `jax`:
#     pip install -r requirements-gpu.txt
#
# This pulls the CUDA-compiled XLA backend. Plain `jax` from requirements.txt
# would silently install the CPU-only backend even on a GPU instance.

-r requirements.txt
jax[cuda12]>=0.4.30
```

The trick: `-r requirements.txt` includes everything from the base file. The `jax[cuda12]` line **upgrades** the plain `jax` install to the CUDA variant (pulls `jax-cuda12-pjrt` and `jax-cuda12-plugin`). pip's resolver handles this correctly — verified pattern.

**Then** edit [requirements.txt](../../requirements.txt) to swap the loose pin for an exact baseline:

- Find: `jax>=0.4.30` (or wherever the current pin lives — check via `grep -n jax requirements.txt`).
- No change required to that line; the GPU overlay just augments it.

### 3.2 Runtime platform self-check

In [lifecycle/__init__.py](../../lifecycle/__init__.py), add this function after the existing `_jax.config.update("jax_compilation_cache_dir", ...)` block (i.e. after the existing cache config and before any solver-side imports — see line 74):

```python
def _check_runtime_platform():
    """Print active JAX platform once at import. Warns if running on CPU when
    the env signals GPU intent (presence of CUDA-related env vars).

    Why this exists: a GPU instance can silently fall through to CPU if
    (a) plain `jax` was installed instead of `jax[cuda12]`, or
    (b) ``XLA_FLAGS=--xla_force_host_platform_device_count=...`` was set
        somewhere up the stack and we didn't catch it. Both produce a
        runnable but slow workload that AWS bills full GPU rates for.
    """
    devices = _jax.devices()
    platforms = sorted({d.platform for d in devices})
    summary = f"{len(devices)} device(s), platform(s)={platforms}"
    print(f"[lifecycle] JAX runtime: {summary}", flush=True)

    cpu_only = platforms == ["cpu"]
    gpu_intent = any(
        _os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "JAX_PLATFORMS",
        )
    ) or _os.environ.get("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "").lower() in ("1", "true", "yes")

    if cpu_only and gpu_intent:
        print(
            "[lifecycle] WARNING: GPU env hints set "
            "(CUDA_VISIBLE_DEVICES / JAX_PLATFORMS / LIFECYCLE_DISABLE_VIRTUAL_CPUS) "
            "but JAX reports CPU only. "
            "Likely causes: (1) `pip install jax` instead of `pip install jax[cuda12]`; "
            "(2) NVIDIA driver / CUDA runtime not present; "
            "(3) `nvidia-smi` would fail. "
            "Run `python -c \"import jax; print(jax.devices())\"` to confirm.",
            flush=True,
        )

_check_runtime_platform()
```

**Design notes:**
- The check runs at import time, after JAX has been imported and configured. By then `jax.devices()` reports the actual platform that XLA selected.
- The "warning" is informational only — it does not raise. We don't want to break legitimate dev runs (e.g. a developer experimenting locally with `JAX_PLATFORMS` set).
- We detect "GPU intent" via the union of three env vars + the explicit opt-out. If none are set, we assume CPU is intentional and stay quiet.
- `flush=True` so the line lands in `cloud-init`/`user-data` logs immediately.

### 3.3 Update [AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md)

Replace **all of §6** ("After the trial / benchmark passes") with the structure below. The current §6 mentions GPU but doesn't give a working userdata.

```markdown
## 6. After the trial / benchmark passes

### 6.A) Same branch, larger CPU run

Adapt `verify_smoke.py` to a bigger config or use `verify_canonical_small.py`
(already in repo at `n_w=40, n_s=40, n_z=5, state=(3,3,3)`). Same `c8a.4xlarge`
or `hpc8a.64xlarge` instance, no recipe change needed.

### 6.B) GPU run on `p4d.24xlarge` (A100) or `p5.48xlarge` (H100)

**Instance:** start with `p4d.24xlarge` (8× A100 40 GB). Single-GPU operation
initially — the codebase doesn't yet shard across multiple GPUs. The other
seven A100s sit idle for now. (Multi-GPU is the work tracked in
[HANDOFF_PMAP_TO_VMAP.md](../handoff/HANDOFF_PMAP_TO_VMAP.md).)

**AMI:** prefer **AWS Deep Learning AMI GPU PyTorch 2.x (Amazon Linux 2023)**.
It ships with the NVIDIA driver, CUDA 12, and cuDNN already configured.
A bare AL2023 AMI works but adds 20-30 min of driver install — skip unless
you have a reason.

**Userdata:**

\```bash
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
\```

**Success criteria** (in `/var/log/jax-gpu-trial.log`):

\```
[lifecycle] JAX runtime: 1 device(s), platform(s)=['cuda']
...
Devices: [CudaDevice(id=0)]
Platforms: ['cuda']
...
  Status: complete  (6/6 ages solved)
  Policy sanity: PASS  (no NaN/Inf in solved ages)
\```

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
```

(The existing §7 paragraph about deleted scripts can stay as-is or move beneath §6.C.)

---

## 4. Verification

### 4.1 Local — confirm the runtime check works

The platform check fires at import. With CPU-only setup:

```bash
python -c "import lifecycle"
# Expected: "[lifecycle] JAX runtime: N device(s), platform(s)=['cpu']"
# No warning (no GPU intent env vars set).
```

With GPU intent but no GPU wheel:

```bash
LIFECYCLE_DISABLE_VIRTUAL_CPUS=1 python -c "import lifecycle"
# Expected:
#   [lifecycle] JAX runtime: 1 device(s), platform(s)=['cpu']
#   [lifecycle] WARNING: GPU env hints set ... but JAX reports CPU only
```

### 4.2 GPU instance — end-to-end smoke

After applying §3.3's userdata to a fresh p4d.24xlarge:

1. Wait for the instance to terminate itself on success (~20-40 min on first run, dominated by JIT compile if the persistent cache is empty).
2. Pull `/var/log/jax-gpu-trial.log` from S3 (if the userdata copies it, per the cache-sync handoff) or via SSH if you kept the instance up:
   ```bash
   aws s3 cp s3://<bucket>/jax-gpu-trial/<timestamp>.log -
   ```
3. Confirm the four required lines from §6.B's success criteria.
4. Confirm alpha ranges match the CPU-trial baseline within 1e-9 (or document the drift — float64 should be deterministic across CPU/GPU under JAX's contract, but verify).

### 4.3 Wheel manifest sanity

After running `pip install -r requirements-gpu.txt`:

```bash
pip list | grep -i 'jax\|cuda\|nvidia' | sort
```

Expected output (subset):
```
jax                       0.4.30+    (or newer)
jax-cuda12-pjrt           0.4.30+
jax-cuda12-plugin         0.4.30+
jaxlib                    0.4.30+
nvidia-cublas-cu12        ...
nvidia-cuda-cupti-cu12    ...
nvidia-cuda-nvcc-cu12     ...
nvidia-cuda-runtime-cu12  ...
nvidia-cudnn-cu12         ...
nvidia-nccl-cu12          ...
nvidia-nvjitlink-cu12     ...
```

**Missing `jax-cuda12-pjrt` or `jax-cuda12-plugin` ⇒ wheel install failed; the CUDA backend is not present.**

---

## 5. Edge cases / gotchas

### 5.1 `JAX_PLATFORMS=cuda` vs `JAX_PLATFORMS=cuda,cpu`

Setting `JAX_PLATFORMS=cuda` makes JAX **fail at import** if CUDA isn't available. This is what we want on a GPU instance — fail-loud.

Setting `JAX_PLATFORMS=cuda,cpu` falls back to CPU if CUDA is missing. **Don't use this on AWS** — silent fallback is exactly what we're trying to prevent. The userdata script in §3.3 uses `JAX_PLATFORMS=cuda` (strict).

### 5.2 `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` is also needed

`JAX_PLATFORMS=cuda` excludes CPU from JAX's device list, but the package's existing `_configure_xla_devices()` in [lifecycle/__init__.py:51-67](../../lifecycle/__init__.py#L51-L67) checks `JAX_PLATFORMS` and self-disables. Belt + braces: set both env vars. If only `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` is set (no `JAX_PLATFORMS`), JAX auto-detects CUDA — also fine, but less explicit.

### 5.3 Plain `pip install jax[cuda12]` on Windows fails

JAX's CUDA wheels are Linux-only. Don't include them in `requirements.txt` (which is the base for Windows dev). Keeping them in `requirements-gpu.txt` (Linux-only AWS overlay) avoids the cross-platform pain.

### 5.4 The Deep Learning AMI's pre-installed JAX may be older

DLAMIs ship with their own JAX install in a system Python. **Always create a fresh venv** (per §3.3 step 3) — don't try to use the system JAX. The version may not match the codebase, and the system install lives outside `pip` for some AMIs.

### 5.5 cudnn version drift

`jax[cuda12]` pins cuDNN through transitive deps. If the host's cuDNN is older (rare on DLAMI), JAX falls back to the bundled wheel — fine. If the host's cuDNN is incompatibly newer (very rare), import errors. Mitigation: check `nvidia-smi` and `python -c "import jax; jax.devices()"` work cleanly **before** the smoke run, in the userdata's step 5.

### 5.6 Single-GPU operation only

The current codebase's `pmap` over `len(jax.devices())` will sit on **one** GPU on p4d.24xlarge. The other 7 A100s are idle. This is intentional for this handoff — multi-GPU sharding is the [pmap-to-vmap handoff](HANDOFF_PMAP_TO_VMAP.md). For the trial, p4d.24xlarge is overkill cost-wise; if the user wants a cheaper single-GPU instance for the bring-up, **`g6.xlarge` (1× L4, 24 GB)** has fp64 throughput acceptable for smoke (NOT for production canonical — fp64 on L4 is 1:64). For benchmarking, stick with A100/H100.

### 5.7 Don't set `XLA_FLAGS` manually on GPU

The package's `_configure_xla_devices()` skips setting `XLA_FLAGS` if it's already in env. If the userdata accidentally exports an old CPU-tuned `XLA_FLAGS=--xla_force_host_platform_device_count=...`, JAX will dutifully use it and hide the GPU. **The userdata in §3.3 explicitly does NOT set `XLA_FLAGS`** — leave it unset.

---

## 6. Files touched

| File | Change | Lines |
|---|---|---|
| `requirements-gpu.txt` | New file | 6 |
| [lifecycle/__init__.py](../../lifecycle/__init__.py) | Add `_check_runtime_platform()` after the existing cache config block | ~30 |
| [docs/agents/AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md) | Replace §6 with structured A/B/C subsections; B is the new GPU recipe | ~80 (replacing existing ~60) |

No solver-side code changes. No test changes.

---

## 7. Implementation checklist (for the agent)

- [ ] Create `requirements-gpu.txt` per §3.1 (6 lines).
- [ ] Add `_check_runtime_platform()` to [lifecycle/__init__.py](../../lifecycle/__init__.py) after the cache config block per §3.2. Make sure the function is called once at import.
- [ ] Replace §6 of [AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md) with the structured A/B/C content from §3.3.
- [ ] Local verification (§4.1):
  - Plain import: confirm `[lifecycle] JAX runtime:` line prints with `platforms=['cpu']`.
  - With `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` set, confirm the warning fires.
- [ ] Wheel manifest verification on Linux (§4.3) — ideally on a clean VM or container; not strictly required if the agent is on Windows. **If skipped, note in the commit message that GPU wheel install was not tested locally.**
- [ ] Single commit:
  ```
  gpu setup: add requirements-gpu.txt overlay + runtime platform check

  - requirements-gpu.txt: thin overlay pulling jax[cuda12] for AWS GPU
    instances. Use INSTEAD of requirements.txt on Linux GPU hosts.
  - lifecycle/__init__.py: print active JAX platform at import; warn
    loud when CPU device is reported despite GPU env hints
    (CUDA_VISIBLE_DEVICES / JAX_PLATFORMS / LIFECYCLE_DISABLE_VIRTUAL_CPUS).
    Catches the silent CPU-fallback failure mode.
  - AWS_TRIAL_JAX.md §6: replace narrative "GPU run" pointer with
    full userdata recipe for p4d.24xlarge — includes nvidia-smi
    preflight, GPU env vars, jax.devices() sanity check before solve.

  Multi-GPU sharding deferred to HANDOFF_PMAP_TO_VMAP.md.
  Tested local CPU-side checks; GPU-side wheel install untested
  (Windows dev, no GPU access here).
  ```
- [ ] Push to `jax-rewrite`.

---

## 8. Performance expectations

This handoff is **plumbing** — no perf change expected from these edits alone. The runtime check costs ~5 ms at import.

The GPU benchmark itself (after this handoff lands and the user runs §3.3's userdata on p4d.24xlarge) should show:

- **JIT compile**: 1-3 min on first run with a cold cache (vs 5-15 min on hpc8a CPU). Compiled trace cached if [HANDOFF_JAX_PERSISTENT_COMPILATION_CACHE.md](HANDOFF_JAX_PERSISTENT_COMPILATION_CACHE.md) has landed.
- **Per-age wall**: depends on whether [HANDOFF_PMAP_TO_VMAP.md](HANDOFF_PMAP_TO_VMAP.md) has landed. Without it, 1× to 1.5× CPU baseline (single-GPU pmap-degenerate is suboptimal). With it, 2-5× faster than CPU.

These numbers go in the GPU-trial completion report, not in this handoff's commit message.

---

## 9. Out of scope / future work

- **Multi-GPU sharding** — see [HANDOFF_PMAP_TO_VMAP.md](HANDOFF_PMAP_TO_VMAP.md).
- **TPU support** — not required for the AWS path.
- **Container deployment** — bare pip install is the current pattern.
- **Driver / CUDA toolkit auto-install** — assume DLAMI; escalate if the user wants bare AL2023.
- **Mixed-precision config** — separate optimization, complementary.
