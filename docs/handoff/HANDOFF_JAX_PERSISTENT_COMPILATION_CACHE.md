# Handoff: Harden JAX Persistent Compilation Cache for AWS Runs

**Branch:** `jax-rewrite`
**Status when this doc was written:** [lifecycle/__init__.py:71-74](../../lifecycle/__init__.py#L71-L74) already enables the JAX persistent compilation cache. The setup is functional but inflexible — hardcoded path, no size cap, no AWS-cross-launch reuse, no startup status line. The four production kernels (terminal/retire/work/boundary) take 5-15 min to JIT-compile from cold on hpc8a; on first-launch GPU benchmarks this dominates wall time before any solve work begins. Caching across EC2 launches via S3 is the high-value win.

---

## 1. Goal

Make the JAX compilation cache:

1. **Configurable** — overridable cache directory via env var, so AWS user-data can drop it on EBS or a mounted volume without code edits.
2. **Bounded** — explicit total-size cap so a long-running instance doesn't fill the disk silently.
3. **Observable** — one printed line on startup showing cache state (path, min-compile-time, max-size, status), so AWS log scrubbing tells you whether you got a cache hit.
4. **Reusable across EC2 launches** — a thin `aws s3 sync` helper that pulls the cache from S3 before any JAX import and pushes it back at run end. Optional and gated behind an env var.

The current 3-line setup gives within-instance (intra-launch) reuse only. The user's stated AWS workflow is single-shot launches that auto-terminate, so without S3 sync each launch eats the full ~10-15 min compile cost.

---

## 2. Scope and non-goals

### In scope

- Edit `lifecycle/__init__.py` to read three env vars (cache dir, min compile time, max size) with sensible defaults.
- Add a single startup print describing cache state.
- Add a small standalone module `lifecycle/_compile_cache_sync.py` with two functions: `pull_from_s3(bucket, prefix)` and `push_to_s3(bucket, prefix)` using `subprocess.run(["aws", "s3", "sync", ...])`. **Not** invoked from inside the package; called explicitly by the user-data script or `verify_*` runner.
- Update [docs/agents/AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md) bootstrap recipe to use the env vars and S3 sync.

### Out of scope (do not implement)

- Auto-pulling from S3 inside `__init__.py` import. Side-effecting network calls in package init are a footgun (slow imports, breaks offline dev, fails confusingly without AWS creds). Sync stays an explicit user-data step.
- Garbage-collection / TTL for cache entries beyond the size cap JAX provides natively.
- Cross-architecture cache sharing (CPU↔GPU). JAX namespaces by hardware/version internally; don't try to second-guess.
- Replacing the existing `XLA_FLAGS` virtual-CPU logic. Untouched.

---

## 3. Existing state — what already works

[lifecycle/__init__.py:67-74](../../lifecycle/__init__.py#L67-L74):

```python
_configure_xla_devices()

import jax as _jax  # noqa: E402

_jax.config.update("jax_enable_x64", True)
_jax.config.update("jax_compilation_cache_dir", _os.path.expanduser("~/.cache/jax_lifecycle"))
_jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
_jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
```

What this delivers:
- ✅ Cache lives at `~/.cache/jax_lifecycle/` (Linux: `/home/ec2-user/.cache/jax_lifecycle/` or `/root/.cache/jax_lifecycle/` depending on userdata user).
- ✅ Only entries with compile time ≥ 1.0s are cached (filters out trivial JITs that aren't worth disk).
- ✅ No min entry size (`-1` means cache all sizes).
- ✅ Cache survives across re-runs **on the same machine**.

What it doesn't deliver:
- ❌ No way to redirect the cache dir from outside — relevant for AWS where you may want it on an attached EBS volume rather than the root volume.
- ❌ No size cap — the cache grows unbounded. At canonical sizes a single solve adds ~50-200 MB; not catastrophic, but worth bounding for long-lived dev machines.
- ❌ No reuse across instances — each fresh EC2 launch starts cold.
- ❌ Silent — nothing in the log tells you whether the cache is enabled or where it points. After a 28-min hpc8a JIT failure with no log indication of cache state, this matters.

---

## 4. Implementation

### 4.1 Modify `lifecycle/__init__.py`

Replace the existing 4-line cache block (lines 71-74) with the following. Keep the rest of the file (XLA_FLAGS logic, file header) untouched.

```python
import jax as _jax  # noqa: E402

_jax.config.update("jax_enable_x64", True)


def _configure_persistent_cache():
    """Configure JAX's persistent compilation cache.

    Cache directory:
        ``LIFECYCLE_JAX_CACHE_DIR`` env var, falling back to
        ``~/.cache/jax_lifecycle/``. Set to the empty string to disable
        (skip all jax_compilation_cache_* config).

    Min compile time threshold (seconds):
        ``LIFECYCLE_JAX_CACHE_MIN_COMPILE_SECS`` env var (float),
        default 1.0. Traces faster than this are not persisted.

    Max cache size (bytes):
        ``LIFECYCLE_JAX_CACHE_MAX_SIZE_BYTES`` env var (int),
        default 10 * 1024**3 (10 GB). Use -1 for unlimited.

    Prints one status line on enable so AWS log scrubbing can confirm.
    """
    cache_dir_raw = _os.environ.get(
        "LIFECYCLE_JAX_CACHE_DIR",
        _os.path.expanduser("~/.cache/jax_lifecycle"),
    )
    if cache_dir_raw == "":
        print("[lifecycle] JAX compilation cache: DISABLED (LIFECYCLE_JAX_CACHE_DIR='')", flush=True)
        return

    cache_dir = _os.path.expanduser(cache_dir_raw)
    try:
        _os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        print(
            f"[lifecycle] JAX compilation cache: COULD NOT CREATE {cache_dir} ({exc}); "
            f"continuing without persistent cache",
            flush=True,
        )
        return

    try:
        min_secs = float(_os.environ.get("LIFECYCLE_JAX_CACHE_MIN_COMPILE_SECS", "1.0"))
    except ValueError:
        min_secs = 1.0
    try:
        max_bytes = int(_os.environ.get("LIFECYCLE_JAX_CACHE_MAX_SIZE_BYTES", str(10 * 1024**3)))
    except ValueError:
        max_bytes = 10 * 1024**3

    _jax.config.update("jax_compilation_cache_dir", cache_dir)
    _jax.config.update("jax_persistent_cache_min_compile_time_secs", min_secs)
    _jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    if max_bytes >= 0:
        # Newer JAX versions support this; older ones ignore the unknown key.
        try:
            _jax.config.update("jax_compilation_cache_max_size_bytes", max_bytes)
        except (AttributeError, KeyError):
            pass

    size_str = "unlimited" if max_bytes < 0 else f"{max_bytes / 1024**3:.1f} GB"
    print(
        f"[lifecycle] JAX compilation cache: enabled at {cache_dir} "
        f"(min_compile={min_secs}s, max_size={size_str})",
        flush=True,
    )


_configure_persistent_cache()
```

**Key design decisions in this block:**

1. **Empty-string disable.** `LIFECYCLE_JAX_CACHE_DIR=""` is the explicit opt-out — useful for benchmarking the cold-compile cost or for sandboxes where disk writes aren't desired.
2. **`makedirs(exist_ok=True)`** so the cache dir is always present before JAX tries to write. Failure here is non-fatal (we log and skip) rather than crashing import — important for environments with weird permissions (containers, read-only home dirs).
3. **Try/except around `jax_compilation_cache_max_size_bytes`** because that flag was added in a recent JAX version (≥ 0.4.30 ish). Older JAX in dev environments will silently ignore it. Don't break dev installs.
4. **`flush=True`** on prints so the line appears in AWS user-data logs even when stdout is being teed.
5. **Status line is exactly one line** — keeps log noise minimal.

### 4.2 Add `lifecycle/_compile_cache_sync.py`

A small standalone module with two functions. **Not auto-invoked** — called explicitly by the user-data script or a verify_* runner.

```python
"""Helpers for reusing the JAX persistent compilation cache across EC2 launches
via S3. Not invoked by ``import lifecycle``; the user-data script (or a
``verify_*`` runner) calls these explicitly before/after the solve.

Usage from a runner script::

    from lifecycle._compile_cache_sync import pull_from_s3, push_to_s3
    pull_from_s3("hugo-thesis-runs", "jax-cache/v1")    # before any heavy JIT
    # ... run the solve ...
    push_to_s3("hugo-thesis-runs", "jax-cache/v1")      # after solve completes
"""
import os
import subprocess


def _resolve_cache_dir():
    return os.path.expanduser(
        os.environ.get("LIFECYCLE_JAX_CACHE_DIR", "~/.cache/jax_lifecycle")
    )


def pull_from_s3(bucket, prefix, region=None):
    """Download the JAX compilation cache from ``s3://<bucket>/<prefix>/``
    into the local cache dir. No-op if the local dir is non-empty (assume
    the cache is already populated and skip the network round-trip).

    Returns the subprocess returncode (0 on success, non-zero on failure).
    Failure is non-fatal — caller should log and continue.
    """
    cache_dir = _resolve_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    if any(os.scandir(cache_dir)):
        print(f"[cache_sync] local cache at {cache_dir} non-empty; skipping s3 pull", flush=True)
        return 0
    cmd = [
        "aws", "s3", "sync",
        f"s3://{bucket}/{prefix.rstrip('/')}/",
        cache_dir,
    ]
    if region:
        cmd += ["--region", region]
    print(f"[cache_sync] pulling cache: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    print(f"[cache_sync] s3 pull rc={rc}", flush=True)
    return rc


def push_to_s3(bucket, prefix, region=None):
    """Upload the local JAX compilation cache to ``s3://<bucket>/<prefix>/``.
    Use ``--size-only`` so unchanged files are skipped (fast).

    Returns the subprocess returncode. Failure is non-fatal.
    """
    cache_dir = _resolve_cache_dir()
    if not os.path.isdir(cache_dir) or not any(os.scandir(cache_dir)):
        print(f"[cache_sync] local cache at {cache_dir} empty; skipping s3 push", flush=True)
        return 0
    cmd = [
        "aws", "s3", "sync",
        cache_dir,
        f"s3://{bucket}/{prefix.rstrip('/')}/",
        "--size-only",
    ]
    if region:
        cmd += ["--region", region]
    print(f"[cache_sync] pushing cache: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    print(f"[cache_sync] s3 push rc={rc}", flush=True)
    return rc
```

**Key design decisions:**

1. **Non-fatal on failure.** A flaky S3 sync should never block a solve. Print + return code is enough.
2. **Idempotent pull (skip if non-empty).** Avoids burning S3 bandwidth on a re-launch that has cache from a previous interactive session.
3. **`--size-only` on push.** JAX cache files are content-addressed — same compile produces the same file. Size comparison is correct and ~5× faster than `--checksum`.
4. **No region default in code.** Caller passes it explicitly (or relies on env / IAM role). Avoid hardcoding `eu-north-1`.
5. **No CLI / __main__ entrypoint.** Keep it library-only; the runner imports the functions.

### 4.3 Update `docs/agents/AWS_TRIAL_JAX.md`

In §2 (Bootstrap script userdata), insert just before step 4 ("Run smoke") :

```bash
# 3b. Configure JAX compilation cache (cross-launch reuse via S3)
export LIFECYCLE_JAX_CACHE_DIR=/home/ec2-user/.cache/jax_lifecycle
export S3_CACHE_BUCKET=hugo-thesis-runs       # set if you want cross-launch reuse
export S3_CACHE_PREFIX=jax-cache/v1

if [ -n "${S3_CACHE_BUCKET:-}" ]; then
    sudo -u ec2-user mkdir -p "$LIFECYCLE_JAX_CACHE_DIR"
    sudo -u ec2-user aws s3 sync \
        "s3://${S3_CACHE_BUCKET}/${S3_CACHE_PREFIX}/" \
        "$LIFECYCLE_JAX_CACHE_DIR" \
        --region eu-north-1 || echo "cache pull failed; continuing"
fi
```

And replace step 6 (auto-terminate) so the cache pushes before shutdown:

```bash
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

Add a brief paragraph in the "What this trial validates" section noting that the cache is enabled by default and may save 5-15 min of JIT compile on second-launch runs.

---

## 5. Verification

### 5.1 Local — confirm the cache works in isolation

1. Wipe the existing cache: `rm -rf ~/.cache/jax_lifecycle`
2. Run `python verify_smoke.py`. Expect a startup line:
   ```
   [lifecycle] JAX compilation cache: enabled at /home/<user>/.cache/jax_lifecycle (min_compile=1.0s, max_size=10.0 GB)
   ```
3. Confirm cache directory is non-empty after first run: `du -sh ~/.cache/jax_lifecycle/` should report ~50-200 MB.
4. Run smoke a second time. **Compile portion** of wall (the time spent before "Terminal condition" prints) should drop from ~3-5 min to under 30 seconds.
5. Confirm with `LIFECYCLE_JAX_CACHE_DIR=""` that the disable path works — should print `JAX compilation cache: DISABLED` and the second run takes the same compile time as the first.

### 5.2 Local — confirm env var override

```bash
LIFECYCLE_JAX_CACHE_DIR=/tmp/jax_cache_test python verify_smoke.py
```
Should print the cache path as `/tmp/jax_cache_test`. Confirm that path exists and contains files after the run.

### 5.3 AWS — confirm S3 round-trip

After applying the doc changes from §4.3:

1. Launch instance with no cache in S3 (first ever run). Smoke should run with full compile cost. After completion, check `aws s3 ls s3://hugo-thesis-runs/jax-cache/v1/` — should now contain ~50-200 MB of cache entries.
2. Launch a second instance with the same userdata. The userdata's `aws s3 sync` should populate `~/.cache/jax_lifecycle/` before Python starts. The smoke run should print `[lifecycle] JAX compilation cache: enabled at ...` and the per-kernel compile messages (if any) should report cache-hit timings (sub-second).

### 5.4 What "successful caching" looks like in JAX logs

JAX itself emits cache messages when `JAX_PERSISTENT_CACHE_LOG_LEVEL=1`. Optional sanity step: set that env var in §4.3 and look for lines like:

```
Loaded compilation cache entry from /home/.../jax_lifecycle/<hash> in 0.12s
```

This confirms the cache is being read, not just present on disk.

---

## 6. Edge cases / gotchas

1. **JAX version compatibility.** `jax_compilation_cache_max_size_bytes` was added in jax ~0.4.30. The try/except around it handles older versions. Verify your dev jax with `pip show jax | grep Version`. The repo's `requirements.txt` should pin `jax>=0.4.30` (already done in commit 8ed9e8b per the AWS_TRIAL_JAX.md context).

2. **Hardware-keyed cache.** JAX hashes the trace key with hardware identity. **A cache populated on hpc8a (CPU) will not produce hits on p4d (A100 GPU).** Use separate S3 prefixes per hardware tier:
   - `jax-cache/v1/cpu-hpc8a/`
   - `jax-cache/v1/gpu-a100/`
   - `jax-cache/v1/gpu-h100/`

   The handoff doesn't enforce this — leave it to the userdata writer to pick the right `S3_CACHE_PREFIX`. Document this in AWS_TRIAL_JAX.md alongside the env var.

3. **Permissions on Linux.** EC2 user-data scripts run as root. The current AWS_TRIAL_JAX.md uses `sudo -u ec2-user` for the Python invocation, so the cache is created as ec2-user. If user-data does the S3 pull as root and chowns later, watch for permissions mismatch — easier to do all cache operations as ec2-user.

4. **Stale entries after JAX upgrade.** JAX should invalidate via the trace key when its version changes, but old entries linger as dead bytes. Bumping the S3 prefix (`v1` → `v2`) on JAX upgrade is the simplest disposal mechanism.

5. **Cache directory on EBS root vs separate volume.** Default `~/.cache/jax_lifecycle/` lives on the root EBS volume. For very large cache sizes (cross-config sweeps), point `LIFECYCLE_JAX_CACHE_DIR` to an attached high-IOPS volume. Out of scope here, but the env var supports it.

6. **Concurrent Python processes writing to the same cache dir.** JAX uses file locking. Safe in practice; do not parallelize cache writes by accident (e.g. two `verify_*` runs simultaneously on the same instance).

7. **S3 bucket policy.** The IAM role on the EC2 instance needs `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on the cache prefix. If you're using the same bucket as runs, this is presumably already configured — flag it during the first launch.

8. **Don't put the cache in `/tmp`.** Some AMIs make `/tmp` a tmpfs; cache eats RAM. Stick with `~/.cache/...` on EBS.

---

## 7. Files touched

| File | Change | Lines (approx) |
|---|---|---|
| [lifecycle/__init__.py](../../lifecycle/__init__.py) | Replace 3-line cache block with `_configure_persistent_cache()` function | ~50 net new lines (replaces ~3 existing) |
| [lifecycle/_compile_cache_sync.py](../../lifecycle/_compile_cache_sync.py) | New file | ~60 lines |
| [docs/agents/AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md) | Insert env-var section + S3 sync in userdata; brief note in trial-validates section | ~30 added lines |

No test files touched (verification is via smoke timing, §5).

---

## 8. Implementation checklist (for the agent)

- [ ] Replace the 3-line block at [lifecycle/__init__.py:72-74](../../lifecycle/__init__.py#L72-L74) with the `_configure_persistent_cache()` function from §4.1. Keep the existing `_configure_xla_devices()` and the `import jax as _jax` line above it untouched.
- [ ] Add new file `lifecycle/_compile_cache_sync.py` exactly as in §4.2. Underscore prefix marks it as internal.
- [ ] Update [docs/agents/AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md) §2 (Bootstrap script) and §3 (Success criteria) per §4.3. Mention `S3_CACHE_BUCKET` / `S3_CACHE_PREFIX` env vars and the hardware-keyed prefix convention from §6.2.
- [ ] Run `verify_smoke.py` twice locally (first run wipes cache via `rm -rf ~/.cache/jax_lifecycle`). Confirm:
  - First run prints `[lifecycle] JAX compilation cache: enabled at ...`
  - Second run wall time is materially shorter (compile portion drops by 80-95%).
  - `du -sh ~/.cache/jax_lifecycle/` reports a non-trivial size after run 1.
- [ ] Run `LIFECYCLE_JAX_CACHE_DIR="" python verify_smoke.py` once, confirm the disable path prints the expected `DISABLED` line.
- [ ] (Optional) Run `LIFECYCLE_JAX_CACHE_DIR=/tmp/jax_test python verify_smoke.py`, confirm path shows in print and files appear at that path.
- [ ] Single commit with message:
  ```
  jax cache: env-var configurable + S3 sync helper for cross-launch reuse

  - lifecycle/__init__.py: replace hardcoded cache config with
    LIFECYCLE_JAX_CACHE_DIR / _MIN_COMPILE_SECS / _MAX_SIZE_BYTES env vars,
    one printed status line on enable, non-fatal failure path.
  - lifecycle/_compile_cache_sync.py: pull_from_s3/push_to_s3 helpers
    using aws s3 sync; idempotent pull, --size-only push.
  - docs/agents/AWS_TRIAL_JAX.md: bootstrap recipe uses the env vars
    and S3 sync; document hardware-keyed prefix convention.

  Verified: smoke run 2 compile drops from ~3 min to <30s on cache hit.
  ```
- [ ] Push to `jax-rewrite`. No PR review needed unless the maintainer asks.

---

## 9. Performance expectations to record in commit/PR message

- **Cold compile (run 1):** ~3-5 min on local CPU smoke; ~5-15 min on hpc8a; ~1-3 min on A100.
- **Cache hit (run 2 same hardware):** sub-30s startup, near-zero compile time. Wall savings = cold compile time minus ~10s of cache-load overhead.
- **S3 sync round-trip:** ~10-30 s for a ~200 MB cache from `eu-north-1`. Net savings on AWS launch: ~3-12 min per cache-hit launch.
- **Cache disk usage at canonical:** ~100-200 MB per (model, hardware) combination.

These are the numbers the agent should put in the commit message and update [docs/agents/AWS_TRIAL_JAX.md](../agents/AWS_TRIAL_JAX.md) with after running §5 verifications.

---

## 10. Out-of-scope, future work

- **Auto-pull from S3 inside `__init__.py`.** Considered and rejected (§2): network at import time is a footgun.
- **Garbage collection beyond JAX's built-in size cap.** If `LIFECYCLE_JAX_CACHE_MAX_SIZE_BYTES` proves insufficient (it triggers JAX-side LRU eviction), revisit then.
- **Encrypted cache for sensitive XLA traces.** Not relevant for this codebase.
- **Cache warming via dry-run compile.** A separate `python -c "from lifecycle.solver import _build_per_age_terminal_kernel; ..."` warm-up step could populate the cache deterministically. Out of scope; the smoke itself is the warm-up.
