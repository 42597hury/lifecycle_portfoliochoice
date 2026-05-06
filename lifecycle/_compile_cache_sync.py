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
