"""
lifecycle — life-cycle portfolio choice model with stocks, nominal bonds, and bills.

Core modules:

  model               LifecyclePortfolioModel + utility/tax/bequest helpers
  var                 VAR estimation, partition, and predictability ablations
  discretization      Rouwenhorst, Judd quadrature, state grid construction
  mortality           earnings-dependent survival calibration
  precompute          build_model() factory + Precompute container
  solver              JAX-pure backward induction (EGM + 2D Newton)
  inf_horizon_solver  CCV-style infinite-horizon benchmark
  simulation          forward simulation
  diagnostics         pre/post-solve calibration + Newton failure reports
  numerics            shared bin-prob helpers
  plots               pre/post-solve figures
  policy_io           policy bundle save/load (saved_runs/<name>/)
  predictability_ablation   systems I-IV (full VAR vs ablations)

JAX runtime configuration: this package configures JAX for float64 arithmetic
and exposes a virtual multi-device CPU pool so the JAX solver can pmap across
cores. Both must be set before any ``jax`` import — they are configured here
at package import time. Users who set ``XLA_FLAGS`` themselves are not
overridden.
"""

import os as _os
import multiprocessing as _mp


def _configure_xla_devices():
    """Expose ``len(jax.devices()) == cpu_count`` so the JAX solver can pmap
    across virtual devices on a CPU-only host. XLA's CPU backend is
    single-threaded per device, so creating one virtual device per core is
    the standard pattern for parallelising a pure-JAX workload on CPU.

    On GPU/TPU hosts this flag must NOT be set: ``--xla_force_host_platform_device_count``
    forces XLA's host platform and HIDES the accelerator. Skipped when:

    - ``XLA_FLAGS`` is already set (user customisation wins);
    - ``JAX_PLATFORMS`` is set to anything that doesn't include ``cpu``
      (e.g. ``cuda``, ``cuda,cpu`` — explicit GPU intent);
    - ``LIFECYCLE_DISABLE_VIRTUAL_CPUS=1`` is set (explicit opt-out for
      GPU/TPU runs on a host where JAX_PLATFORMS isn't otherwise customised).

    On AWS p4d/p5 instances the standard recipe is::

        export LIFECYCLE_DISABLE_VIRTUAL_CPUS=1
        python verify/smoke.py    # JAX picks up the GPU automatically
    """
    if "XLA_FLAGS" in _os.environ:
        return
    platforms = _os.environ.get("JAX_PLATFORMS", "").lower()
    # If JAX_PLATFORMS is set and doesn't include CPU, the user wants
    # accelerator-only — don't fight that.
    if platforms and "cpu" not in [p.strip() for p in platforms.split(",")]:
        return
    if _os.environ.get("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "").lower() in ("1", "true", "yes"):
        return
    try:
        n = _mp.cpu_count()
    except NotImplementedError:
        n = 1
    _os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n}"


_configure_xla_devices()

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


def _check_runtime_platform():
    """Print active JAX platform once at import. Warns if running on CPU when
    the env signals GPU intent (presence of CUDA-related env vars).

    Why this exists: a GPU instance can silently fall through to CPU if
    (a) plain ``jax`` was installed instead of ``jax[cuda12]``, or
    (b) ``XLA_FLAGS=--xla_force_host_platform_device_count=...`` was set
        somewhere up the stack and we didn't catch it. Both produce a
        runnable but slow workload that AWS bills full GPU rates for.
    """
    try:
        devices = _jax.devices()
    except RuntimeError as exc:
        print(f"[lifecycle] WARNING: jax.devices() raised: {exc}", flush=True)
        print(
            "[lifecycle] WARNING: GPU env hints set but JAX runtime failed to "
            "enumerate devices. Likely cause: CUDA driver/runtime mismatch "
            "with jax[cuda12] wheel. Run nvidia-smi and "
            "`python -c 'import jax; print(jax.devices())'` to confirm.",
            flush=True,
        )
        return
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