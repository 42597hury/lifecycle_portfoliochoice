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
        python verify_smoke.py    # JAX picks up the GPU automatically
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
_jax.config.update("jax_compilation_cache_dir", _os.path.expanduser("~/.cache/jax_lifecycle"))
_jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
_jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)