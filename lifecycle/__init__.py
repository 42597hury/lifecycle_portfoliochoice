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
    across virtual devices on a single CPU. XLA's CPU backend is single-threaded
    per device, so creating one virtual device per core is the standard pattern
    for parallelising a pure-JAX workload on CPU.

    Skipped if ``XLA_FLAGS`` is already set so user-supplied flags win.
    """
    if "XLA_FLAGS" in _os.environ:
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