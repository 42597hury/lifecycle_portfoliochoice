"""
lifecycle — life-cycle portfolio choice model with stocks, nominal bonds, and bills.

Core modules (importable as lifecycle.X or via root shims X for backward compat):

  model               LifecyclePortfolioModel + utility/tax/bequest helpers
  var                 VAR estimation, partition, and predictability ablations
  discretization      Rouwenhorst, Judd quadrature, state grid construction
  mortality           earnings-dependent survival calibration
  precompute          build_model() factory + Precompute container
  solver              backward induction (EGM + 2D Newton)
  inf_horizon_solver  CCV-style infinite-horizon benchmark
  simulation          forward simulation
  diagnostics         pre/post-solve calibration + Newton failure reports
  numerics            shared PCHIP and bin-prob helpers
  plots               pre/post-solve figures
  policy_io           policy bundle save/load (saved_runs/<name>/)
  predictability_ablation   systems I-IV (full VAR vs ablations)

The package is intended to be imported from the repo root, which is on
sys.path either via tests/conftest.py, scripts/run_solve.py boot, or
direct `pip install -e .` (no pyproject.toml is shipped — sys.path
discovery is sufficient for the use cases).

Backward compatibility: each module also has a top-level shim file at
the repo root (e.g. `model.py`) that aliases `sys.modules[__name__]`
to the lifecycle implementation. This preserves both `from model import X`
imports across the codebase and the pickle qualified names embedded in
existing saved_runs/ bundles.
"""
