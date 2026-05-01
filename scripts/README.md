# Scripts

This folder holds ad hoc runners that are useful during development but are
not part of the automated `pytest` suite.

## Layout

- `scripts/benchmarks/` for timing and performance comparisons
- `scripts/diagnostics/` for exploratory diagnostics and bundle inspection
- `scripts/investigation/` for focused experiments and temporary research work
- `scripts/smoke/` for quick sanity checks
- `scripts/validation/` for heavier correctness and audit scripts

## Running

Run these modules from the repo root with `python -m ...` so repo-root imports
continue to work after being moved out of the top level.

Examples:

```powershell
python -m scripts.diagnostics._diag_grid_quad_sweep
python -m scripts.diagnostics._diag_quadrature_cloud
python -m scripts.benchmarks.time_retirement
python -m scripts.validation.test_terminal_correctness
```
