# Computational Complexity Analysis — Handoff

## Task

Analyze the computational cost of the lifecycle solver, focusing on how the
recent change from discrete Markov transitions (`Pi_state`) to Gauss-Hermite
state quadrature affected runtime. The old system had 2 return dimensions
(n_ret=2) with the bill rate known; the new system has 3 return dimensions
(n_ret=3) with the bill rate uncertain (integrated via quadrature). Quantify
the cost increase and identify the binding bottleneck.

---

## What changed (old vs new)

### Old system (5-variable VAR, n_ret=2)

- State variables: `[rtb, y_nom, dp]` (3 state vars, rtb was known/riskless)
- Returns: `[xr, xb]` (2 returns, integrated via quadrature)
- **State transitions:** discrete Markov chain `Pi_state[i_s, j_s]` of size
  `N_state x N_state`. Inner loop: `for j_s in range(N_state)`.
- **Return quadrature:** 2D, K^2 nodes (K=1 gave 1 node, K=2 gave 4)
- **Per-FOC cost (retirement):** `N_state * K^2` iterations
- **Per-FOC cost (working):** `N_state * K^2 * n_eta * n_eps`
- Policy lookup at next state: direct array index `c_next_full[j_s, :]`

### New system (6-variable VAR, n_ret=3)

- State variables: `[y_1, spr, cy]` (3 state vars)
- Returns: `[rtb, xr, xb]` (3 returns, ALL uncertain including bill rate)
- **State transitions:** Gauss-Hermite quadrature over `v^s ~ N(0, Sigma_ss)`.
  Inner loop: `for k_v in range(n_state_quad)` where `n_state_quad = K_s^3`.
- **Return quadrature:** 3D, K_r^3 nodes
- **Per-FOC cost (retirement):** `K_s^3 * K_r^3` iterations, each with
  trilinear interpolation (8 corner lookups + weights)
- **Per-FOC cost (working):** `K_s^3 * K_r^3 * n_eta * n_eps`, each with
  trilinear interpolation
- Policy lookup at next state: **trilinear interpolation** via
  `bracket_state_3d` + 8 weighted lookups (replaces direct index)

---

## Default configuration values

From `model.py` `DiscretizationConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_grid_sizes` | (5,5,5) | Rouwenhorst grid per state dimension. N_state = 125 |
| `n_z` | 7 | Persistent income grid points |
| `n_savings` | 150 | EGM savings grid |
| `n_wealth` | 150 | Cash-on-hand interpolation grid |
| `n_state_quad_nodes` | 3 | GH order per state dim. Total = 3^3 = 27 nodes |
| `n_ret_nodes_1d` | 2 | GH order per return dim. Total = 2^3 = 8 nodes |
| `n_eps_nodes` | 3 | GH per mixture component for transitory shock |
| `n_eta_nodes` | 3 | GH per mixture component for persistent shock |

**Income quadrature (mixture-normal, 2 components each):**
- eps: `n_eps_nodes=3` per component, 2 components -> `n_eps = 6` total nodes
- eta: `n_eta_nodes=3` per component, 2 components -> `n_eta = 6` total nodes

**Lifecycle:**
- `start_age=22`, `retire_age=67`, `terminal_age=99`
- Working periods: 45 (ages 22-66)
- Retirement periods: 32 (ages 67-98)
- Terminal: 1 (age 99)
- Total: 78 periods

---

## Inner loop iteration counts (per FOC/Jacobian call)

### Retirement

```
for k_v in range(27):           # state innovation quadrature (3^3)
    bracket_state_3d(...)       # trilinear: 8 corner lookups
    for k_r in range(8):        # return residual quadrature (2^3)
        [6 accumulators]
```

**Total iterations:** 27 * 8 = **216**
**Trilinear interpolations:** 27 (each touches 8 grid corners)
**Comparison:** Old system did `N_state * K^2` = 125 * 1 = 125 (at K=1) or
125 * 4 = 500 (at K=2). But old had direct index lookup, no interpolation.

### Working age

```
for k_v in range(27):           # state innovation quadrature
    bracket_state_3d(...)       # trilinear interpolation setup
    for k_r in range(8):        # return residual quadrature
        [bequest accumulators]
        for k_eta in range(6):  # persistent income innovation
            [bracket z_next]
            for i_e in range(6): # transitory shock
                [alive accumulators]
```

**Total innermost iterations:** 27 * 8 * 6 * 6 = **7,776**
**Comparison:** Old system did `N_state * K^2 * n_eta * n_eps` = 125 * 1 * 6 * 6
= 4,500 (at K=1) or 125 * 4 * 6 * 6 = 18,000 (at K=2).

---

## Full period cost

### Per savings grid point (one Newton solve)

Assume ~4 Newton iterations on average (typical for constrained solver):

| Phase | Inner iters/FOC | Newton iters | Total FOC iters/s_val |
|-------|----------------|--------------|----------------------|
| Retirement | 216 | ~4 | ~864 |
| Working | 7,776 | ~4 | ~31,104 |

### Per period (one age)

Outer loops: `N_state * n_z * n_savings` = 125 * 7 * 150 = **131,250** Newton solves

| Phase | FOC iters/s_val | Newton solves/period | Total FOC iters/period |
|-------|----------------|---------------------|----------------------|
| Retirement | 216 | 131,250 | ~28M |
| Working | 7,776 | 131,250 | ~1.0B |

### Full lifecycle

| Phase | Periods | Total FOC iterations |
|-------|---------|---------------------|
| Working (ages 22-66) | 45 | ~46B |
| Retirement (ages 67-98) | 32 | ~0.9B |
| **Total** | **78** | **~47B** |

Working-age periods dominate by ~50x.

---

## Production grid (7x7x7)

With `state_grid_sizes=(7,7,7)`, N_state = 343:

Outer loops: 343 * 7 * 150 = **360,150** Newton solves per period.

Working-age total: 45 * 360,150 * 4 * 7,776 = **~504B** FOC iterations.

Retirement total: 32 * 360,150 * 4 * 216 = **~10B** FOC iterations.

---

## Key cost drivers to analyze

1. **Trilinear interpolation overhead.** Each of the 27 state quad nodes
   requires `bracket_state_3d` (6 binary searches + clamping) and then
   8 corner lookups with bilinear weight computation. In the old system,
   `c_next_full[j_s, :]` was a single array index. Quantify the per-node
   overhead ratio.

2. **3D vs 2D return quadrature.** Old: K^2 = 1 or 4. New: K^3 = 8.
   This is a 2x-8x multiplier on the return inner loop.

3. **State quad (27) vs Pi_state (125 or 343).** The quadrature loop is
   shorter (27 vs 125), but each iteration is more expensive (interpolation).
   Net effect depends on the interpolation-to-lookup cost ratio.

4. **Parallelization.** `prange` over `i_s` (N_state). With 125 states on
   e.g. 8 cores, each core handles ~16 states. With 343 states, ~43 per core.

---

## Measured timing data (2026-04-26, current quadrature implementation)

Benchmark: `time_retirement.py` — terminal age + 5 retirement periods.
Machine: user's Windows 11 desktop. First period includes Numba JIT compilation.

### 5x5x5 grid (N_state=125, n_z=7, 27 state quad, 8 return quad)

| Period | Time |
|--------|------|
| Terminal (incl. JIT) | 9.28s |
| Age 98 (incl. JIT) | 32.80s |
| Age 97 | 2.06s |
| Age 96 | 2.08s |
| Age 95 | 2.13s |
| Age 94 | 2.11s |
| **Steady-state avg** | **~2.1s** |

### 7x7x7 grid (N_state=343, n_z=7, 27 state quad, 8 return quad)

| Period | Time |
|--------|------|
| Terminal (incl. JIT) | 35.61s |
| Age 98 | 4.14s |
| Age 97 | 4.05s |
| Age 96 | 4.16s |
| Age 95 | 4.06s |
| Age 94 | 4.04s |
| **Steady-state avg** | **~4.1s** |

### Key observations

- **7x7x7 is only ~2x slower** despite 2.7x more states. Per-state time is
  actually lower (11.4us vs 62.8us) — `prange` over 343 states saturates
  CPU cores better than 125.
- **JIT compilation** adds ~30s on first period (one-time cost).
- **Projected full solve (retirement only):** 5x5x5: ~67s, 7x7x7: ~131s.
- **Projected working-age periods** (est ~36x retirement due to income loops):
  5x5x5: ~3.7hr, 7x7x7: ~1.8hr for 45 working periods.
  The 36x multiplier = n_eta(6) * n_eps(6); actual ratio may differ due to
  Newton convergence differences.
- **No historical timing data** for comparison with the old Pi_state
  implementation. Previous runs did not persist wall-clock times.

---

## Key files to read

| File | What to look at | Lines |
|------|----------------|-------|
| `solver.py` | `compute_foc_jac_retirement_quad` — retirement inner loop | ~328-460 |
| `solver.py` | `compute_foc_jac_working_quad` — working-age inner loop | ~468-660 |
| `solver.py` | `bracket_state_3d` — trilinear interpolation | ~219-260 |
| `solver.py` | `_solve_retirement_step_quad_jit` — retirement outer loops | ~1667-1850 |
| `solver.py` | `_solve_working_age_step_quad_jit` — working outer loops | ~1855-2050 |
| `solver.py` | `run_lifecycle_solver` — master loop + timing | ~2160-2260 |
| `model.py` | `DiscretizationConfig` — default node counts | 92-115 |
| `model.py` | `SolverConfig` — Newton iteration limits | 121-160 |
| `contextfiles/DESIGN.md` | Section 4 (solver) and Section 6 (computational notes) | |
| `contextfiles/RETURNS.md` | Section 5 (quadrature structure, precomputed arrays) | |
