"""
plots.py -- Visual diagnostic figures for the lifecycle portfolio choice model.

Pre-solve (call after Precompute):
  plot_income_pre(model, pc)       -- 3 figures: income lifecycle, replacement curve, AIME pipeline
  plot_income_lifecycle(model, pc)       -- income lifecycle profile (low/mid/high z)
  plot_replacement_curve(model, pc)      -- replacement rate vs z
  plot_aime_pipeline(model, pc)          -- AIME pipeline: raw earnings, AIME, PIA, net pension

Post-simulation (call after simulate_lifecycle):
  plot_income_post(model, pc, sim) -- 4 figures: z dist, income fan, replacement dist, survivor selection
  plot_z_distribution(model, pc, sim)    -- z histogram at key ages
  plot_income_fan(model, pc, sim)        -- income percentile bands by age
  plot_replacement_dist(model, pc, sim)  -- realized replacement rate histogram
  plot_survivor_selection(model, pc, sim)-- mean z by age

Dependencies: matplotlib, numpy
Never imported by diagnostics.py.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Pre-solve figures
# =============================================================================

def plot_income_lifecycle(model, pc):
    """Income lifecycle profile: 3 lines (low/mid/high z) showing E_eps[Y_net] by age."""
    iz_lo, iz_mid, iz_hi = 0, pc.n_z // 2, pc.n_z - 1
    ages = np.arange(model.start_age, model.terminal_age + 1)
    retire_t = model.retire_age - model.start_age

    y_lines = {}
    for label, iz in [("low z", iz_lo), ("median z", iz_mid), ("high z", iz_hi)]:
        y = np.empty(len(ages))
        for i, age in enumerate(ages):
            t = age - model.start_age
            if age <= model.retire_age:
                y[i] = float(np.dot(pc.eps_weights, pc.working_income[t, iz, :]))
            else:
                y[i] = float(pc.pension_after_tax[t, iz])
        y_lines[label] = y

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, y in y_lines.items():
        ax.plot(ages, y, label=label)
    ax.axvline(model.retire_age, color='gray', linestyle='--', alpha=0.5, label='Retirement')
    ax.set_xlabel('Age')
    ax.set_ylabel('After-tax income (model units)')
    ax.set_title('Income Lifecycle Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_replacement_curve(model, pc):
    """Replacement rate (pension / career-avg after-tax income) vs z-grid."""
    working_ages = np.arange(model.start_age, model.retire_age)
    retire_t = model.retire_age - model.start_age
    n_z = pc.n_z
    avg_det = float(np.mean(np.exp(
        model.b0 + model.b1 * working_ages + model.b2 * working_ages**2 / 10.0
        + model.b3 * working_ages**3 / 100.0
    )))

    repl_rates = np.empty(n_z)
    for iz in range(n_z):
        career_avg = float(np.mean([
            np.dot(pc.eps_weights, pc.working_income[t, iz, :])
            for t in range(len(working_ages))
        ]))
        pens = float(pc.pension_after_tax[0, iz])
        repl_rates[iz] = pens / career_avg if career_avg > 1e-10 else np.nan

    # Mark where AIME cap binds
    z_cap = np.log(2.5 / avg_det)
    cap_idx = np.searchsorted(pc.z_grid, z_cap)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pc.z_grid, repl_rates * 100, 'b-o', markersize=4)
    if 0 <= cap_idx < n_z:
        ax.axvline(z_cap, color='red', linestyle='--', alpha=0.6, label=f'AIME cap binds (z={z_cap:.2f})')
    ax.set_xlabel('Persistent income state z')
    ax.set_ylabel('Replacement rate (%)')
    ax.set_title('Social Security Replacement Rate by Earnings Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_aime_pipeline(model, pc):
    """AIME pipeline: raw earnings proxy, AIME (capped), gross PIA, net pension vs z."""
    working_ages = np.arange(model.start_age, model.retire_age)
    avg_det = float(np.mean(np.exp(
        model.b0 + model.b1 * working_ages + model.b2 * working_ages**2 / 10.0
        + model.b3 * working_ages**3 / 100.0
    )))

    n_z = pc.n_z
    raw_earnings = np.exp(pc.z_grid) * avg_det
    aime = np.minimum(raw_earnings, 2.5)

    pia_gross = np.empty(n_z)
    for iz in range(n_z):
        a = aime[iz]
        if a <= 0.21:
            pia_gross[iz] = a * 0.90
        elif a <= 1.25:
            pia_gross[iz] = 0.90 * 0.21 + 0.32 * (a - 0.21)
        else:
            pia_gross[iz] = 0.90 * 0.21 + 0.32 * (1.25 - 0.21) + 0.15 * (a - 1.25)

    net_pension = pc.pension_after_tax[0, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pc.z_grid, raw_earnings, 'b-', label='exp(z) x avg_det (raw earnings)')
    ax.plot(pc.z_grid, aime, 'r--', label='AIME (capped at 2.5)')
    ax.plot(pc.z_grid, pia_gross, 'g-.', label='Gross PIA')
    ax.plot(pc.z_grid, net_pension, 'k-', linewidth=2, label='Net pension')
    ax.axhline(2.5, color='red', alpha=0.3, linestyle=':')
    ax.set_xlabel('Persistent income state z')
    ax.set_ylabel('Model units')
    ax.set_title('AIME Pipeline: Earnings to Net Pension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_income_pre(model, pc):
    """All 3 pre-solve labour/SS figures. Returns list of figures."""
    return [
        plot_income_lifecycle(model, pc),
        plot_replacement_curve(model, pc),
        plot_aime_pipeline(model, pc),
    ]


# =============================================================================
# Post-simulation figures
# =============================================================================

def plot_z_distribution(model, pc, sim):
    """Histogram of z (continuous) at key ages."""
    sim_z = sim["z"]
    sim_alive = sim["alive"]
    ages = sim["ages"]
    key_ages = [22, 45, 67, 85]
    key_ages = [a for a in key_ages if model.start_age <= a <= model.terminal_age]

    fig, axes = plt.subplots(1, len(key_ages), figsize=(4 * len(key_ages), 4), sharey=True)
    if len(key_ages) == 1:
        axes = [axes]

    for ax, age in zip(axes, key_ages):
        t = age - model.start_age
        alive_mask = sim_alive[:, t].astype(bool)
        z_vals = sim_z[alive_mask, t]
        ax.hist(z_vals, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'Age {age} (N={int(alive_mask.sum())})')
        ax.set_xlabel('z')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)

    axes[0].set_ylabel('Density')
    fig.suptitle('Persistent Income State Distribution at Key Ages', y=1.02)
    fig.tight_layout()
    return fig


def plot_income_fan(model, pc, sim):
    """Income fan chart: percentile bands (p10/p25/p50/p75/p90) by age."""
    sim_income = sim["income"]
    sim_alive = sim["alive"]
    ages = sim["ages"]
    n_age = len(ages)

    pctiles = [10, 25, 50, 75, 90]
    pct_vals = np.full((len(pctiles), n_age), np.nan)

    for t in range(n_age):
        alive_mask = sim_alive[:, t].astype(bool)
        if alive_mask.sum() < 10:
            continue
        y_alive = sim_income[alive_mask, t]
        for j, p in enumerate(pctiles):
            pct_vals[j, t] = np.percentile(y_alive, p)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(ages, pct_vals[0], pct_vals[4], alpha=0.15, color='blue', label='p10-p90')
    ax.fill_between(ages, pct_vals[1], pct_vals[3], alpha=0.3, color='blue', label='p25-p75')
    ax.plot(ages, pct_vals[2], 'b-', linewidth=2, label='Median')
    ax.axvline(model.retire_age, color='gray', linestyle='--', alpha=0.5, label='Retirement')
    ax.set_xlabel('Age')
    ax.set_ylabel('After-tax income (model units)')
    ax.set_title('Simulated Income Fan Chart')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_replacement_dist(model, pc, sim):
    """Histogram of realized replacement rates (pension / last-working-year income)."""
    sim_income = sim["income"]
    sim_alive = sim["alive"]
    retire_t = model.retire_age - model.start_age
    n_age = sim_income.shape[1]

    t_last_work = retire_t - 1
    t_first_pens = retire_t + 1

    fig, ax = plt.subplots(figsize=(8, 5))

    if t_first_pens < n_age:
        mask = sim_alive[:, t_last_work].astype(bool) & sim_alive[:, t_first_pens].astype(bool)
        y_last = sim_income[mask, t_last_work]
        y_pens = sim_income[mask, t_first_pens]
        safe = y_last > 1e-6
        repl = y_pens[safe] / y_last[safe]

        ax.hist(repl, bins=60, density=True, alpha=0.7, edgecolor='black', linewidth=0.5,
                range=(0, min(3.0, np.percentile(repl, 99))))
        median_r = float(np.median(repl))
        ax.axvline(median_r, color='red', linestyle='--', label=f'Median = {median_r:.1%}')
    else:
        ax.text(0.5, 0.5, 'Simulation too short', transform=ax.transAxes, ha='center')

    ax.set_xlabel('Replacement rate (pension / last working year income)')
    ax.set_ylabel('Density')
    ax.set_title('Realized Replacement Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_survivor_selection(model, pc, sim):
    """Mean z among alive agents by age -- should rise if mortality is income-dependent."""
    sim_z = sim.get("z", None)
    sim_z_idx = sim["z_idx"]
    sim_alive = sim["alive"]
    ages = sim["ages"]
    n_age = len(ages)

    mean_z = np.full(n_age, np.nan)
    for t in range(n_age):
        alive_mask = sim_alive[:, t].astype(bool)
        if alive_mask.sum() < 5:
            continue
        if sim_z is not None:
            mean_z[t] = float(np.mean(sim_z[alive_mask, t]))
        else:
            mean_z[t] = float(np.mean(pc.z_grid[sim_z_idx[alive_mask, t]]))

    fig, ax = plt.subplots(figsize=(10, 5))
    valid = ~np.isnan(mean_z)
    ax.plot(ages[valid], mean_z[valid], 'b-o', markersize=3)
    ax.axvline(model.retire_age, color='gray', linestyle='--', alpha=0.5, label='Retirement')
    ax.axhline(0, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('Age')
    ax.set_ylabel('Mean z among survivors')
    ax.set_title('Survivor Selection Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_income_post(model, pc, sim):
    """All 4 post-sim labour/SS figures. Returns list of figures."""
    return [
        plot_z_distribution(model, pc, sim),
        plot_income_fan(model, pc, sim),
        plot_replacement_dist(model, pc, sim),
        plot_survivor_selection(model, pc, sim),
    ]
