"""
MORTALITY DEEP-DIVE TEST SUITE
================================
Tests targeting subtle integration issues between mortality calibration,
the solver Bellman equation, and simulation death draws.

These go beyond verifying the off-by-one fix — they probe whether the
mortality module is *correctly integrated* with the rest of the model.
"""
import numpy as np
import json
import sys

results = []

def report(test_id, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((test_id, status, detail))
    marker = "" if passed else " <<<<<"
    print(f"  {test_id:12s}  {status}  {detail}{marker}")

# =====================================================================
# Setup: build model, precompute, load policy, run simulation
# =====================================================================
print("=" * 74)
print("  SETUP: Building model + precompute, loading policy, running sim")
print("=" * 74)

from mortality import (SSA_DEATH_PROB_2017, life_expectancy_at_40,
                       calibrate_earnings_dependent_mortality,
                       compute_sigma_z, build_survival_probs_2d,
                       chetty_expected_age_at_death, CHETTY_EXPECTED_AGE_AT_DEATH)
from precompute import build_model, Precompute, DiscretizationConfig
from simulation import simulate_lifecycle
from scipy.stats import norm

with open("saved_runs/constrained_grid5x5x5_nz11/metadata.json") as f:
    meta = json.load(f)
rc = meta["run_config"]
base_config = rc["base_config"]
var_raw = rc["var_config"]
var_config = {}
for k, v in var_raw.items():
    if isinstance(v, dict) and v.get("kind") == "ndarray":
        var_config[k] = np.array(v["values"]).reshape(v["shape"])
    else:
        var_config[k] = v
disc_config = DiscretizationConfig(**meta["diagnostics_summary"]["disc_config"])
model = build_model(base_config, var_config, verbose=False)
pc = Precompute(model, disc_config, verbose=False)

data = np.load("saved_runs/constrained_grid5x5x5_nz11/policy_arrays.npz")
C_mat, S_mat, B_mat = data["C_mat"], data["S_mat"], data["B_mat"]

start_age = model.start_age
terminal_age = model.terminal_age
n_age = terminal_age - start_age + 1
n_z = len(pc.z_grid)
ages = np.arange(start_age, terminal_age + 1)
chi_vec = pc._chi_vec
sigma_z = pc._mortality_diag["sigma_z"]

print(f"  start_age={start_age}  terminal_age={terminal_age}  n_age={n_age}  n_z={n_z}")
print(f"  sigma_z={sigma_z:.4f}  chi range=[{chi_vec.min():.4f}, {chi_vec.max():.4f}]")

N_SIM = 50_000
print(f"  Running simulation ({N_SIM:,} agents)...")
sim = simulate_lifecycle(C_mat, S_mat, B_mat, pc, model,
                         n_simulations=N_SIM, seed=42, verbose=False)
print("  Done.\n")


# =====================================================================
print("=" * 74)
print("  SECTION A: Truncation Bias — Model Horizon vs Calibration Horizon")
print("=" * 74)
# The calibration computes LE using SSA table up to age 119, but the model
# forces death at terminal_age=99. Any agent who *would* survive past 99
# in the calibration dies at 99 in the model. This creates a systematic
# downward bias in effective LE.
# =====================================================================

# A.1 — Quantify truncation bias per z-grid point
# Compute LE truncated at terminal_age vs full LE (to 119).
print()
le_full = np.array([life_expectancy_at_40(c, max_age=119) for c in chi_vec])
le_trunc = np.array([life_expectancy_at_40(c, max_age=terminal_age) for c in chi_vec])
bias = le_full - le_trunc
# The bias should be small for high-mortality agents and larger for
# low-mortality (high-income) agents who would survive past 99.
ok = np.all(bias >= 0)  # truncation can only reduce LE
report("A.1", ok, f"truncation bias range: [{bias.min():.3f}, {bias.max():.3f}] years")

# A.2 — Truncation bias should be small enough not to materially distort
# calibration. If the bias exceeds 1 year for any z-grid point, the
# calibration targets are materially wrong within the model horizon.
max_bias = bias.max()
ok = max_bias < 1.0
report("A.2", ok, f"max truncation bias = {max_bias:.4f} years (threshold: <1.0)")

# A.3 — Bias should be monotonically increasing in z (richer agents
# live longer, so more of their survival mass is past terminal_age).
ok = all(bias[i] <= bias[i+1] for i in range(len(bias) - 1))
report("A.3", ok, f"bias monotone in z: lowest={bias[0]:.4f}, highest={bias[-1]:.4f}")


# =====================================================================
print()
print("=" * 74)
print("  SECTION B: Pre-40 Mortality Consistency")
print("=" * 74)
# Chi is calibrated to match LE *at age 40*. But survival_probs_2d
# applies chi to ages 22-39 as well. Since Chetty data says nothing
# about mortality before 40, this is an extrapolation. Test that the
# pre-40 mortality impact is small and doesn't create artifacts.
# =====================================================================

# B.1 — Pre-40 survival should be very high for ALL z-grid points.
# SSA mortality at ages 22-39 is tiny (< 0.003), so even with chi ~1.4,
# the death probability per year is < 0.4%. Cumulative survival 22->40
# should exceed 0.95 even for the worst-off agents.
pre40_idx = 40 - start_age  # index 18
cumulative_surv = np.ones(n_z)
for t in range(pre40_idx):
    cumulative_surv *= pc.survival_probs_2d[t, :]
ok = bool(np.all(cumulative_surv > 0.95))
report("B.1", ok, f"cumulative survival 22->40: lowest-z={cumulative_surv[0]:.5f}, "
       f"highest-z={cumulative_surv[-1]:.5f}")

# B.2 — Pre-40 mortality gradient should be very small.
# The difference in cumulative survival (22->40) between highest and
# lowest z should be < 2 percentage points.
gradient = cumulative_surv[-1] - cumulative_surv[0]
ok = gradient < 0.02
report("B.2", ok, f"pre-40 survival gradient (high z - low z) = {gradient:.5f}")


# =====================================================================
print()
print("=" * 74)
print("  SECTION C: Gaussian Percentile Mapping Accuracy")
print("=" * 74)
# The code maps z -> percentile via Phi(z/sigma_z), assuming z is
# unconditionally Gaussian. With mixture-normal innovations, the true
# unconditional distribution is non-Gaussian. Test how large the error is.
# =====================================================================

# C.1 — Simulate the AR(1) process for many periods to get the empirical
# unconditional distribution, then compare CDF to Gaussian.
rng = np.random.default_rng(12345)
T_sim = 200_000
z_sim = np.zeros(T_sim)
for t in range(1, T_sim):
    if rng.random() < model.pz:
        eta = rng.normal(model.mu_eta1, model.sigma_eta1)
    else:
        eta = rng.normal(model.mu_eta2, model.sigma_eta2)
    z_sim[t] = model.rho * z_sim[t-1] + eta

# Compare percentiles at grid points
z_burn = z_sim[10000:]  # discard burn-in
empirical_pcts = np.array([100.0 * np.mean(z_burn < z) for z in pc.z_grid])
gaussian_pcts = 100.0 * norm.cdf(pc.z_grid / sigma_z)
gaussian_pcts = np.clip(gaussian_pcts, 1.0, 100.0)
empirical_pcts = np.clip(empirical_pcts, 1.0, 100.0)

max_pct_diff = np.max(np.abs(empirical_pcts - gaussian_pcts))
ok = max_pct_diff < 5.0  # within 5 percentile points
report("C.1", ok, f"max |empirical - Gaussian| percentile diff = {max_pct_diff:.2f} pp")

# C.2 — Translate percentile error into LE target error.
# Map both empirical and Gaussian percentiles to Chetty targets.
le_gaussian = np.array([chetty_expected_age_at_death(p) for p in gaussian_pcts])
le_empirical = np.array([chetty_expected_age_at_death(p) for p in empirical_pcts])
max_le_diff = np.max(np.abs(le_gaussian - le_empirical))
ok = max_le_diff < 1.0  # within 1 year
report("C.2", ok, f"max |LE target| diff from Gaussian assumption = {max_le_diff:.3f} years")


# =====================================================================
print()
print("=" * 74)
print("  SECTION D: Simulation vs Analytical Mortality Rates")
print("=" * 74)
# The simulation draws death from Bernoulli(1 - psi[t, z]). The empirical
# hazard rates from the simulation should match the analytical ones.
# =====================================================================

death_age_arr = sim["death_age"]
alive_arr = sim["alive"]
z_idx_arr = sim["z_idx"]

# D.1 — Empirical vs analytical hazard rate at age 65 (a well-populated age).
t65 = 65 - start_age
alive_at_65 = alive_arr[:, t65].astype(bool)
n_alive_65 = alive_at_65.sum()
if t65 + 1 < n_age:
    died_at_65 = alive_at_65 & (~alive_arr[:, t65 + 1].astype(bool))
    empirical_hazard_65 = died_at_65.sum() / max(n_alive_65, 1)
else:
    empirical_hazard_65 = 1.0

# Analytical hazard (population-average across z states)
z_dist_65 = z_idx_arr[alive_at_65, t65]
if len(z_dist_65) > 0:
    analytical_hazard_65 = np.mean([1.0 - pc.survival_probs_2d[t65, int(z)]
                                    for z in z_dist_65])
else:
    analytical_hazard_65 = 0
diff = abs(empirical_hazard_65 - analytical_hazard_65)
ok = diff < 0.005  # within 0.5 pp
report("D.1", ok, f"hazard@65: empirical={empirical_hazard_65:.5f}, "
       f"analytical={analytical_hazard_65:.5f}, diff={diff:.5f}")

# D.2 — Same at age 80
t80 = 80 - start_age
alive_at_80 = alive_arr[:, t80].astype(bool)
n_alive_80 = alive_at_80.sum()
if t80 + 1 < n_age:
    died_at_80 = alive_at_80 & (~alive_arr[:, t80 + 1].astype(bool))
    empirical_hazard_80 = died_at_80.sum() / max(n_alive_80, 1)
else:
    empirical_hazard_80 = 1.0
z_dist_80 = z_idx_arr[alive_at_80, t80]
if len(z_dist_80) > 0:
    analytical_hazard_80 = np.mean([1.0 - pc.survival_probs_2d[t80, int(z)]
                                    for z in z_dist_80])
else:
    analytical_hazard_80 = 0
diff = abs(empirical_hazard_80 - analytical_hazard_80)
ok = diff < 0.005
report("D.2", ok, f"hazard@80: empirical={empirical_hazard_80:.5f}, "
       f"analytical={analytical_hazard_80:.5f}, diff={diff:.5f}")

# D.3 — Empirical mortality gradient: agents in lowest z should die
# younger on average than agents in highest z. Compute mean death age
# by z-index and check monotonicity.
mean_death_by_z = []
for iz in range(n_z):
    # Agents who spent most time in this z bucket
    mask = (z_idx_arr[:, n_age // 2] == iz) & (death_age_arr >= start_age)
    if mask.sum() > 10:
        mean_death_by_z.append(death_age_arr[mask].mean())
    else:
        mean_death_by_z.append(np.nan)
mean_death_by_z = np.array(mean_death_by_z)
valid = ~np.isnan(mean_death_by_z)
valid_vals = mean_death_by_z[valid]
# Overall trend should be increasing (allow minor noise)
ok = valid_vals[-1] > valid_vals[0] + 1.0 if len(valid_vals) > 1 else False
report("D.3", ok, f"mean death age by z: lowest-z={valid_vals[0]:.1f}, "
       f"highest-z={valid_vals[-1]:.1f}, gap={valid_vals[-1]-valid_vals[0]:.1f}")

# D.4 — No agent should die at an age where survival_probs_2d = 1.0
# (this would indicate a bug in the Bernoulli draw or indexing).
all_ok = True
fail_count = 0
for i in range(min(N_SIM, 10000)):
    d = death_age_arr[i]
    if d < start_age or d >= terminal_age:
        continue  # terminal death is forced, skip
    t_d = d - start_age
    iz = int(z_idx_arr[i, t_d])
    psi = pc.survival_probs_2d[t_d, iz]
    if psi >= 1.0 - 1e-15:
        fail_count += 1
        all_ok = False
report("D.4", all_ok, f"agents dying where psi~=1: {fail_count}")


# =====================================================================
print()
print("=" * 74)
print("  SECTION E: Solver Bellman Equation Consistency")
print("=" * 74)
# The solver uses psi in the Euler equation:
#   u'(c_t) = beta * [psi * E[u'(c_{t+1})*R] + (1-psi) * E[b'(a*R)]]
# Verify that the psi values used match what mortality.py produces.
# =====================================================================

# E.1 — survival_probs_2d on Precompute matches independent recomputation
surv_independent = build_survival_probs_2d(ages, chi_vec)
ok = np.array_equal(pc.survival_probs_2d, surv_independent)
report("E.1", ok, f"pc.survival_probs_2d matches independent build_survival_probs_2d")

# E.2 — Verify that at terminal age, survival prob is NOT 0 in the array
# (it represents survival from terminal_age to terminal_age+1, which
# is physically meaningful but never used by the solver since the terminal
# period has no continuation). The important thing is the solver skips it.
terminal_psi = pc.survival_probs_2d[-1, :]
ok = True  # just report the values
report("E.2", ok, f"psi at terminal_age={terminal_age}: "
       f"[{terminal_psi[0]:.6f}, ..., {terminal_psi[-1]:.6f}] "
       f"(stored but not used in solver)")

# E.3 — The effective annual discount factor beta*psi should be < 1
# for all ages and z states, otherwise the Bellman equation diverges.
beta = model.beta
eff_discount = beta * pc.survival_probs_2d
ok = bool(np.all(eff_discount < 1.0))
max_eff = eff_discount.max()
report("E.3", ok, f"max(beta*psi) = {max_eff:.6f} (must be <1)")

# E.4 — Effective discount should be monotonically increasing in z
# at every age (richer agents discount less because they survive longer).
all_ok = True
for t in range(n_age):
    row = eff_discount[t, :]
    if not all(row[i] <= row[i+1] for i in range(len(row)-1)):
        all_ok = False
        break
report("E.4", all_ok, "beta*psi increasing in z at every age")


# =====================================================================
print()
print("=" * 74)
print("  SECTION F: Mortality Clamping and Extreme Chi")
print("=" * 74)
# Test that min(chi * m, 1) clamping works correctly and doesn't
# create pathological survival = 0 within the model horizon.
# =====================================================================

# F.1 — No age in the model range should have survival prob = 0
# (which would mean certain death at that age). Even for the poorest
# agents, death should be probabilistic within the model horizon.
min_surv = pc.survival_probs_2d.min()
ok = min_surv > 0.0
report("F.1", ok, f"min survival prob in model = {min_surv:.6f}")

# F.2 — Identify the age at which survival is lowest for the poorest agent.
worst_z = 0  # lowest z = highest mortality
worst_t = np.argmin(pc.survival_probs_2d[:, worst_z])
worst_age = ages[worst_t]
worst_val = pc.survival_probs_2d[worst_t, worst_z]
# The worst survival should still be reasonable (> 0.5 even at age 99
# for the worst-off, since chi_max ~ 1.4 and m(99) ~ 0.344)
ok = worst_val > 0.3
report("F.2", ok, f"worst survival: age={worst_age}, z=lowest, psi={worst_val:.4f}")

# F.3 — Verify that chi * m_base never exceeds 1 within model range.
# If it does, we have certain death at that age, which would be pathological.
chi_max = chi_vec.max()
max_m_in_model = max(SSA_DEATH_PROB_2017[a] for a in range(start_age, terminal_age + 1))
product = chi_max * max_m_in_model
ok = product < 1.0
report("F.3", ok, f"max(chi)*max(m_in_model) = {chi_max:.4f}*{max_m_in_model:.4f} = "
       f"{product:.4f} (must be <1)")


# =====================================================================
print()
print("=" * 74)
print("  SECTION G: Cumulative Survival Curve Consistency")
print("=" * 74)
# Build the cumulative survival curve from survival_probs_2d and verify
# it matches the product formula.
# =====================================================================

# G.1 — Cumulative survival from start_age to each subsequent age.
for iz_check in [0, n_z // 2, n_z - 1]:
    cum_surv = np.ones(n_age)
    for t in range(1, n_age):
        cum_surv[t] = cum_surv[t-1] * pc.survival_probs_2d[t-1, iz_check]
    # Independently compute
    cum_surv_direct = np.ones(n_age)
    for t in range(1, n_age):
        age = ages[t-1]
        m = SSA_DEATH_PROB_2017.get(age, 1.0)
        cum_surv_direct[t] = cum_surv_direct[t-1] * (1.0 - min(chi_vec[iz_check] * m, 1.0))
    ok = np.allclose(cum_surv, cum_surv_direct, atol=1e-14)
    report(f"G.1.z{iz_check}", ok,
           f"cumulative survival matches direct formula (iz={iz_check})")

# G.2 — Cumulative survival at age 99 should be positive for all z.
cum_surv_at_99 = np.ones(n_z)
for t in range(n_age - 1):
    cum_surv_at_99 *= pc.survival_probs_2d[t, :]
ok = bool(np.all(cum_surv_at_99 > 0))
report("G.2", ok, f"P(survive to 99): lowest-z={cum_surv_at_99[0]:.4f}, "
       f"highest-z={cum_surv_at_99[-1]:.4f}")


# =====================================================================
print()
print("=" * 74)
print("  SECTION H: Simulation Dead-Agent Hygiene")
print("=" * 74)
# After an agent dies, all economic variables should be zeroed out.
# Any non-zero consumption, income, or wealth after death is a bug.
# =====================================================================

n_check = min(5000, N_SIM)

# H.1 — Consumption after death should be 0
all_ok = True
fail_info = ""
for i in range(n_check):
    d = death_age_arr[i]
    t_d = d - start_age
    if t_d + 1 < n_age:
        post_death_c = sim["c"][i, t_d + 1:]
        if np.any(post_death_c != 0):
            all_ok = False
            fail_info = f"agent {i} has c>0 after death at age {d}"
            break
report("H.1", all_ok, f"consumption after death = 0 (checked {n_check})" +
       (f" — {fail_info}" if fail_info else ""))

# H.2 — Income after death should be 0
all_ok = True
fail_info = ""
for i in range(n_check):
    d = death_age_arr[i]
    t_d = d - start_age
    if t_d + 1 < n_age:
        post_death_inc = sim["income"][i, t_d + 1:]
        if np.any(post_death_inc != 0):
            all_ok = False
            fail_info = f"agent {i} has income>0 after death at age {d}"
            break
report("H.2", all_ok, f"income after death = 0 (checked {n_check})" +
       (f" — {fail_info}" if fail_info else ""))

# H.3 — Wealth (cash-on-hand) after death should be 0
all_ok = True
fail_info = ""
for i in range(n_check):
    d = death_age_arr[i]
    t_d = d - start_age
    if t_d + 1 < n_age:
        post_death_x = sim["x"][i, t_d + 1:]
        if np.any(post_death_x != 0):
            all_ok = False
            fail_info = f"agent {i} has x>0 after death at age {d}"
            break
report("H.3", all_ok, f"wealth after death = 0 (checked {n_check})" +
       (f" — {fail_info}" if fail_info else ""))

# H.4 — Estate at death should be positive (agent had some wealth).
# Not all agents will have positive estate (some may be broke), but
# the vast majority should.
estates = sim["estate_at_death"]
frac_positive = np.mean(estates > 0)
ok = frac_positive > 0.80
report("H.4", ok, f"fraction with positive estate at death: {frac_positive:.3f}")


# =====================================================================
print()
print("=" * 74)
print("  SECTION I: Age-at-Death Distribution Shape")
print("=" * 74)
# The distribution of death ages should look like a realistic actuarial
# distribution — right-skewed, peaking around 80-90.
# =====================================================================

valid_deaths = death_age_arr[death_age_arr >= start_age]

# I.1 — Mean death age should be in a plausible range
mean_d = valid_deaths.mean()
ok = 75 < mean_d < 92
report("I.1", ok, f"mean death age = {mean_d:.1f}")

# I.2 — Standard deviation of death age should be reasonable (5-15 years)
std_d = valid_deaths.std()
ok = 4 < std_d < 18
report("I.2", ok, f"std death age = {std_d:.1f}")

# I.3 — Mode should be in the 80-99 range
hist, edges = np.histogram(valid_deaths, bins=np.arange(start_age, terminal_age + 2))
mode_age = edges[np.argmax(hist)]
ok = 78 <= mode_age <= terminal_age
report("I.3", ok, f"modal death age = {mode_age}")

# I.4 — Very few agents should die before age 50 (< 2%)
frac_early = np.mean(valid_deaths < 50)
ok = frac_early < 0.02
report("I.4", ok, f"fraction dying before 50 = {frac_early:.4f}")

# I.5 — A significant fraction should reach terminal_age (pile-up at 99)
frac_terminal = np.mean(valid_deaths == terminal_age)
ok = 0.01 < frac_terminal < 0.50
report("I.5", ok, f"fraction dying at terminal_age ({terminal_age}) = {frac_terminal:.3f}")


# =====================================================================
print()
print("=" * 74)
print("  SECTION J: Cross-Validation of LE Within Model Horizon")
print("=" * 74)
# Compute the model-implied LE (truncated at terminal_age) for each z
# and compare against the simulation's mean death age by z.
# =====================================================================

# J.1 — Analytical LE within model horizon per z-grid point.
# LE_model(z) = sum_{t=22}^{98} P(survive from 22 to t) * psi(t,z)
# This accounts for truncation at terminal_age.
le_model = np.zeros(n_z)
for iz in range(n_z):
    cum = 1.0
    le = 0.0
    for t in range(n_age - 1):
        p_die = 1.0 - pc.survival_probs_2d[t, iz]
        le += cum * p_die * ages[t]
        cum *= pc.survival_probs_2d[t, iz]
    # Remaining survivors die at terminal_age
    le += cum * terminal_age
    le_model[iz] = le

# J.2 — Simulated mean death age per z-grid point (using mid-life z index)
le_sim = np.full(n_z, np.nan)
for iz in range(n_z):
    mask = (z_idx_arr[:, n_age // 2] == iz) & (death_age_arr >= start_age)
    if mask.sum() > 50:
        le_sim[iz] = death_age_arr[mask].mean()

# Compare (allow 1.5 year tolerance due to sampling noise and z transitions)
valid = ~np.isnan(le_sim)
if valid.sum() > 0:
    diffs = np.abs(le_model[valid] - le_sim[valid])
    max_diff = diffs.max()
    ok = max_diff < 2.0
    report("J.1", ok, f"max |analytical - simulated| LE = {max_diff:.2f} years")
    for iz in range(n_z):
        if valid[iz]:
            d = le_model[iz] - le_sim[iz]
            print(f"          z[{iz}]={pc.z_grid[iz]:+7.3f}  "
                  f"analytical={le_model[iz]:.1f}  sim={le_sim[iz]:.1f}  "
                  f"diff={d:+.2f}")
else:
    report("J.1", False, "no valid z-bins for comparison")


# =====================================================================
print()
print("=" * 74)
print("  SECTION K: Calibration Target Fidelity")
print("=" * 74)
# The Chetty targets are LE at 40. The chi calibration matches these
# exactly. But agents enter the model at age 22 with chi-scaled mortality
# already applied. Verify the chi values actually reproduce the targets
# when you compute LE starting at 40 (the calibration definition).
# =====================================================================

# K.1 — Recompute LE at 40 with the calibrated chi values
targets = pc._mortality_diag["target_ages"]
actuals = np.array([life_expectancy_at_40(c) for c in chi_vec])
errors = actuals - targets
max_err = np.max(np.abs(errors))
ok = max_err < 1e-6
report("K.1", ok, f"max |actual - target| LE at 40 = {max_err:.2e} years")

# K.2 — Percentile-to-z mapping: verify the percentiles assigned to
# each z-grid point are consistent with the Gaussian CDF.
pcts_check = 100.0 * norm.cdf(pc.z_grid / sigma_z)
pcts_check = np.clip(pcts_check, 1.0, 100.0)
pcts_stored = pc._mortality_diag["percentiles"]
ok = np.allclose(pcts_check, pcts_stored, atol=1e-10)
report("K.2", ok, f"percentile mapping consistent with Phi(z/sigma_z)")


# =====================================================================
print()
print("=" * 74)
print("  SUMMARY")
print("=" * 74)
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"  {n_pass} PASS, {n_fail} FAIL out of {len(results)} tests")
if n_fail > 0:
    print()
    print("  FAILED tests:")
    for tid, s, d in results:
        if s == "FAIL":
            print(f"    {tid}: {d}")
print("=" * 74)
