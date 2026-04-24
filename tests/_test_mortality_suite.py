"""
MORTALITY MODULE — FULL VERIFICATION TEST SUITE (Sections 1-6, 9)
"""
import numpy as np
from mortality import (SSA_DEATH_PROB_2017, CHETTY_EXPECTED_AGE_AT_DEATH,
                       chetty_expected_age_at_death, compute_sigma_z,
                       life_expectancy_at_40, calibrate_chi_vector,
                       calibrate_earnings_dependent_mortality,
                       build_survival_probs_2d)

# Production parameters
rho = 0.991
pz = 0.176
mu_eta1 = -0.524
sigma_eta1 = 0.113
mu_eta2 = -(pz / (1.0 - pz)) * mu_eta1
sigma_eta2 = 0.046
start_age = 22
terminal_age = 99
n_z = 11
n_stds = 3.0

sigma_z = compute_sigma_z(rho, pz, mu_eta1, sigma_eta1, mu_eta2, sigma_eta2)
z_grid = np.linspace(-n_stds * sigma_z, n_stds * sigma_z, n_z)

results = []

def report(test_id, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((test_id, status, detail))
    print(f"  {test_id:12s}  {status}  {detail}")

# =====================================================================
print("=" * 70)
print("SECTION 1: SSA Baseline Table Integrity")
print("=" * 70)

# 1.1
ok = len(SSA_DEATH_PROB_2017) == 120 and all(a in SSA_DEATH_PROB_2017 for a in range(120))
report("1.1", ok, f"len={len(SSA_DEATH_PROB_2017)}")

# 1.2
ok = all(0 < q <= 1 for q in SSA_DEATH_PROB_2017.values())
report("1.2", ok, "all 0<q<=1")

# 1.3
ok = all(SSA_DEATH_PROB_2017[a+1] >= SSA_DEATH_PROB_2017[a] for a in range(40, 119))
report("1.3", ok, "monotone 40-118")

# 1.4
ok = SSA_DEATH_PROB_2017[119] == 1.0
report("1.4", ok, f"q(119)={SSA_DEATH_PROB_2017[119]}")

# 1.5
checks = [
    (40, 0.00258, 0.001),
    (65, 0.01446, 0.001),
    (85, 0.08868, 0.005),
]
all_ok = True
details = []
for age, target, tol in checks:
    diff = abs(SSA_DEATH_PROB_2017[age] - target)
    ok_i = diff < tol
    all_ok = all_ok and ok_i
    details.append(f"q({age})={SSA_DEATH_PROB_2017[age]:.5f} vs {target}")
report("1.5", all_ok, "; ".join(details))

# =====================================================================
print()
print("=" * 70)
print("SECTION 2: Chetty Target Data Integrity")
print("=" * 70)

# 2.1
ok = set(CHETTY_EXPECTED_AGE_AT_DEATH.keys()) == set(range(1, 101))
report("2.1", ok, f"keys 1-100, count={len(CHETTY_EXPECTED_AGE_AT_DEATH)}")

# 2.2
mn = min(CHETTY_EXPECTED_AGE_AT_DEATH.values())
mx = max(CHETTY_EXPECTED_AGE_AT_DEATH.values())
ok = all(70 < v < 95 for v in CHETTY_EXPECTED_AGE_AT_DEATH.values())
report("2.2", ok, f"range [{mn:.1f}, {mx:.1f}]")

# 2.3
ok1 = CHETTY_EXPECTED_AGE_AT_DEATH[100] > CHETTY_EXPECTED_AGE_AT_DEATH[1] + 10
ok2 = CHETTY_EXPECTED_AGE_AT_DEATH[75] > CHETTY_EXPECTED_AGE_AT_DEATH[25]
report("2.3", ok1 and ok2, f"p100-p1={CHETTY_EXPECTED_AGE_AT_DEATH[100]-CHETTY_EXPECTED_AGE_AT_DEATH[1]:.1f}, p75>p25={ok2}")

# 2.4
gap = CHETTY_EXPECTED_AGE_AT_DEATH[100] - CHETTY_EXPECTED_AGE_AT_DEATH[1]
ok = 11.0 < gap < 14.0
report("2.4", ok, f"gap={gap:.2f}")

# 2.5
ok1 = chetty_expected_age_at_death(1) == CHETTY_EXPECTED_AGE_AT_DEATH[1]
ok2 = chetty_expected_age_at_death(50) == CHETTY_EXPECTED_AGE_AT_DEATH[50]
ok3 = chetty_expected_age_at_death(100) == CHETTY_EXPECTED_AGE_AT_DEATH[100]
mid = chetty_expected_age_at_death(50.5)
ok4 = CHETTY_EXPECTED_AGE_AT_DEATH[50] < mid < CHETTY_EXPECTED_AGE_AT_DEATH[51]
report("2.5", ok1 and ok2 and ok3 and ok4, f"exact@1,50,100; interp@50.5={mid:.4f}")

# =====================================================================
print()
print("=" * 70)
print("SECTION 3: Unconditional Standard Deviation sigma_z")
print("=" * 70)

# 3.1
sz_test = compute_sigma_z(rho=0.97, pz=1.0, mu_eta1=0.0, sigma_eta1=0.10,
                          mu_eta2=0.0, sigma_eta2=999.0)
expected = 0.10 / np.sqrt(1 - 0.97**2)
ok = abs(sz_test - expected) < 1e-12
report("3.1", ok, f"got={sz_test:.10f}, expected={expected:.10f}")

# 3.2
sz_test = compute_sigma_z(rho=0.97, pz=0.5, mu_eta1=0.05, sigma_eta1=0.10,
                          mu_eta2=-0.05, sigma_eta2=0.10)
expected = np.sqrt(0.0125 / (1 - 0.97**2))
ok = abs(sz_test - expected) < 1e-12
report("3.2", ok, f"got={sz_test:.10f}, expected={expected:.10f}")

# 3.3
ok = sigma_z > 0 and np.isfinite(sigma_z)
report("3.3", ok, f"sigma_z={sigma_z:.6f}")

# =====================================================================
print()
print("=" * 70)
print("SECTION 4: Life Expectancy Formula (post-fix)")
print("=" * 70)

# 4.1
m_toy = {40: 0.5, 41: 0.5, 42: 1.0}
r = life_expectancy_at_40(chi=1.0, m_table=m_toy, max_age=42)
ok = abs(r - 40.75) < 1e-12
report("4.1", ok, f"toy={r} (expected 40.75)")

# 4.2
m_instant = {40: 1.0}
r = life_expectancy_at_40(chi=1.0, m_table=m_instant, max_age=40)
ok = abs(r - 40.0) < 1e-12
report("4.2", ok, f"instant death={r} (expected 40.0)")

# 4.3
m_zero = {a: 0.0 for a in range(40, 120)}
m_zero[119] = 1.0
r = life_expectancy_at_40(chi=1.0, m_table=m_zero, max_age=119)
ok = abs(r - 119.0) < 1e-10
report("4.3", ok, f"immortal={r} (expected 119.0)")

# 4.4
S = {40: 1.0}
for a in range(40, 120):
    S[a+1] = S[a] * (1 - SSA_DEATH_PROB_2017[a])
E_direct = sum(a * (S[a] - S.get(a+1, 0.0)) for a in range(40, 121))
r = life_expectancy_at_40(chi=1.0)
ok = abs(r - E_direct) < 1e-8
report("4.4", ok, f"func={r:.8f}, direct={E_direct:.8f}, diff={abs(r-E_direct):.2e}")

# 4.5
le_low = life_expectancy_at_40(chi=0.5)
le_mid = life_expectancy_at_40(chi=1.0)
le_high = life_expectancy_at_40(chi=2.0)
ok = le_low > le_mid > le_high
report("4.5", ok, f"chi=0.5:{le_low:.2f} > chi=1.0:{le_mid:.2f} > chi=2.0:{le_high:.2f}")

# 4.6
le_a = life_expectancy_at_40(chi=1.0)
le_b = life_expectancy_at_40(chi=1.001)
ok = abs(le_a - le_b) < 0.1
report("4.6", ok, f"delta={abs(le_a-le_b):.6f} for 0.1% chi change")

# =====================================================================
print()
print("=" * 70)
print("SECTION 5: Chi Calibration (Brent Solver)")
print("=" * 70)

surv_2d, chi_vec, diag = calibrate_earnings_dependent_mortality(
    start_age=start_age, terminal_age=terminal_age, z_grid=z_grid,
    rho=rho, pz=pz, mu_eta1=mu_eta1, sigma_eta1=sigma_eta1,
    mu_eta2=mu_eta2, sigma_eta2=sigma_eta2, verbose=False)

# 5.1
errors = diag["actual_ages"] - diag["target_ages"]
max_err = np.max(np.abs(errors))
ok = max_err < 1e-6
report("5.1", ok, f"max round-trip error={max_err:.2e} years")

# 5.2
ok = all(chi_vec[i] > chi_vec[i+1] for i in range(len(chi_vec)-1))
report("5.2", ok, f"chi monotone decreasing: [{chi_vec[0]:.4f}, ..., {chi_vec[-1]:.4f}]")

# 5.3
ok = all(0.1 < c < 5.0 for c in chi_vec)
report("5.3", ok, f"chi range [{chi_vec.min():.4f}, {chi_vec.max():.4f}]")

# 5.4
pcts = diag["percentiles"]
ok = pcts[0] < 5.0 and pcts[-1] > 95.0
report("5.4", ok, f"percentile range [{pcts[0]:.1f}, {pcts[-1]:.1f}]")

# =====================================================================
print()
print("=" * 70)
print("SECTION 6: 2D Survival Probability Array")
print("=" * 70)

ages = np.arange(start_age, terminal_age + 1)
n_age = len(ages)

# 6.1
ok = surv_2d.shape == (n_age, n_z)
report("6.1", ok, f"shape={surv_2d.shape}, expected=({n_age}, {n_z})")

# 6.2
ok = bool(np.all(surv_2d > 0) and np.all(surv_2d <= 1))
report("6.2", ok, f"min={surv_2d.min():.8f}, max={surv_2d.max():.8f}")

# 6.3
age_40_idx = 40 - start_age
all_ok = True
for iz in range(n_z):
    col = surv_2d[age_40_idx:, iz]
    if not all(col[i] >= col[i+1] for i in range(len(col)-1)):
        all_ok = False
        break
report("6.3", all_ok, "decreasing in age (from 40) for all z")

# 6.4
all_ok = True
for t in range(n_age):
    row = surv_2d[t, :]
    if not all(row[i] <= row[i+1] for i in range(len(row)-1)):
        all_ok = False
        break
report("6.4", all_ok, "increasing in z for all ages")

# 6.5
spot_checks = [(0, 0), (0, n_z-1), (n_age//2, 0), (n_age//2, n_z//2), (n_age-1, 0), (n_age-1, n_z-1)]
all_ok = True
for t_check, iz_check in spot_checks:
    age = ages[t_check]
    m_base = SSA_DEATH_PROB_2017.get(age, 1.0)
    expected = 1.0 - min(chi_vec[iz_check] * m_base, 1.0)
    if abs(surv_2d[t_check, iz_check] - expected) > 1e-15:
        all_ok = False
report("6.5", all_ok, f"spot-checked {len(spot_checks)} entries against formula")

# =====================================================================
print()
print("=" * 70)
print("SECTION 9: End-to-End Regression Check")
print("=" * 70)

# 9.1
out1 = calibrate_earnings_dependent_mortality(
    start_age=start_age, terminal_age=terminal_age, z_grid=z_grid,
    rho=rho, pz=pz, mu_eta1=mu_eta1, sigma_eta1=sigma_eta1,
    mu_eta2=mu_eta2, sigma_eta2=sigma_eta2, verbose=False)
out2 = calibrate_earnings_dependent_mortality(
    start_age=start_age, terminal_age=terminal_age, z_grid=z_grid,
    rho=rho, pz=pz, mu_eta1=mu_eta1, sigma_eta1=sigma_eta1,
    mu_eta2=mu_eta2, sigma_eta2=sigma_eta2, verbose=False)
ok1 = np.array_equal(out1[0], out2[0])
ok2 = np.array_equal(out1[1], out2[1])
report("9.1", ok1 and ok2, f"bitwise identical surv_2d={ok1}, chi_vec={ok2}")

# 9.2
_, chi_lo, _ = calibrate_earnings_dependent_mortality(
    start_age=start_age, terminal_age=terminal_age, z_grid=z_grid,
    rho=0.95, pz=pz, mu_eta1=mu_eta1, sigma_eta1=sigma_eta1,
    mu_eta2=mu_eta2, sigma_eta2=sigma_eta2, verbose=False)
_, chi_hi, _ = calibrate_earnings_dependent_mortality(
    start_age=start_age, terminal_age=terminal_age, z_grid=z_grid,
    rho=0.99, pz=pz, mu_eta1=mu_eta1, sigma_eta1=sigma_eta1,
    mu_eta2=mu_eta2, sigma_eta2=sigma_eta2, verbose=False)
range_lo = max(chi_lo) - min(chi_lo)
range_hi = max(chi_hi) - min(chi_hi)
ok = range_hi >= range_lo
report("9.2", ok, f"rho=0.95 range={range_lo:.4f}, rho=0.99 range={range_hi:.4f}")

# =====================================================================
print()
print("=" * 70)
print("SUMMARY (Sections 1-6, 9)")
print("=" * 70)
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"  {n_pass} PASS, {n_fail} FAIL out of {len(results)} tests")
if n_fail > 0:
    print("  FAILED tests:")
    for tid, s, d in results:
        if s == "FAIL":
            print(f"    {tid}: {d}")
print("=" * 70)
