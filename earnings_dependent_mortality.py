"""
EARNINGS-DEPENDENT MORTALITY CALIBRATION
=========================================

Implements Catherine (2025) eq. (35):
    m_{it} = min{ chi(earnings_rank_i) * m(age_t),  1 }

This module uses the persistent income component z_i (already on the DP
grid) as the conditioning variable instead of Catherine's AIYE/L_bar.
Since z freezes at retirement in our model — exactly as AIYE does in
Catherine's — the two implementations are economically equivalent.

Data sources
------------
1. Baseline mortality m(age):
   SSA Period Life Table, 2017, as used in the 2020 Trustees Report.
   Gender-averaged: q(age) = 0.5 * q_male(age) + 0.5 * q_female(age).
   Source: https://www.ssa.gov/oact/STATS/table4c6.html
   (Select "2017 (2020 TR)" from the dropdown.)

   Catherine (2025) Sec. 5.1: "m(age_it) is the average mortality rate at
   that age, which we calibrate as the average across genders from the
   2017 Social Security actuarial life tables."

2. Calibration targets — life expectancy at age 40 by income percentile:
   Chetty, R., Stepner, M., Abraham, S., Lin, S., Scuderi, B., Turner, N.,
   Bergeron, A., & Cutler, D. (2016). "The Association Between Income and
   Life Expectancy in the United States, 2001-2014." JAMA, 315(16), 1750-1766.
   doi:10.1001/jama.2016.4226

   Data downloaded from the Health Inequality Project:
   https://healthinequality.org/dl/health_ineq_online_table_1.csv
   (Table 1: National life expectancy estimates by gender and income
   percentile, pooled 2001-2014. Released under CC0 license.)

   Values used: race-adjusted life expectancy (le_raceadj), gender-averaged
   as 0.5 * le_raceadj_male + 0.5 * le_raceadj_female at each percentile.

References
----------
Catherine, S. (2025). "Interest Rate Risk and Household Portfolios."
    SSRN 4117856. Equation (35), Section 5.1, Appendix D.2.
Chetty, R. et al. (2016). JAMA, 315(16), 1750-1766.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# =============================================================================
# 1. SSA 2017 PERIOD LIFE TABLE — gender-averaged death probabilities
# =============================================================================
# q(age) = 0.5 * q_male(age) + 0.5 * q_female(age)
# Source: SSA 2017 period life table (2020 Trustees Report).
#
# NOTE: Catherine (2025) uses this exact table. If you want the 2022 table
# instead, replace these values from the SSA website (differences are < 0.2 pp
# at all ages). The code works identically with either vintage.
#
# The 2017 male and female q(x) values are sourced from:
#   https://www.ssa.gov/oact/STATS/table4c6.html (select "2017 (2020 TR)")
# We store the gender-averaged values directly. Ages 25-119.

SSA_DEATH_PROB_2017 = {
    # fmt: off
    # Age: 0.5 * q_male + 0.5 * q_female  (2017 period life table, 2020 TR)
    #
    # Ages 0-24: young-age mortality (SSA 2017 period life table, gender-averaged)
    0: 0.005765, 1: 0.000393, 2: 0.000241, 3: 0.000185, 4: 0.000150,
    5: 0.000134, 6: 0.000123, 7: 0.000113, 8: 0.000102, 9: 0.000094,
    10: 0.000091, 11: 0.000099, 12: 0.000125, 13: 0.000173, 14: 0.000240,
    15: 0.000322, 16: 0.000414, 17: 0.000517, 18: 0.000624, 19: 0.000727,
    20: 0.000835, 21: 0.000944, 22: 0.001035, 23: 0.001094, 24: 0.001149,
    # Ages 25-39: low mortality, slowly rising
    25: 0.001238, 26: 0.001312, 27: 0.001391, 28: 0.001475, 29: 0.001572,
    30: 0.001660, 31: 0.001749, 32: 0.001843, 33: 0.001926, 34: 0.001990,
    35: 0.002058, 36: 0.002149, 37: 0.002244, 38: 0.002344, 39: 0.002459,
    # Ages 40-59: working-age mortality accelerates
    40: 0.002578, 41: 0.002702, 42: 0.002826, 43: 0.002964, 44: 0.003110,
    45: 0.003264, 46: 0.003452, 47: 0.003689, 48: 0.003955, 49: 0.004241,
    50: 0.004537, 51: 0.004856, 52: 0.005228, 53: 0.005656, 54: 0.006147,
    55: 0.006694, 56: 0.007272, 57: 0.007905, 58: 0.008589, 59: 0.009308,
    # Ages 60-79: retirement mortality rises steeply
    60: 0.010058, 61: 0.010854, 62: 0.011721, 63: 0.012626, 64: 0.013540,
    65: 0.014458, 66: 0.015380, 67: 0.016373, 68: 0.017541, 69: 0.018886,
    70: 0.020427, 71: 0.022087, 72: 0.023978, 73: 0.026156, 74: 0.028657,
    75: 0.031534, 76: 0.035140, 77: 0.038818, 78: 0.042841, 79: 0.047222,
    # Ages 80-99: old-age mortality
    80: 0.052408, 81: 0.058057, 82: 0.064317, 83: 0.071402, 84: 0.079445,
    85: 0.088683, 86: 0.099187, 87: 0.110770, 88: 0.123535, 89: 0.137037,
    90: 0.152763, 91: 0.169529, 92: 0.188188, 93: 0.208612, 94: 0.230267,
    95: 0.252645, 96: 0.275567, 97: 0.299013, 98: 0.322159, 99: 0.344466,
    # Age 100+: absorbing (or use SSA extrapolation)
    100: 0.367322, 101: 0.390629, 102: 0.414283, 103: 0.438166, 104: 0.462152,
    105: 0.487462, 106: 0.514169, 107: 0.542352, 108: 0.572093, 109: 0.603478,
    110: 0.636599, 111: 0.671553, 112: 0.708442, 113: 0.747374, 114: 0.788125,
    115: 0.827531, 116: 0.868907, 117: 0.912353, 118: 0.957970, 119: 1.000000,
    # fmt: on
}


# =============================================================================
# 2. CHETTY ET AL. (2016) — expected age at death by income percentile
# =============================================================================
# Race-adjusted, gender-averaged (0.5*male + 0.5*female), pooled 2001-2014.
# Source: https://healthinequality.org/dl/health_ineq_online_table_1.csv
# Variable: le_raceadj (race- and ethnicity-adjusted life expectancy at 40).
# Expected age at death = 40 + le_raceadj.  Values below ARE expected age
# at death (i.e. le_raceadj + 40 already applied by the source).
#
# License: CC0 (public domain). Citation required per data provider request:
#   Chetty et al. (2016), JAMA 315(16), 1750-1766.

CHETTY_EXPECTED_AGE_AT_DEATH = {
    # percentile: 0.5 * le_raceadj_female + 0.5 * le_raceadj_male
    1: 75.763309, 2: 77.478988, 3: 78.309616, 4: 78.684387, 5: 78.831105,
    6: 78.927548, 7: 78.993477, 8: 78.992005, 9: 79.046978, 10: 79.070305,
    11: 79.240059, 12: 79.358486, 13: 79.549072, 14: 79.583939, 15: 79.730244,
    16: 79.895939, 17: 79.990459, 18: 80.059986, 19: 80.286095, 20: 80.317074,
    21: 80.464024, 22: 80.599274, 23: 80.685429, 24: 80.841339, 25: 80.913651,
    26: 80.982712, 27: 81.101296, 28: 81.260556, 29: 81.405350, 30: 81.505241,
    31: 81.552456, 32: 81.673573, 33: 81.806541, 34: 81.941910, 35: 82.077366,
    36: 82.140446, 37: 82.267631, 38: 82.347244, 39: 82.367508, 40: 82.588318,
    41: 82.672646, 42: 82.617161, 43: 82.801159, 44: 82.941078, 45: 82.886265,
    46: 83.044335, 47: 83.135143, 48: 83.217624, 49: 83.256656, 50: 83.354004,
    51: 83.344730, 52: 83.528660, 53: 83.482239, 54: 83.567871, 55: 83.670089,
    56: 83.735207, 57: 83.916004, 58: 83.878571, 59: 84.097138, 60: 84.059105,
    61: 83.988396, 62: 84.268509, 63: 84.342495, 64: 84.328243, 65: 84.476604,
    66: 84.484875, 67: 84.476796, 68: 84.686535, 69: 84.677925, 70: 84.725842,
    71: 84.921368, 72: 84.943535, 73: 85.041588, 74: 85.033775, 75: 85.078629,
    76: 85.232964, 77: 85.311436, 78: 85.285045, 79: 85.408848, 80: 85.434781,
    81: 85.484096, 82: 85.659832, 83: 85.750450, 84: 85.855004, 85: 85.879418,
    86: 86.212257, 87: 86.204437, 88: 86.218575, 89: 86.372261, 90: 86.431870,
    91: 86.703694, 92: 86.828930, 93: 86.974483, 94: 87.104309, 95: 87.226860,
    96: 87.366111, 97: 87.528461, 98: 87.617111, 99: 87.762055, 100: 88.105545,
}
# Verification (matches Chetty et al. 2016 headline numbers):
#   p100 - p1 gap, men only:   14.59 years  (paper: 14.6)
#   p100 - p1 gap, women only: 10.09 years  (paper: 10.1)
#   p100 - p1 gap, averaged:   12.34 years


# =============================================================================
# 3. INTERPOLATION OF CHETTY TARGETS
# =============================================================================

def chetty_expected_age_at_death(percentile):
    """
    Interpolate expected age at death from Chetty et al. (2016) data.

    Parameters
    ----------
    percentile : float
        Income percentile in [1, 100].

    Returns
    -------
    float
        Expected age at death (gender-averaged, race-adjusted).
    """
    pcts = sorted(CHETTY_EXPECTED_AGE_AT_DEATH.keys())
    vals = [CHETTY_EXPECTED_AGE_AT_DEATH[p] for p in pcts]

    percentile = np.clip(percentile, pcts[0], pcts[-1])

    for i in range(len(pcts) - 1):
        if pcts[i] <= percentile <= pcts[i + 1]:
            t = (percentile - pcts[i]) / (pcts[i + 1] - pcts[i])
            return vals[i] + t * (vals[i + 1] - vals[i])
    return vals[-1]


# =============================================================================
# 4. LIFE EXPECTANCY COMPUTATION FROM MORTALITY SCHEDULE
# =============================================================================

def life_expectancy_at_40(chi, m_table=None, max_age=119):
    """
    Compute expected age at death for a 40-year-old given mortality scaling chi.

    Catherine (2025) eq. (35): m(age) = min(chi * m_baseline(age), 1)
    Life expectancy at 40 = sum_{a=40}^{max_age} S(a)
    where S(a) = prod_{s=40}^{a-1} (1 - min(chi * m(s), 1))

    Parameters
    ----------
    chi : float
        Mortality scaling factor (chi > 1 => higher mortality).
    m_table : dict, optional
        {age: death_prob}. Defaults to SSA_DEATH_PROB_2017.
    max_age : int
        Maximum age (119 for SSA tables, where q(119)=1).

    Returns
    -------
    float
        Expected age at death = 40 + sum of survival probabilities.
    """
    if m_table is None:
        m_table = SSA_DEATH_PROB_2017

    survival = 1.0
    le_sum = 0.0

    for age in range(40, max_age + 1):
        m_base = m_table.get(age, 1.0)
        m_adj = min(chi * m_base, 1.0)
        le_sum += survival
        survival *= (1.0 - m_adj)
        if survival < 1e-15:
            break

    return 40.0 + le_sum


# =============================================================================
# 5. CALIBRATE CHI AT A SINGLE PERCENTILE
# =============================================================================

def _solve_chi(target_age_at_death, m_table=None, max_age=119,
               chi_lo=0.01, chi_hi=20.0):
    """
    Brent root-finding: find chi such that LE@40 matches a Chetty target.
    """
    if m_table is None:
        m_table = SSA_DEATH_PROB_2017

    def residual(chi):
        return life_expectancy_at_40(chi, m_table, max_age) - target_age_at_death

    r_lo = residual(chi_lo)
    r_hi = residual(chi_hi)

    if r_lo * r_hi > 0:
        return chi_lo if abs(r_lo) < abs(r_hi) else chi_hi

    return brentq(residual, chi_lo, chi_hi, xtol=1e-10, maxiter=200)


# =============================================================================
# 6. MAIN CALIBRATION: z grid -> chi vector -> 2D survival probs
# =============================================================================

def calibrate_chi_vector(z_grid, sigma_z, m_table=None, max_age=119):
    """
    Calibrate chi(z) at each z grid point.

    z -> percentile via Phi(z / sigma_z) -> Chetty target -> chi via Brent.

    Parameters
    ----------
    z_grid : ndarray, shape (n_z,)
        Persistent income grid points.
    sigma_z : float
        Unconditional standard deviation of z.
    m_table : dict, optional
        Baseline SSA death probabilities.
    max_age : int
        Maximum age for life expectancy computation.

    Returns
    -------
    chi_vec : ndarray, shape (n_z,)
    percentiles : ndarray, shape (n_z,)
    target_ages : ndarray, shape (n_z,)
    """
    if m_table is None:
        m_table = SSA_DEATH_PROB_2017

    n_z = len(z_grid)
    percentiles = 100.0 * norm.cdf(z_grid / sigma_z)
    percentiles = np.clip(percentiles, 1.0, 100.0)

    target_ages = np.array([chetty_expected_age_at_death(p) for p in percentiles])

    chi_vec = np.empty(n_z, dtype=float)
    for i in range(n_z):
        chi_vec[i] = _solve_chi(target_ages[i], m_table, max_age)

    return chi_vec, percentiles, target_ages


def compute_sigma_z(rho, pz, mu_eta1, sigma_eta1, mu_eta2, sigma_eta2):
    """
    Unconditional standard deviation of the persistent income component z.

    For a mixture-normal AR(1): Var(z) = Var(eta) / (1 - rho^2).
    """
    mu_eta = pz * mu_eta1 + (1.0 - pz) * mu_eta2
    var_eta = (pz * (sigma_eta1**2 + (mu_eta1 - mu_eta)**2)
               + (1.0 - pz) * (sigma_eta2**2 + (mu_eta2 - mu_eta)**2))
    return np.sqrt(var_eta / (1.0 - rho**2))


def build_survival_probs_2d(ages, chi_vec, m_table=None):
    """
    Build 2D survival probability array: shape (n_age, n_z).

    survival_probs_2d[t, iz] = 1 - min(chi[iz] * m(ages[t]), 1)

    Drop-in replacement for the old 1D survival_probs.

    Parameters
    ----------
    ages : ndarray, shape (n_age,)
        Array of ages [start_age, ..., terminal_age].
    chi_vec : ndarray, shape (n_z,)
        Calibrated mortality scaling factors.
    m_table : dict, optional
        Baseline SSA death probabilities.

    Returns
    -------
    ndarray, shape (n_age, n_z)
    """
    if m_table is None:
        m_table = SSA_DEATH_PROB_2017

    n_age = len(ages)
    n_z = len(chi_vec)
    out = np.empty((n_age, n_z), dtype=np.float64)

    for t, age in enumerate(ages):
        m_base = m_table.get(age, 1.0)
        for iz in range(n_z):
            out[t, iz] = 1.0 - min(chi_vec[iz] * m_base, 1.0)

    return out


# =============================================================================
# 7. ONE-CALL ENTRY POINT
# =============================================================================

def calibrate_earnings_dependent_mortality(
    start_age, terminal_age, z_grid,
    rho, pz, mu_eta1, sigma_eta1, mu_eta2, sigma_eta2,
    max_age=119, verbose=True
):
    """
    Full calibration: income parameters + z grid -> 2D survival probs.

    Parameters
    ----------
    start_age, terminal_age : int
    z_grid : ndarray, shape (n_z,)
    rho, pz, mu_eta1, sigma_eta1, mu_eta2, sigma_eta2 : float
        Income process parameters.
    max_age : int
        Maximum age for LE calculation (119 = full SSA table).
    verbose : bool

    Returns
    -------
    survival_probs_2d : ndarray, shape (n_age, n_z)
    chi_vec : ndarray, shape (n_z,)
    diagnostics : dict
        Contains percentiles, targets, actual LE values, errors.
    """
    sigma_z = compute_sigma_z(rho, pz, mu_eta1, sigma_eta1, mu_eta2, sigma_eta2)
    chi_vec, percentiles, target_ages = calibrate_chi_vector(
        z_grid, sigma_z, max_age=max_age
    )

    ages = np.arange(start_age, terminal_age + 1)
    survival_probs_2d = build_survival_probs_2d(ages, chi_vec)

    # Compute actual LE for verification
    actual_ages = np.array([
        life_expectancy_at_40(chi_vec[i], max_age=max_age)
        for i in range(len(z_grid))
    ])
    errors = actual_ages - target_ages

    diagnostics = {
        "sigma_z": sigma_z,
        "percentiles": percentiles,
        "target_ages": target_ages,
        "actual_ages": actual_ages,
        "errors": errors,
        "chi_vec": chi_vec,
    }

    if verbose:
        W = 74
        print()
        print("=" * W)
        print("  EARNINGS-DEPENDENT MORTALITY CALIBRATION")
        print("  Catherine (2025) eq. (35), conditioned on persistent income z")
        print("=" * W)
        print(f"  Income params: rho={rho:.4f}  sigma_z={sigma_z:.4f}")
        print(f"  z grid: {len(z_grid)} pts  [{z_grid.min():.4f}, {z_grid.max():.4f}]")
        print(f"  SSA table: 2017 (2020 TR), gender-averaged")
        print(f"  Targets: Chetty et al. (2016), race-adjusted, gender-averaged")
        print()
        print(f"  {'z':>8} {'pctile':>7} {'chi':>8} {'target':>8} "
              f"{'actual':>8} {'err(yr)':>8}")
        print(f"  {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for i in range(len(z_grid)):
            print(f"  {z_grid[i]:8.4f} {percentiles[i]:7.1f} {chi_vec[i]:8.5f} "
                  f"{target_ages[i]:8.2f} {actual_ages[i]:8.2f} "
                  f"{errors[i]:+8.5f}")

        print()
        print(f"  Survival probs at key ages (lowest z → highest z):")
        for age in [40, 50, 60, 70, 80, 90]:
            if start_age <= age <= terminal_age:
                t = age - start_age
                lo, hi = survival_probs_2d[t, 0], survival_probs_2d[t, -1]
                print(f"    Age {age:3d}:  low-z={lo:.6f}  high-z={hi:.6f}  "
                      f"gap={hi - lo:+.6f}")

        # Life expectancy gap summary
        le_gap = actual_ages[-1] - actual_ages[0]
        print(f"\n  LE gap (top z − bottom z): {le_gap:.1f} years")
        print(f"  Max calibration error:     {np.max(np.abs(errors)):.2e} years")
        print("=" * W)

    return survival_probs_2d, chi_vec, diagnostics


# =============================================================================
# 8. SELF-TEST
# =============================================================================
if __name__ == "__main__":
    # Run with your current notebook parameters
    rho = 0.97
    pz, mu_eta1, sigma_eta1 = 0.9, -0.01, 0.12
    mu_eta2, sigma_eta2 = 0.09, 0.05

    sigma_z = compute_sigma_z(rho, pz, mu_eta1, sigma_eta1, mu_eta2, sigma_eta2)

    # Simulate the z grid your Precompute would create (5 points, ±3 sigma)
    n_z = 7
    z_grid = np.linspace(-3 * sigma_z, 3 * sigma_z, n_z)

    surv_2d, chi, diag = calibrate_earnings_dependent_mortality(
        start_age=25, terminal_age=80,
        z_grid=z_grid,
        rho=rho, pz=pz,
        mu_eta1=mu_eta1, sigma_eta1=sigma_eta1,
        mu_eta2=mu_eta2, sigma_eta2=sigma_eta2,
        verbose=True,
    )

    print(f"\nOutput shapes:")
    print(f"  survival_probs_2d : {surv_2d.shape}  (n_age × n_z)")
    print(f"  chi_vec           : {chi.shape}")
    print(f"\n  Ready for integration into LifecyclePortfolioModel.")
