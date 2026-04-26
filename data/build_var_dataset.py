"""
Build var_dataset.csv for the 6-variable VAR system.

Columns: [y_1, spr, cy, rtb, xr, xb]
  States:  y_1(0), spr(1), cy(2)   — known at end of year t
  Returns: rtb(3), xr(4), xb(5)    — realized during year t+1

Timing (row labelled year T):
  levels (y_1, spr, cy) = end-of-year T values
  returns (rtb, xr, xb) = realized during year T, using y_1[T-1] as the bill yield

Sources: DGS1.csv, AAA.csv, CPIAUCSL.csv, ie_data.xls (Shiller)
"""

import pandas as pd
import numpy as np
import sys

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

# DGS1: 1-year Treasury yield, daily, % p.a.
dgs1_raw = pd.read_csv('Thesisdata/DGS1.csv',
                        index_col='observation_date', parse_dates=True)
dgs1_raw['DGS1'] = pd.to_numeric(dgs1_raw['DGS1'], errors='coerce')

# AAA: Moody's AAA corporate bond yield, monthly, % p.a.
aaa_raw = pd.read_csv('Thesisdata/AAA.csv',
                       index_col='observation_date', parse_dates=True)
aaa_raw['AAA'] = pd.to_numeric(aaa_raw['AAA'], errors='coerce')

# CPI: CPIAUCSL, monthly
cpi_raw = pd.read_csv('Thesisdata/CPIAUCSL.csv',
                       index_col='observation_date', parse_dates=True)
cpi_raw['CPIAUCSL'] = pd.to_numeric(cpi_raw['CPIAUCSL'], errors='coerce')

# Shiller: P (nominal price), D (nominal dividends, annual rate), CAPE, monthly
shiller = pd.read_excel('Thesisdata/ie_data.xls', sheet_name='Data', header=7)
shiller = shiller.rename(columns={shiller.columns[0]: 'date_raw'})
shiller = shiller.dropna(subset=['date_raw'])
shiller = shiller[shiller['date_raw'].apply(lambda x: isinstance(x, (int, float)))]
shiller['date_raw'] = shiller['date_raw'].astype(float)
shiller['year'] = shiller['date_raw'].apply(lambda x: int(x))
shiller['month'] = shiller['date_raw'].apply(lambda x: max(1, round((x % 1) * 100)))
shiller['date'] = pd.to_datetime(shiller[['year', 'month']].assign(day=1))
shiller = shiller.set_index('date')
shiller['P_nom'] = pd.to_numeric(shiller['P'], errors='coerce')
shiller['D_nom'] = pd.to_numeric(shiller['D'], errors='coerce')
shiller['CAPE'] = pd.to_numeric(shiller['CAPE'], errors='coerce')

# ============================================================
# 2. RESAMPLE TO END-OF-YEAR (December values)
# ============================================================

# DGS1: last available observation in December each year
y_1 = dgs1_raw['DGS1'].resample('YE-DEC').last() / 100.0
y_1.index = y_1.index.year
y_1.name = 'y_1'

# AAA: last available observation in December each year
y_20 = aaa_raw['AAA'].resample('YE-DEC').last() / 100.0
y_20.index = y_20.index.year
y_20.name = 'y_20'

# CPI: December value each year
cpi_dec = cpi_raw['CPIAUCSL'].resample('YE-DEC').last()
cpi_dec.index = cpi_dec.index.year
cpi_dec.name = 'cpi_dec'

# Shiller CAPE: December value each year (for cy)
shiller_dec_cape = shiller[shiller.index.month == 12][['CAPE']].copy()
shiller_dec_cape.index = shiller_dec_cape.index.year

# Nominal stock return: sum of 12 monthly log returns from P and D
# nom_ret_m = log((P_t + D_t/12) / P_{t-1})  — pure nominal, no CPI
shiller['nom_ret_m'] = np.log(
    (shiller['P_nom'] + shiller['D_nom'] / 12) / shiller['P_nom'].shift(1)
)
annual_nom_stock = shiller.groupby(shiller.index.year)['nom_ret_m'].agg(['sum', 'count'])
annual_nom_stock = annual_nom_stock[annual_nom_stock['count'] == 12]['sum']
annual_nom_stock.name = 'nominal_stock'

# ============================================================
# 3. CONSTRUCT VARIABLES
# ============================================================

# --- State variables (levels, end of year T) ---
spr = y_20 - y_1
spr.name = 'spr'

cy = -np.log(shiller_dec_cape['CAPE'])
cy.name = 'cy'

# --- Annual log inflation: pi[T] = log(CPI_Dec_T / CPI_Dec_{T-1}) ---
pi = np.log(cpi_dec / cpi_dec.shift(1))
pi.name = 'pi'

# --- r_1 = log(1 + y_1): log gross nominal bill return ---
r_1 = np.log(1 + y_1)
r_1.name = 'r_1'

# --- rtb[T] = log(1 + y_1[T-1]) - pi[T] ---
rtb = r_1.shift(1) - pi
rtb.name = 'rtb'

# --- Bond return: CCV loglinear approximation ---
# D_t = (1 - (1+Y_t)^{-20}) / (1 - (1+Y_t)^{-1})  where Y_t = y_20 (AAA decimal)
n_bond = 20
D = (1 - (1 + y_20)**(-n_bond)) / (1 - (1 + y_20)**(-1))
y_log_20 = np.log(1 + y_20)

# r_bond[T] = D[T-1] * y_log_20[T-1] - (D[T-1] - 1) * y_log_20[T]
r_bond = D.shift(1) * y_log_20.shift(1) - (D.shift(1) - 1) * y_log_20
r_bond.name = 'r_bond'

# xb[T] = r_bond[T] - log(1 + y_1[T-1])
xb = r_bond - r_1.shift(1)
xb.name = 'xb'

# --- Nominal stock return from Shiller P and D (CPI-free) ---
# nominal_stock[T] = sum of 12 monthly log((P+D/12)/P_lag) for year T
nominal_stock = annual_nom_stock

# xr[T] = nominal_stock[T] - log(1 + y_1[T-1])
xr = nominal_stock - r_1.shift(1)
xr.name = 'xr'

# ============================================================
# 4. ASSEMBLE DATASET
# ============================================================

df = pd.DataFrame({
    'y_1': y_1,
    'spr': spr,
    'cy': cy,
    'rtb': rtb,
    'xr': xr,
    'xb': xb,
}, index=y_1.index)

df = df.dropna()

print(f"Dataset: years {df.index.min()} to {df.index.max()}, T={len(df)} rows")
print()
print("First 3 rows:")
print(df.head(3).to_string())
print()
print("Last 3 rows:")
print(df.tail(3).to_string())
print()

# ============================================================
# 5. VERIFICATION IDENTITIES (hard-fail on any)
# ============================================================

errors = []

# V1: rtb identity — rtb[T] + pi[T] = log(1 + y_1[T-1])
v1_resid = (df['rtb'] + pi.reindex(df.index) - r_1.shift(1).reindex(df.index)).dropna()
v1_max = v1_resid.abs().max()
print(f"V1 (rtb identity): max |resid| = {v1_max:.3e}  {'PASS' if v1_max < 1e-10 else 'FAIL'}")
if v1_max >= 1e-10:
    errors.append(f"V1 FAIL: {v1_max:.3e}")

# V2: bond identity — reconstruct xb from y_20
r_bond_check = D.shift(1) * y_log_20.shift(1) - (D.shift(1) - 1) * y_log_20
xb_check = r_bond_check - r_1.shift(1)
v2_resid = (xb_check.reindex(df.index) - df['xb']).dropna()
v2_max = v2_resid.abs().max()
print(f"V2 (bond identity): max |resid| = {v2_max:.3e}  {'PASS' if v2_max < 1e-10 else 'FAIL'}")
if v2_max >= 1e-10:
    errors.append(f"V2 FAIL: {v2_max:.3e}")

# V3: real stock identity — rtb + xr = nominal_stock - pi  (exact by construction)
real_stock_recovered = df['rtb'] + df['xr']
real_stock_direct = (nominal_stock - pi).reindex(df.index)
v3_resid = (real_stock_recovered - real_stock_direct).dropna()
v3_max = v3_resid.abs().max()
print(f"V3 (real stock identity): max |resid| = {v3_max:.3e}  {'PASS' if v3_max < 1e-10 else 'FAIL'}")
if v3_max >= 1e-10:
    errors.append(f"V3 FAIL: {v3_max:.3e}")

# V+: real bond recovery — rtb + xb = r_bond - pi
real_bond_recovered = df['rtb'] + df['xb']
real_bond_direct = (r_bond - pi).reindex(df.index)
v_rb_resid = (real_bond_recovered - real_bond_direct).dropna()
v_rb_max = v_rb_resid.abs().max()
print(f"V+ (real bond recovery): max |resid| = {v_rb_max:.3e}  {'PASS' if v_rb_max < 1e-10 else 'FAIL'}")
if v_rb_max >= 1e-10:
    errors.append(f"V+ FAIL: {v_rb_max:.3e}")

print()

# ============================================================
# 6. SANITY CHECKS
# ============================================================
print("=== Sample statistics ===")
for col in ['y_1', 'spr', 'cy', 'rtb', 'xr', 'xb']:
    s = df[col]
    if col != 'cy':
        print(f"{col:4s}: mean={s.mean():+.4f} ({s.mean()*100:+.2f}%), "
              f"std={s.std():.4f}, min={s.min():+.4f}, max={s.max():+.4f}")
    else:
        print(f"{col:4s}: mean={s.mean():+.4f}, "
              f"std={s.std():.4f}, min={s.min():+.4f}, max={s.max():+.4f}")

print()

# Duration check
D_mean = D.reindex(df.index).mean()
print(f"Mean AAA bond duration: {D_mean:.2f} years (expect ~13-14 for 20yr par bond)")
print(f"Mean y_20 (AAA): {y_20.reindex(df.index).mean()*100:.2f}%")

# ============================================================
# 7. SAVE
# ============================================================
if errors:
    print()
    print("*** VERIFICATION FAILURES — NOT SAVING ***")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)

df.index.name = 'year'
df.to_csv('var_dataset.csv')
print()
print("Saved to data/var_dataset.csv")
