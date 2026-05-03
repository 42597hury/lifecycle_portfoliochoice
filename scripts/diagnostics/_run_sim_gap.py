"""Simulation + Euler equation gap measurement for diagnostic run."""
import numpy as np
import io, sys

# Suppress verbose precompute output
old_stdout = sys.stdout
sys.stdout = io.StringIO()
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model, Precompute
from lifecycle.simulation import simulate_lifecycle
var_config, _, _ = build_nominal_system1_var_config(csv_path="data/var_dataset.csv")
base_config = {
    "beta": 0.96, "gamma": 3.0, "b_bar": 10,
    "start_age": 22, "retire_age": 67, "terminal_age": 99,
    "b0": -6.142, "b1": 0.304, "b2": -0.051, "b3": 0.002586,
    "rho": 0.991, "pz": 0.176,
    "mu_eta1": -0.524, "sigma_eta1": 0.113,
    "mu_eta2": -(0.176/(1-0.176))*(-0.524), "sigma_eta2": 0.046,
    "pe": 0.044, "mu_eps1": 0.134, "sigma_eps1": 0.762,
    "mu_eps2": 0.0, "sigma_eps2": 0.055, "constrained": True,
}
disc_config = DiscretizationConfig(
    n_wealth=150, n_savings=150, state_grid_sizes=(3,3,2), n_z=11,
    n_eps_nodes=3, n_eta_nodes=3, n_ret_nodes_1d=2,
)
model = build_model(base_config, var_config)
pc = Precompute(model, disc_config)
sys.stdout = old_stdout

# Load policies
d = np.load("saved_runs/_diag_332_nz11/policy_arrays.npz")
C_mat, S_mat, B_mat = d["C_mat"], d["S_mat"], d["B_mat"]

# Simulate
print("Running simulation (10k households)...")
sim = simulate_lifecycle(
    C_mat, S_mat, B_mat, pc, model,
    n_simulations=10_000, initial_wealth=0.1,
    initial_z="normal", initial_z_normal_std=0.652, initial_state="median",
    seed=42, verbose=False,
)

c_sim = sim["c"]
x_sim = sim["x"]
inc_sim = sim["income"]
sav_sim = sim["savings"]
z_sim = sim["z"]
alive = sim["alive"]
R_port = sim["R_port"]

n_sim, n_periods = c_sim.shape
start_age = 22
ages = np.arange(start_age, start_age + n_periods)
print(f"Sim: {n_sim} households x {n_periods} periods")

# === AGE PROFILES ===
print("\n=== AGE PROFILES (alive households only) ===")
print(f"{'Age':>4s}  {'N_alive':>7s}  {'mean_c':>8s}  {'mean_x':>8s}  {'c/x':>6s}  {'mean_inc':>8s}")
for t in range(0, n_periods, 5):
    mask = alive[:, t]
    n_al = mask.sum()
    if n_al < 10:
        continue
    mc = np.mean(c_sim[mask, t])
    mx = np.mean(x_sim[mask, t])
    mi = np.mean(inc_sim[mask, t])
    cx = mc / mx if mx > 0 else float("nan")
    print(f"{ages[t]:4d}  {n_al:7d}  {mc:8.4f}  {mx:8.4f}  {cx:6.3f}  {mi:8.4f}")

# === EULER EQUATION ERRORS ===
gamma = 3.0
beta_val = 0.96

euler_errors = []
for t in range(n_periods - 1):
    mask = alive[:, t] & alive[:, t+1]
    if mask.sum() < 100:
        continue
    c_t = c_sim[mask, t]
    c_t1 = c_sim[mask, t+1]
    R_t1 = R_port[mask, t]

    valid = (c_t > 1e-8) & (c_t1 > 1e-8) & (R_t1 > 0)
    if valid.sum() < 50:
        continue

    euler_resid = beta_val * (c_t1[valid] / c_t[valid]) ** (-gamma) * R_t1[valid] - 1.0
    euler_errors.append({
        "age": ages[t],
        "mean": np.mean(euler_resid),
        "abs_mean": np.mean(np.abs(euler_resid)),
        "median_abs": np.median(np.abs(euler_resid)),
        "p95_abs": np.percentile(np.abs(euler_resid), 95),
        "n": valid.sum(),
    })

print("\n=== EULER EQUATION ERRORS (Den Haan 2010) ===")
print(f"{'Age':>4s}  {'mean':>10s}  {'|mean|':>10s}  {'med|e|':>10s}  {'p95|e|':>10s}  {'N':>6s}")
for e in euler_errors[::5]:
    print(f"{e['age']:4d}  {e['mean']:10.4f}  {e['abs_mean']:10.4f}  {e['median_abs']:10.4f}  {e['p95_abs']:10.4f}  {e['n']:6d}")

# Overall summary
all_abs = np.array([e["abs_mean"] for e in euler_errors])
all_med = np.array([e["median_abs"] for e in euler_errors])
work_idx = [i for i, e in enumerate(euler_errors) if e["age"] < 67]
ret_idx = [i for i, e in enumerate(euler_errors) if e["age"] >= 67]

print(f"\n--- SUMMARY ---")
print(f"Overall mean |euler|:     {np.mean(all_abs):.4f}")
print(f"Overall median |euler|:   {np.mean(all_med):.4f}")
if work_idx:
    print(f"Working mean |euler|:     {np.mean(all_abs[work_idx]):.4f}")
if ret_idx:
    print(f"Retirement mean |euler|:  {np.mean(all_abs[ret_idx]):.4f}")
worst_idx = np.argmax(all_abs)
print(f"Worst age mean |euler|:   age {euler_errors[worst_idx]['age']}, {np.max(all_abs):.4f}")

# === BY WEALTH QUANTILE (working ages) ===
print("\n=== EULER ERROR BY WEALTH QUANTILE (working ages 30-60) ===")
work_t = [t for t in range(n_periods) if 30 <= ages[t] <= 60]
z_pool, x_pool, ee_pool = [], [], []
for t in work_t:
    if t >= n_periods - 1:
        continue
    mask = alive[:, t] & alive[:, t+1]
    c_t = c_sim[mask, t]
    c_t1 = c_sim[mask, t+1]
    R_t1 = R_port[mask, t]
    valid = (c_t > 1e-8) & (c_t1 > 1e-8) & (R_t1 > 0)
    ee = beta_val * (c_t1[valid] / c_t[valid]) ** (-gamma) * R_t1[valid] - 1.0
    z_pool.append(z_sim[mask, t][valid])
    x_pool.append(x_sim[mask, t][valid])
    ee_pool.append(ee)

z_pool = np.concatenate(z_pool)
x_pool = np.concatenate(x_pool)
ee_pool = np.concatenate(ee_pool)

print(f"\n  By cash-on-hand quantile:")
pcts = [0, 25, 50, 75, 90, 100]
x_edges = np.percentile(x_pool, pcts)
print(f"  {'x_bin':>12s}  {'x_range':>20s}  {'mean|e|':>10s}  {'med|e|':>10s}  {'N':>8s}")
for i in range(len(pcts)-1):
    mask = (x_pool >= x_edges[i]) & (x_pool < x_edges[i+1] + (1 if i==len(pcts)-2 else 0))
    if mask.sum() == 0:
        continue
    x_lo, x_hi = x_edges[i], x_edges[i+1]
    mae = np.mean(np.abs(ee_pool[mask]))
    mde = np.median(np.abs(ee_pool[mask]))
    print(f"  p{pcts[i]:02d}-p{pcts[i+1]:02d}      [{x_lo:7.3f},{x_hi:7.3f}]  {mae:10.4f}  {mde:10.4f}  {mask.sum():8d}")

print(f"\n  By z quantile:")
z_edges = np.percentile(z_pool, pcts)
print(f"  {'z_bin':>12s}  {'z_range':>20s}  {'mean|e|':>10s}  {'med|e|':>10s}  {'N':>8s}")
for i in range(len(pcts)-1):
    mask = (z_pool >= z_edges[i]) & (z_pool < z_edges[i+1] + (1 if i==len(pcts)-2 else 0))
    if mask.sum() == 0:
        continue
    z_lo, z_hi = z_edges[i], z_edges[i+1]
    mae = np.mean(np.abs(ee_pool[mask]))
    mde = np.median(np.abs(ee_pool[mask]))
    print(f"  p{pcts[i]:02d}-p{pcts[i+1]:02d}      [{z_lo:+7.3f},{z_hi:+7.3f}]  {mae:10.4f}  {mde:10.4f}  {mask.sum():8d}")
