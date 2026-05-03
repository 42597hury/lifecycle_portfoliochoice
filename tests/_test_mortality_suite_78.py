"""
MORTALITY MODULE — VERIFICATION TEST SUITE (Sections 7-8)
Requires building full model + running simulation with saved policy arrays.
"""
import numpy as np
import json
import sys

results = []

def report(test_id, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((test_id, status, detail))
    print(f"  {test_id:12s}  {status}  {detail}")

# =====================================================================
# Build model and precompute
# =====================================================================
print("Building model and precompute...")

with open("saved_runs/constrained_grid5x5x5_nz11/metadata.json") as f:
    meta = json.load(f)

rc = meta["run_config"]
base_config = rc["base_config"]

# Reconstruct var_config with numpy arrays
var_raw = rc["var_config"]
var_config = {}
for k, v in var_raw.items():
    if isinstance(v, dict) and v.get("kind") == "ndarray":
        var_config[k] = np.array(v["values"]).reshape(v["shape"])
    else:
        var_config[k] = v

from lifecycle.precompute import build_model, Precompute

# Reconstruct disc_config
disc_raw = meta["diagnostics_summary"]["disc_config"]
from lifecycle.precompute import DiscretizationConfig
disc_config = DiscretizationConfig(**disc_raw)

model = build_model(base_config, var_config, verbose=False)
pc = Precompute(model, disc_config, verbose=False)

start_age = model.start_age
terminal_age = model.terminal_age
n_age = terminal_age - start_age + 1
n_z = len(pc.z_grid)

print(f"  Model built: start_age={start_age}, terminal_age={terminal_age}, n_age={n_age}, n_z={n_z}")

# =====================================================================
print()
print("=" * 70)
print("SECTION 7: Solver Integration")
print("=" * 70)

# 7.1
ok1 = hasattr(pc, "survival_probs_2d")
ok2 = pc.survival_probs_2d.shape == (n_age, n_z) if ok1 else False
report("7.1", ok1 and ok2, f"has survival_probs_2d={ok1}, shape={pc.survival_probs_2d.shape if ok1 else 'N/A'}")

# 7.2 — Load saved policy arrays and verify solver completed for all ages
data = np.load("saved_runs/constrained_grid5x5x5_nz11/policy_arrays.npz")
C_mat = data["C_mat"]
S_mat = data["S_mat"]
B_mat = data["B_mat"]
expected_shape = (n_age, n_z, pc.N_state, pc.n_w)
ok = C_mat.shape == expected_shape
report("7.2", ok, f"policy shape={C_mat.shape}, expected={expected_shape}")

# 7.3 — Terminal age handling (code inspection equivalent: check terminal period has valid values)
# At terminal age, consumption should equal wealth (eat everything + bequest)
# The terminal slice should have non-zero consumption
terminal_c = C_mat[-1]
ok = np.all(np.isfinite(terminal_c)) and np.any(terminal_c > 0)
report("7.3", ok, f"terminal C finite={np.all(np.isfinite(terminal_c))}, has positive={np.any(terminal_c > 0)}")

# =====================================================================
print()
print("=" * 70)
print("SECTION 8: Simulation Consistency")
print("=" * 70)

print("Running simulation (10,000 agents)...")
from lifecycle.simulation import simulate_lifecycle

sim = simulate_lifecycle(
    C_mat, S_mat, B_mat, pc, model,
    n_simulations=10000,
    seed=42,
    verbose=False,
)
print("  Simulation complete.")

death_age = sim["death_age"]
alive = sim["alive"]
ages = sim["ages"]

# 8.1
valid_deaths = death_age[death_age >= 0]
if len(valid_deaths) > 0:
    median_death = np.median(valid_deaths)
else:
    median_death = -1
ok = 75 < median_death < 95
report("8.1", ok, f"median death age={median_death:.1f}")

# 8.2
z_init = sim["z"][:, 0]
med_z = np.median(z_init)
bottom = z_init < med_z
top = ~bottom
deaths_bottom = death_age[bottom & (death_age >= 0)]
deaths_top = death_age[top & (death_age >= 0)]
med_bottom = np.median(deaths_bottom) if len(deaths_bottom) > 0 else -1
med_top = np.median(deaths_top) if len(deaths_top) > 0 else -1
ok = med_top > med_bottom
report("8.2", ok, f"median death: bottom-z={med_bottom:.1f}, top-z={med_top:.1f}")

# 8.3
ok = bool(np.all(death_age <= terminal_age))
report("8.3", ok, f"max death_age={death_age.max()}, terminal_age={terminal_age}")

# 8.4
n_check = min(1000, len(death_age))
all_ok = True
fail_detail = ""
for i in range(n_check):
    d = death_age[i]
    if d < start_age:
        continue
    t_death = d - start_age
    # alive up to and including death period
    if t_death < n_age:
        pre = alive[i, :t_death + 1]
        if not np.all(pre == 1):
            all_ok = False
            fail_detail = f"agent {i}: alive before death not all 1"
            break
    if t_death < n_age - 1:
        post = alive[i, t_death + 1:]
        if not np.all(post == 0):
            all_ok = False
            fail_detail = f"agent {i}: alive after death not all 0"
            break
report("8.4", all_ok, f"checked {n_check} agents" + (f" — {fail_detail}" if fail_detail else ""))

# 8.5
check_ages = [(65, (0.80, 0.99)), (80, (0.50, 0.90)), (95, (0.05, 0.50))]
all_ok = True
details = []
for check_age, (lo, hi) in check_ages:
    t = check_age - start_age
    if t < n_age:
        frac_alive = np.mean(alive[:, t])
        in_range = lo < frac_alive < hi
        all_ok = all_ok and in_range
        details.append(f"age {check_age}: {frac_alive:.3f} in ({lo},{hi})={'OK' if in_range else 'FAIL'}")
report("8.5", all_ok, "; ".join(details))

# =====================================================================
print()
print("=" * 70)
print("SUMMARY (Sections 7-8)")
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
