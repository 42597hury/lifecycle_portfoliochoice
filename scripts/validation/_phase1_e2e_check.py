"""End-to-end smoke test for per-axis state_n_stds.

Verifies the full pipeline works with a per-axis tuple:
  1. DiscretizationConfig accepts and stores tuple
  2. Precompute builds without errors
  3. JSON round-trip via _to_jsonable preserves per-axis values
  4. Re-build from deserialized config gives bit-identical state_grid
  5. Old scalar bundles still load (backward compat)
"""
import json
import sys
import numpy as np

sys.path.insert(0, '.')

from lifecycle.model import DiscretizationConfig
from lifecycle.precompute import Precompute, build_model
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.policy_io import _to_jsonable

# ----- model -----
base = {
    'beta': 0.96, 'gamma': 3.0, 'b_bar': 10,
    'start_age': 22, 'retire_age': 67, 'terminal_age': 99,
    'b0': -6.142, 'b1': 0.3040, 'b2': -0.051, 'b3': 0.002586,
    'rho': 0.991, 'pz': 0.176, 'mu_eta1': -0.524, 'sigma_eta1': 0.113,
    'mu_eta2': -(0.176/(1.0-0.176))*(-0.524), 'sigma_eta2': 0.046,
    'pe': 0.044, 'mu_eps1': 0.134, 'sigma_eps1': 0.762,
    'mu_eps2': 0.0, 'sigma_eps2': 0.055,
    'constrained': False,
}
var_config, _, _ = build_nominal_system1_var_config(csv_path='data/var_dataset.csv')
model = build_model(base, var_config, verbose=False)

print("=" * 70)
print("PHASE 1 END-TO-END VERIFICATION")
print("=" * 70)

# ----- Test 1: tuple config builds Precompute -----
disc_tuple = DiscretizationConfig(
    n_wealth=20, n_savings=20,
    state_grid_sizes=(5, 5, 5),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 1.25, 1.5),
    n_z=5, n_eps_nodes=2, n_eta_nodes=2,
    n_ret_nodes_1d=(3, 5, 3), n_state_quad_nodes=2,
)
pc_tuple = Precompute(model, disc_tuple, verbose=False)
print(f"[1] Tuple config Precompute built. N_state = {pc_tuple.N_state}")
print(f"    bracket[0] half-width = {pc_tuple.state_bracket_grids[0].max():.3f} (expect 2.00)")
print(f"    bracket[1] half-width = {pc_tuple.state_bracket_grids[1].max():.3f} (expect 1.25)")
print(f"    bracket[2] half-width = {pc_tuple.state_bracket_grids[2].max():.3f} (expect 1.50)")
ok1 = (np.isclose(pc_tuple.state_bracket_grids[0].max(), 2.0)
       and np.isclose(pc_tuple.state_bracket_grids[1].max(), 1.25)
       and np.isclose(pc_tuple.state_bracket_grids[2].max(), 1.50))
print(f"    [{'PASS' if ok1 else 'FAIL'}]")

# ----- Test 2: JSON round-trip via _to_jsonable -----
ser = _to_jsonable(disc_tuple)
print(f"\n[2] Serialized state_n_stds: {ser['state_n_stds']!r} (type={type(ser['state_n_stds']).__name__})")
print(f"    JSON round-trip:")
roundtrip = json.loads(json.dumps(ser))
print(f"      after json.loads: {roundtrip['state_n_stds']!r} (type={type(roundtrip['state_n_stds']).__name__})")
ok2 = roundtrip['state_n_stds'] == [2.0, 1.25, 1.5]
print(f"    [{'PASS' if ok2 else 'FAIL'}]")

# ----- Test 3: Re-build from deserialized list gives same grid -----
disc_loaded = DiscretizationConfig(**{k: (tuple(v) if isinstance(v, list) and k in ('state_grid_sizes', 'n_ret_nodes_1d') else v)
                                       for k, v in roundtrip.items()})
pc_loaded = Precompute(model, disc_loaded, verbose=False)
ok3 = (np.array_equal(pc_loaded.state_grid, pc_tuple.state_grid)
       and np.array_equal(pc_loaded.state_stationary_probs, pc_tuple.state_stationary_probs))
print(f"\n[3] Re-built Precompute from JSON-restored config:")
print(f"    state_grid bit-equal:        {np.array_equal(pc_loaded.state_grid, pc_tuple.state_grid)}")
print(f"    stationary_probs bit-equal:  {np.array_equal(pc_loaded.state_stationary_probs, pc_tuple.state_stationary_probs)}")
print(f"    [{'PASS' if ok3 else 'FAIL'}]")

# ----- Test 4: scalar config still works (backward compat) -----
disc_scalar = DiscretizationConfig(
    n_wealth=20, n_savings=20,
    state_grid_sizes=(5, 5, 5),
    state_grid_mode="cholesky",
    state_n_stds=2.0,
    n_z=5, n_eps_nodes=2, n_eta_nodes=2,
    n_ret_nodes_1d=(3, 5, 3), n_state_quad_nodes=2,
)
pc_scalar = Precompute(model, disc_scalar, verbose=False)
disc_uniform_tuple = disc_scalar._replace(state_n_stds=(2.0, 2.0, 2.0))
pc_uniform_tuple = Precompute(model, disc_uniform_tuple, verbose=False)
ok4 = (np.array_equal(pc_scalar.state_grid, pc_uniform_tuple.state_grid)
       and np.array_equal(pc_scalar.state_stationary_probs, pc_uniform_tuple.state_stationary_probs))
print(f"\n[4] Scalar 2.0 vs tuple (2.0,2.0,2.0) bit-equivalence:")
print(f"    state_grid bit-equal:        {np.array_equal(pc_scalar.state_grid, pc_uniform_tuple.state_grid)}")
print(f"    stationary_probs bit-equal:  {np.array_equal(pc_scalar.state_stationary_probs, pc_uniform_tuple.state_stationary_probs)}")
print(f"    [{'PASS' if ok4 else 'FAIL'}]")

# ----- Test 5: NamedTuple._replace() with new state_n_stds works -----
disc_replaced = disc_tuple._replace(state_n_stds=(1.0, 1.0, 1.0))
print(f"\n[5] NamedTuple._replace works: state_n_stds={disc_replaced.state_n_stds}")
pc_repl = Precompute(model, disc_replaced, verbose=False)
ok5 = np.isclose(pc_repl.state_bracket_grids[0].max(), 1.0)
print(f"    [{'PASS' if ok5 else 'FAIL'}]")

# ----- Test 6: list input also works (deserialization path) -----
disc_list = disc_scalar._replace(state_n_stds=[2.0, 1.25, 1.5])
pc_list = Precompute(model, disc_list, verbose=False)
ok6 = (np.array_equal(pc_list.state_grid, pc_tuple.state_grid))
print(f"\n[6] List input matches tuple input:")
print(f"    state_grid bit-equal:        {np.array_equal(pc_list.state_grid, pc_tuple.state_grid)}")
print(f"    [{'PASS' if ok6 else 'FAIL'}]")

# ----- Test 7: lyapunov-axis with per-axis n_stds -----
disc_la = DiscretizationConfig(
    n_wealth=20, n_savings=20,
    state_grid_sizes=(5, 5, 5),
    state_grid_mode="lyapunov-axis",
    state_n_stds=(2.0, 1.0, 3.0),
    n_z=5, n_eps_nodes=2, n_eta_nodes=2,
    n_ret_nodes_1d=(3, 5, 3), n_state_quad_nodes=2,
)
pc_la = Precompute(model, disc_la, verbose=False)
sig = pc_la.state_grid_sigma_z
mu = pc_la.state_grid_mu_s
ns_arr = np.array([2.0, 1.0, 3.0])
ok7 = all(
    np.isclose(pc_la.state_bracket_grids[d].max(), mu[d] + ns_arr[d] * sig[d])
    and np.isclose(pc_la.state_bracket_grids[d].min(), mu[d] - ns_arr[d] * sig[d])
    for d in range(3)
)
print(f"\n[7] lyapunov-axis per-axis matches +/-n_stds[d]*sigma_stat[d]:")
for d in range(3):
    expected_max = mu[d] + ns_arr[d] * sig[d]
    actual_max = pc_la.state_bracket_grids[d].max()
    print(f"    axis {d}: actual={actual_max:+.5f}, expected={expected_max:+.5f}")
print(f"    [{'PASS' if ok7 else 'FAIL'}]")

print()
print("=" * 70)
all_ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7
print(f"OVERALL: {'ALL PASS' if all_ok else 'SOME FAIL'}")
print("=" * 70)
