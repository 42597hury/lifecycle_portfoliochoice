import pickle, sys
bundle = sys.argv[1] if len(sys.argv) > 1 else "saved_runs/system_iv_full_var_unconstrained_cholesky_grid7x7x7_nz11_v11_state_k7_z7_wide"
with open(f"{bundle}/diagnostics.pkl", "rb") as f:
    d = pickle.load(f)
print(f"wall_time_sec:    {d.get('wall_time_sec'):.1f}")
print(f"n_ages_solved:    {d.get('n_ages_solved')}")
dc = d['disc_config']
print()
print("--- discretization ---")
for k in ("n_wealth", "wealth_min", "wealth_max", "n_savings",
         "state_grid_sizes", "state_grid_mode", "state_n_stds",
         "n_z", "n_eps_nodes", "n_eta_nodes",
         "n_ret_nodes_1d", "ret_lobatto_Z",
         "n_state_quad_nodes", "state_lobatto_Z"):
    print(f"  {k:22s}= {dc.get(k)}")
