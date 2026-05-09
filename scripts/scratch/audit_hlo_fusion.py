"""HLO fusion audit on the per-age single-device kernels.

Builds the terminal / retirement / working kernels at a tiny config matching
``verify/smoke.py`` scale, lowers each to HLO, and emits coarse summary
statistics + the post-XLA HLO text + the pre-XLA StableHLO text under
``docs/scans/hlo_dumps/``.

Read-only; no code changes. The audit report at
``docs/scans/HLO_FUSION_AUDIT_2026-05-07.md`` consumes the dumps + summary.
"""
import os
# Single-device CPU before any JAX/lifecycle import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import re
from pathlib import Path

import jax
import jax.numpy as jnp

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig, DELTA_BEQUEST
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import (
    _pc_to_jnp,
    ModelParams,
    _precompute_per_is_tensors,
    _build_per_age_terminal_kernel_vmap_only,
    _build_per_age_retirement_kernel_vmap_only,
    _build_per_age_working_kernel_vmap_only,
)


# ---- Tiny config matching verify/smoke.py scale ----
disc = DiscretizationConfig(
    n_wealth=12, wealth_min=0.13, wealth_max=200.0,
    n_savings=12,
    state_grid_sizes=(2, 3, 2, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=3,
    n_eps_nodes=2,
    n_eta_nodes=2,
    n_ret_nodes_1d=(2, 2),
    n_state_quad_nodes=(2, 3, 2, 3),
)
base = dict(BASE_CONFIG)
# Boundary ages need to span retirement so the working kernel build is meaningful.
# Keep the model itself wide; we are only building kernels, not running solves.
var = build_nominal_system1_var_config_hardcoded()
model = build_model(base, var, verbose=False)
pc = build_precompute(model, disc, verbose=False)

print(f"tiny model: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, "
      f"n_w={pc.n_w}, n_savings={pc.n_s}")
print(f"  state_grid_sizes={disc.state_grid_sizes}, n_state_quad={disc.n_state_quad_nodes}, "
      f"n_ret_1d={disc.n_ret_nodes_1d}")
print(f"  n_eta={disc.n_eta_nodes}, n_eps={disc.n_eps_nodes}")
print(f"devices: {jax.devices()}")

sc = CANONICAL_SOLVER._replace(
    max_iter=30, gather_precision="f64",
)
delta = sc.delta_bequest if sc.delta_bequest >= 0.0 else DELTA_BEQUEST
pcj = _pc_to_jnp(pc, delta)
mp = ModelParams(
    gamma=jnp.float64(model.gamma),
    beta=jnp.float64(model.beta),
    b_bar=jnp.float64(model.b_bar),
    delta=jnp.float64(delta),
    rho=jnp.float64(model.rho),
)
per_is_tensors = _precompute_per_is_tensors(pcj)


# Output directory.
DUMP_DIR = Path("docs/scans/hlo_dumps")
DUMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_jit(call_fn):
    """Find the @jit'd inner function inside a kernel-builder return value.

    Same trick as verify/mixed_precision_tiny.py: each builder returns a
    Python ``call`` whose closure holds the @jit'd ``per_chunk``.
    """
    for cell in (call_fn.__closure__ or ()):
        obj = cell.cell_contents
        if hasattr(obj, "lower") and callable(getattr(obj, "lower", None)):
            return obj
    raise RuntimeError("Could not locate @jit inside kernel builder closure.")


def summarize_hlo(hlo_text: str) -> dict:
    """Coarse fusion-structure stats for a post-XLA HLO text.

    Counts:
      - ``fusion_count``: ``= fusion(`` lines (top-level XLA fusion regions).
      - ``while_count``: ``while(...)`` lines (lax.fori / while -> XLA while).
      - ``reduce_count``: ``reduce(...)`` ops (jnp.sum -> reduction kernels).
        High count *not* embedded in fusions = unfused boundary signal.
      - ``copy_count``: ``copy(...)`` HBM→HBM moves.
      - ``transpose_count``, ``reshape_count``, ``bitcast_count``: layout ops.
      - ``convert_count``: dtype conversions.
      - ``dynamic_slice_count``: hot-path scalar gathers (expected high; the
        fp32-cast plumbing produces these).
    """
    return dict(
        size_chars=len(hlo_text),
        fusion_count=len(re.findall(r"=\s*\S+\s+fusion\(", hlo_text)),
        while_count=len(re.findall(r"=\s*\S+\s+while\(", hlo_text)),
        reduce_count=len(re.findall(r"=\s*\S+\s+reduce\(", hlo_text)),
        copy_count=len(re.findall(r"=\s*\S+\s+copy\(", hlo_text)),
        transpose_count=len(re.findall(r"=\s*\S+\s+transpose\(", hlo_text)),
        reshape_count=len(re.findall(r"=\s*\S+\s+reshape\(", hlo_text)),
        bitcast_count=len(re.findall(r"=\s*\S+\s+bitcast\(", hlo_text)),
        convert_count=len(re.findall(r"=\s*\S+\s+convert\(", hlo_text)),
        dynamic_slice_count=len(re.findall(r"=\s*\S+\s+dynamic-slice\(", hlo_text)),
        gather_count=len(re.findall(r"=\s*\S+\s+gather\(", hlo_text)),
    )


def dump_kernel(name: str, jit_fn, args):
    """Lower JIT to HLO + StableHLO; write both to disk; return summary stats."""
    print(f"\n--- {name} ---")
    lowered = jit_fn.lower(*args)

    # StableHLO (pre-XLA) — what JAX submitted.
    stablehlo_text = lowered.as_text()
    stablehlo_path = DUMP_DIR / f"{name}.stablehlo.txt"
    stablehlo_path.write_text(stablehlo_text, encoding="utf-8")

    # HLO (post-XLA fusion) — what XLA actually compiled.
    compiled = lowered.compile()
    hlo_text = compiled.as_text()
    hlo_path = DUMP_DIR / f"{name}.hlo.txt"
    hlo_path.write_text(hlo_text, encoding="utf-8")

    stats = summarize_hlo(hlo_text)
    print(f"  StableHLO -> {stablehlo_path}  ({len(stablehlo_text)} chars)")
    print(f"  HLO       -> {hlo_path}  ({stats['size_chars']} chars)")
    print(f"  fusions={stats['fusion_count']}  whiles={stats['while_count']}  "
          f"reduces={stats['reduce_count']}  copies={stats['copy_count']}")
    print(f"  transposes={stats['transpose_count']}  reshapes={stats['reshape_count']}  "
          f"bitcasts={stats['bitcast_count']}  converts={stats['convert_count']}")
    print(f"  dynamic-slice={stats['dynamic_slice_count']}  gathers={stats['gather_count']}")
    return stats


# ---------------------------------------------------------------------------
# 1) Terminal kernel
# ---------------------------------------------------------------------------
terminal_call = _build_per_age_terminal_kernel_vmap_only(pcj, mp, sc)
terminal_jit = find_jit(terminal_call)
# per_chunk(log_Rb_c, lxs_c, lxb_c, ann_c) — closure constants are the full
# padded tensors built inside the kernel builder.
# We can find them via the same closure walk.
terminal_args = []
for cell in (terminal_call.__closure__ or ()):
    obj = cell.cell_contents
    if isinstance(obj, jnp.ndarray):
        terminal_args.append(obj)
# Should be exactly 4 jnp arrays: log_R_bill_jnp, log_x_s_jnp, log_x_b_jnp, ann_jnp.
# (Order is closure-cell order; for a clean kernel build at n_chunks=1 this
# matches the per_chunk call signature.)
if len(terminal_args) != 4:
    print(f"  WARN: expected 4 closure-jnp args for terminal; got {len(terminal_args)}; "
          f"falling back to dummy zeros.")
    n_is = pc.N_state
    n_kv = pcj.weight_kv_kr.shape[0]
    n_kr = pcj.weight_kv_kr.shape[1]
    terminal_args = [
        jnp.zeros((n_is, n_kv, n_kr), dtype=jnp.float64),
        jnp.zeros((n_is, n_kv, n_kr), dtype=jnp.float64),
        jnp.zeros((n_is, n_kv, n_kr), dtype=jnp.float64),
        jnp.ones((n_is,), dtype=jnp.float64),
    ]
terminal_stats = dump_kernel("terminal_kernel", terminal_jit, terminal_args)


# ---------------------------------------------------------------------------
# 2) Retirement kernel
# ---------------------------------------------------------------------------
retirement_call = _build_per_age_retirement_kernel_vmap_only(
    pcj, mp, sc, pc.n_z, pc.N_state, per_is_tensors,
)
retirement_jit = find_jit(retirement_call)

# per_chunk signature:
#   (c_next, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr,
#    z_chunk, is_chunk)
init_arr_shape = (pc.n_z, pc.N_state, pc.n_w)
ret_args = (
    jnp.zeros(init_arr_shape, dtype=jnp.float64),                        # c_next
    jnp.zeros(pc.n_z, dtype=jnp.float64),                                # pension_next_by_z
    jnp.ones(pc.n_z, dtype=jnp.float64),                                 # psi_per_z
    jnp.full(init_arr_shape, 0.5, dtype=jnp.float64),                    # init_a_s_arr
    jnp.full(init_arr_shape, 0.4, dtype=jnp.float64),                    # init_a_b_arr
    jnp.arange(pc.n_z * pc.N_state, dtype=jnp.int64) // pc.N_state,      # z_chunk
    jnp.arange(pc.n_z * pc.N_state, dtype=jnp.int64) % pc.N_state,       # is_chunk
)
retirement_stats = dump_kernel("retirement_kernel", retirement_jit, ret_args)


# ---------------------------------------------------------------------------
# 3) Working kernel (use_pension_next=False — pure working trace)
# ---------------------------------------------------------------------------
working_call = _build_per_age_working_kernel_vmap_only(
    pcj, mp, sc, pc.n_z, pc.N_state, use_pension_next=False,
    per_is_tensors=per_is_tensors,
)
working_jit = find_jit(working_call)

# per_chunk signature:
#   (c_next, income_next_table_z, pension_next_by_z, psi_per_z,
#    init_a_s_arr, init_a_b_arr, z_chunk, is_chunk)
n_eta = disc.n_eta_nodes
n_eps = disc.n_eps_nodes
work_args = (
    jnp.zeros(init_arr_shape, dtype=jnp.float64),                        # c_next
    jnp.zeros((pc.n_z, n_eta, n_eps), dtype=jnp.float64),                # income_next_table_z
    jnp.zeros(pc.n_z, dtype=jnp.float64),                                # pension_next_by_z
    jnp.ones(pc.n_z, dtype=jnp.float64),                                 # psi_per_z
    jnp.full(init_arr_shape, 0.5, dtype=jnp.float64),                    # init_a_s_arr
    jnp.full(init_arr_shape, 0.4, dtype=jnp.float64),                    # init_a_b_arr
    jnp.arange(pc.n_z * pc.N_state, dtype=jnp.int64) // pc.N_state,      # z_chunk
    jnp.arange(pc.n_z * pc.N_state, dtype=jnp.int64) % pc.N_state,       # is_chunk
)
working_stats = dump_kernel("working_kernel", working_jit, work_args)


print("\n=== summary ===")
for name, s in [("terminal", terminal_stats),
                 ("retirement", retirement_stats),
                 ("working", working_stats)]:
    print(f"{name:>11}: fusions={s['fusion_count']:>4}  while={s['while_count']}  "
          f"reduce={s['reduce_count']}  copy={s['copy_count']}  "
          f"trans={s['transpose_count']}  reshape={s['reshape_count']}  "
          f"convert={s['convert_count']}  dyn-slice={s['dynamic_slice_count']}  "
          f"size={s['size_chars']}")
