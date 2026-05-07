"""smolyak_feasibility_jax.py — second-opinion test of Smolyak quadrature
on the JAX branch's 4-state / 2-return geometry.

Three questions answered numerically:
  1. Moment recovery: does a Smolyak rule integrate Gaussian moments
     exactly to expected polynomial-exactness order?
  2. Discrete-cloud arbitrage: does the Smolyak NODE SET (positions only,
     ignoring signed weights) span the support enough that the
     hull/dominance gaps stay near zero on the canonical state grid?
  3. Integration accuracy: at near-cap |alpha|, how does the FOC
     integrand `f(z) = u'(W) * V(s_next) * exp(r_p^CCV)` integrated by
     Smolyak compare to the dense tensor baseline?

Run (from project root):
    python -m scripts.scratch.smolyak_feasibility_jax
"""
from __future__ import annotations

import sys
from itertools import product
from math import comb

import numpy as np
from scipy.special import roots_hermite

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Build model + precompute (Lobatto disabled to avoid the canonical_jax
# K=4 / Lobatto Z=7 mismatch that's unrelated to this investigation).
# ---------------------------------------------------------------------------

from configs._canonical_jax import CANONICAL_DISC, BASE_CONFIG
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute


def _build():
    disc = CANONICAL_DISC._replace(ret_lobatto_Z=None, state_lobatto_Z=None)
    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    return model, pc, disc


# ---------------------------------------------------------------------------
# 1D Gauss-Hermite rule (standard normal). K=1 -> single 0 node.
# ---------------------------------------------------------------------------

def gh_1d(K: int) -> tuple[np.ndarray, np.ndarray]:
    if K == 1:
        return np.zeros(1), np.ones(1)
    z, w = roots_hermite(int(K))
    return z * np.sqrt(2.0), w / np.sqrt(np.pi)


def lobatto_1d(K: int, Z: float) -> tuple[np.ndarray, np.ndarray]:
    """1D Gauss-Hermite-Lobatto rule with prescribed tail nodes at +/-Z."""
    from lifecycle.quadrature_with_tails import gauss_hermite_prescribed_tails
    return gauss_hermite_prescribed_tails(int(K), float(Z))


def tensor_mixed_rule(K_per_axis, Z_per_axis):
    """Tensor product where axis i uses Lobatto if Z_per_axis[i] is not None, else GH."""
    grids = []
    for K, Z in zip(K_per_axis, Z_per_axis):
        if Z is None:
            grids.append(gh_1d(K))
        else:
            grids.append(lobatto_1d(K, Z))
    z_grids, w_grids = zip(*grids)
    meshes_z = np.meshgrid(*z_grids, indexing="ij")
    meshes_w = np.meshgrid(*w_grids, indexing="ij")
    nodes = np.stack([m.ravel() for m in meshes_z], axis=1)
    weights = np.prod(np.stack(meshes_w, axis=0), axis=0).ravel()
    return nodes, weights


# ---------------------------------------------------------------------------
# Anisotropic Smolyak (Wasilkowski-Wozniakowski combination form).
#
# Multi-index k = (k_1, ..., k_d) with k_i >= 1.
# Anisotropic budget: sum_i (k_i - 1) * w_i <= L_eff
# Standard isotropic: w_i = 1 and budget = L - d + (d-1) = L - 1 -> use L-d for
#   compatibility with the usual "level L = polynomial exactness 2L-1 in 1D".
#
# Combination coefficient (Smolyak):
#   c(k) = sum_{j: k+j is in admissible set, j_i in {0,1}} (-1)^|j|
# Each k contributes the tensor product Q_{k_1} ⊗ ... ⊗ Q_{k_d} with
# coefficient c(k); duplicate node positions across multi-indices simply
# accumulate weights (we deduplicate at the end for support analysis).
# ---------------------------------------------------------------------------

def _admissible(k_tuple, w_axis, L):
    """k_tuple is 1-indexed; level cost = sum (k_i - 1) * w_i."""
    return sum((ki - 1) * wi for ki, wi in zip(k_tuple, w_axis)) <= L + 1e-12


def smolyak_anisotropic(L: float, w_axis: tuple[float, ...],
                        K_max_per_axis: tuple[int, ...] | None = None,
                        rule_1d=gh_1d) -> tuple[np.ndarray, np.ndarray]:
    """Build anisotropic Smolyak rule on standard-normal R^d.

    Parameters
    ----------
    L : float
        Smolyak level (anisotropic budget for ``sum (k_i - 1) * w_i <= L``).
    w_axis : tuple of floats
        Per-axis anisotropy weights. w_i >= 1; larger w_i means less
        refinement on axis i.
    K_max_per_axis : optional tuple of ints
        Per-axis maximum K (clamps how high any 1D rule can go).
    rule_1d : callable
        K -> (nodes, weights). Default: standard Gauss-Hermite.

    Returns
    -------
    nodes : (N, d) array — d-dim node positions in standard-normal space.
    weights : (N,) array — signed Smolyak weights.
    """
    d = len(w_axis)
    if K_max_per_axis is None:
        K_max_per_axis = (max(2, int(L) + 1),) * d

    # Enumerate all admissible multi-indices (each k_i in [1, K_max_i])
    indices = []
    ranges = [range(1, Kmax + 1) for Kmax in K_max_per_axis]
    for k_tuple in product(*ranges):
        if _admissible(k_tuple, w_axis, L):
            indices.append(k_tuple)
    if not indices:
        raise ValueError(f"No admissible multi-indices for L={L}, w={w_axis}")

    indices_set = set(indices)

    # Combination coefficient: c(k) = sum_{j in {0,1}^d, k+j admissible} (-1)^|j|.
    coeffs = {}
    for k in indices:
        total = 0
        for j in product((0, 1), repeat=d):
            kj = tuple(ki + ji for ki, ji in zip(k, j))
            if kj in indices_set or all((kj[i] - 1) * w_axis[i] <= L + 1e-12 for i in range(d)):
                # The set is closed under "remove a level" so admissible set
                # is downward closed; we just need the combination over j in {0,1}^d
                if all(kj[i] <= K_max_per_axis[i] for i in range(d)):
                    if _admissible(kj, w_axis, L):
                        total += (-1) ** sum(j)
        coeffs[k] = total

    # Build (node, weight) entries by tensor-product on each multi-index
    raw_nodes_list = []
    raw_weights_list = []
    one_d_cache = {}
    for k, c in coeffs.items():
        if c == 0:
            continue
        per_axis = []
        for axis_i, ki in enumerate(k):
            key = (axis_i, ki)
            if key not in one_d_cache:
                one_d_cache[key] = rule_1d(ki)
            per_axis.append(one_d_cache[key])
        # Tensor product
        z_grids, w_grids = zip(*per_axis)
        meshes_z = np.meshgrid(*z_grids, indexing="ij")
        meshes_w = np.meshgrid(*w_grids, indexing="ij")
        nodes_block = np.stack([m.ravel() for m in meshes_z], axis=1)   # (P, d)
        weights_block = np.prod(np.stack(meshes_w, axis=0), axis=0).ravel()
        weights_block = weights_block * c
        raw_nodes_list.append(nodes_block)
        raw_weights_list.append(weights_block)

    raw_nodes = np.concatenate(raw_nodes_list, axis=0)
    raw_weights = np.concatenate(raw_weights_list, axis=0)

    # Deduplicate by exact node position (round to 12 decimals to coalesce
    # numerical duplicates, then sum weights).
    keys = np.round(raw_nodes, decimals=12)
    keys_struct = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    ).ravel()
    _, inverse = np.unique(keys_struct, return_inverse=True)
    n_unique = inverse.max() + 1
    nodes_unique = np.zeros((n_unique, d))
    weights_unique = np.zeros(n_unique)
    for n_orig, idx in enumerate(inverse):
        weights_unique[idx] += raw_weights[n_orig]
        nodes_unique[idx] = raw_nodes[n_orig]
    return nodes_unique, weights_unique


# ---------------------------------------------------------------------------
# Tensor-product baseline (for comparison).
# ---------------------------------------------------------------------------

def tensor_product_rule(K_per_axis):
    grids = [gh_1d(K) for K in K_per_axis]
    z_grids, w_grids = zip(*grids)
    meshes_z = np.meshgrid(*z_grids, indexing="ij")
    meshes_w = np.meshgrid(*w_grids, indexing="ij")
    nodes = np.stack([m.ravel() for m in meshes_z], axis=1)
    weights = np.prod(np.stack(meshes_w, axis=0), axis=0).ravel()
    return nodes, weights


# ---------------------------------------------------------------------------
# Test 1: moment recovery on standard normal R^6
# ---------------------------------------------------------------------------

def _even_moment(k):
    if k == 0:
        return 1.0
    out = 1.0
    for i in range(1, 2 * k, 2):
        out *= i
    return out


def test_moment_recovery(rules):
    print("=" * 78)
    print("Test 1: moment recovery on standard normal R^6 (mean=0, cov=I)")
    print("=" * 78)
    for label, (nodes, weights) in rules.items():
        sw = weights.sum()
        # E[z]
        mean = nodes.T @ weights
        # E[z z^T]
        cov = (nodes * weights[:, None]).T @ nodes
        # diagonal-only check at high order: E[z_i^4] (per axis)
        m4 = (nodes ** 4 * weights[:, None]).sum(axis=0)   # should be 3 for each axis
        m6 = (nodes ** 6 * weights[:, None]).sum(axis=0)   # should be 15
        m8 = (nodes ** 8 * weights[:, None]).sum(axis=0)   # should be 105
        # cross moment E[z_0^2 z_3^2] should be 1
        m22 = ((nodes[:, 0] ** 2) * (nodes[:, 3] ** 2) * weights).sum()
        print(f"  {label:<28} N={len(weights):>5}  "
              f"|sumw-1|={abs(sw-1.0):.2e}  "
              f"|mean|max={np.max(np.abs(mean)):.2e}  "
              f"|cov-I|max={np.max(np.abs(cov - np.eye(nodes.shape[1]))):.2e}")
        print(f"      m4 max err = {np.max(np.abs(m4 - 3.0)):.2e}, "
              f"m6 max err = {np.max(np.abs(m6 - 15.0)):.2e}, "
              f"m8 max err = {np.max(np.abs(m8 - 105.0)):.2e}, "
              f"E[z0^2 z3^2] err = {abs(m22 - 1.0):.2e}")


# ---------------------------------------------------------------------------
# Test 2: discrete-cloud arbitrage at every state cell.
# ---------------------------------------------------------------------------

def _convex_hull_arb_gap(R_bill, R_stock, R_bond, n_angles=360):
    Xs = (R_stock - R_bill).reshape(R_stock.shape[0], -1)
    Xb = (R_bond - R_bill).reshape(R_bond.shape[0], -1)
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    cos = np.cos(angles)
    sin = np.sin(angles)
    N_state, n_nodes = Xs.shape
    gap = np.full(N_state, -np.inf)
    # Loop over angles one at a time; small per-angle cost, no big allocation
    for a in range(n_angles):
        proj = cos[a] * Xs + sin[a] * Xb        # (N_state, n_nodes)
        chunk_min = proj.min(axis=1)            # (N_state,)
        gap = np.maximum(gap, chunk_min)
    return np.maximum(gap, 0.0)


def _axis_dom_gap(R_bill, R_stock, R_bond):
    flat = lambda R: R.reshape(R.shape[0], -1)
    R = {"bill": flat(R_bill), "stock": flat(R_stock), "bond": flat(R_bond)}
    R_min = {a: R[a].min(axis=1) for a in R}
    R_max = {a: R[a].max(axis=1) for a in R}
    worst = -np.inf
    worst_lbl = ""
    for i in R:
        for j in R:
            if i == j:
                continue
            g = R_min[i] - R_max[j]
            mg = float(g.max())
            if mg > worst:
                worst = mg
                worst_lbl = f"{i}_dom_{j}"
    return worst, worst_lbl


def _build_cloud_with_rule(model, pc, joint_z):
    """Map joint-z (N, 6) to gross returns at every state cell.

    Layout: joint_z[:, :4] = state z, joint_z[:, 4:] = return z.
    """
    L_s = np.linalg.cholesky(0.5 * (model.Sigma_ss + model.Sigma_ss.T))
    L_r = np.linalg.cholesky(0.5 * (model.Sigma_r_cond + model.Sigma_r_cond.T))
    v_nodes = joint_z[:, :model.n_state] @ L_s.T            # (N, 4)
    r_nodes = joint_z[:, model.n_state:] @ L_r.T            # (N, 2)
    state_grid = np.asarray(pc.state_grid, dtype=np.float64)
    Phi_0 = np.asarray(model.Phi_0_state, dtype=np.float64)
    Phi_11 = np.asarray(model.Phi_11, dtype=np.float64)
    rtb_idx = int(model.rtb_index_in_state)
    xr_pos = list(model.ret_names).index("xr")
    xb_pos = list(model.ret_names).index("xb")
    M = np.asarray(model.M, dtype=np.float64)
    A_r = np.asarray(model.A_r if hasattr(model, "A_r") else model.Phi_21, dtype=np.float64)
    const_r = np.asarray(model.Phi_0_ret, dtype=np.float64)

    # base_mu_r[i_s, :] = const_r + A_r @ s_t   (N_state, n_ret)
    base_mu_r = const_r[None, :] + state_grid @ A_r.T

    # M_v: (N, n_ret)
    M_v = v_nodes @ M.T   # (N, 2)
    # log_x stock and bond at each (i_s, n)
    mu_r_per = base_mu_r[:, None, :] + M_v[None, :, :]   # (N_state, N, 2)
    log_x_s = mu_r_per[:, :, xr_pos] + r_nodes[None, :, xr_pos]
    log_x_b = mu_r_per[:, :, xb_pos] + r_nodes[None, :, xb_pos]

    # log_R_bill[i_s, n] = (Phi_0 + Phi_11 @ s_t + v[n])[rtb_idx]
    s_next_rtb = (Phi_0[None, :] + state_grid @ Phi_11.T)[:, rtb_idx][:, None] \
                 + v_nodes[None, :, rtb_idx]
    R_bill = np.exp(s_next_rtb)
    R_stock = R_bill * np.exp(log_x_s)
    R_bond = R_bill * np.exp(log_x_b)
    # Reshape to (N_state, n, 1) trivial extra dim for the API
    return R_bill[:, :, None], R_stock[:, :, None], R_bond[:, :, None]


def test_arbitrage_cloud(model, pc, rules):
    print()
    print("=" * 78)
    print("Test 2: discrete-cloud arbitrage at every state cell (positions only)")
    print("=" * 78)
    print(f"  N_state cells = {pc.N_state} (state_grid_sizes = {pc.state_grid_sizes})")
    for label, (joint_z, _w) in rules.items():
        R_bill, R_stock, R_bond = _build_cloud_with_rule(model, pc, joint_z)
        hull = _convex_hull_arb_gap(R_bill, R_stock, R_bond, n_angles=360)
        dom, dlbl = _axis_dom_gap(R_bill, R_stock, R_bond)
        max_hull = float(hull.max())
        n_arb = int((hull > 0).sum())
        max_dom = dom
        worst = max(max_hull, max_dom)
        verdict = "PASS" if worst < 1e-6 else ("CONCERNING" if worst < 1e-4 else "FAIL")
        print(f"  {label:<32} N_nodes={len(joint_z):>5}  "
              f"max_hull={max_hull:+.3e}  arb_cells={n_arb}/{pc.N_state}  "
              f"max_dom={max_dom:+.3e} ({dlbl})  -> {verdict}")


# ---------------------------------------------------------------------------
# Test 3: integrate the actual production FOC integrand at near-cap alpha.
# ---------------------------------------------------------------------------

def _ccv_log_return(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
                    sigma2_xr, sigma2_xb, sigma_xrxb):
    """Match `_ccv_log_return_and_grad` from solver.py — minus the gradients."""
    # excess log return per asset = log_x_s, log_x_b
    # CCV portfolio log return:
    #   r_p = log_R_bill + alpha_s * log_x_s + alpha_b * log_x_b
    #         + 0.5 * (alpha_s - alpha_s^2) * sigma_xr^2
    #         + 0.5 * (alpha_b - alpha_b^2) * sigma_xb^2
    #         - alpha_s * alpha_b * sigma_xrxb
    r_p = (log_R_bill + alpha_s * log_x_s + alpha_b * log_x_b
           + 0.5 * (alpha_s - alpha_s ** 2) * sigma2_xr
           + 0.5 * (alpha_b - alpha_b ** 2) * sigma2_xb
           - alpha_s * alpha_b * sigma_xrxb)
    return r_p


def test_integration_accuracy(model, pc, rules, base_label):
    """Test integration of `g(z) = u'(W) * V(s_next) * R_p` style integrand
    at canonical and stress alphas, comparing each rule against a very dense
    tensor-product baseline.
    """
    print()
    print("=" * 78)
    print("Test 3: FOC integrand integration accuracy at stress alphas")
    print("=" * 78)
    # Pick a representative state cell — body of distribution
    i_s = pc.N_state // 2
    s_t = np.asarray(pc.state_grid[i_s], dtype=np.float64)

    # Build per-cell context
    L_s = np.linalg.cholesky(0.5 * (model.Sigma_ss + model.Sigma_ss.T))
    L_r = np.linalg.cholesky(0.5 * (model.Sigma_r_cond + model.Sigma_r_cond.T))
    Phi_0 = np.asarray(model.Phi_0_state, dtype=np.float64)
    Phi_11 = np.asarray(model.Phi_11, dtype=np.float64)
    rtb_idx = int(model.rtb_index_in_state)
    xr_pos = list(model.ret_names).index("xr")
    xb_pos = list(model.ret_names).index("xb")
    A_r = np.asarray(model.Phi_21, dtype=np.float64)
    const_r = np.asarray(model.Phi_0_ret, dtype=np.float64)
    M = np.asarray(model.M, dtype=np.float64)
    base_mu_r = const_r + A_r @ s_t

    sigma2_xr = float(model.Sigma_rr[xr_pos, xr_pos])
    sigma2_xb = float(model.Sigma_rr[xb_pos, xb_pos])
    sigma_xrxb = float(model.Sigma_rr[xr_pos, xb_pos])

    gamma = float(getattr(model, "gamma", 5.0))

    def _eval_integrand(joint_z, alpha_s, alpha_b, s_val=10.0, w_floor=1e-3):
        """Evaluate g(z) = u'(W_next) * V_next(s_next) * R_p at every joint z node.

        u'(W) = W^{-gamma}
        V_next ~ a smooth synthetic V on s_next (use a small Gaussian-like profile
        to mimic value-function shape). For arbitrage/integration this only
        needs to be smooth and bounded — it's the curvature in z that matters.
        """
        v_nodes = joint_z[:, :model.n_state] @ L_s.T
        r_nodes = joint_z[:, model.n_state:] @ L_r.T
        s_next = Phi_0 + Phi_11 @ s_t + v_nodes              # (N, 4)
        log_R_bill = s_next[:, rtb_idx]
        Mv = v_nodes @ M.T
        log_x_s = base_mu_r[xr_pos] + Mv[:, xr_pos] + r_nodes[:, xr_pos]
        log_x_b = base_mu_r[xb_pos] + Mv[:, xb_pos] + r_nodes[:, xb_pos]
        r_p = _ccv_log_return(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
                              sigma2_xr, sigma2_xb, sigma_xrxb)
        R_p = np.exp(r_p)
        W = np.maximum(s_val * R_p, w_floor)
        # Synthetic V_next: smooth function of s_next, monotone in cy and y_1.
        # Doesn't matter what exactly — we just need finite, smooth.
        V = np.exp(-0.3 * s_next[:, 0] - 0.1 * s_next[:, 3]) * 0.1
        u_prime = W ** (-gamma)
        return u_prime * V * R_p

    # Build a very dense tensor-product baseline (truth)
    truth_K = (7, 7, 7, 7, 7, 7)   # 7^6 = 117,649 nodes
    print(f"  Building truth tensor with K={truth_K} ({np.prod(truth_K)} nodes)...")
    truth_z, truth_w = tensor_product_rule(truth_K)

    alphas = [
        ("at zero", 0.0, 0.0),
        ("body", 0.5, 0.5),
        ("stress mid", 1.5, 1.0),
        ("stress high", 3.0, 2.0),
        ("near cap", 5.0, -3.0),
        ("near cap +/+", 6.0, 6.0),
    ]
    for label_a, a_s, a_b in alphas:
        truth_integrand = _eval_integrand(truth_z, a_s, a_b)
        truth_val = float(truth_w @ truth_integrand)
        line = f"  alpha=(a_s={a_s:+.1f}, a_b={a_b:+.1f}) [{label_a}]  truth={truth_val:.6e}"
        print(line)
        for label_r, (z, w) in rules.items():
            integ = _eval_integrand(z, a_s, a_b)
            val = float(w @ integ)
            relerr = abs(val - truth_val) / max(abs(truth_val), 1e-30)
            print(f"      {label_r:<28} N={len(w):>5}  val={val:+.6e}  relerr={relerr:.3e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building model + precompute (canonical_jax with Lobatto disabled)...")
    model, pc, disc = _build()
    print(f"  N_state={pc.N_state}, n_state_quad={pc.n_state_quad}, n_ret_quad={pc.n_ret_quad}")
    print(f"  Tensor baseline = {pc.n_state_quad}*{pc.n_ret_quad} = "
          f"{pc.n_state_quad * pc.n_ret_quad} nodes")

    # Build a portfolio of rules to compare. All on R^6 (joint state+return).
    # Axis order: (cy, spr, rtb, y_1, xr, xb)
    rules = {}

    # Tensor baselines
    rules["Tensor (3,3,3,3,4,4) GH"] = tensor_product_rule((3, 3, 3, 3, 4, 4))
    rules["Tensor (3,3,3,5,4,4) GH"] = tensor_product_rule((3, 3, 3, 5, 4, 4))
    rules["Tensor (3,3,3,5,3,5) GH"] = tensor_product_rule((3, 3, 3, 5, 3, 5))
    rules["Tensor (5,3,3,5,3,5) GH"] = tensor_product_rule((5, 3, 3, 5, 3, 5))
    rules["Tensor (3,5,3,5,5,5) GH"] = tensor_product_rule((3, 5, 3, 5, 5, 5))
    rules["Tensor (5,5,5,5,5,5) GH"] = tensor_product_rule((5, 5, 5, 5, 5, 5))

    # Mixed Lobatto + GH rules. Lobatto requires K in {3,5,7}.
    # Z values from numba investigation: Z=2.93 on state axes (matches state_n_stds),
    # Z=2.86 on return axes (matches GH-K=5 max-tail node).
    rules["Mixed (3,3,3,5,3,5) Lobatto y_1=2.93, xb=2.86"] = tensor_mixed_rule(
        (3, 3, 3, 5, 3, 5), (None, None, None, 2.93, None, 2.86)
    )
    rules["Mixed (5,3,3,5,3,5) Lobatto cy=2.93, y_1=2.93, xb=2.86"] = tensor_mixed_rule(
        (5, 3, 3, 5, 3, 5), (2.93, None, None, 2.93, None, 2.86)
    )
    rules["Mixed (5,3,3,5,5,5) Lobatto cy,y_1,xr,xb"] = tensor_mixed_rule(
        (5, 3, 3, 5, 5, 5), (2.93, None, None, 2.93, 2.86, 2.86)
    )

    # Smolyak isotropic at various L
    for L in (3, 4, 5):
        try:
            z, w = smolyak_anisotropic(L=L, w_axis=(1, 1, 1, 1, 1, 1),
                                       K_max_per_axis=(L + 1,) * 6)
            rules[f"Smolyak iso L={L}"] = (z, w)
        except Exception as e:
            print(f"  iso L={L} failed: {e}")

    # Anisotropic Smolyak — y_1 is the dominant axis on bond integrand
    # via M[xb, y_1] = -8.84. Refine y_1 (axis 3) and the bond residual
    # (axis 5) more than the others.
    aniso_specs = [
        ("Smolyak aniso A L=4", 4, (2, 4/3, 2, 1, 4/3, 1)),       # mostly refine y_1, xb
        ("Smolyak aniso B L=4", 4, (2, 4/3, 2, 1, 1, 1)),         # refine y_1, xr, xb
        ("Smolyak aniso C L=5", 5, (2, 4/3, 4/3, 1, 1, 1)),       # refine y_1, spr, xr, xb
        ("Smolyak aniso D L=5", 5, (2, 4/3, 2, 1, 4/3, 1)),       # like A, more level
        ("Smolyak aniso E L=6", 6, (2, 4/3, 4/3, 1, 1, 1)),       # high level
    ]
    for label, L, w_ax in aniso_specs:
        try:
            z, w = smolyak_anisotropic(L=L, w_axis=w_ax, K_max_per_axis=(7,) * 6)
            rules[label] = (z, w)
        except Exception as e:
            print(f"  {label} failed: {e}")

    # Run tests
    test_moment_recovery(rules)
    test_arbitrage_cloud(model, pc, rules)
    test_integration_accuracy(model, pc, rules, base_label="Tensor (3,3,3,3,4,4)")


if __name__ == "__main__":
    main()
