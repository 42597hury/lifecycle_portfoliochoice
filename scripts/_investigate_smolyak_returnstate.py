"""Investigation: anisotropic Smolyak-Hermite for the joint return x state-innovation expectation.

Investigation only. Does NOT touch lifecycle/, configs/_canonical.py, or any
production module. Run standalone:

    python scripts/_investigate_smolyak_returnstate.py

Question: can a 6-D anisotropic Smolyak-Hermite rule replace the nested
state-quad x return-quad outer product currently used in
[solver.py:874-928] without losing accuracy on CCV-relevant integrands?

Baseline (canonical, configs/_canonical.py):
    n_state_quad_nodes = (3, 4, 4)  ->  48 state-innovation nodes
    n_ret_nodes_1d     = (3, 5, 5)  ->  75 return-residual nodes
    Joint per-cell     = 3,600 nodes outer product.

The 6-D integrand is over z = (z_v0, z_v1, z_v2, z_r0, z_r1, z_r2) ~ N(0, I_6),
mapped to physical (v, r) via Cholesky:  v = L_state z_v,  r = L_ret z_r.

Anisotropy rationale (configs/_canonical.py:68-71 + LOBATTO_CONFIG_TRACKER §11):
    Axis    Channel       Target K   Why
    z_v0    cy resid      3          M[xb, cy]  = -0.005  (weak)
    z_v1    spr resid     4          M[xb, spr] = -8.514  (dominant)
    z_v2    y_1 resid     4          M[xb, y_1] = -8.718  (dominant)
    z_r0    rtb resid     3          small residual variance
    z_r1    xr resid      5          largest residual variance
    z_r2    xb resid      5          bond leg, high gamma|alpha_b| stress

Univariate rule schedule:  R_i = Gauss-Hermite of order n(i) = i over N(0,1).
                          R_0 ≡ 0,   Δ_i = R_i - R_{i-1}.

Anisotropic Smolyak set:
    A(L; w) = { i ∈ N^d : 1 ≤ i_k ≤ K_k_max,  Σ_k w_k (i_k - 1) ≤ L }

Combination technique (non-nested):  Q_A = Σ_j c_j ⊗_k R_{j_k}  with
    c_j = Σ_{q ∈ {0,1}^d : (j+q) ∈ A} (-1)^{|q|}
restricted to j with all j_k ≥ 1 (so R_{j_k} is well-defined).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import roots_hermite


# --------------------------------------------------------------------------- #
# Univariate Gauss-Hermite rule on N(0,1).                                    #
# --------------------------------------------------------------------------- #
def _gh_rule(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Order-K Gauss-Hermite nodes/weights for N(0,1) (zero-mean unit-var)."""
    if K == 0:
        return np.zeros(0), np.zeros(0)
    z, w = roots_hermite(K)
    return z * np.sqrt(2.0), w / np.sqrt(np.pi)


# --------------------------------------------------------------------------- #
# Smolyak set + combination coefficients.                                     #
# --------------------------------------------------------------------------- #
def smolyak_admissible(L: float, w: tuple[float, ...], K_max: tuple[int, ...]) -> list[tuple[int, ...]]:
    """All multi-indices i with 1 ≤ i_k ≤ K_max[k] and Σ w_k (i_k - 1) ≤ L."""
    d = len(w)
    out: list[tuple[int, ...]] = []
    def recurse(prefix: list[int], k: int, slack: float) -> None:
        if k == d:
            out.append(tuple(prefix))
            return
        for i_k in range(1, K_max[k] + 1):
            cost = w[k] * (i_k - 1)
            if cost > slack + 1e-12:
                break
            recurse(prefix + [i_k], k + 1, slack - cost)
    recurse([], 0, L)
    return out


def smolyak_combination_coefs(A: list[tuple[int, ...]]) -> dict[tuple[int, ...], int]:
    """c_j = Σ_{q ∈ {0,1}^d : (j+q) ∈ A} (-1)^{|q|}  (Wasilkowski-Wozniakowski)."""
    A_set = set(A)
    d = len(A[0])
    j_set: set[tuple[int, ...]] = set()
    for i in A:
        for q in itertools.product((0, 1), repeat=d):
            j = tuple(i[k] - q[k] for k in range(d))
            if all(jk >= 1 for jk in j):
                j_set.add(j)
    coefs: dict[tuple[int, ...], int] = {}
    for j in j_set:
        c = 0
        for q in itertools.product((0, 1), repeat=d):
            i = tuple(j[k] + q[k] for k in range(d))
            if i in A_set:
                c += (-1) ** sum(q)
        if c != 0:
            coefs[j] = c
    return coefs


@dataclass
class SmolyakRule:
    """6-D Smolyak rule on N(0, I_d) as a list of signed tensor-product terms."""
    coefs: dict[tuple[int, ...], int]
    d: int

    @classmethod
    def build(cls, L: float, w: tuple[float, ...], K_max: tuple[int, ...]) -> "SmolyakRule":
        A = smolyak_admissible(L, w, K_max)
        coefs = smolyak_combination_coefs(A)
        return cls(coefs=coefs, d=len(w))

    @property
    def n_terms(self) -> int:
        return len(self.coefs)

    @property
    def n_signed_nodes(self) -> int:
        """Total weighted-node evaluations: sum over tensor terms of Π_k n(j_k)."""
        return sum(int(np.prod(j)) * 1 for j, c in self.coefs.items())

    def apply(self, integrand) -> float:
        """Evaluate Σ_j c_j (⊗_k R_{j_k} integrand). integrand: ndarray(d,) -> float."""
        total = 0.0
        for j, c in self.coefs.items():
            grids = [_gh_rule(jk) for jk in j]
            zs = [g[0] for g in grids]
            ws = [g[1] for g in grids]
            for idx in itertools.product(*[range(jk) for jk in j]):
                z = np.array([zs[k][idx[k]] for k in range(self.d)])
                wprod = float(np.prod([ws[k][idx[k]] for k in range(self.d)]))
                total += c * wprod * float(integrand(z))
        return total


# --------------------------------------------------------------------------- #
# Full tensor product reference (canonical baseline).                          #
# --------------------------------------------------------------------------- #
def full_tensor_apply(K: tuple[int, ...], integrand) -> float:
    """Standard full-tensor Gauss-Hermite quadrature on N(0, I_d)."""
    grids = [_gh_rule(k) for k in K]
    zs = [g[0] for g in grids]
    ws = [g[1] for g in grids]
    total = 0.0
    for idx in itertools.product(*[range(k) for k in K]):
        z = np.array([zs[d][idx[d]] for d in range(len(K))])
        wprod = float(np.prod([ws[d][idx[d]] for d in range(len(K))]))
        total += wprod * float(integrand(z))
    return total


# --------------------------------------------------------------------------- #
# Test integrand bank.                                                         #
# --------------------------------------------------------------------------- #
def make_mgf_integrand(a: np.ndarray):
    """exp(a · z), z ~ N(0, I_d).   Exact integral = exp(0.5 a · a)."""
    a = np.asarray(a, float)
    def f(z):
        return float(np.exp(np.dot(a, z)))
    exact = float(np.exp(0.5 * np.dot(a, a)))
    return f, exact


def make_log_portfolio_integrand(L_state: np.ndarray, M: np.ndarray, L_ret: np.ndarray,
                                 alpha_s: float, alpha_b: float):
    """exp(alpha · log_R_excess) where log_R_excess[k] = (M L_state z_v + L_ret z_r)[k+1].

    Mimics the FOC integrand:  E[exp(α^T(M v + r))] under z ~ N(0, I_6),
    with v = L_state z_v, r = L_ret z_r.
    Returns (integrand, exact_value).  Exact via MGF since linear in z.
    """
    M_xr = M[1, :] @ L_state            # shape (3,)
    M_xb = M[2, :] @ L_state            # shape (3,)
    L_xr = L_ret[1, :]                  # shape (3,) — but only first 2 are nonzero (lower triangular)
    L_xb = L_ret[2, :]
    a_v = alpha_s * M_xr + alpha_b * M_xb
    a_r = np.zeros(3)
    a_r += alpha_s * L_xr
    a_r += alpha_b * L_xb
    a = np.concatenate([a_v, a_r])
    return make_mgf_integrand(a)


def make_polynomial_integrand(powers: tuple[int, ...]):
    """Π_k z_k^{p_k}.  Exact = Π_k Hermite_moment(p_k)."""
    def hermite_moment(p):
        if p % 2 == 1:
            return 0.0
        # E[Z^{2m}] = (2m-1)!! for Z~N(0,1)
        if p == 0:
            return 1.0
        prod = 1.0
        for k in range(1, p, 2):
            prod *= k
        return prod
    exact = 1.0
    for p in powers:
        exact *= hermite_moment(p)
    def f(z):
        out = 1.0
        for k, p in enumerate(powers):
            out *= z[k] ** p
        return out
    return f, exact


# --------------------------------------------------------------------------- #
# Canonical empirical M, Sigma values from explore-agent findings.             #
# --------------------------------------------------------------------------- #
def canonical_M_and_choleskys() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirically observed M, L_state, L_ret_cond from this codebase."""
    M = np.array([
        [-0.0357, -0.6210, -0.9365],
        [-0.9324, -0.9411,  0.0949],
        [-0.0054, -8.5139, -8.7180],
    ])
    Sigma_ss = np.array([
        [0.027839,  0.000010,  0.000437],
        [0.000010,  0.000126, -0.000153],
        [0.000437, -0.000153,  0.000247],
    ])
    Sigma_r_cond = np.array([
        [ 0.000239, -0.000281, -0.000030],
        [-0.000281,  0.000957,  0.000071],
        [-0.000030,  0.000071,  0.000511],
    ])
    L_state = np.linalg.cholesky(Sigma_ss)
    L_ret = np.linalg.cholesky(Sigma_r_cond)
    return M, L_state, L_ret


# --------------------------------------------------------------------------- #
# Main investigation.                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    np.set_printoptions(precision=4, suppress=True)
    M, L_state, L_ret = canonical_M_and_choleskys()
    print("=" * 78)
    print("Canonical empirical inputs")
    print("=" * 78)
    print(f"M (return-from-state-innov, rows=rtb,xr,xb; cols=cy,spr,y_1):\n{M}\n")
    print(f"L_state diag = {np.diag(L_state)}  (cy, spr, y_1 residual std-dev)")
    print(f"L_ret   diag = {np.diag(L_ret)}    (rtb, xr, xb residual std-dev)\n")

    # Canonical full-tensor baseline.
    K_v_canonical = (3, 4, 4)
    K_r_canonical = (3, 5, 5)
    K_full = K_v_canonical + K_r_canonical
    n_full = int(np.prod(K_full))
    print("=" * 78)
    print("Baseline reference rule")
    print("=" * 78)
    print(f"Full tensor product 6-D Gauss-Hermite at K = {K_full}")
    print(f"  Total nodes = {n_full}  (= {np.prod(K_v_canonical)} state-quad x {np.prod(K_r_canonical)} return-quad)\n")

    # Anisotropic Smolyak candidates.
    # Axis order: (z_v0, z_v1, z_v2, z_r0, z_r1, z_r2)
    # Per-axis K_max copied from canonical.
    K_max = (3, 4, 4, 3, 5, 5)

    # Weight choice rationale: w_k = L_target / (K_max_k - 1) so the
    # "anchor" multi-index (3,4,4,3,5,5) sits exactly on the budget boundary
    # at L = L_target. Using L_target = 4 gives:
    #   w = (4/2, 4/3, 4/3, 4/2, 4/4, 4/4) = (2.0, 4/3, 4/3, 2.0, 1.0, 1.0)
    # Smaller w_k = more refinement on axis k. This places the heaviest
    # refinement on z_r1, z_r2 (xr/xb residuals — largest L_ret diag entries)
    # and the lightest refinement on z_v0, z_r0 (cy and rtb — weak channels).
    L_target = 4.0
    w_canonical = tuple(L_target / (K_max[k] - 1) for k in range(6))
    print("=" * 78)
    print("Anisotropic Smolyak weight choice")
    print("=" * 78)
    print(f"Per-axis K_max  = {K_max}")
    print(f"Per-axis weights w = {tuple(round(x, 3) for x in w_canonical)}")
    print("  (smaller = more refinement; chosen so anchor i*=K_max sits at boundary L_target)\n")

    candidates: list[tuple[str, SmolyakRule]] = []
    for L in (2.0, 3.0, 4.0, 5.0):
        rule = SmolyakRule.build(L, w_canonical, K_max)
        candidates.append((f"Smolyak L={L:.0f}", rule))

    # An "isotropic Smolyak" comparator with the same K_max but uniform weights.
    rule_iso = SmolyakRule.build(4.0, (1.0,) * 6, K_max)
    candidates.append(("Smolyak L=4 isotropic", rule_iso))

    # Lighter anisotropic level (L=3) but with explicit cap so xr/xb get full K=5
    # and cy/rtb stay at K=3:
    w_aggressive = (2.0, 1.5, 1.5, 2.0, 1.0, 1.0)
    rule_aggr = SmolyakRule.build(3.0, w_aggressive, K_max)
    candidates.append(("Smolyak L=3 aggressive", rule_aggr))

    print("=" * 78)
    print("Candidate Smolyak rules: node counts")
    print("=" * 78)
    print(f"{'rule':<28} {'#tensor terms':>15} {'#signed nodes':>16}  speed-up")
    print("-" * 78)
    for name, rule in candidates:
        speedup = n_full / max(rule.n_signed_nodes, 1)
        print(f"{name:<28} {rule.n_terms:>15d} {rule.n_signed_nodes:>16d}  {speedup:>6.1f}x")
    print(f"{'full tensor (baseline)':<28} {1:>15d} {n_full:>16d}  {1.0:>6.1f}x\n")

    # ----- Polynomial exactness check ---------------------------------------
    print("=" * 78)
    print("Polynomial exactness check: max |Smolyak - exact| over a basis")
    print("=" * 78)
    # All 6-D monomials with Σ p_k ≤ P_total and per-axis p_k ≤ 2 K_max[k] - 1.
    # 1-D GH of order K integrates polynomials of degree ≤ 2K-1 exactly.
    # The Smolyak rule's polynomial-exactness varies by multi-index; we test
    # empirically by sweeping degrees up to 2 max(K_max) - 1 = 9.
    def poly_max_err(rule_apply, max_total_degree: int) -> tuple[float, tuple[int, ...]]:
        worst = 0.0
        worst_p = (0,) * 6
        for total in range(max_total_degree + 1):
            for p in itertools.product(range(total + 1), repeat=6):
                if sum(p) != total:
                    continue
                if max(p) > 9:
                    continue
                f, exact = make_polynomial_integrand(p)
                est = rule_apply(f)
                err = abs(est - exact)
                if err > worst:
                    worst = err
                    worst_p = p
        return worst, worst_p

    for name, rule in candidates:
        err, p_worst = poly_max_err(lambda f: rule.apply(f), max_total_degree=6)
        print(f"{name:<28}  max poly err (deg<=6) = {err:.3e}   worst p = {p_worst}")
    err_full, p_full = poly_max_err(lambda f: full_tensor_apply(K_full, f), max_total_degree=6)
    print(f"{'full tensor (baseline)':<28}  max poly err (deg<=6) = {err_full:.3e}   worst p = {p_full}\n")

    # ----- Stress integrand: log-portfolio MGF ------------------------------
    print("=" * 78)
    print("Stress integrand: E[exp(alpha . log_R_excess)] (closed-form MGF)")
    print("=" * 78)
    print("This is the dominant nonlinearity in the FOC: integrating exp(log_R)")
    print("against the joint Gaussian z. Higher |alpha| amplifies tail mass.")
    stress_alphas = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (3.0, 3.0),
        (5.0, -3.0),
        (-3.0, 5.0),
        (6.0, 6.0),     # Cap-bound stress
        (6.0, -6.0),
    ]
    print(f"\n{'alpha=(a_s,a_b)':<20} {'exact':>12}  " +
          "  ".join(f"{name:>14}" for name, _ in candidates) + f"  {'full':>14}")
    print("-" * (20 + 14 + 16 * (len(candidates) + 1)))
    for a_s, a_b in stress_alphas:
        f, exact = make_log_portfolio_integrand(L_state, M, L_ret, a_s, a_b)
        row = [f"({a_s:+.0f},{a_b:+.0f})".rjust(20)]
        row.append(f"{exact:>12.6f}")
        for _, rule in candidates:
            est = rule.apply(f)
            err = est - exact
            row.append(f"{err:>+14.3e}")
        est_full = full_tensor_apply(K_full, f)
        err_full = est_full - exact
        row.append(f"{err_full:>+14.3e}")
        print("  ".join(row))
    print()

    # ----- Recommendation ---------------------------------------------------
    print("=" * 78)
    print("Recommendation")
    print("=" * 78)
    print("Look for the candidate with:")
    print("  - <= 5e-3 relative error on the (6,6)/(6,-6) stress integrand (high-|alpha| regime)")
    print("  - >= 2x reduction in signed-node count vs the 3,600-node baseline")
    print("  - polynomial exactness on all degree-<=4 monomials (covers Ito/Jensen terms")
    print("    and the linear FOC residual)\n")
    print("If no candidate is satisfying, the level vector needs more nodes on the")
    print("z_r1/z_r2 axes specifically -- try setting w_r1 = w_r2 = 0.8 and re-running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
