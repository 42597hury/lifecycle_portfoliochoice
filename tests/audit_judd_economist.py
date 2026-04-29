"""Economist's audit of the Judd-mixture quadrature.

Polynomial-exactness tests (Tier 1) only verify the rule integrates monomials
up to degree 2n-1 exactly. The integrals an economist actually cares about
are:

  1) CRRA marginal-utility expectations:  E[exp(-gamma * X)]
  2) Income-risk insurance:                E[exp(-gamma * (eta + eps))]
  3) Loss probabilities / tail mass:       P(X <= q) for q deep in the tail
  4) CDF fidelity at risk-relevant quantiles

This audit does NOT replace test_judd_quadrature.py. It is an out-of-band
diagnostic — run it directly:

    python tests/audit_judd_economist.py

It compares four rules at production calibration:

  - Judd, n=3   (new, default)         exactness 5
  - Judd, n=5   (new, opt-in)          exactness 9
  - GH-mixture K=3, 6 nodes (old)       exactness 5  (same poly order as Judd n=3)
  - GH-mixture K=5, 10 nodes (old)      exactness 9  (same poly order as Judd n=5)

Reference: 200-point Gauss-Hermite per mixture component (400 nodes per shock,
exact through ~degree 399 against each component) — essentially machine
precision for smooth integrands.
"""

from __future__ import annotations

import numpy as np
from scipy.special import roots_hermite
from scipy.stats import norm
import sys
from pathlib import Path

# allow running as a script from project root or tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discretization import _judd_mixture_quadrature  # noqa: E402

ETA_CAL = dict(p=0.176, mu1=-0.524, sigma1=0.113, sigma2=0.046)
EPS_CAL = dict(p=0.044, mu1=0.134,  sigma1=0.762, sigma2=0.055)


def zero_mean_components(p, mu1, sigma1, sigma2):
    return [p, 1.0 - p], [mu1, -(p / (1.0 - p)) * mu1], [sigma1, sigma2]


def gh_mixture(probs, mus, sigmas, K_per_component):
    """Old concatenated-Gauss-Hermite construction. 2K total nodes."""
    z, w = roots_hermite(K_per_component)
    z = z * np.sqrt(2.0)
    w = w / np.sqrt(np.pi)
    nodes_all = []
    weights_all = []
    for p, mu, s in zip(probs, mus, sigmas):
        nodes_all.append(z * s + mu)
        weights_all.append(w * p)
    return np.concatenate(nodes_all), np.concatenate(weights_all)


def reference_rule(probs, mus, sigmas, K=200):
    return gh_mixture(probs, mus, sigmas, K)


# ---------------------------------------------------------------------------

def report_marginal_crra(name, cfg):
    print(f"\n=== Marginal CRRA: E[exp(-gamma * {name})] ===")
    probs, mus, sigmas = zero_mean_components(**cfg)
    rules = {
        "TRUTH (GH 400)":       reference_rule(probs, mus, sigmas, K=200),
        "Judd n=3 (new)":       _judd_mixture_quadrature(probs, mus, sigmas, 3),
        "Judd n=5 (new)":       _judd_mixture_quadrature(probs, mus, sigmas, 5),
        "GH K=3 (old, 6)":      gh_mixture(probs, mus, sigmas, 3),
        "GH K=5 (old, 10)":     gh_mixture(probs, mus, sigmas, 5),
    }
    print(f"{'gamma':>5s}  {'TRUTH':>14s}  "
          f"{'Judd 3 rel':>12s}  {'Judd 5 rel':>12s}  "
          f"{'GH K=3 rel':>12s}  {'GH K=5 rel':>12s}")
    for gamma in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        truth = float(np.sum(rules["TRUTH (GH 400)"][1]
                             * np.exp(-gamma * rules["TRUTH (GH 400)"][0])))
        approx = {}
        for label, (n, w) in rules.items():
            approx[label] = float(np.sum(w * np.exp(-gamma * n)))
        rel = lambda a: (a - truth) / truth
        print(f"{gamma:5.1f}  {truth:14.6e}  "
              f"{rel(approx['Judd n=3 (new)']):+12.3e}  "
              f"{rel(approx['Judd n=5 (new)']):+12.3e}  "
              f"{rel(approx['GH K=3 (old, 6)']):+12.3e}  "
              f"{rel(approx['GH K=5 (old, 10)']):+12.3e}")


def report_joint_crra():
    print("\n=== Joint CRRA: E[exp(-gamma * (eta + eps))] ===")
    probs_e, mus_e, sigmas_e = zero_mean_components(**ETA_CAL)
    probs_x, mus_x, sigmas_x = zero_mean_components(**EPS_CAL)

    def tensor(rule_eta, rule_eps):
        nE, wE = rule_eta
        nX, wX = rule_eps
        N = np.add.outer(nE, nX).ravel()
        W = np.outer(wE, wX).ravel()
        return N, W

    rules = {
        "TRUTH (GH 400 x 400)": tensor(reference_rule(probs_e, mus_e, sigmas_e, 200),
                                       reference_rule(probs_x, mus_x, sigmas_x, 200)),
        "Judd 3 x 3 (new, 9)":  tensor(_judd_mixture_quadrature(probs_e, mus_e, sigmas_e, 3),
                                       _judd_mixture_quadrature(probs_x, mus_x, sigmas_x, 3)),
        "Judd 5 x 5 (new, 25)": tensor(_judd_mixture_quadrature(probs_e, mus_e, sigmas_e, 5),
                                       _judd_mixture_quadrature(probs_x, mus_x, sigmas_x, 5)),
        "GH K=3 x K=3 (old, 36)": tensor(gh_mixture(probs_e, mus_e, sigmas_e, 3),
                                          gh_mixture(probs_x, mus_x, sigmas_x, 3)),
        "GH K=5 x K=5 (old, 100)": tensor(gh_mixture(probs_e, mus_e, sigmas_e, 5),
                                           gh_mixture(probs_x, mus_x, sigmas_x, 5)),
    }
    print(f"{'gamma':>5s}  {'TRUTH':>14s}  "
          f"{'Judd 3x3 rel':>14s}  {'Judd 5x5 rel':>14s}  "
          f"{'GH K=3 rel':>12s}  {'GH K=5 rel':>12s}")
    for gamma in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        truth = float(np.sum(rules["TRUTH (GH 400 x 400)"][1]
                             * np.exp(-gamma * rules["TRUTH (GH 400 x 400)"][0])))
        approx = {}
        for label, (n, w) in rules.items():
            approx[label] = float(np.sum(w * np.exp(-gamma * n)))
        rel = lambda a: (a - truth) / truth
        print(f"{gamma:5.1f}  {truth:14.6e}  "
              f"{rel(approx['Judd 3 x 3 (new, 9)']):+14.3e}  "
              f"{rel(approx['Judd 5 x 5 (new, 25)']):+14.3e}  "
              f"{rel(approx['GH K=3 x K=3 (old, 36)']):+12.3e}  "
              f"{rel(approx['GH K=5 x K=5 (old, 100)']):+12.3e}")


def report_tail_mass(name, cfg):
    """Compare discrete tail mass against true CDF at risk-relevant quantiles."""
    print(f"\n=== Tail mass for {name} (P(X <= q)) ===")
    probs, mus, sigmas = zero_mean_components(**cfg)

    def cdf(x):
        return sum(p * norm.cdf(x, loc=mu, scale=s) for p, mu, s in zip(probs, mus, sigmas))

    # Find true quantiles by bisection
    def quantile(q):
        lo, hi = -10.0, 10.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        return mid

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quants = [quantile(q) for q in qs]

    rules = {
        "Judd 3":  _judd_mixture_quadrature(probs, mus, sigmas, 3),
        "Judd 5":  _judd_mixture_quadrature(probs, mus, sigmas, 5),
        "GH K=3":  gh_mixture(probs, mus, sigmas, 3),
        "GH K=5":  gh_mixture(probs, mus, sigmas, 5),
    }

    print(f"{'q':>5s}  {'X<=q':>9s}  {'true':>9s}  "
          f"{'Judd 3':>9s}  {'Judd 5':>9s}  "
          f"{'GH 3':>9s}  {'GH 5':>9s}")
    for q, qv in zip(qs, quants):
        line = f"{q:5.2f}  {qv:+9.4f}  {q:9.4f}  "
        for lbl in ["Judd 3", "Judd 5", "GH K=3", "GH K=5"]:
            n, w = rules[lbl]
            mass = float(np.sum(w[n <= qv]))
            line += f"{mass:9.4f}  "
        print(line)


def show_node_layout():
    print("\n=== Node layout (production calibration, n=3) ===")
    for label, cfg in [("eta", ETA_CAL), ("eps", EPS_CAL)]:
        probs, mus, sigmas = zero_mean_components(**cfg)
        n, w = _judd_mixture_quadrature(probs, mus, sigmas, 3)
        print(f"{label}: nodes  = {np.array2string(n, precision=4, sign='+')}")
        print(f"{label}: weights= {np.array2string(w, precision=4)}")
        # density at the nodes
        f = np.array([sum(p * norm.pdf(x, loc=mu, scale=s)
                          for p, mu, s in zip(probs, mus, sigmas))
                      for x in n])
        print(f"{label}: f(x_i) = {np.array2string(f, precision=4)}")


if __name__ == "__main__":
    np.set_printoptions(linewidth=140)
    show_node_layout()
    report_marginal_crra("eta", ETA_CAL)
    report_marginal_crra("eps", EPS_CAL)
    report_joint_crra()
    report_tail_mass("eta", ETA_CAL)
    report_tail_mass("eps", EPS_CAL)
