"""
DEBUG: Check 4 — chi sensitivity unit test.

Calls static_condensation() directly with a synthetic nonzero trace/source
for two values of k1 (differing by 3 orders of magnitude) and compares
flux_jump[0:2] (the u-equation flux, where chi enters).

If the two flux_jumps differ significantly → chi is working mathematically,
and the issue is physical (fields are zero at runtime).

If they do NOT differ → structural bug in matrix construction or chi is
not being passed through correctly.

Run with:
    python tests/debug_chi_sensitivity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal stubs so we can instantiate StaticCondensationOOC without a full
# simulation setup.
# ---------------------------------------------------------------------------

def make_elementary_matrices(h=1.0):
    """Return a mock elementary matrix provider with plausible 2x2 matrices."""
    rng = np.random.default_rng(42)

    def spd(n=2):
        A = rng.normal(size=(n, n))
        return A @ A.T + n * np.eye(n)

    mats = {
        'M':    spd(),
        'Mb':   spd(),
        'Gb':   rng.normal(size=(2, 2)),
        'T':    rng.normal(size=(2, 2)),
        'D':    rng.normal(size=(2, 2)),
        'IM':   np.linalg.inv(spd()) * h,
        'Av':   np.ones((1, 2)) / 2,
        'Ntil': rng.normal(size=(2, 2)),
        'Nhat': rng.normal(size=(2, 2)),
        'QUAD': spd(),
    }

    em = MagicMock()
    em.get_matrix.side_effect = lambda name: mats[name]
    return em


def make_problem(k1, k2=5e-6, nu=200.0, mu=900.0, epsilon=900.0, sigma=1e-9,
                 a=1e-4, b=0.2, c=1e-4, d=0.1):
    prob = MagicMock()
    prob.parameters = np.array([nu, mu, epsilon, sigma, a, b, c, d, 1.0])
    prob.chi  = lambda x: k1 / (nu * (k2 + x)**2)
    prob.dchi = lambda x: -2.0 * k1 / (nu * (k2 + x)**3)
    prob.lambda_function  = lambda omega: np.ones_like(omega)
    prob.dlambda_function = lambda omega: np.zeros_like(omega)
    return prob


def make_discretization(h=15.0, tau=None):
    disc = MagicMock()
    disc.element_length = h
    disc.tau = tau or [0.5, 0.5, 0.5, 0.5]
    return disc


def make_global_disc(h=15.0, tau=None, dt=64.0):
    """Wrap a discretization inside a global_disc mock as the base class expects."""
    disc = make_discretization(h=h, tau=tau)
    global_disc = MagicMock()
    global_disc.spatial_discretizations = [disc]
    global_disc.dt = dt
    return global_disc


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

def run_sc(k1, dt=64.0, seed=7):
    """Build matrices and run one static_condensation call; return flux_jump."""
    from bionetflux.core.static_condensation_ooc import StaticCondensationOOC

    h = 15.0
    problem = make_problem(k1=k1)
    global_disc = make_global_disc(h=h, dt=dt)
    em      = make_elementary_matrices(h=h)

    sc = StaticCondensationOOC(problem, global_disc, em)
    # sc.dt is set by base __init__ from global_disc.dt

    sc.build_matrices()

    rng = np.random.default_rng(seed)
    # Nonzero trace and source — same for both k1 values
    local_trace  = rng.normal(scale=1e-3, size=8)
    local_source = rng.normal(scale=1e-3, size=8)

    _, _, flux_jump, jacobian = sc.static_condensation(local_trace, local_source)
    return flux_jump, jacobian


if __name__ == '__main__':
    k1_lo = 3.9e-1
    k1_hi = 3.9e2   # 1000x larger

    print("=" * 60)
    print(f"Check 4: Chi sensitivity test")
    print(f"  k1_lo = {k1_lo},  k1_hi = {k1_hi}")
    print("=" * 60)

    fj_lo, jac_lo = run_sc(k1_lo)
    fj_hi, jac_hi = run_sc(k1_hi)

    print("\nflux_jump (k1_lo):", fj_lo)
    print("flux_jump (k1_hi):", fj_hi)

    diff = np.abs(fj_hi - fj_lo)
    rel  = diff / (np.abs(fj_lo) + 1e-30)
    print(f"\nAbsolute difference:  {diff}")
    print(f"Relative difference:  {rel}")

    if np.max(rel[:2]) > 0.01:
        print("\n[PASS] flux_jump[0:2] (u-equation) changes with k1 → chi IS active.")
        print("       Look for zero fields at runtime (Check 2 prints).")
    else:
        print("\n[FAIL] flux_jump[0:2] does NOT change with k1.")
        print("       Likely structural bug in matrix construction or chi routing.")

    # Also check jacobian rows
    djac_rel = np.abs(jac_hi - jac_lo) / (np.abs(jac_lo) + 1e-30)
    print(f"\nJacobian max relative change row 0: {djac_rel[0].max():.4e}")
    print(f"Jacobian max relative change row 1: {djac_rel[1].max():.4e}")
