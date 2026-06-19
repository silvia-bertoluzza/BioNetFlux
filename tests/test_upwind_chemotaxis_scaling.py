#!/usr/bin/env python3
"""Regression test for the nu-scaling of the chemotaxis upwind velocity in
``StaticCondensationOOCUpwind``.

Bug (fixed): the upwind stabilization in the u-equation used the rescaled
``chi_code = chi_true / nu`` as the advection velocity, while the physical flux
``j`` uses ``chi_true = nu * chi_code``. The diffusive stabilization carries nu
(``tu = nu * tau / h``) but the advective (upwind) part did not, so with nu < 1
the chemotaxis transport was mis-scaled by 1/nu — manifesting as needing to
multiply the forcing by nu to recover the manufactured solution.

The fix uses the physical velocity ``w = nu * chi_code * grad_phi`` in the
upwind stabilization. This test reconstructs the upwind stabilization
coefficient from the returned flux jump and asserts it matches the physical
velocity (and is distinguishable from the buggy, un-rescaled value).
"""
import numpy as np
import pytest

from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.utils.elementary_matrices import ElementaryMatrices
from bionetflux.core.static_condensation_ooc_upwind import StaticCondensationOOCUpwind


# Physical/test parameters
NU = 0.1
K1 = 2.0
K2 = 1.0
H = 0.5            # element length (single element, domain_length = H)
TAU = [0.5, 0.5, 0.5, 0.5]
DT = 0.1


class _MockProblem:
    """Minimal problem exposing what StaticCondensationOOCUpwind reads."""

    def __init__(self):
        # parameters: [nu, mu, epsilon, sigma, a, b, c, d, 1.0]
        self.parameters = np.array([NU, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.neq = 4
        # chi is the rescaled chi_code = chi_true / nu  (receptor_saturation)
        self.chi = lambda x: K1 / (NU * (K2 + x) ** 2)
        self.dchi = lambda x: -2.0 * K1 / (NU * (K2 + x) ** 3)


def _build_sc():
    disc = Discretization(n_elements=1, domain_start=0.0, domain_length=H)
    disc.set_tau(TAU)
    gdisc = GlobalDiscretization([disc])
    gdisc.set_time_parameters(dt=DT, T=1.0)
    em = ElementaryMatrices(orthonormal_basis=False)
    sc = StaticCondensationOOCUpwind(_MockProblem(), gdisc, em, ipb=0)
    sc.build_matrices()
    return sc


def test_upwind_velocity_uses_physical_chi():
    """Reconstructed upwind coefficient = tu - min(0, nu*chi_code*grad_phi*n)."""
    sc = _build_sc()

    # Trace [u(2), omega(2), v(2), phi(2)] and a nonzero u-source so the bulk u
    # differs from its trace (=> the upwind term acts on a nonzero jump).
    phi0, phi1 = 0.0, 0.5
    local_trace = np.array([1.0, 1.1, 0.0, 0.0, 0.0, 0.0, phi0, phi1])
    local_source = np.array([0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Picard freeze: phi from previous iterate (u4) sets chi_frozen / grad_phi.
    prev = np.array([1.0, 1.1, 0.0, 0.0, 0.0, 0.0, phi0, phi1])

    bulk, flux, flux_jump, _jac = sc.static_condensation(
        local_trace, local_source, prev_local_solution=prev)

    U = np.asarray(bulk).flatten()
    u1 = U[:2]                       # bulk u (P1 nodal values)
    hu_u = local_trace[:2]           # u trace
    T = sc.sc_matrices['T']
    delta = T @ u1 - hu_u            # the jump the upwind term multiplies
    assert abs(delta[0]) > 1e-12, "Need a nonzero bulk-trace jump to probe Tau"

    # hj = B5.T @ j + Tau_Upwind @ delta, with B5 = normali = [-1, 1] and
    # j the scalar P0 u-flux (first entry of `flux`).
    hj = np.asarray(flux_jump).flatten()[:2]
    j = float(np.asarray(flux).flatten()[0])
    normali = np.array([-1.0, 1.0])
    residual = hj - normali * j      # = Tau_Upwind @ delta (Tau_Upwind diagonal)

    tau_diag0 = residual[0] / delta[0]

    # Expected coefficient with the *physical* chemotaxis velocity.
    tu = NU * TAU[0] / H
    grad_phi = (phi1 - phi0) / H
    chi_code0 = K1 / (NU * (K2 + phi0) ** 2)
    w_phys0 = NU * chi_code0 * grad_phi
    expected = tu - min(0.0, w_phys0 * normali[0])

    # Buggy variant (velocity without the nu rescaling) — must NOT match.
    w_bug0 = chi_code0 * grad_phi
    buggy = tu - min(0.0, w_bug0 * normali[0])

    assert np.isclose(tau_diag0, expected, rtol=1e-9, atol=1e-12), (
        f"upwind coeff {tau_diag0} != physical {expected}")
    assert not np.isclose(tau_diag0, buggy, rtol=1e-6), (
        "upwind coeff matches the un-rescaled (buggy) velocity — nu dropped out")


def test_upwind_velocity_invariant_to_nu_at_fixed_chi_true():
    """With chi_true fixed, the physical upwind velocity nu*chi_code is nu-independent.

    chi_code = chi_true / nu, so nu * chi_code = chi_true. The reconstructed
    upwind coefficient's advective part must therefore be identical for two
    different nu values (the diffusive part tu = nu*tau/h is subtracted out).
    """
    coeffs = {}
    for nu in (0.1, 0.5):
        disc = Discretization(n_elements=1, domain_start=0.0, domain_length=H)
        disc.set_tau(TAU)
        gdisc = GlobalDiscretization([disc])
        gdisc.set_time_parameters(dt=DT, T=1.0)
        em = ElementaryMatrices(orthonormal_basis=False)

        class _P:
            def __init__(self, nu_):
                self.parameters = np.array([nu_, 1, 1, 1, 1, 1, 1, 1, 1.0])
                self.neq = 4
                # chi_true = K1/(K2+x)^2 held fixed => chi_code = chi_true/nu
                self.chi = lambda x, _nu=nu_: (K1 / (K2 + x) ** 2) / _nu
                self.dchi = lambda x, _nu=nu_: (-2.0 * K1 / (K2 + x) ** 3) / _nu

        sc = StaticCondensationOOCUpwind(_P(nu), gdisc, em, ipb=0)
        sc.build_matrices()

        phi0, phi1 = 0.0, 0.5
        local_trace = np.array([1.0, 1.1, 0, 0, 0, 0, phi0, phi1])
        local_source = np.array([0.2, 0.3, 0, 0, 0, 0, 0, 0.0])
        prev = np.array([1.0, 1.1, 0, 0, 0, 0, phi0, phi1])

        bulk, flux, flux_jump, _ = sc.static_condensation(
            local_trace, local_source, prev_local_solution=prev)
        U = np.asarray(bulk).flatten()
        delta = sc.sc_matrices['T'] @ U[:2] - local_trace[:2]
        hj = np.asarray(flux_jump).flatten()[:2]
        j = float(np.asarray(flux).flatten()[0])
        residual = hj - np.array([-1.0, 1.0]) * j
        tau0 = residual[0] / delta[0]
        # Advective part = Tau - tu (subtract the diffusive piece).
        coeffs[nu] = tau0 - nu * TAU[0] / H

    assert np.isclose(coeffs[0.1], coeffs[0.5], rtol=1e-9, atol=1e-12), (
        f"physical upwind velocity should be nu-independent: {coeffs}")
