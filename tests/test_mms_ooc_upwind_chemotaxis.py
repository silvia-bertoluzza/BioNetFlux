#!/usr/bin/env python3
"""Tests for the chemotaxis-coupled MMS generation in
``validation_scripts/mms_ooc_upwind.py``.

These cover the symbolic chemotaxis sensitivity helper ``_chi_expr`` and the
forcing produced by ``compute_mms`` when ``chi`` is active. The reference
chemotaxis definitions mirror ``src/bionetflux/problems/ooc_problem_upwind.py``:

    constant:            chi(phi) = k1
    receptor_saturation: chi(phi) = k1 / (nu * (k2 + phi)**2)

The chemotaxis term enters the immune-cell flux as
``flux_u = nu*du/ds - chi(phi)*u*dphi/ds``.
"""
import importlib.util
import os

import pytest
import sympy as sp

# ---------------------------------------------------------------------------
# Load the generator module by file path (validation_scripts/ is not a package).
# ---------------------------------------------------------------------------
_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "validation_scripts", "mms_ooc_upwind.py"
)
_spec = importlib.util.spec_from_file_location("mms_ooc_upwind", _SCRIPT_PATH)
mms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mms)

s, t = mms.s, mms.t


@pytest.mark.unit
class TestChiExpr:
    """The symbolic chemotaxis sensitivity helper."""

    def test_zero_k1_returns_none(self):
        """k1 == 0 disables chi, reproducing the legacy chi-free forcing."""
        assert mms._chi_expr("constant", 0.0, 1.0, 1.0, t * sp.sin(s)) is None
        assert mms._chi_expr("receptor_saturation", 0.0, 2.0, 1.0, t * sp.sin(s)) is None

    def test_constant_returns_k1(self):
        """Constant chemotaxis is the constant k1, independent of phi."""
        chi = mms._chi_expr("constant", 1.5, 1.0, 1.0, t * sp.sin(s))
        assert sp.simplify(chi - sp.Float(1.5)) == 0

    def test_receptor_saturation_matches_solver(self):
        """Receptor-saturation chi(phi) = k1 / (nu * (k2 + phi)**2)."""
        phi = t * sp.sin(s)
        chi = mms._chi_expr("receptor_saturation", 1.0, 2.0, 1.0, phi)
        expected = sp.Float(1.0) / (sp.Float(1.0) * (sp.Float(2.0) + phi) ** 2)
        assert sp.simplify(chi - expected) == 0

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            mms._chi_expr("nonsense", 1.0, 1.0, 1.0, t * sp.sin(s))


@pytest.mark.unit
class TestCoupledChiForcing:
    """``compute_mms`` with an active chemotaxis term."""

    def test_constant_chi_forcing(self):
        """For u = phi = t*sin(s), nu = 1, chi = 1, the immune-cell forcing is
        f_u = d(u)/dt - d/ds(nu*du/ds - chi*u*dphi/ds).

        Hand-derived reference:
            f_u = sin(s) + t*sin(s) + t**2*cos(s)**2 - t**2*sin(s)**2.
        """
        u = t * sp.sin(s)
        phi = t * sp.sin(s)
        params = dict(nu=1.0, mu=1.0, epsilon=1.0, sigma=1.0,
                      a=1.0, b=0.0, c=1.0, d=0.0)
        chi = mms._chi_expr("constant", 1.0, 1.0, params["nu"], phi)
        result = mms.compute_mms(
            u_ms=u, omega_ms=mms.ZERO, v_ms=mms.ZERO, phi_ms=phi,
            params=params, chi_ms=chi,
        )
        expected_f_u = (sp.sin(s) + t * sp.sin(s)
                        + t ** 2 * sp.cos(s) ** 2 - t ** 2 * sp.sin(s) ** 2)
        assert sp.simplify(result["f_u"] - expected_f_u) == 0

    def test_flux_u_includes_chemotaxis(self):
        """The immune-cell flux carries the -chi*u*dphi/ds chemotaxis term."""
        u = t * sp.sin(s)
        phi = t * sp.sin(s)
        params = dict(nu=1.0, mu=1.0, epsilon=1.0, sigma=1.0,
                      a=1.0, b=0.0, c=1.0, d=0.0)
        chi = mms._chi_expr("constant", 1.0, 1.0, params["nu"], phi)
        result = mms.compute_mms(
            u_ms=u, omega_ms=mms.ZERO, v_ms=mms.ZERO, phi_ms=phi,
            params=params, chi_ms=chi,
        )
        expected_flux = (params["nu"] * sp.diff(u, s)
                         - chi * u * sp.diff(phi, s))
        assert sp.simplify(result["flux_u"] - expected_flux) == 0

    def test_no_chi_matches_legacy(self):
        """Passing chi_ms=None reproduces the pure-diffusion immune-cell flux."""
        u = t * sp.sin(s)
        phi = t * sp.sin(s)
        params = dict(nu=1.0, mu=1.0, epsilon=1.0, sigma=1.0,
                      a=1.0, b=0.0, c=1.0, d=0.0)
        result = mms.compute_mms(
            u_ms=u, omega_ms=mms.ZERO, v_ms=mms.ZERO, phi_ms=phi,
            params=params, chi_ms=None,
        )
        assert sp.simplify(result["flux_u"] - params["nu"] * sp.diff(u, s)) == 0


@pytest.mark.unit
class TestNewCasesRegistered:
    """The four coupled-chemotaxis cases are present and configured."""

    def test_cases_exist(self):
        names = {c["name"] for c in mms.CASES}
        assert {
            "chemo_const_sin",
            "chemo_satur_sin",
            "fully_coupled_chi_const",
            "fully_coupled_chi_satur",
        } <= names

    def test_chi_types(self):
        by_name = {c["name"]: c for c in mms.CASES}
        assert by_name["chemo_const_sin"]["chi_type"] == "constant"
        assert by_name["chemo_satur_sin"]["chi_type"] == "receptor_saturation"
        assert by_name["fully_coupled_chi_const"]["chi_type"] == "constant"
        assert by_name["fully_coupled_chi_satur"]["chi_type"] == "receptor_saturation"

    def test_chemotaxis_active(self):
        """Each new case has k1 > 0 and nonzero u and phi, so chi is active."""
        by_name = {c["name"]: c for c in mms.CASES}
        for name in ("chemo_const_sin", "chemo_satur_sin",
                     "fully_coupled_chi_const", "fully_coupled_chi_satur"):
            case = by_name[name]
            assert case["k1"] > 0.0
            assert case["u"] != mms.ZERO
            assert case["phi"] != mms.ZERO

    def test_receptor_saturation_nonsingular(self):
        """receptor_saturation k2 keeps (k2 + phi) > 0 for phi = sin in [-1, 1]."""
        by_name = {c["name"]: c for c in mms.CASES}
        for name in ("chemo_satur_sin", "fully_coupled_chi_satur"):
            assert by_name[name]["k2"] > 1.0
