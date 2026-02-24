#!/usr/bin/env python3
"""
Tests for the per-equation flux polynomial order metadata
stored in StaticCondensationBase and its subclasses.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.utils.elementary_matrices import ElementaryMatrices
from bionetflux.core.static_condensation_factory import StaticCondensationFactory
from bionetflux.core.static_condensation_keller_segel import KellerSegelStaticCondensation
from bionetflux.core.static_condensation_ooc import StaticCondensationOOC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ks_problem() -> Problem:
    """Create a minimal Keller-Segel problem (neq=2)."""
    p = Problem(neq=2, domain_start=0.0, domain_length=1.0,
                parameters=np.array([1.0, 1.0, 0.0, 0.0]),
                problem_type="keller_segel", name="ks_test")
    return p


def _make_ooc_problem() -> Problem:
    """Create a minimal Organ-on-Chip problem (neq=4)."""
    p = Problem(neq=4, domain_start=0.0, domain_length=1.0,
                parameters=np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                problem_type="organ_on_chip", name="ooc_test")
    return p


def _make_global_disc(problem: Problem, n_elements: int = 4,
                      dt: float = 0.01, T: float = 0.1) -> GlobalDiscretization:
    """Create a GlobalDiscretization wrapping a single domain."""
    disc = Discretization(n_elements, problem.domain_start, problem.domain_length)
    disc.set_tau([1.0] * problem.neq)
    gd = GlobalDiscretization([disc])
    gd.set_time_parameters(dt, T)
    return gd


def _make_elementary_matrices() -> ElementaryMatrices:
    return ElementaryMatrices(orthonormal_basis=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFluxOrdersKellerSegel:
    """Flux order metadata for the 2-equation Keller-Segel system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.problem = _make_ks_problem()
        self.global_disc = _make_global_disc(self.problem)
        self.em = _make_elementary_matrices()
        self.sc = KellerSegelStaticCondensation(
            self.problem, self.global_disc, self.em, ipb=0)

    def test_flux_orders_values(self):
        """Equation 0 (u) has P0 flux, equation 1 (phi) has P1 flux."""
        assert self.sc.flux_orders == [0, 1]

    def test_flux_orders_length(self):
        """Length of flux_orders must equal neq."""
        assert len(self.sc.flux_orders) == self.problem.neq

    def test_flux_dofs_per_element(self):
        """P0 → 1 DOF, P1 → 2 DOFs."""
        assert self.sc.flux_dofs_per_element == [1, 2]

    def test_total_flux_dofs_per_element(self):
        assert self.sc.total_flux_dofs_per_element == 3

    def test_validation_passes(self):
        """Validation should pass for correctly set flux_orders."""
        self.sc._validate_flux_orders()  # should not raise


class TestFluxOrdersOOC:
    """Flux order metadata for the 4-equation Organ-on-Chip system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.problem = _make_ooc_problem()
        self.global_disc = _make_global_disc(self.problem)
        self.em = _make_elementary_matrices()
        self.sc = StaticCondensationOOC(
            self.problem, self.global_disc, self.em, ipb=0)

    def test_flux_orders_values(self):
        """Eq 0 (u) P0, eqs 1-3 (omega, v, phi) P1."""
        assert self.sc.flux_orders == [0, 1, 1, 1]

    def test_flux_orders_length(self):
        assert len(self.sc.flux_orders) == self.problem.neq

    def test_flux_dofs_per_element(self):
        assert self.sc.flux_dofs_per_element == [1, 2, 2, 2]

    def test_total_flux_dofs_per_element(self):
        assert self.sc.total_flux_dofs_per_element == 7


class TestFluxOrdersFactory:
    """Ensure flux_orders is set when SC is created through the factory."""

    def test_factory_ks(self):
        problem = _make_ks_problem()
        gd = _make_global_disc(problem)
        em = _make_elementary_matrices()
        sc = StaticCondensationFactory.create(problem, gd, em, 0)
        assert sc.flux_orders == [0, 1]

    def test_factory_ooc(self):
        problem = _make_ooc_problem()
        gd = _make_global_disc(problem)
        em = _make_elementary_matrices()
        sc = StaticCondensationFactory.create(problem, gd, em, 0)
        assert sc.flux_orders == [0, 1, 1, 1]


class TestFluxOrdersValidation:
    """Validation catches misconfigured flux_orders."""

    def _make_sc(self):
        problem = _make_ks_problem()
        gd = _make_global_disc(problem)
        em = _make_elementary_matrices()
        return KellerSegelStaticCondensation(problem, gd, em, ipb=0)

    def test_wrong_length_raises(self):
        sc = self._make_sc()
        sc.flux_orders = [0]  # too short
        with pytest.raises(ValueError, match="flux_orders has length 1"):
            sc._validate_flux_orders()

    def test_invalid_value_raises(self):
        sc = self._make_sc()
        sc.flux_orders = [0, 2]  # 2 is not a valid order
        with pytest.raises(ValueError, match="expected 0.*or 1"):
            sc._validate_flux_orders()

    def test_empty_raises(self):
        sc = self._make_sc()
        sc.flux_orders = []
        with pytest.raises(ValueError, match="flux_orders has length 0"):
            sc._validate_flux_orders()
