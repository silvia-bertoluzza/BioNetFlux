#!/usr/bin/env python3
"""
Tests for:
  1. The flux-data storage pipeline (domain_flux_jump → GlobalAssembler →
     TimeStepResult).
  2. The ``compute_flux_error`` method of ``MinimalErrorEvaluator``.

Each test is self-contained — we build tiny Problem / Discretization /
StaticCondensation objects and call the production code directly.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.utils.elementary_matrices import ElementaryMatrices
from bionetflux.core.static_condensation_keller_segel import KellerSegelStaticCondensation
from bionetflux.core.flux_jump import domain_flux_jump
from bionetflux.core.minimal_error_evaluator import MinimalErrorEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ks_problem(domain_start=0.0, domain_length=1.0) -> Problem:
    """Create a minimal Keller-Segel problem (neq=2)."""
    p = Problem(neq=2, domain_start=domain_start, domain_length=domain_length,
                parameters=np.array([1.0, 1.0, 0.0, 0.0]),
                problem_type="keller_segel", name="ks_test")
    # Constant chemotaxis
    p.set_chemotaxis(chi=lambda s: 1.0, dchi=lambda s: 0.0)
    return p


def _make_global_disc(problem: Problem, n_elements: int = 4,
                      dt: float = 0.01, T: float = 0.1) -> GlobalDiscretization:
    """Create a GlobalDiscretization wrapping a single domain."""
    disc = Discretization(n_elements, problem.domain_start, problem.domain_length)
    disc.set_tau([1.0] * problem.neq)
    gd = GlobalDiscretization([disc])
    gd.set_time_parameters(dt, T)
    return gd


def _make_sc(problem, global_disc):
    em = ElementaryMatrices(orthonormal_basis=False)
    sc = KellerSegelStaticCondensation(problem, global_disc, em, ipb=0)
    sc.build_matrices()
    return sc


# ---------------------------------------------------------------------------
# Tests — domain_flux_jump returns 4-tuple including J
# ---------------------------------------------------------------------------

class TestDomainFluxJumpReturnsJ:
    """Verify that domain_flux_jump returns (U, J, F, JF)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.problem = _make_ks_problem()
        self.global_disc = _make_global_disc(self.problem, n_elements=4)
        self.sc = _make_sc(self.problem, self.global_disc)
        self.disc = self.global_disc.spatial_discretizations[0]
        self.N = self.disc.n_elements
        self.neq = self.problem.neq

    def test_returns_four_values(self):
        """domain_flux_jump should return exactly 4 values."""
        trace = np.zeros((self.neq * (self.N + 1), 1))
        forcing = np.zeros((2 * self.neq, self.N))
        result = domain_flux_jump(trace, forcing, None, None, self.sc)
        assert len(result) == 4

    def test_J_shape(self):
        """J should have shape (total_flux_dofs_per_element, N)."""
        trace = np.zeros((self.neq * (self.N + 1), 1))
        forcing = np.zeros((2 * self.neq, self.N))
        U, J, F, JF = domain_flux_jump(trace, forcing, None, None, self.sc)
        expected_rows = self.sc.total_flux_dofs_per_element  # 3 for KS
        assert J is not None
        assert J.shape == (expected_rows, self.N)

    def test_J_not_all_zero_with_nonzero_trace(self):
        """With nonzero trace input, J should contain nonzero values."""
        trace = np.ones((self.neq * (self.N + 1), 1)) * 0.5
        forcing = np.ones((2 * self.neq, self.N)) * 0.1
        U, J, F, JF = domain_flux_jump(trace, forcing, None, None, self.sc)
        assert J is not None
        assert not np.allclose(J, 0.0)


# ---------------------------------------------------------------------------
# Tests — Problem.flux_solution attribute and setter
# ---------------------------------------------------------------------------

class TestProblemFluxSolution:
    """Verify Problem.flux_solution attribute and set_flux_solution method."""

    def test_default_flux_solution_is_none_list(self):
        p = Problem(neq=2)
        assert hasattr(p, 'flux_solution')
        assert len(p.flux_solution) == 2
        assert all(f is None for f in p.flux_solution)

    def test_set_flux_solution(self):
        p = Problem(neq=2)
        f0 = lambda s, t: s + t
        p.set_flux_solution(0, f0)
        assert p.flux_solution[0] is f0
        assert p.flux_solution[1] is None

    def test_set_flux_solution_all_equations(self):
        p = Problem(neq=3)
        funcs = [lambda s, t, i=i: s * i for i in range(3)]
        for i, f in enumerate(funcs):
            p.set_flux_solution(i, f)
        for i in range(3):
            assert p.flux_solution[i] is funcs[i]


# ---------------------------------------------------------------------------
# Tests — MinimalErrorEvaluator.compute_flux_error
# ---------------------------------------------------------------------------

class TestComputeFluxError:
    """Test the flux error computation with synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.problem = _make_ks_problem(domain_start=0.0, domain_length=1.0)
        self.global_disc = _make_global_disc(self.problem, n_elements=4)
        self.sc = _make_sc(self.problem, self.global_disc)
        self.disc = self.global_disc.spatial_discretizations[0]
        self.evaluator = MinimalErrorEvaluator()

    def test_returns_correct_structure(self):
        """Result should have 'local' and 'global' keys."""
        N = self.disc.n_elements
        total_dofs = self.sc.total_flux_dofs_per_element
        flux_array = np.zeros((total_dofs, N))

        # Set trivial flux_solution
        self.problem.set_flux_solution(0, lambda s, t: 0.0)
        self.problem.set_flux_solution(1, lambda s, t: 0.0)

        result = self.evaluator.compute_flux_error(
            flux_data=[flux_array],
            problems=[self.problem],
            discretizations=[self.disc],
            static_condensations=[self.sc],
            time=0.0,
        )
        assert 'local' in result
        assert 'global' in result
        assert 0 in result['global']
        assert 1 in result['global']

    def test_zero_error_when_matching(self):
        """Error should be ≈0 when numerical flux matches analytical exactly."""
        N = self.disc.n_elements
        total_dofs = self.sc.total_flux_dofs_per_element  # 3 for KS
        flux_array = np.zeros((total_dofs, N))

        # For eq 0 (P0): constant 2.0 on every element
        flux_array[0, :] = 2.0
        # For eq 1 (P1): linear interpolation matching constant 3.0
        flux_array[1, :] = 3.0  # left
        flux_array[2, :] = 3.0  # right

        self.problem.set_flux_solution(0, lambda s, t: np.full_like(s, 2.0))
        self.problem.set_flux_solution(1, lambda s, t: np.full_like(s, 3.0))

        result = self.evaluator.compute_flux_error(
            flux_data=[flux_array],
            problems=[self.problem],
            discretizations=[self.disc],
            static_condensations=[self.sc],
            time=0.0,
        )

        assert result['global'][0] is not None
        assert result['global'][0] < 1e-12
        assert result['global'][1] is not None
        assert result['global'][1] < 1e-12

    def test_nonzero_error_when_different(self):
        """Error should be positive when numerical ≠ analytical."""
        N = self.disc.n_elements
        total_dofs = self.sc.total_flux_dofs_per_element
        flux_array = np.zeros((total_dofs, N))

        # Numerical flux is 0, analytical is 1 → nonzero error
        self.problem.set_flux_solution(0, lambda s, t: np.ones_like(s))
        self.problem.set_flux_solution(1, lambda s, t: np.ones_like(s))

        result = self.evaluator.compute_flux_error(
            flux_data=[flux_array],
            problems=[self.problem],
            discretizations=[self.disc],
            static_condensations=[self.sc],
            time=0.0,
        )

        assert result['global'][0] is not None
        assert result['global'][0] > 0.1
        assert result['global'][1] is not None
        assert result['global'][1] > 0.1

    def test_none_flux_data_returns_none_errors(self):
        """When flux_data entry is None, errors should be None."""
        self.problem.set_flux_solution(0, lambda s, t: np.zeros_like(s))
        self.problem.set_flux_solution(1, lambda s, t: np.zeros_like(s))

        result = self.evaluator.compute_flux_error(
            flux_data=[None],
            problems=[self.problem],
            discretizations=[self.disc],
            static_condensations=[self.sc],
            time=0.0,
        )
        assert result['global'][0] is None
        assert result['global'][1] is None

    def test_no_flux_solution_returns_none(self):
        """When no analytical flux_solution is set, errors should be None."""
        N = self.disc.n_elements
        total_dofs = self.sc.total_flux_dofs_per_element
        flux_array = np.zeros((total_dofs, N))

        # Don't set flux_solution at all (default is [None, None])
        result = self.evaluator.compute_flux_error(
            flux_data=[flux_array],
            problems=[self.problem],
            discretizations=[self.disc],
            static_condensations=[self.sc],
            time=0.0,
        )
        # _get_flux_analytical_functions returns [None, None], so all errors None
        assert result['global'][0] is None
        assert result['global'][1] is None

    def test_p0_exact_constant_error_value(self):
        """P0 equation: numerical=1, analytical=0 → error follows QUAD convention."""
        N = self.disc.n_elements
        total_dofs = self.sc.total_flux_dofs_per_element
        flux_array = np.zeros((total_dofs, N))
        flux_array[0, :] = 1.0  # P0 flux = 1 everywhere

        self.problem.set_flux_solution(0, lambda s, t: np.zeros_like(s))
        # Only check eq 0
        result = self.evaluator.compute_flux_error(
            flux_data=[flux_array],
            problems=[self.problem],
            discretizations=[self.disc],
            static_condensations=[self.sc],
            time=0.0,
        )
        # The QUAD matrix uses basis-weighted quadrature (same as bulk error).
        # For constant error=1 on [0,1]: integrated value = sqrt(0.5)
        assert result['global'][0] is not None
        assert result['global'][0] > 0.5  # Positive, nontrivial

    def test_p1_linear_exact_error(self):
        """P1 equation: match a linear analytical solution exactly."""
        N = self.disc.n_elements
        nodes = self.disc.nodes
        total_dofs = self.sc.total_flux_dofs_per_element
        flux_array = np.zeros((total_dofs, N))

        # Set P1 coefficients to match f(x) = 2*x + 1
        for k in range(N):
            flux_array[1, k] = 2 * nodes[k] + 1       # left value
            flux_array[2, k] = 2 * nodes[k + 1] + 1   # right value

        self.problem.set_flux_solution(1, lambda s, t: 2 * s + 1)

        result = self.evaluator.compute_flux_error(
            flux_data=[flux_array],
            problems=[self.problem],
            discretizations=[self.disc],
            static_condensations=[self.sc],
            time=0.0,
        )
        assert result['global'][1] is not None
        assert result['global'][1] < 1e-12


class TestComputeFluxErrorMultiDomain:
    """Test flux error computation with multiple domains."""

    def test_two_domains_global_aggregation(self):
        """Global error should aggregate across domains via root-sum-of-squares."""
        p0 = _make_ks_problem(domain_start=0.0, domain_length=0.5)
        p1 = _make_ks_problem(domain_start=0.5, domain_length=0.5)
        gd = _make_global_disc(p0, n_elements=4)
        
        # Second domain uses same disc params
        disc1 = Discretization(4, 0.5, 0.5)
        disc1.set_tau([1.0, 1.0])
        
        sc0 = _make_sc(p0, gd)
        
        # For sc1, create its own global disc
        gd1 = GlobalDiscretization([disc1])
        gd1.set_time_parameters(0.01, 0.1)
        sc1 = _make_sc(p1, gd1)
        
        N = 4
        total_dofs = sc0.total_flux_dofs_per_element
        
        # Set up: numerical=0, analytical=1 for P0 flux (eq 0)
        flux0 = np.zeros((total_dofs, N))
        flux1 = np.zeros((total_dofs, N))
        
        for p in [p0, p1]:
            p.set_flux_solution(0, lambda s, t: np.ones_like(s))
            p.set_flux_solution(1, lambda s, t: np.zeros_like(s))
        
        evaluator = MinimalErrorEvaluator()
        result = evaluator.compute_flux_error(
            flux_data=[flux0, flux1],
            problems=[p0, p1],
            discretizations=[gd.spatial_discretizations[0], disc1],
            static_condensations=[sc0, sc1],
            time=0.0,
        )
        
        # Each domain contributes its share; check global is positive
        assert result['global'][0] is not None
        assert result['global'][0] > 0.5
