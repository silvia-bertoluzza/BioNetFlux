"""
Minimal smoke tests for core modules that lack dedicated test files.

These are import + construction tests — they verify that modules can be
loaded and basic objects can be instantiated, but they do NOT test
mathematical correctness.
"""

import pytest
import numpy as np

from bionetflux.core.constraints import ConstraintManager, Constraint, ConstraintType
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.time_integration.newton_solver import NewtonSolver, NewtonResult
from bionetflux.time_integration.time_stepper import TimeStepResult


# ---------------------------------------------------------------------------
#  Constraints
# ---------------------------------------------------------------------------
class TestConstraintManagerSmoke:
    """Smoke tests for the ConstraintManager."""

    def test_create_empty_manager(self):
        """ConstraintManager can be instantiated with no arguments."""
        cm = ConstraintManager()
        assert len(cm.constraints) == 0

    def test_add_dirichlet(self):
        """Add a Dirichlet constraint and verify it is stored."""
        cm = ConstraintManager()
        idx = cm.add_dirichlet(equation_index=0, domain_index=0, position=0.0)
        assert len(cm.constraints) == 1
        assert cm.constraints[idx].type == ConstraintType.DIRICHLET

    def test_add_neumann(self):
        """Add a Neumann constraint."""
        cm = ConstraintManager()
        idx = cm.add_neumann(equation_index=0, domain_index=0, position=1.0)
        assert len(cm.constraints) == 1
        assert cm.constraints[idx].type == ConstraintType.NEUMANN

    def test_add_trace_continuity(self):
        """Add a trace-continuity constraint between two domains."""
        cm = ConstraintManager()
        idx = cm.add_trace_continuity(
            equation_index=0,
            domain1_index=0, domain2_index=1,
            position1=1.0, position2=0.0,
        )
        assert len(cm.constraints) == 1


# ---------------------------------------------------------------------------
#  Discretization
# ---------------------------------------------------------------------------
class TestDiscretizationSmoke:
    """Smoke tests for Discretization and GlobalDiscretization."""

    def test_create_discretization(self):
        """Discretization can be created with basic parameters."""
        d = Discretization(n_elements=4, domain_start=0.0, domain_length=1.0)
        assert d.n_elements == 4
        assert d.element_length == pytest.approx(0.25)

    def test_create_global_discretization(self):
        """GlobalDiscretization wraps a list of spatial discretizations."""
        d1 = Discretization(n_elements=4)
        d2 = Discretization(n_elements=8)
        gd = GlobalDiscretization([d1, d2])
        assert gd.n_domains == 2
        assert gd.total_elements == 12

    def test_set_time_parameters(self):
        """Time parameters can be set on GlobalDiscretization."""
        d = Discretization(n_elements=4)
        gd = GlobalDiscretization([d])
        gd.set_time_parameters(dt=0.01, T=1.0)
        assert gd.dt == pytest.approx(0.01)
        assert gd.T == pytest.approx(1.0)


# ---------------------------------------------------------------------------
#  Newton solver
# ---------------------------------------------------------------------------
class TestNewtonSolverSmoke:
    """Smoke tests for the NewtonSolver."""

    def test_create_solver(self):
        """NewtonSolver can be instantiated with defaults."""
        ns = NewtonSolver()
        assert ns.tolerance == pytest.approx(1e-10)
        assert ns.max_iterations == 20

    def test_create_solver_custom(self):
        """NewtonSolver accepts custom tolerance and max_iterations."""
        ns = NewtonSolver(tolerance=1e-8, max_iterations=50, verbose=True)
        assert ns.tolerance == pytest.approx(1e-8)
        assert ns.max_iterations == 50

    def test_newton_result_dataclass(self):
        """NewtonResult dataclass can be instantiated."""
        result = NewtonResult(
            converged=True,
            iterations=5,
            final_solution=np.zeros(10),
            final_residual_norm=1e-12,
            residual_history=[1.0, 0.1, 0.01, 0.001, 1e-12],
            step_norms=[1.0, 0.5, 0.1, 0.01, 0.001],
        )
        assert result.converged is True
        assert result.iterations == 5


# ---------------------------------------------------------------------------
#  Time stepper result
# ---------------------------------------------------------------------------
class TestTimeStepResultSmoke:
    """Smoke tests for TimeStepResult."""

    def test_create_result(self):
        """TimeStepResult dataclass can be instantiated."""
        result = TimeStepResult(
            converged=True,
            iterations=3,
            final_residual_norm=1e-11,
            updated_solution=np.zeros(10),
            updated_bulk_data=[],
            computation_time=0.05,
        )
        assert result.converged is True
        assert result.computation_time == pytest.approx(0.05)
