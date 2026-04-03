#!/usr/bin/env python3
"""
Pytest-compatible test script for static condensation setup and matrices.
Tests the complete initialization pipeline before entering the solution process.

Usage:
    pytest test_static_condensation_setup.py
    pytest test_static_condensation_setup.py -v  # verbose output
    pytest test_static_condensation_setup.py -s  # show print statements
    pytest test_static_condensation_setup.py::TestStaticCondensationSetup::test_problem_creation  # specific test
"""

import sys
import os
import pytest
import numpy as np

# sys.path hack — commented out, use pip install -e . instead
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.utils.elementary_matrices import ElementaryMatrices
from bionetflux.core.static_condensation_factory import StaticCondensationFactory
from bionetflux.core.constraints import ConstraintManager
from bionetflux.problems.test_problem import create_global_framework


class TestStaticCondensationSetup:
    """Test class for static condensation setup process."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test."""
        self.problems = None
        self.global_disc = None
        self.constraint_manager = None
        self.problem_name = None
        self.elementary_matrices = None
        self.sc_implementations = []

    def test_problem_creation(self):
        """Test creation of the problem framework."""
        # Create problem using the problems module
        self.problems, self.global_disc, self.constraint_manager, self.problem_name = create_global_framework()
        
        assert self.problems is not None, "Problems list should not be None"
        assert len(self.problems) > 0, "Should have at least one domain"
        assert self.global_disc is not None, "Global discretization should not be None"
        assert self.problem_name is not None, "Problem name should not be None"
        
        discretizations = self.global_disc.spatial_discretizations
        assert len(discretizations) == len(self.problems), "Number of discretizations should match number of problems"
        
        # Verify each domain
        for i, (problem, discretization) in enumerate(zip(self.problems, discretizations)):
            assert hasattr(problem, 'domain_start'), f"Domain {i} should have domain_start"
            assert hasattr(problem, 'domain_end'), f"Domain {i} should have domain_end"
            assert hasattr(problem, 'neq'), f"Domain {i} should have neq"
            assert hasattr(problem, 'type'), f"Domain {i} should have type"
            assert discretization.n_elements > 0, f"Domain {i} should have positive number of elements"

    def test_constraint_manager_setup(self):
        """Test constraint manager setup and functionality."""
        if not hasattr(self, 'problems') or self.problems is None:
            self.test_problem_creation()
        
        if self.constraint_manager is not None:
            assert hasattr(self.constraint_manager, 'n_constraints'), "Should have n_constraints attribute"
            assert hasattr(self.constraint_manager, 'n_multipliers'), "Should have n_multipliers attribute"
            
            # Test constraint residual evaluation if there are constraints
            if self.constraint_manager.n_constraints > 0:
                discretizations = self.global_disc.spatial_discretizations
                
                # Create test trace solutions for all domains
                test_trace_solutions = []
                for i, (problem, discretization) in enumerate(zip(self.problems, discretizations)):
                    n_nodes = discretization.n_elements + 1
                    trace_size = problem.neq * n_nodes
                    test_trace = np.ones((trace_size,)) * (i + 1) * 0.1
                    test_trace_solutions.append(test_trace)
                
                # Create test multiplier values
                n_multipliers = self.constraint_manager.n_multipliers
                test_multipliers = np.random.rand(n_multipliers) * 0.01
                test_time = 0.5
                
                # Test constraint residual computation
                residuals = self.constraint_manager.compute_constraint_residuals(
                    test_trace_solutions, test_multipliers, test_time, discretizations
                )
                
                assert residuals is not None, "Residuals should not be None"
                assert len(residuals) == n_multipliers, "Residual vector length should match number of multipliers"
                assert isinstance(residuals, np.ndarray), "Residuals should be numpy array"

    def test_elementary_matrices_creation(self):
        """Test creation of elementary matrices."""
        self.elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
        
        assert self.elementary_matrices is not None, "Elementary matrices should be created"
        
        # Test that we can get matrices
        all_matrices = self.elementary_matrices.get_all_matrices()
        assert isinstance(all_matrices, dict), "Should return a dictionary of matrices"
        assert len(all_matrices) > 0, "Should have at least some matrices"
        
        # Test specific matrix access
        mass_matrix = self.elementary_matrices.get_matrix('M')
        assert mass_matrix is not None, "Mass matrix should exist"
        assert isinstance(mass_matrix, np.ndarray), "Mass matrix should be numpy array"

    def test_static_condensation_factory(self):
        """Test static condensation factory creation for all domains."""
        if not hasattr(self, 'problems') or self.problems is None:
            self.test_problem_creation()
        
        if self.elementary_matrices is None:
            self.test_elementary_matrices_creation()
        
        self.sc_implementations = []
        
        for i, (problem, discretization) in enumerate(zip(self.problems, self.global_disc.spatial_discretizations)):
            # Create static condensation implementation
            sc = StaticCondensationFactory.create(problem, self.global_disc, self.elementary_matrices, i)
            
            assert sc is not None, f"Static condensation should be created for domain {i}"
            assert hasattr(sc, 'build_matrices'), f"Domain {i} SC should have build_matrices method"
            assert hasattr(sc, 'static_condensation'), f"Domain {i} SC should have static_condensation method"
            
            self.sc_implementations.append(sc)

    def test_static_condensation_matrices(self):
        """Test building of static condensation matrices."""
        if not self.sc_implementations:
            self.test_static_condensation_factory()
        
        for i, sc in enumerate(self.sc_implementations):
            # Build matrices
            sc_matrices = sc.build_matrices()
            
            assert sc_matrices is not None, f"SC matrices should be built for domain {i}"
            assert isinstance(sc_matrices, dict), f"SC matrices should be a dictionary for domain {i}"
            assert len(sc_matrices) > 0, f"Should have at least some SC matrices for domain {i}"
            
            # Check that matrices have appropriate shapes and types
            for matrix_name, matrix in sc_matrices.items():
                if matrix is not None:
                    assert isinstance(matrix, np.ndarray), f"Matrix {matrix_name} should be numpy array"
                    assert matrix.shape[0] > 0 and matrix.shape[1] > 0, f"Matrix {matrix_name} should have positive dimensions"

    def test_trace_solution_initialization(self):
        """Test initialization of trace solutions."""
        if not hasattr(self, 'problems') or self.problems is None:
            self.test_problem_creation()
        
        trace_solutions = []
        total_trace_dofs = 0
        
        for i, (problem, discretization) in enumerate(zip(self.problems, self.global_disc.spatial_discretizations)):
            n_nodes = discretization.n_elements + 1
            trace_size = problem.neq * n_nodes
            trace_solution = np.zeros((trace_size, 1))
            trace_solutions.append(trace_solution)
            
            assert trace_solution.shape == (trace_size, 1), f"Trace solution shape should be ({trace_size}, 1) for domain {i}"
            assert trace_solution.dtype == np.float64, f"Trace solution should be float64 for domain {i}"
            
            total_trace_dofs += trace_size
        
        assert len(trace_solutions) == len(self.problems), "Should have trace solution for each domain"
        assert total_trace_dofs > 0, "Total trace DOFs should be positive"

    @pytest.mark.slow
    def test_static_condensation_process(self):
        """Test the static condensation process on sample data."""
        if not self.sc_implementations:
            self.test_static_condensation_factory()
        
        # Create test trace solutions
        trace_solutions = []
        for i, (problem, discretization) in enumerate(zip(self.problems, self.global_disc.spatial_discretizations)):
            n_nodes = discretization.n_elements + 1
            trace_size = problem.neq * n_nodes
            test_trace = np.zeros((trace_size, 1))
            test_trace[::2] = 1.0  # Set every other entry to 1 for testing
            trace_solutions.append(test_trace)
        
        # Test static condensation on each domain
        for i, (sc, trace_solution, problem, discretization) in enumerate(zip(
            self.sc_implementations, trace_solutions, self.problems, self.global_disc.spatial_discretizations
        )):
            # Test on first element of each domain
            k = 0  # First element
            
            # Extract local trace for element k
            local_trace = np.zeros((2 * problem.neq, 1))
            
            for eq in range(problem.neq):
                n_nodes = discretization.n_elements + 1
                left_idx = eq * n_nodes + k
                right_idx = eq * n_nodes + (k + 1)
                
                local_trace[eq * 2] = trace_solution[left_idx]
                local_trace[eq * 2 + 1] = trace_solution[right_idx]
            
            try:
                # Ensure matrices are built
                sc_matrices = sc.build_matrices()
                
                # Check if required matrices exist
                required_matrices = ['L1', 'B1', 'L2', 'B2', 'C2']  # Based on MATLAB StaticC.m
                missing_matrices = [mat for mat in required_matrices if mat not in sc_matrices]
                
                if missing_matrices:
                    pytest.skip(f"Domain {i}: Missing required matrices {missing_matrices} - implementation incomplete")
                
                # Apply static condensation
                local_solution, flux, flux_trace, jacobian = sc.static_condensation(local_trace)
                
                # Verify outputs
                assert local_solution is not None, f"Local solution should not be None for domain {i}"
                assert flux is not None, f"Flux should not be None for domain {i}"
                assert flux_trace is not None, f"Flux trace should not be None for domain {i}"
                assert jacobian is not None, f"Jacobian should not be None for domain {i}"
                
                assert isinstance(local_solution, np.ndarray), f"Local solution should be numpy array for domain {i}"
                assert isinstance(flux, np.ndarray), f"Flux should be numpy array for domain {i}"
                assert isinstance(flux_trace, np.ndarray), f"Flux trace should be numpy array for domain {i}"
                assert isinstance(jacobian, np.ndarray), f"Jacobian should be numpy array for domain {i}"
                
                # Check dimensions
                assert local_solution.shape[0] > 0, f"Local solution should have positive size for domain {i}"
                assert flux.shape[0] > 0, f"Flux should have positive size for domain {i}"
                assert flux_trace.shape[0] > 0, f"Flux trace should have positive size for domain {i}"
                assert jacobian.shape[0] > 0 and jacobian.shape[1] > 0, f"Jacobian should have positive dimensions for domain {i}"
            
            except KeyError as e:
                pytest.skip(f"Domain {i}: Static condensation incomplete - missing matrix {e}")
            except NotImplementedError as e:
                pytest.skip(f"Domain {i}: Static condensation not fully implemented - {e}")
            except Exception as e:
                pytest.fail(f"Domain {i}: Unexpected error in static condensation - {e}")

    def test_complete_setup_integration(self):
        """Test the complete setup process integration."""
        # Run all setup steps in sequence
        self.test_problem_creation()
        self.test_constraint_manager_setup()
        self.test_elementary_matrices_creation()
        self.test_static_condensation_factory()
        self.test_static_condensation_matrices()
        self.test_trace_solution_initialization()
        
        # Verify everything is properly set up
        assert self.problems is not None
        assert self.global_disc is not None
        assert self.elementary_matrices is not None
        assert len(self.sc_implementations) == len(self.problems)
        
        # Calculate total DOFs
        total_dofs = sum(
            problem.neq * (discretization.n_elements + 1) 
            for problem, discretization in zip(self.problems, self.global_disc.spatial_discretizations)
        )
        
        assert total_dofs > 0, "Total DOFs should be positive"


# Fixtures for common test setups
@pytest.fixture
def setup_framework():
    """Fixture providing initialized framework components."""
    problems, global_disc, constraint_manager, problem_name = create_global_framework()
    elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
    
    return {
        'problems': problems,
        'global_disc': global_disc,
        'constraint_manager': constraint_manager,
        'problem_name': problem_name,
        'elementary_matrices': elementary_matrices
    }


@pytest.fixture
def static_condensation_setup(setup_framework):
    """Fixture providing static condensation implementations."""
    framework = setup_framework
    problems = framework['problems']
    global_disc = framework['global_disc']
    elementary_matrices = framework['elementary_matrices']
    
    sc_implementations = []
    for i, (problem, discretization) in enumerate(zip(problems, global_disc.spatial_discretizations)):
        sc = StaticCondensationFactory.create(problem, global_disc, elementary_matrices, i)
        sc.build_matrices()  # Pre-build matrices
        sc_implementations.append(sc)
    
    framework['sc_implementations'] = sc_implementations
    return framework


class TestFixtures:
    """Test using fixtures."""

    def test_setup_framework_fixture(self, setup_framework):
        """Test the setup framework fixture."""
        assert 'problems' in setup_framework
        assert 'global_disc' in setup_framework
        assert 'elementary_matrices' in setup_framework
        assert len(setup_framework['problems']) > 0
        assert setup_framework['elementary_matrices'] is not None

    def test_static_condensation_fixture(self, static_condensation_setup):
        """Test the static condensation fixture."""
        framework = static_condensation_setup
        assert 'sc_implementations' in framework
        assert len(framework['sc_implementations']) == len(framework['problems'])
        
        # Test that matrices are pre-built
        for sc in framework['sc_implementations']:
            matrices = sc.build_matrices()
            assert matrices is not None
            assert len(matrices) > 0

    @pytest.mark.parametrize("domain_index", [0])  # Test first domain
    def test_parametrized_domain_condensation(self, static_condensation_setup, domain_index):
        """Test static condensation for specific domain using parametrization."""
        framework = static_condensation_setup
        
        if domain_index >= len(framework['problems']):
            pytest.skip(f"Domain {domain_index} does not exist")
        
        problem = framework['problems'][domain_index]
        discretization = framework['global_disc'].spatial_discretizations[domain_index]
        sc = framework['sc_implementations'][domain_index]
        
        # Create test local trace
        local_trace = np.ones((2 * problem.neq, 1)) * 0.1
        
        # Test static condensation
        local_solution, flux, flux_trace, jacobian = sc.static_condensation(local_trace)
        
        assert local_solution is not None
        assert flux is not None
        assert flux_trace is not None
        assert jacobian is not None


# Performance markers and configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


class TestPerformance:
    """Performance tests for static condensation setup."""

    @pytest.mark.slow
    def test_setup_performance(self):
        """Test the complete setup process performance."""
        import time
        
        def setup_process():
            problems, global_disc, constraint_manager, problem_name = create_global_framework()
            elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
            
            sc_implementations = []
            for i, (problem, discretization) in enumerate(zip(problems, global_disc.spatial_discretizations)):
                sc = StaticCondensationFactory.create(problem, global_disc, elementary_matrices, i)
                sc.build_matrices()
                sc_implementations.append(sc)
            
            return sc_implementations
        
        # Time the setup process
        start_time = time.time()
        result = setup_process()
        elapsed_time = time.time() - start_time
        
        # Performance assertions
        assert len(result) > 0, "Should have created SC implementations"
        assert elapsed_time < 5.0, f"Setup too slow: {elapsed_time:.3f}s > 5.0s"

    @pytest.mark.slow
    def test_matrix_building_performance(self, setup_framework):
        """Test performance of matrix building."""
        problems = setup_framework['problems']
        global_disc = setup_framework['global_disc']
        elementary_matrices = setup_framework['elementary_matrices']
        
        import time
        start_time = time.time()
        
        for i, (problem, discretization) in enumerate(zip(problems, global_disc.spatial_discretizations)):
            sc = StaticCondensationFactory.create(problem, global_disc, elementary_matrices, i)
            sc.build_matrices()
        
        elapsed_time = time.time() - start_time
        
        # Performance assertion - should complete reasonably quickly
        max_time_per_domain = 1.0  # seconds
        expected_max_time = len(problems) * max_time_per_domain
        
        assert elapsed_time < expected_max_time, f"Matrix building too slow: {elapsed_time:.3f}s > {expected_max_time:.3f}s"


def main():
    """Run tests using pytest."""
    print("This file is now pytest-compatible!")
    print("Usage:")
    print("  pytest test_static_condensation_setup.py")
    print("  pytest test_static_condensation_setup.py -v")
    print("  pytest test_static_condensation_setup.py -s  # show prints")
    print("  pytest test_static_condensation_setup.py -m \"not slow\"  # skip slow tests")
    print("  pytest test_static_condensation_setup.py::TestStaticCondensationSetup::test_problem_creation")
    
    # Run with pytest for backwards compatibility
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    main()

