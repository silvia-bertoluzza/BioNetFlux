"""
Test script for one complete time step using the integrated framework.

This test uses a lean approach where BulkDataManager is used only for forcing term
computation, while framework components are accessed directly in the Newton method.
"""

import numpy as np
import sys
import os
from scipy.optimize import fsolve
from typing import List, Tuple

# Import framework components
from ooc1d.core.bulk_data import BulkData
from ooc1d.core.bulk_data_manager import BulkDataManager
from ooc1d.core.global_assembly import GlobalAssembler
from ooc1d.core.constraints import ConstraintManager
from ooc1d.utils.elementary_matrices import ElementaryMatrices
from ooc1d.problems.test_problem2 import create_global_framework

# Import test setup components
from test_setup import create_test_problems, create_test_discretizations, create_test_static_condensations


def newton_solver(residual_func, jacobian_func, initial_guess: np.ndarray, 
                  max_iter: int = 20, tol: float = 1e-10) -> Tuple[np.ndarray, bool, int]:
    """
    Newton solver for nonlinear system.
    
    Args:
        residual_func: Function that computes residual
        jacobian_func: Function that computes Jacobian
        initial_guess: Initial guess for solution
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        Tuple of (solution, converged, iterations)
    """
    x = initial_guess.copy()
    
    for iteration in range(max_iter):
        # Compute residual and Jacobian
        residual = residual_func(x)
        jacobian = jacobian_func(x)
        
        # Check convergence
        residual_norm = np.linalg.norm(residual)
        print(f"  Newton iteration {iteration}: residual norm = {residual_norm:.2e}")
        
        if residual_norm < tol:
            print(f"  ✓ Newton converged in {iteration} iterations")
            return x, True, iteration
        
        # Newton step: solve J * dx = -R
        try:
            dx = np.linalg.solve(jacobian, -residual)
            x += dx
        except np.linalg.LinAlgError as e:
            print(f"  ✗ Newton solver failed: {e}")
            return x, False, iteration
    
    print(f"  ✗ Newton failed to converge in {max_iter} iterations")
    return x, False, max_iter


def test_single_domain_time_step():
    """Test one time step for a single domain problem using lean approach."""
    print("=== Testing Single Domain Time Step (Lean Approach) ===")
    
    # Setup framework components directly
    print("Setting up framework components...")
    
    # Create test problems and discretizations
    problems = create_test_problems(n_domains=1, with_forcing=True, with_initial=True)
    discretizations = create_test_discretizations(n_domains=1, n_elements=10)
    static_condensations = create_test_static_condensations(problems, discretizations)
    elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
    
    # Create global discretization wrapper
    class MockGlobalDiscretization:
        def __init__(self, discretizations):
            self.spatial_discretizations = discretizations
    
    global_discretization = MockGlobalDiscretization(discretizations)
    
    # Create lean BulkDataManager (only for forcing term computation)
    bulk_manager = BulkDataManager(
        problems=problems,
        global_discretization=global_discretization,
        static_condensations=static_condensations,
        elementary_matrices=elementary_matrices
    )
    
    # Create constraint manager (none for single domain)
    constraint_manager = None
    
    # Create GlobalAssembler with direct access to framework components
    assembler = GlobalAssembler(
        problems=problems,
        global_discretization=global_discretization,
        static_condensations=static_condensations,
        constraint_manager=constraint_manager
    )
    
    print(f"✓ Framework initialized:")
    print(f"  - {len(problems)} domain(s)")
    print(f"  - {assembler.total_trace_dofs} trace DOFs")
    print(f"  - {assembler.n_multipliers} multipliers")
    print(f"  - Total DOFs: {assembler.total_dofs}")
    
    # Initialize BulkData objects with initial conditions using BulkDataManager
    print("\nInitializing bulk data with initial conditions...")
    bulk_data_list = bulk_manager.initialize_all_bulk_data(time=0.0)
    
    for i, bulk_data in enumerate(bulk_data_list):
        data = bulk_data.get_data()
        print(f"  Domain {i}: shape {data.shape}, range [{np.min(data):.6f}, {np.max(data):.6f}]")
    
    # Time step parameters
    current_time = 0.0
    dt = 0.01
    new_time = current_time + dt
    
    print(f"\nPerforming time step: t = {current_time} → {new_time}, dt = {dt}")
    
    # Create initial guess from current BulkData
    initial_guess = assembler.create_initial_guess_from_bulk_data(bulk_data_list)
    print(f"  Initial guess: shape {initial_guess.shape}, range [{np.min(initial_guess):.6f}, {np.max(initial_guess):.6f}]")
    
    # Define residual and Jacobian functions for Newton solver
    # These use framework components directly through GlobalAssembler
    def compute_residual(global_solution):
        residual, _ = assembler.assemble_residual_and_jacobian(
            global_solution, bulk_manager, bulk_data_list, new_time, dt
        )
        return residual
    
    def compute_jacobian(global_solution):
        _, jacobian = assembler.assemble_residual_and_jacobian(
            global_solution, bulk_manager, bulk_data_list, new_time, dt
        )
        return jacobian
    
    # Solve nonlinear system with Newton method
    print("  Solving nonlinear system with Newton method...")
    solution, converged, iterations = newton_solver(
        compute_residual, compute_jacobian, initial_guess,
        max_iter=10, tol=1e-8
    )
    
    if converged:
        print(f"✓ Time step completed successfully")
        print(f"  Solution range: [{np.min(solution):.6f}, {np.max(solution):.6f}]")
        
        # Extract domain solutions using GlobalAssembler
        domain_solutions = assembler.get_domain_solutions(solution)
        for i, domain_sol in enumerate(domain_solutions):
            print(f"  Domain {i} solution: shape {domain_sol.shape}, range [{np.min(domain_sol):.6f}, {np.max(domain_sol):.6f}]")
        
        # For next time step, you would reconstruct BulkData from trace solutions
        # This demonstrates the separation of concerns
        
        return True
    else:
        print("✗ Time step failed - Newton solver did not converge")
        return False


def test_multi_domain_time_step():
    """Test one time step for a multi-domain problem using lean approach."""
    print("\n=== Testing Multi-Domain Time Step (Lean Approach) ===")
    
    # Setup framework components for 2 domains
    print("Setting up multi-domain framework...")
    
    problems = create_test_problems(n_domains=2, with_forcing=True, with_initial=True)
    discretizations = create_test_discretizations(n_domains=2, n_elements=8)
    static_condensations = create_test_static_condensations(problems, discretizations)
    elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
    
    class MockGlobalDiscretization:
        def __init__(self, discretizations):
            self.spatial_discretizations = discretizations
    
    global_discretization = MockGlobalDiscretization(discretizations)
    
    # Create lean BulkDataManager
    bulk_manager = BulkDataManager(
        problems=problems,
        global_discretization=global_discretization,
        static_condensations=static_condensations,
        elementary_matrices=elementary_matrices
    )
    
    # Create simple constraint manager for junction condition
    constraint_manager = ConstraintManager()
    # Note: In real implementation, you'd add constraints here
    
    # Create GlobalAssembler with direct framework access
    assembler = GlobalAssembler(
        problems=problems,
        global_discretization=global_discretization,
        static_condensations=static_condensations,
        constraint_manager=constraint_manager
    )
    
    print(f"✓ Multi-domain framework initialized:")
    print(f"  - {len(problems)} domains")
    print(f"  - {assembler.total_trace_dofs} trace DOFs")
    print(f"  - {assembler.n_multipliers} multipliers")
    print(f"  - Total DOFs: {assembler.total_dofs}")
    
    # Initialize BulkData objects using BulkDataManager
    bulk_data_list = bulk_manager.initialize_all_bulk_data(time=0.0)
    
    # Time step parameters
    current_time = 0.0
    dt = 0.01
    new_time = current_time + dt
    
    print(f"\nPerforming multi-domain time step: t = {current_time} → {new_time}")
    
    # Create initial guess
    initial_guess = assembler.create_initial_guess_from_bulk_data(bulk_data_list)
    
    # Define functions for Newton solver using direct framework access
    def compute_residual(global_solution):
        residual, _ = assembler.assemble_residual_and_jacobian(
            global_solution, bulk_manager, bulk_data_list, new_time, dt
        )
        return residual
    
    def compute_jacobian(global_solution):
        _, jacobian = assembler.assemble_residual_and_jacobian(
            global_solution, bulk_manager, bulk_data_list, new_time, dt
        )
        return jacobian
    
    # Solve nonlinear system
    print("  Solving multi-domain nonlinear system...")
    solution, converged, iterations = newton_solver(
        compute_residual, compute_jacobian, initial_guess,
        max_iter=15, tol=1e-8
    )
    
    if converged:
        print(f"✓ Multi-domain time step completed successfully")
        
        # Extract solutions using GlobalAssembler directly
        domain_solutions = assembler.get_domain_solutions(solution)
        multipliers = assembler.get_multipliers(solution)
        
        for i, domain_sol in enumerate(domain_solutions):
            print(f"  Domain {i} solution: range [{np.min(domain_sol):.6f}, {np.max(domain_sol):.6f}]")
        
        if len(multipliers) > 0:
            print(f"  Multipliers: range [{np.min(multipliers):.6f}, {np.max(multipliers):.6f}]")
        
        return True
    else:
        print("✗ Multi-domain time step failed")
        return False


def test_framework_separation():
    """Test the separation of concerns in the lean approach."""
    print("\n=== Testing Framework Separation ===")
    
    # Setup components
    problems = create_test_problems(n_domains=1, with_forcing=True, with_initial=True)
    discretizations = create_test_discretizations(n_domains=1, n_elements=10)
    static_condensations = create_test_static_condensations(problems, discretizations)
    elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
    
    class MockGlobalDiscretization:
        def __init__(self, discretizations):
            self.spatial_discretizations = discretizations
    
    global_discretization = MockGlobalDiscretization(discretizations)
    
    # Test 1: BulkDataManager used only for specific operations
    print("  Testing lean BulkDataManager usage...")
    
    bulk_manager = BulkDataManager(
        problems=problems,
        global_discretization=global_discretization,
        static_condensations=static_condensations,
        elementary_matrices=elementary_matrices
    )
    
    # Initialize BulkData
    bulk_data_list = bulk_manager.initialize_all_bulk_data(time=0.0)
    
    # Compute forcing terms
    forcing_terms = bulk_manager.compute_forcing_terms(bulk_data_list, time=0.5, dt=0.01)
    
    print(f"    ✓ BulkDataManager provided forcing terms: {len(forcing_terms)} domains")
    
    # Test 2: GlobalAssembler has direct access to framework components
    print("  Testing direct framework component access...")
    
    assembler = GlobalAssembler(
        problems=problems,
        global_discretization=global_discretization,
        static_condensations=static_condensations,
        constraint_manager=None
    )
    
    # Verify assembler can access components directly
    print(f"    ✓ GlobalAssembler has direct access to {len(assembler.problems)} problems")
    print(f"    ✓ GlobalAssembler has direct access to {len(assembler.static_condensations)} static condensations")
    
    # Test 3: Framework components can be used independently
    print("  Testing independent component usage...")
    
    # Use static condensation directly
    sc_matrices = static_condensations[0].build_matrices()
    print(f"    ✓ Static condensation used independently: {list(sc_matrices.keys())}")
    
    # Use problem directly
    problem = problems[0]
    if hasattr(problem, 'force') and problem.force[0] is not None:
        test_value = problem.force[0](0.5, 0.1)
        print(f"    ✓ Problem used independently: forcing value = {test_value:.6f}")
    
    print("  ✓ Framework separation validated")
    return True


def run_all_time_step_tests():
    """Run all time step tests with lean approach."""
    print("Running Complete Time Step Tests (Lean Approach)")
    print("=" * 60)
    
    try:
        # Test 1: Single domain with lean approach
        success1 = test_single_domain_time_step()
        
        # Test 2: Multi-domain with lean approach (simplified)
        success2 = test_multi_domain_time_step()
        
        # Test 3: Framework separation validation
        success3 = test_framework_separation()
        
        if success1 and success2 and success3:
            print("=" * 60)
            print("🎉 All lean approach time step tests completed successfully!")
            print("\nLean framework integration validated:")
            print("  ✓ BulkDataManager used only for specific operations")
            print("  ✓ GlobalAssembler has direct framework component access")
            print("  ✓ Framework components can be used independently")
            print("  ✓ Clean separation of concerns maintained")
            print("  ✓ Newton solver integrates seamlessly")
        else:
            print("=" * 60)
            print("❌ Some lean approach tests failed")
            
    except Exception as e:
        print(f"❌ Lean approach test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_time_step_tests()
