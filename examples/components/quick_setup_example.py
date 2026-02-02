import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from setup_solver import quick_setup, SolverSetup
import numpy as np

def complete_solver_example():
    """Complete example of solver setup and Newton iteration."""
    
    # Step 1: Initialize solver with validation
    print("Setting up solver...")
    setup = quick_setup("bionetflux.problems.ooc_test_problem", validate=True)
    
    # Step 2: Get problem information
    info = setup.get_problem_info()
    print(f"\nProblem: {info['problem_name']}")
    print(f"Domains: {info['num_domains']}")
    print(f"Total DOFs: {info['total_trace_dofs'] + info['num_constraints']}")
    
    # Step 3: Create initial conditions
    trace_solutions, multipliers = setup.create_initial_conditions()
    global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
    
    print(f"\nInitial conditions created:")
    print(f"  Global solution shape: {global_solution.shape}")
    print(f"  Initial residual norm: {np.linalg.norm(global_solution):.6e}")
    
    # Step 4: Setup Newton iteration
    assembler = setup.global_assembler
    bulk_manager = setup.bulk_data_manager
    static_condensations = setup.static_condensations
    
    # Step 5: Newton iteration loop
    tolerance = 1e-10
    max_iterations = 20
    current_time = 0.0
    dt = setup.global_discretization.dt
    
    print(f"\nStarting Newton iterations (tol={tolerance:.0e})...")
    
    for iteration in range(max_iterations):
        # Create bulk data for forcing terms
        bulk_data_list = bulk_manager.initialize_all_bulk_data(
            setup.problems,
            setup.global_discretization.spatial_discretizations,
            time=current_time
        )
        
        # Compute forcing terms
        forcing_terms = bulk_manager.compute_forcing_terms(
            bulk_data_list, setup.problems,
            setup.global_discretization.spatial_discretizations,
            current_time, dt
        )
        
        # Assemble system
        residual, jacobian = assembler.assemble_residual_and_jacobian(
            global_solution=global_solution,
            forcing_terms=forcing_terms,
            static_condensations=static_condensations,
            time=current_time
        )
        
        # Check convergence
        residual_norm = np.linalg.norm(residual)
        print(f"  Iteration {iteration}: ||R|| = {residual_norm:.6e}")
        
        if residual_norm < tolerance:
            print("  ✓ Newton solver converged")
            break
        
        # Newton update
        try:
            delta = np.linalg.solve(jacobian, -residual)
            global_solution += delta
        except np.linalg.LinAlgError:
            print("  ✗ Newton solver failed: singular Jacobian")
            break
    
    # Step 6: Extract final solutions
    final_traces, final_multipliers = setup.extract_domain_solutions(global_solution)
    
    print(f"\nSolver completed:")
    print(f"  Final residual norm: {residual_norm:.6e}")
    print(f"  Domain solutions: {len(final_traces)}")
    print(f"  Constraint multipliers: {len(final_multipliers)}")
    
    return setup, global_solution, final_traces, final_multipliers

# Usage
setup, solution, traces, multipliers = complete_solver_example()