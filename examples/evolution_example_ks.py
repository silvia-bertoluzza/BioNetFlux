"""
Evolution + Plotting Example using new Time Stepper Module

This example demonstrates the same functionality as evolution+plotting_example.py
but using the new TimeStepper module for cleaner, more maintainable code.

The time advancement logic is replaced with a single TimeStepper class that
encapsulates all the Newton iteration and bulk data management.
"""

import sys
import os

from scipy import setup
# Add the python_port directory to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from setup_solver import quick_setup, SolverSetup
from bionetflux.time_integration import TimeStepper
from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry, build_grid_geometry
from bionetflux.core.minimal_error_evaluator import MinimalErrorEvaluator
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional


def plot_trace_solution(trace_solutions, problems, discretizations, current_time, equation_idx):
    """
    Plot trace solution (nodal values) with analytical solution overlay.
    
    Args:
        trace_solutions: List of trace solutions per domain
        problems: Problem instances
        discretizations: Spatial discretifications
        current_time: Current time value
        equation_idx: Equation index to plot
    """
    plt.figure(figsize=(10, 6))
    
    for domain_idx, (trace_sol, problem, disc) in enumerate(zip(trace_solutions, problems, discretizations)):
        # Get nodes for this domain
        nodes = disc.nodes
        
        # Extract trace values for this equation
        # Trace solution is stacked by equation: [u0_node0, u0_node1, ..., u1_node1, ...]
        n_equations = problem.neq
        n_nodes = len(nodes)
        
        # Extract values for this equation from stacked format
        start_idx = equation_idx * n_nodes
        end_idx = start_idx + n_nodes
        trace_values = trace_sol[start_idx:end_idx]
        
        # Ensure we have the right number of values
        if len(trace_values) != n_nodes:
            print(f"Warning: Domain {domain_idx}, equation {equation_idx}: expected {n_nodes} values, got {len(trace_values)}")
            continue
        
        # Plot numerical solution
        plt.plot(nodes, trace_values, 'bo-', label=f'Domain {domain_idx} - Numerical', 
                markersize=4, linewidth=2)
        
        # Plot analytical solution if available
        if hasattr(problem, 'solution') and equation_idx < len(problem.solution):
            try:
                analytical_values = problem.solution[equation_idx](nodes, current_time)
                plt.plot(nodes, analytical_values, 'r--', 
                        label=f'Domain {domain_idx} - Analytical', linewidth=2)
            except:
                pass  # No analytical solution available
    
    plt.xlabel('x')
    plt.ylabel(f'Trace Solution - Equation {equation_idx}')
    plt.title(f'Trace Solution Comparison - Equation {equation_idx} at t = {current_time:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_bulk_solution(bulk_solutions, problems, discretizations, current_time, equation_idx):
    """
    Plot bulk solution (discontinuous piecewise linear per element) with analytical solution overlay.
    
    Args:
        bulk_solutions: List of BulkData objects per domain
        problems: Problem instances  
        discretizations: Spatial discretifications
        current_time: Current time value
        equation_idx: Equation index to plot
    """
    plt.figure(figsize=(10, 6))
    
    for domain_idx, (bulk_data, problem, disc) in enumerate(zip(bulk_solutions, problems, discretizations)):
        # Get elements and nodes
        elements = disc.elements
        nodes = disc.nodes
        n_equations = problem.neq
        n_elements = len(elements)
        
        # Get bulk solution data from BulkData object
        bulk_sol = bulk_data.get_data()
        
        # Plot each element segment independently (discontinuous)
        for elem_idx, element in enumerate(elements):
            # Get element nodes
            node_indices = element
            elem_nodes = [nodes[node_indices[0]], nodes[node_indices[1]]]
            
            # Extract bulk values for this element and equation
            # Bulk solution format: shape (2*n_equations, n_elements)
            # For each element, we have 2*n_equations values (left and right node for each equation)
            bulk_values = []
            for node_in_elem in range(2):  # 2 nodes per element
                # For bulk solution: row = equation*2 + node_in_element, col = element_index
                bulk_idx = equation_idx * 2 + node_in_elem
                if bulk_idx < bulk_sol.shape[0] and elem_idx < bulk_sol.shape[1]:
                    bulk_values.append(bulk_sol[bulk_idx, elem_idx])
                else:
                    print(f"Warning: Bulk solution index out of bounds for domain {domain_idx}, element {elem_idx}")
                    bulk_values.append(0.0)
            
            # Plot element segment
            if elem_idx == 0 and domain_idx == 0:  # Only add label once
                plt.plot(elem_nodes, bulk_values, 'b-', linewidth=2, 
                        label='Numerical (discontinuous)')
            else:
                plt.plot(elem_nodes, bulk_values, 'b-', linewidth=2)
        
        # Plot analytical solution if available
        if hasattr(problem, 'solution') and equation_idx < len(problem.solution):
            try:
                # Create fine grid for smooth analytical solution
                x_fine = np.linspace(nodes[0], nodes[-1], 200)
                analytical_values = problem.solution[equation_idx](x_fine, current_time)
                if domain_idx == 0:  # Only add label once
                    plt.plot(x_fine, analytical_values, 'r--', 
                            label='Analytical', linewidth=2)
                else:
                    plt.plot(x_fine, analytical_values, 'r--', linewidth=2)
            except:
                pass  # No analytical solution available
    
    plt.xlabel('x')
    plt.ylabel(f'Bulk Solution - Equation {equation_idx}')
    plt.title(f'Bulk Solution Comparison - Equation {equation_idx} at t = {current_time:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def run_evolution_with_time_stepper(config_file: Optional[str] = None):
    """
    Main function demonstrating time evolution with the new TimeStepper module.
    
    Args:
        config_file: Optional TOML configuration file path
    """
    
    # ============================================================================
    # STEP 1: SOLVER SETUP (Enhanced with config file support and error handling)
    # ============================================================================
    
    geometry = build_arc_sequence_geometry(N=1, start=0.5, length=1.0)

    # Try to call quick_setup with error handling for config compatibility    
    try:
        # Use quick_setup with both geometry and config file support
        setup = quick_setup(
            problem_module="bionetflux.problems.ks_problem",
            validate=True,
            config_file=config_file,  # Pass config file
            geometry=geometry         # Pass geometry
        )
    except ValueError as e:
        # Handle configuration compatibility errors gracefully
        if "not compatible with" in str(e) or "problem type" in str(e):
            print(f"Configuration Error: {e}")
            print("Suggestions: Check problem_type in config file matches problem module")
            return None, None, None, None
        else:
            # Re-raise other ValueError types
            raise
    except Exception as e:
        # Handle other setup errors
        print(f"Setup Error: {e}")
        return None, None, None, None

    # Get problem information
    info = setup.get_problem_info()
    
    # ============================================================================
    # STEP 2: TIME STEPPER INITIALIZATION
    # ============================================================================
    
    # Create time stepper with Newton solver configuration
    time_stepper = TimeStepper(setup, verbose=True)
    
    # Initialize solution at t=0
    current_solution, current_bulk_data = time_stepper.initialize_solution()
    
    # Initialize error evaluator
    # error_evaluator = ErrorEvaluator(setup.problems, setup.global_discretization.spatial_discretizations)
    
        
    # ============================================================================
    # STEP 3: TIME EVOLUTION
    # ============================================================================
    
    # Time evolution parameters
    current_time = 0.0
    dt = setup.global_discretization.dt
    T = setup.global_discretization.T  
    max_time_steps = int(T / dt) + 1
    
    # Solution history for analysis
    solution_history = [current_solution.copy()]
    time_history = [current_time]
    
    print(f"Evolution Parameters: t ∈ [0, {T}], dt = {dt}, max steps = {max_time_steps}")
    
    # TIME EVOLUTION LOOP
    time_step = 0
    
    while current_time < T - dt/2 and time_step < max_time_steps:
        time_step += 1
        
        # SINGLE CALL REPLACES ~50 LINES OF COMPLEX NEWTON ITERATION CODE!
        result = time_stepper.advance_time_step(
            current_solution=current_solution,
            current_bulk_data=current_bulk_data,
            current_time=current_time,
            dt=dt
        )
        
        current_time += dt

        # Handle result
        if result.converged:
            # Update state for next iteration
            old_bulk_data = current_bulk_data
            current_solution = result.updated_solution
            current_bulk_data = result.updated_bulk_data
            
            # Store history
            solution_history.append(current_solution.copy())
            time_history.append(current_time)
            
        else:
            print(f"Step {time_step} failed: Newton its={result.iterations}, ||R||={result.final_residual_norm:.2e}, time={result.computation_time:.4f}s")
            break
    
    # ============================================================================
    # FINAL RESULTS AND ERROR ANALYSIS
    # ============================================================================
    
    successful_steps = len(solution_history) - 1  # Subtract initial condition
    print(f"Evolution completed: {successful_steps}/{max_time_steps} steps, final time: {current_time:.6f}")
    
    # Extract final solutions
    final_traces, final_multipliers = setup.extract_domain_solutions(current_solution)
    final_bulk_data = current_bulk_data
    
    # ERROR ANALYSIS using MinimalErrorEvaluator
    


   
    error_evaluator = MinimalErrorEvaluator()
    
    # Compute trace errors
    trace_errors = error_evaluator.compute_trace_error(
        trace_solutions=final_traces,
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        time=current_time
    )
    
    # Compute bulk errors
    bulk_errors = error_evaluator.compute_bulk_error(
        bulk_solutions=final_bulk_data,
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        time=current_time
    )
    
    # Print computed errors
    print("\n=== ERROR ANALYSIS RESULTS ===")
    print(f"Time: {current_time:.6f}")
    print(f"Time step: {dt:.2e}")
    n_elements = setup.global_discretization.spatial_discretizations[0].n_elements
    print(f"Elements: {n_elements}")
    
    print("\nTRACE ERRORS (weighted Euclidean):")
    for eq_idx, global_error in trace_errors['global'].items():
        if global_error is not None:
            print(f"  Equation {eq_idx}: {global_error:.6e}")
            for domain_idx, local_error in trace_errors['local'].items():
                if eq_idx in local_error and local_error[eq_idx] is not None:
                    print(f"    Domain {domain_idx}: {local_error[eq_idx]:.6e}")
        else:
            print(f"  Equation {eq_idx}: No analytical solution available")
    
    print("\nBULK ERRORS (L2 norm):")
    for eq_idx, global_error in bulk_errors['global'].items():
        if global_error is not None:
            print(f"  Equation {eq_idx}: {global_error:.6e}")
            for domain_idx, local_error in bulk_errors['local'].items():
                if eq_idx in local_error and local_error[eq_idx] is not None:
                    print(f"    Domain {domain_idx}: {local_error[eq_idx]:.6e}")
        else:
            print(f"  Equation {eq_idx}: No analytical solution available")
    
    # Check if any errors were computed
    has_trace_errors = any(err is not None for err in trace_errors['global'].values())
    has_bulk_errors = any(err is not None for err in bulk_errors['global'].values())
    
    if not has_trace_errors and not has_bulk_errors:
        print("\nNo analytical solutions available for error computation")
    else:
        print("\nError analysis completed successfully")

    # ============================================================================
    # PLOTTING RESULTS
    # ============================================================================
    
    print("\n=== GENERATING SOLUTION PLOTS ===")
    
    # Get number of equations from problem
    problem = setup.problems[0]
    n_equations = problem.neq
    
    print(f"Problem has {n_equations} equations")
    print(f"Trace solutions shapes: {[trace.shape for trace in final_traces]}")
    print(f"Bulk solutions shapes: {[bulk.get_data().shape for bulk in final_bulk_data]}")
    
    # Plot solutions for each equation
    for eq_idx in range(n_equations):
        print(f"Plotting equation {eq_idx}...")
        
        # Plot trace solution
        plot_trace_solution(
            trace_solutions=final_traces,
            problems=setup.problems,
            discretizations=setup.global_discretization.spatial_discretizations,
            current_time=current_time,
            equation_idx=eq_idx
        )
        
        # Plot bulk solution  
        plot_bulk_solution(
            bulk_solutions=final_bulk_data,
            problems=setup.problems,
            discretizations=setup.global_discretization.spatial_discretizations,
            current_time=current_time,
            equation_idx=eq_idx
        )
    
    print(f"Generated plots for {n_equations} equations")
    plt.show()

    
    return setup, time_stepper, solution_history, time_history


if __name__ == "__main__":
    """Main execution with config file support."""
    
    # Check for config file argument
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"Error: Configuration file '{config_file}' not found")
            sys.exit(1)
    else:
        # Default to ooc_parameters.toml if no argument provided
        config_file = "config/ooc_parameters.toml"
        if not os.path.exists(config_file):
            config_file = None
    
    try:
        # Main evolution example with config file
        result = run_evolution_with_time_stepper(config_file)
        
        # Check if setup failed due to configuration error
        if result[0] is None:
            print(f"Stopping execution due to configuration error")
            sys.exit(1)
        
        setup, time_stepper, sol_history, time_hist = result
        
    except KeyboardInterrupt:
        print(f"Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Example failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
