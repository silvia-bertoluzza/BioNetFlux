"""
Evolution + Plotting Example using new Time Stepper Module

This example demonstrates the same functionality as evolution+plotting_example.py
but using the new TimeStepper module for cleaner, more maintainable code.

The time advancement logic is replaced with a single TimeStepper class that
encapsulates all the Newton iteration and bulk data management.
"""

import sys
import os
# Add the python_port directory to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from setup_solver import quick_setup, SolverSetup
from bionetflux.time_integration import TimeStepper
from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry, build_grid_geometry
from bionetflux.analysis.error_evaluation import ErrorEvaluator
import numpy as np
import time
from typing import Optional


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
    error_evaluator = ErrorEvaluator(setup.problems, setup.global_discretization.spatial_discretizations)
    
        
    # ============================================================================
    # STEP 3: TIME EVOLUTION
    # ============================================================================
    
    # Time evolution parameters
    current_time = 0.0
    dt = setup.global_discretization.dt
    T = min(0.5, setup.global_discretization.T)  # Limit runtime for demo
    max_time_steps = int(T / dt) + 1
    
    # Solution history for analysis
    solution_history = [current_solution.copy()]
    time_history = [current_time]
    
    print(f"Evolution Parameters: t ∈ [0, {T}], dt = {dt}, max steps = {max_time_steps}")
    
    # TIME EVOLUTION LOOP
    time_step = 0
    
    while current_time + dt <= T and time_step < max_time_steps:
        time_step += 1
        
        # SINGLE CALL REPLACES ~50 LINES OF COMPLEX NEWTON ITERATION CODE!
        result = time_stepper.advance_time_step(
            current_solution=current_solution,
            current_bulk_data=current_bulk_data,
            current_time=current_time,
            dt=dt
        )
        
        # Handle result
        if result.converged:
            # Update state for next iteration
            current_time += dt
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
    
    # ERROR ANALYSIS
    trace_errors = error_evaluator.compute_trace_error(
                            numerical_solutions=final_traces, 
                            time=current_time,
                            analytical_functions=None  # Auto-detect if available
                            )
    
    bulk_errors = error_evaluator.compute_bulk_error(
                            bulk_solutions = final_bulk_data, 
                            time=current_time
                            )



    # Generate comprehensive error report with both trace and bulk errors
    if trace_errors is not None or bulk_errors is not None:
        error_report = error_evaluator.generate_error_report(
            trace_errors=trace_errors, 
            bulk_errors=bulk_errors
        )
        print(error_report)
        
        # Extract data for file output
        n_elements = setup.global_discretization.spatial_discretizations[0].n_elements
        euclidean_trace_error = trace_errors[0]['euclidean'] if trace_errors and len(trace_errors) > 0 else float('nan')
        l2_bulk_error_eq0 = bulk_errors[0]['L2'] if bulk_errors and len(bulk_errors) > 0 else float('nan')
        l2_bulk_error_eq1 = bulk_errors[1]['L2'] if bulk_errors and len(bulk_errors) > 1 else float('nan')
        
        # Append results to file
        results_file = "evolution_results.txt"
        with open(results_file, "a") as f:
            f.write(f"{dt:.6e}\t{n_elements}\t{euclidean_trace_error:.6e}\t{l2_bulk_error_eq0:.6e}\t{l2_bulk_error_eq1:.6e}\n")
        
        print(f"Results appended to {results_file}")
    else:
        print("No analytical solution available for error computation")
        
        # Still log basic parameters even without errors
        n_elements = setup.global_discretization.spatial_discretizations[0].n_elements
        results_file = "evolution_results.txt"
        with open(results_file, "a") as f:
            f.write(f"{dt:.6e}\t{n_elements}\tnan\tnan\tnan\n")
        print(f"Basic parameters appended to {results_file}")
    
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
