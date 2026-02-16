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
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry, build_grid_geometry
from bionetflux.analysis.error_evaluation import ErrorEvaluator
import numpy as np
import time
from typing import Optional


def run_evolution_with_time_stepper(config_file: Optional[str] = None, dt: Optional[float] = None, n_elements: Optional[int] = None):
    """
    Main function demonstrating time evolution with the new TimeStepper module.
    
    Args:
        config_file: Optional TOML configuration file path
        dt: Optional time step size (overrides config file value)
        n_elements: Optional number of elements (overrides config file value)
    """
    # Show override parameters for convergence study
    if dt is not None or n_elements is not None:
        overrides = []
        if dt is not None:
            overrides.append(f"dt={dt}")
        if n_elements is not None:
            overrides.append(f"n_elements={n_elements}")
        print(f"Running with: {', '.join(overrides)}")
    
    
    # Build geometry (always N=1 for this experiment)
    geometry = build_arc_sequence_geometry(N=1, start=2, length=1.0)

    # Try to call quick_setup with error handling for config compatibility    
    try:
        # Setup solver
        setup = quick_setup(
            problem_module="bionetflux.problems.ks_problem",
            validate=True,
            config_file=config_file,
            geometry=geometry
        )
        
        # Override parameters after setup but before validation
        if dt is not None:
            setup.global_discretization.dt = dt
            
        if n_elements is not None:
            # Override spatial discretization for all domains
            for discretization, domain_data in zip(setup.global_discretization.spatial_discretizations, setup._domain_data):
                discretization.n_elements = n_elements
                domain_data.n_elements = n_elements
                
                
            # Map constraints to discretizations
            setup.constraint_manager.map_to_discretizations(setup.global_discretization.spatial_discretizations)
            
            
            
            # Also update any existing domain data to match
            for problem in setup.problems:
                if hasattr(problem, 'domain_data'):
                    problem.domain_data.n_elements = n_elements
    except ValueError as e:
        # Handle configuration compatibility errors gracefully
        if "not compatible with" in str(e) or "problem type" in str(e):
            print(f"\n❌ Configuration Error:")
            print(f"   {e}")
            print(f"\n💡 Suggestions:")
            print(f"   - Check that problem_type in your config file matches the problem module")
            print(f"   - For ooc_problem.py, use problem_type = \"ooc\"")
            print(f"   - For ks_problem.py, use problem_type = \"ks\"")
            print(f"   - Or run without a config file to use defaults")
            return None, None, None, None
        else:
            # Re-raise other ValueError types
            raise
    except Exception as e:
        # Handle other setup errors
        print(f"\n❌ Setup Error: {e}")
        print(f"💡 Try running with default parameters (no config file)")
        return None, None, None, None

    # Setup time stepper and initialize solution
    time_stepper = TimeStepper(setup, verbose=False)  # Reduce verbosity for convergence study
    current_solution, current_bulk_data = time_stepper.initialize_solution()
    
    # Setup plotter and error evaluator (minimal output)
    plotter = LeanMatplotlibPlotter(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        equation_names=None,
        figsize=(15, 10)
    )
    
    setup.compute_geometry_from_problems()
    
    error_evaluator = ErrorEvaluator(setup.problems, setup.global_discretization.spatial_discretizations)
    
        
    # Time evolution parameters
    current_time = 0.0
    dt = setup.global_discretization.dt
    T = min(0.5, setup.global_discretization.T)
    max_time_steps = int(T / dt) + 1
    
    # Solution history for analysis
    solution_history = [current_solution.copy()]
    time_history = [current_time]
    
    # Time evolution loop (minimal output for convergence study)
    time_step = 0
    failed = False
    
    while current_time + dt <= T and time_step < max_time_steps:
        time_step += 1
        
        result = time_stepper.advance_time_step(
            current_solution=current_solution,
            current_bulk_data=current_bulk_data,
            current_time=current_time,
            dt=dt
        )
        
        if result.converged:
            # Update state for next iteration
            current_time += dt
            current_solution = result.updated_solution
            current_bulk_data = result.updated_bulk_data
            
            # Store history
            solution_history.append(current_solution.copy())
            time_history.append(current_time)
        else:
            failed = True
            break
    
    # Report convergence failure for convergence study
    if failed:
        print(f"  ✗ Newton convergence failure at step {time_step}")
        return None, None, None, None
    
    # Extract final solutions and compute errors for convergence study
    final_traces, final_multipliers = setup.extract_domain_solutions(current_solution)
    final_bulk_data = current_bulk_data
    
    # Compute errors (silent for convergence study)
    trace_errors = error_evaluator.compute_trace_error(
        numerical_solutions=final_traces, 
        time=current_time,
        analytical_functions=None
    )
    
    bulk_errors = error_evaluator.compute_bulk_error(
        bulk_solutions=final_bulk_data, 
        time=current_time
    )
    
    return setup, time_stepper, solution_history, time_history


def demonstrate_multiple_steps(config_file: Optional[str] = None):
    """
    Demonstrate the advance_multiple_steps functionality.
    
    Args:
        config_file: Optional TOML configuration file path
    """
    print("\n" + "="*80)
    print("DEMONSTRATING MULTIPLE STEPS ADVANCEMENT")
    print("="*80)
    
    # Quick setup with config file support (geometry can be passed here too if needed)
    setup = quick_setup(
        problem_module="bionetflux.problems.ooc_problem", 
        validate=True,
        config_file=config_file  # Pass config file
    )
    time_stepper = TimeStepper(setup, verbose=True)
    
    # Initialize
    initial_solution, initial_bulk_data = time_stepper.initialize_solution()
    
    # Advance multiple steps in one call
    dt = setup.global_discretization.dt
    n_steps = 5
    
    print(f"Advancing {n_steps} time steps with dt = {dt}")
    
    results = time_stepper.advance_multiple_steps(
        initial_solution=initial_solution,
        initial_bulk_data=initial_bulk_data,
        start_time=0.0,
        dt=dt,
        n_steps=n_steps,
        stop_on_failure=True
    )
    
    # Analyze results
    successful_steps = sum(1 for r in results if r.converged)
    print(f"\nMultiple steps results:")
    print(f"  Steps attempted: {len(results)}")
    print(f"  Steps successful: {successful_steps}")
    print(f"  Success rate: {successful_steps/len(results)*100:.1f}%")
    
    # Show per-step details
    for i, result in enumerate(results):
        status = "✓" if result.converged else "✗"
        print(f"  Step {i+1}: {status} {result.iterations} Newton its, "
              f"||R|| = {result.final_residual_norm:.2e}")
    
    return results


def run_convergence_study(config_file: Optional[str] = None):
    """
    Run a convergence study with different dt and n_elements values.
    
    Args:
        config_file: Optional TOML configuration file path
    """
    print("="*80)
    print("CONVERGENCE STUDY")
    print("="*80)
    
    # Define convergence study parameters
    dt_values = [0.1, 0.05, 0.025, 0.0125]  # Time step refinement
    n_elements_values = [4, 8, 16, 32]       # Spatial refinement
    
    results = {}
    
    for dt_val in dt_values:
        for n_elem in n_elements_values:
            print(f"\n{'='*60}")
            print(f"RUNNING: dt = {dt_val}, n_elements = {n_elem}")
            print(f"{'='*60}")
            
            try:
                # Run with specific parameters
                result = run_evolution_with_time_stepper(
                    config_file=config_file,
                    dt=dt_val,
                    n_elements=n_elem
                )
                
                if result[0] is not None:  # Check if setup succeeded
                    setup, time_stepper, sol_history, time_hist = result
                    results[(dt_val, n_elem)] = {
                        'success': True,
                        'final_time': time_hist[-1],
                        'n_timesteps': len(time_hist) - 1,
                        'setup': setup,
                        'solution_history': sol_history
                    }
                    print(f"✓ Success: dt={dt_val}, n_elements={n_elem}")
                else:
                    results[(dt_val, n_elem)] = {'success': False}
                    print(f"✗ Failed: dt={dt_val}, n_elements={n_elem}")
                    
            except Exception as e:
                print(f"✗ Error with dt={dt_val}, n_elements={n_elem}: {e}")
                results[(dt_val, n_elem)] = {'success': False, 'error': str(e)}
    
    # Print convergence study summary
    print(f"\n{'='*80}")
    print("CONVERGENCE STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"{'dt':<10} {'n_elements':<12} {'Status':<8} {'Final Time':<12} {'Timesteps'}")
    print("-" * 60)
    
    for (dt_val, n_elem), result in results.items():
        if result['success']:
            status = "✓"
            final_time = f"{result['final_time']:.4f}"
            timesteps = f"{result['n_timesteps']}"
        else:
            status = "✗"
            final_time = "N/A"
            timesteps = "N/A"
        
        print(f"{dt_val:<10} {n_elem:<12} {status:<8} {final_time:<12} {timesteps}")
    
    return results


if __name__ == "__main__":
    """Main execution focused on convergence study."""
    
    # Check for config file argument
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"❌ Error: Configuration file '{config_file}' not found")
            sys.exit(1)
    else:
        # Default to ks_parameters.toml if no argument provided
        config_file = "../config/ks_parameters.toml"
        if os.path.exists(config_file):
            config_file = config_file
        else:
            config_file = None
    
    try:
        # Run convergence study
        convergence_results = run_convergence_study(config_file)
        
        print(f"\n🎉 Convergence study completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        sys.exit(1)
