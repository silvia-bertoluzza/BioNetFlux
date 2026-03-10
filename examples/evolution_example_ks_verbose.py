"""
Evolution + Plotting Example using new Time Stepper Module

This example demonstrates the same functionality as evolution+plotting_example.py
but using the new TimeStepper module for cleaner, more maintainable code.

The time advancement logic is replaced with a single TimeStepper class that
encapsulates all the Newton iteration and bulk data management.
"""

import sys
import os
# sys.path hack — commented out, use pip install -e . instead
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from setup_solver import quick_setup, SolverSetup
from bionetflux.time_integration import TimeStepper
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
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
    print("="*80)
    print("EVOLUTION + PLOTTING EXAMPLE WITH TIME STEPPER")
    print("="*80)
    print("Time evolution using the new TimeStepper module")
    if config_file:
        print(f"Using configuration file: {config_file}")
    else:
        print("Using default parameters")
    print()
    
    # ============================================================================
    # STEP 1: SOLVER SETUP (Enhanced with config file support and error handling)
    # ============================================================================
    
    print("Step 1: Setting up solver...")
    
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

    # Get problem information
    info = setup.get_problem_info()
    print(f"✓ Problem loaded: {info['problem_name']}")
    print(f"  Domains: {info['num_domains']}")
    print(f"  Total DOFs: {info['total_trace_dofs'] + info['num_constraints']}")
    print(f"  Time discretization: dt={info['time_discretization']['dt']}, T={info['time_discretization']['T']}")
    
    # ============================================================================
    # STEP 2: TIME STEPPER INITIALIZATION (NEW!)
    # ============================================================================
    
    print("\nStep 2: Initializing time stepper...")
    
    # Create time stepper with Newton solver configuration
    time_stepper = TimeStepper(setup, verbose=True)
    
    # Initialize solution at t=0 (replaces Steps 3-4 and lines 226-233 from original)
    current_solution, current_bulk_data = time_stepper.initialize_solution()
    
    print("✓ Time stepper initialized")
    print(f"✓ Initial solution: shape {current_solution.shape}")
    print(f"✓ Initial bulk data: {len(current_bulk_data)} domains")
    
    # ============================================================================
    # STEP 3: VISUALIZATION SETUP (Same as original)
    # ============================================================================
    
    print("\nStep 3: Setting up visualization...")
    
    # Initialize plotter
    plotter = LeanMatplotlibPlotter(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        equation_names=None,  # Auto-detect
        figsize=(15, 10)
    )
    
    
    
    print(f"✓ Plotter initialized for {plotter.ndom} domains, {plotter.neq} equations")
    print(f"✓ Equation names: {plotter.equation_names}")
    
    # Plot geometry
    print("\nPlotting geometry...")
    
    setup.compute_geometry_from_problems()
    plotter.plot_geometry_with_indices(geometry=setup.geometry,
                                       save_filename="geometry_with_indices.png")
    print("✓ Geometry plot created")
    
    # # Plot initial state
    # initial_traces, _ = setup.extract_domain_solutions(current_solution)
    
    # print("\nPlotting initial state...")
    # plotter.plot_2d_curves(
    #     initial_traces,
    #     title="Initial Solution State",
    #     save_filename="initial_2d_curves.png"
    # )
    
    # for eq_idx in range(plotter.neq):
    #     plotter.plot_birdview(
    #         initial_traces,
    #         equation_idx=eq_idx,
    #         time=0.0,
    #         save_filename=f"initial_birdview_eq{eq_idx}.png"
    #     )
    
    # print("✓ Initial state plots created")

    
    error_evaluator = ErrorEvaluator(setup.problems, setup.global_discretization.spatial_discretizations)
    
        
    # ============================================================================
    # STEP 4: TIME EVOLUTION (MASSIVELY SIMPLIFIED!)
    # ============================================================================
    
    print("\nStep 4: Starting time evolution...")
    
    # Time evolution parameters
    current_time = 0.0
    dt = setup.global_discretization.dt
    T = min(0.5, setup.global_discretization.T)  # Limit runtime for demo
    max_time_steps = int(T / dt) + 1
    
    # Solution history for analysis
    solution_history = [current_solution.copy()]
    time_history = [current_time]
    
    print(f"Time evolution: t ∈ [0, {T}], dt = {dt}")
    print(f"Maximum time steps: {max_time_steps}")
    print()
    
    initial_traces, initial_multipliers = setup.extract_domain_solutions(current_solution)
    for eq_idx in range(plotter.neq):
        plotter.plot_birdview(
            initial_traces,
            equation_idx=eq_idx,
            time=current_time,
            save_filename=f"final_birdview_eq{eq_idx}.png"
        )
    
    # TIME EVOLUTION LOOP - SIMPLIFIED TO ONE LINE PER TIME STEP!
    time_step = 0
    
    while current_time + dt <= T and time_step < max_time_steps:
        time_step += 1
        print(f"\n--- Time Step {time_step}: t = {current_time:.6f} → {current_time + dt:.6f} ---")
        
        # SINGLE CALL REPLACES ~50 LINES OF COMPLEX NEWTON ITERATION CODE!
        result = time_stepper.advance_time_step(
            current_solution=current_solution,
            current_bulk_data=current_bulk_data,
            current_time=current_time,
            dt=dt
        )
        
        # Handle result
        if result.converged:
            print(f"  ✓ Time step successful!")
            print(f"    Newton iterations: {result.iterations}")
            print(f"    Final residual norm: {result.final_residual_norm:.6e}")
            print(f"    Computation time: {result.computation_time:.4f}s")
            
            # Update state for next iteration
            current_time += dt
            current_solution = result.updated_solution
            current_bulk_data = result.updated_bulk_data
            
            # Store history
            solution_history.append(current_solution.copy())
            time_history.append(current_time)
            
        else:
            print(f"  ✗ Time step failed!")
            print(f"    Newton iterations: {result.iterations}")
            print(f"    Final residual norm: {result.final_residual_norm:.6e}")
            print(f"    Computation time: {result.computation_time:.4f}s")
            print("  Stopping time evolution due to convergence failure")
            break
    
    # ============================================================================
    # STEP 5: FINAL RESULTS AND VISUALIZATION
    # ============================================================================
    
    print(f"\n" + "="*50)
    print("TIME EVOLUTION COMPLETED")
    print("="*50)
    
    successful_steps = len(solution_history) - 1  # Subtract initial condition
    print(f"Successful time steps: {successful_steps}/{max_time_steps}")
    print(f"Final time: {current_time:.6f}")
    print(f"Total solution history: {len(solution_history)} time points")
    
    # Extract final solutions
    final_traces, final_multipliers = setup.extract_domain_solutions(current_solution)
    final_bulk_data = current_bulk_data
    
    # ============================================================================
    # STEP 6a: ERROR ANALYSIS (NEW!)
    # ============================================================================
    
    print(f"\nStep 6a: Evaluating final solution error...")  
    
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
    else:
        print("✓ No analytical solution available for error computation")
    
    # ============================================================================
    # STEP 6b: FINAL VISUALIZATION
    # ============================================================================
    
    print(f"\nStep 6: Creating final visualization...")
    

    
    for eq_idx in range(plotter.neq):
        plotter.plot_birdview(
            final_traces,
            equation_idx=eq_idx,
            time=current_time,
            save_filename=f"final_birdview_eq{eq_idx}.png"
        )
    
    # Evolution comparison
 
    print("✓ Final visualization completed")
    
    # ============================================================================
    # STEP 7: ANALYSIS AND SUMMARY
    # ============================================================================
    
    print(f"\nStep 7: Solution analysis...")
    
    
    # Show all plots
    print(f"\nDisplaying all generated plots...")
    plotter.show_all()
    
    print(f"\n🎉 Evolution example completed successfully!")
    print(f"📊 Key improvements with TimeStepper:")
    print(f"   - Time advancement: 1 line instead of ~50 lines")
    print(f"   - Automatic error handling and reporting")
    print(f"   - Clean separation of concerns")
    print(f"   - Detailed convergence information")
    print(f"   - Easy to extend with adaptive time stepping")
    
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


if __name__ == "__main__":
    """Main execution with multiple demonstrations and config file support."""
    
    # Check for config file argument
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"❌ Error: Configuration file '{config_file}' not found")
            print(f"💡 Please check the file path and try again")
            sys.exit(1)
        print(f"Using configuration file: {config_file}")
    else:
        # Default to ooc_parameters.toml if no argument provided
        config_file = "config/ooc_parameters.toml"
        if os.path.exists(config_file):
            print(f"Using default configuration file: {config_file}")
        else:
            print(f"Default config file '{config_file}' not found, using defaults")
            config_file = None
    
    try:
        # Main evolution example with config file
        result = run_evolution_with_time_stepper(config_file)
        
        # Check if setup failed due to configuration error
        if result[0] is None:
            print(f"\n🛑 Stopping execution due to configuration error")
            sys.exit(1)
        
        setup, time_stepper, sol_history, time_hist = result
        
        # Additional demonstrations
        print("\n" + "🔬" * 40)
        
        # Multiple steps demonstration with config file
        # multi_results = demonstrate_multiple_steps(config_file)
        
        print(f"\n🎉 All demonstrations completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Example failed with unexpected error:")
        print(f"   {type(e).__name__}: {e}")
        print(f"\n🔧 Debug information:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
