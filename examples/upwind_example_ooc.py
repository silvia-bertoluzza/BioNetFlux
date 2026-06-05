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
from bionetflux.time_integration.picard_solver import PicardSolver
from bionetflux.time_integration.time_stepper import AdaptiveTimeStepper
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry, build_grid_geometry, create_maze_geometry
from bionetflux.problems.ooc_config_manager import OoCConfigManager
import bionetflux.geometry.domain_geometry as _geom_module
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time
from datetime import datetime
from typing import Optional, List, Dict, Any


def _create_run_output_dir(config_file: Optional[str] = None) -> str:
    """Create a timestamped output directory for this run under outputs/.

    The directory is named ``run_YYYYMMDD_HHMMSS`` and, if a configuration
    file is provided, a copy is placed inside it.

    Args:
        config_file: Path to the TOML configuration file (may be None).

    Returns:
        Path of the newly created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    if config_file is not None and os.path.isfile(config_file):
        shutil.copy2(config_file, os.path.join(run_dir, os.path.basename(config_file)))

    return run_dir


# Config key order matches OoC equation order (u, omega, v, phi)
_OOC_EQ_KEYS = ['u', 'omega', 'v', 'phi']


def _overlay_exact_solutions(
    fig: plt.Figure,
    plotter: LeanMatplotlibPlotter,
    exact_solutions: Dict[str, Any],
    current_time: float,
) -> bool:
    """Superpose exact solutions on an existing plot_2d_curves figure.

    For each equation whose entry in *exact_solutions* is callable, the exact
    solution is evaluated at the domain nodes at *current_time* and drawn as a
    dashed line (same colour as the discrete curve) on the corresponding axis.

    Args:
        fig: Figure returned by :meth:`LeanMatplotlibPlotter.plot_2d_curves`.
        plotter: The plotter instance that produced *fig*.
        exact_solutions: Dict mapping equation keys to callables or ``None``.
        current_time: The time at which the exact solution is evaluated.

    Returns:
        ``True`` if at least one exact solution was overlaid, ``False`` otherwise.
    """
    axes = fig.axes
    coord_start = 0
    any_drawn = False

    for domain_idx in range(plotter.ndom):
        ax = axes[domain_idx]
        domain_info = plotter.domain_info[domain_idx]
        n_nodes = domain_info['n_nodes']
        domain_coords = plotter.all_coords[coord_start: coord_start + n_nodes]

        domain_drawn = False
        for eq_idx, eq_key in enumerate(_OOC_EQ_KEYS[:plotter.neq]):
            func = exact_solutions.get(eq_key)
            if callable(func):
                exact_vals = func(domain_coords, current_time)
                color = plotter.equation_colors[eq_idx % len(plotter.equation_colors)]
                ax.plot(
                    domain_coords, exact_vals,
                    color=color, linewidth=1.5, linestyle='--',
                    label=f'{plotter.equation_names[eq_idx]} (exact)',
                    alpha=0.9,
                )
                domain_drawn = True

        if domain_drawn:
            ax.legend()
            any_drawn = True

        coord_start += n_nodes

    return any_drawn


def _load_exact_solutions(config_file: Optional[str]) -> Dict[str, Any]:
    """Load the OoC config and return the resolved ``exact_solutions`` dict."""
    cfg_manager = OoCConfigManager()
    config = cfg_manager.load_config(config_file)
    return config.get('exact_solutions', {})


def run_evolution_with_time_stepper(
    config_file: Optional[str] = None,
    arc_number: int = 3,
    arc_length: float = 500.0,
):
    """
    Main function demonstrating time evolution with the new TimeStepper module.

    Args:
        config_file: Optional TOML configuration file path
        arc_number: Number of arcs in the sequence geometry (N parameter)
        arc_length: Length of each arc in the sequence geometry
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
    
    geometry = build_arc_sequence_geometry(N=arc_number, length=arc_length)
    # _geom_dir = os.path.dirname(_geom_module.__file__)
    # geometry = create_maze_geometry(
    #     data_dir=os.path.join(_geom_dir, "maze_3_data"),
    #     length=50.0,
    # )   


    try:
        # Use quick_setup with both geometry and config file support
        setup = quick_setup(
            problem_module="bionetflux.problems.ooc_problem_upwind",
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
    
    # Create Picard solver and pass it to the time stepper
    picard_solver = PicardSolver(tolerance=1.e-7, max_iterations=50, verbose=False)
    time_stepper = TimeStepper(setup, picard_solver=picard_solver, verbose=True)
    
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
    # plotter.plot_geometry_with_indices(geometry=setup.geometry,
    #                                    save_filename="geometry_with_indices.png")
    # print("✓ Geometry plot created")
    
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
    
    # ============================================================================
    # STEP 4: TIME EVOLUTION (MASSIVELY SIMPLIFIED!)
    # ============================================================================
    
    print("\nStep 4: Starting time evolution...")
    
    # Time evolution parameters
    current_time = 0.0
    dt = setup.global_discretization.dt
    T = setup.global_discretization.T # Limit runtime for demo
    max_time_steps = int(T / dt) + 1
    
    # Solution history for analysis
    solution_history = [current_solution.copy()]
    time_history = [current_time]
    
    print(f"Time evolution: t ∈ [0, {T}], dt = {dt}")
    print(f"Maximum time steps: {max_time_steps}")
    print()
    
    # TIME EVOLUTION LOOP - SIMPLIFIED TO ONE LINE PER TIME STEP!
    time_step = 0
    
    while current_time + dt <= T + 1e-12 and time_step <= max_time_steps:
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
            print(f"    Picard iterations: {result.iterations}")
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
    
    print(f"\nFinal solution characteristics:")
    for i, trace in enumerate(final_traces):
        trace_norm = np.linalg.norm(trace)
        print(f"  Domain {i}: ||trace|| = {trace_norm:.6e}")
    
    if len(final_multipliers) > 0:
        multiplier_norm = np.linalg.norm(final_multipliers)
        print(f"  Multipliers: ||λ|| = {multiplier_norm:.6e}")
    
    # ============================================================================
    # STEP 6: FINAL VISUALIZATION
    # ============================================================================
    
    print(f"\nStep 6: Creating final visualization...")

    fig_final = plotter.plot_2d_curves(
        final_traces,
        title=f"Final 2D Curves at t={current_time:.2f}",
        save_filename="final_2d_curves.png",
        bulk_solutions=current_bulk_data
    )

    # Superpose exact solution if available
    exact_solutions = _load_exact_solutions(config_file)
    if _overlay_exact_solutions(fig_final, plotter, exact_solutions, current_time):
        save_path = plotter._get_save_path("final_2d_curves.png")
        if save_path:
            fig_final.savefig(save_path, dpi=300, bbox_inches='tight')
        print("✓ Exact solution overlay added")

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


def run_evolution_with_adaptive_time_stepper(
    config_file: Optional[str] = None,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    safety_factor: float = 0.8,
    arc_number: int = 3,
    arc_length: float = 500.0,
):
    """Time evolution using the AdaptiveTimeStepper.

    This mirrors :func:`run_evolution_with_time_stepper` but lets the time
    step shrink (and grow) automatically based on Newton convergence.

    If *dt_min* / *dt_max* are not provided they default to
    ``dt_config * 1e-4`` and ``dt_config`` respectively, where
    ``dt_config`` is the value read from the configuration file.

    Args:
        config_file: Optional TOML configuration file path.
        dt_min: Minimum allowed time step (default: dt_config * 1e-4).
        dt_max: Maximum allowed time step (default: dt_config).
        safety_factor: Safety factor for time step adjustment.
        arc_number: Number of arcs in the sequence geometry (N parameter).
        arc_length: Length of each arc in the sequence geometry.

    Returns:
        (setup, time_stepper, solution_history, time_history, dt_history)
    """
    print("=" * 80)
    print("EVOLUTION WITH ADAPTIVE TIME STEPPER")
    print("=" * 80)
    if config_file:
        print(f"Using configuration file: {config_file}")
    else:
        print("Using default parameters")
    print()

    # --- Create per-run output directory ---
    run_dir = _create_run_output_dir(config_file)
    print(f"Run output directory: {run_dir}")

    # ====================================================================
    # STEP 1: SOLVER SETUP
    # ====================================================================

    print("Step 1: Setting up solver...")

    geometry = build_arc_sequence_geometry(N=arc_number, length=arc_length)

    try:
        setup = quick_setup(
            problem_module="bionetflux.problems.ooc_problem_upwind",
            validate=True,
            config_file=config_file,
            geometry=geometry,
        )
    except ValueError as e:
        if "not compatible with" in str(e) or "problem type" in str(e):
            print(f"\n\u274c Configuration Error: {e}")
            return None, None, None, None, None
        raise
    except Exception as e:
        print(f"\n\u274c Setup Error: {e}")
        return None, None, None, None, None

    info = setup.get_problem_info()
    print(f"\u2713 Problem loaded: {info['problem_name']}")
    print(f"  Domains: {info['num_domains']}")
    print(f"  Total DOFs: {info['total_trace_dofs'] + info['num_constraints']}")
    print(f"  Time discretization: dt={info['time_discretization']['dt']}, "
          f"T={info['time_discretization']['T']}")

    # ====================================================================
    # STEP 2: ADAPTIVE TIME STEPPER INITIALIZATION
    # ====================================================================

    print("\nStep 2: Initializing adaptive time stepper...")
    
    # Create Picard solver and pass it to the time stepper
    picard_solver = PicardSolver(tolerance=1.e-7, max_iterations=50, verbose=False)
    time_stepper = AdaptiveTimeStepper(setup, picard_solver=picard_solver, verbose=True,
                               dt_min=dt_min,
                               dt_max=dt_max,
                               safety_factor=safety_factor,)
    

    # time_stepper = AdaptiveTimeStepper(
    #     setup,
    #     verbose=True,
    #     dt_min=dt_min,
    #     dt_max=dt_max,
    #     safety_factor=safety_factor,
    # )

    current_solution, current_bulk_data = time_stepper.initialize_solution()
    print(f"\u2713 Adaptive time stepper initialized "
          f"(dt_min={time_stepper.dt_min:.6e}, dt_max={time_stepper.dt_max:.6e})")
    print(f"\u2713 Initial solution: shape {current_solution.shape}")
    print(f"\u2713 Initial bulk data: {len(current_bulk_data)} domains")

    # ====================================================================
    # STEP 3: VISUALIZATION SETUP
    # ====================================================================

    print("\nStep 3: Setting up visualization...")

    plotter = LeanMatplotlibPlotter(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        equation_names=None,
        figsize=(15, 10),
        output_dir=run_dir,
    )
    print(f"\u2713 Plotter initialized for {plotter.ndom} domains, {plotter.neq} equations")

    setup.compute_geometry_from_problems()
    # plotter.plot_geometry_with_indices(
    #     geometry=setup.geometry, save_filename="geometry_with_indices.png"
    # )
    # print("\u2713 Geometry plot created")

    # ====================================================================
    # STEP 4: TIME EVOLUTION (ADAPTIVE)
    # ====================================================================

    print("\nStep 4: Starting adaptive time evolution...")

    current_time = 0.0
    dt = setup.global_discretization.dt  # initial dt from config
    T = setup.global_discretization.T
    # Safety cap: independent of dt so adaptive shrinking cannot exhaust it.
    max_time_steps = int(T / time_stepper.dt_min) + 100

    solution_history = [current_solution.copy()]
    time_history = [current_time]
    dt_history: List[float] = []  # dt actually used for each step

    print(f"Time evolution: t \u2208 [0, {T}], initial dt = {dt}")
    print()

    time_step = 0
    dt_suggested = dt

    while current_time < T - 1e-14 and time_step < max_time_steps:
        time_step += 1
        # Clip so we land exactly on T
        dt_try = min(dt_suggested, T - current_time)

        print(f"\n--- Time Step {time_step}: t = {current_time:.6f} \u2192 "
              f"{current_time + dt_try:.6f}  (dt = {dt_try:.6e}) ---")

        result, dt_next = time_stepper.advance_time_step_adaptive(
            current_solution=current_solution,
            current_bulk_data=current_bulk_data,
            current_time=current_time,
            dt_suggested=dt_try,
        )

        if result.converged:
            # The actual dt used may differ from dt_try if retries occurred;
            # read it back from GlobalDiscretization (the single source of truth).
            dt_used = setup.global_discretization.dt

            print(f"  \u2713 Time step successful!")
            print(f"    dt used: {dt_used:.6e}")
            print(f"    Newton iterations: {result.iterations}")
            print(f"    Final residual norm: {result.final_residual_norm:.6e}")
            print(f"    Computation time: {result.computation_time:.4f}s")

            current_time += dt_used
            current_solution = result.updated_solution
            current_bulk_data = result.updated_bulk_data

            solution_history.append(current_solution.copy())
            time_history.append(current_time)
            dt_history.append(dt_used)

            dt_suggested = dt_next

        else:
            print(f"  \u2717 Time step failed after retries!")
            print(f"    Newton iterations: {result.iterations}")
            print(f"    Final residual norm: {result.final_residual_norm:.6e}")
            print("  Stopping time evolution due to convergence failure")
            break

    # ====================================================================
    # STEP 5: SUMMARY
    # ====================================================================

    successful_steps = len(solution_history) - 1
    print(f"\n" + "=" * 50)
    print("ADAPTIVE TIME EVOLUTION COMPLETED")
    print("=" * 50)
    print(f"Successful time steps: {successful_steps}")
    print(f"Final time: {current_time:.6f}")
    if dt_history:
        print(f"dt range: [{min(dt_history):.6e}, {max(dt_history):.6e}]")
    print(f"Total solution history: {len(solution_history)} time points")

    final_traces, final_multipliers = setup.extract_domain_solutions(current_solution)
    for i, trace in enumerate(final_traces):
        print(f"  Domain {i}: ||trace|| = {np.linalg.norm(trace):.6e}")
    if len(final_multipliers) > 0:
        print(f"  Multipliers: ||\u03bb|| = {np.linalg.norm(final_multipliers):.6e}")

    # ====================================================================
    # STEP 6: FINAL VISUALIZATION
    # ====================================================================

    print(f"\nStep 6: Creating final visualization...")

    # --- Final 2D curves ---
    exact_solutions = _load_exact_solutions(config_file)
    fig_final = plotter.plot_2d_curves(
        final_traces,
        title=f"Final 2D Curves at t={current_time:.2f}",
        save_filename="final_2d_curves.png",
        bulk_solutions=current_bulk_data
    )
    
    print(f"Current time: {current_time:.6f}")
    
    if _overlay_exact_solutions(fig_final, plotter, exact_solutions, current_time):
        save_path = plotter._get_save_path("final_2d_curves.png")
        if save_path:
            fig_final.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  \u2713 Exact solution overlay added")
    print(f"  \u2713 Final snapshot at t = {current_time:.2f}")

    # ====================================================================
    # STEP 7: DT EVOLUTION PLOT
    # ====================================================================

    # if dt_history:
    #     print("\nStep 7: Plotting dt evolution...")
    #     fig_dt, ax_dt = plt.subplots(figsize=(10, 4))
    #     # Each dt_history[i] was used for the step ending at time_history[i+1]
    #     ax_dt.step(time_history[1:], dt_history, where='pre', color='tab:blue')
    #     ax_dt.set_xlabel("Time")
    #     ax_dt.set_ylabel("dt")
    #     ax_dt.set_yscale("log")
    #     ax_dt.set_title("Adaptive time step size")
    #     ax_dt.grid(True, alpha=0.3)
    #     fig_dt.tight_layout()
    #     dt_plot_path = os.path.join(run_dir, "dt_evolution.png")
    #     fig_dt.savefig(dt_plot_path, dpi=150)
    #     print(f"\u2713 dt evolution plot saved to {dt_plot_path}")

    # Display all figures, then close
    plt.show()
    plotter.close_all()
    plt.close('all')
    print(f"\n\u2713 All figures saved to {run_dir} and closed.")
    print(f"\n\U0001f389 Adaptive evolution completed successfully!")

    return setup, time_stepper, solution_history, time_history, dt_history


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
    """Main execution with multiple demonstrations and config file support.

    Usage:
        python evolution_example_ooc.py [config.toml]              # fixed dt
        python evolution_example_ooc.py --adaptive [config.toml]   # adaptive dt
        python evolution_example_ooc.py --arc-number 5 --arc-length 300.0 [config.toml]
    """

    # Parse a simple --adaptive flag
    args = sys.argv[1:]
    use_adaptive = False
    if "--adaptive" in args:
        use_adaptive = True
        args.remove("--adaptive")

    # Parse --arc-number and --arc-length flags
    arc_number = 3
    arc_length = 500.0
    for flag, attr, conv in (
        ("--arc-number", "arc_number", int),
        ("--arc-length", "arc_length", float),
    ):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 >= len(args):
                print(f"Error: {flag} requires a value")
                sys.exit(1)
            if flag == "--arc-number":
                arc_number = conv(args[idx + 1])
            else:
                arc_length = conv(args[idx + 1])
            del args[idx:idx + 2]

    # Check for config file argument
    config_file = None
    if args:
        config_file = args[0]
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
        if use_adaptive:
            result = run_evolution_with_adaptive_time_stepper(
                config_file, arc_number=arc_number, arc_length=arc_length
            )

            if result[0] is None:
                print(f"\n🛑 Stopping execution due to configuration error")
                sys.exit(1)

            setup, time_stepper, sol_history, time_hist, dt_hist = result
        else:
            result = run_evolution_with_time_stepper(
                config_file, arc_number=arc_number, arc_length=arc_length
            )

            if result[0] is None:
                print(f"\n🛑 Stopping execution due to configuration error")
                sys.exit(1)

            setup, time_stepper, sol_history, time_hist = result

        # Additional demonstrations
        print("\n" + "🔬" * 40)
        
        # Multiple steps demonstration with config file
        # multi_results = demonstrate_multiple_steps(config_file)
        
        print(f"\n🎉 All demonstrations completed successfully!")

        # Play a system sound to signal completion (macOS)
        os.system("afplay /System/Library/Sounds/Glass.aiff &")
        
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
