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
from bionetflux.time_integration.time_stepper import AdaptiveTimeStepper
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
from bionetflux.geometry.domain_geometry import build_grid_geometry, create_maze_geometry
import bionetflux.geometry.domain_geometry as _geom_module
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# Mass monitoring helpers (left / right split of the maze)
# ---------------------------------------------------------------------------

def classify_domains_left_right(
    setup: "SolverSetup",
    length: float,
) -> List[Dict]:
    """Classify each domain as left, right, midline, or straddling the midline.

    Only domains whose y-extent lies within [0, 6*length] are included.
    The midline is at x = 3*length.

    For straddling (horizontal-crossing) domains the classification is
    refined element-by-element so that the trapezoidal integral can be
    split exactly at the midline.

    Args:
        setup: The SolverSetup object (provides problems and discretizations).
        length: The scaling length used when building the maze geometry.

    Returns:
        A list (one entry per domain) of dicts with keys:
            included  – bool, whether the domain contributes to the integral.
            side      – 'left' | 'right' | 'midline' | 'straddle' | None.
            elements  – For 'straddle' domains only: a list (per element) of
                        dicts {'side': 'left'|'right'|'split',
                               'frac_left': float} where *frac_left* is the
                        fraction of the element length on the left of the
                        midline (relevant only when side == 'split').
    """
    midline_x = 3.0 * length
    y_min = 0.0
    y_max = 6.0 * length
    tol = 1e-10

    n_domains = len(setup.problems)
    classification: List[Dict] = []

    for d in range(n_domains):
        prob = setup.problems[d]
        (x0, y0) = prob.extrema[0]
        (x1, y1) = prob.extrema[1]

        # ---- y-range filter ----
        seg_y_min = min(y0, y1)
        seg_y_max = max(y0, y1)
        if seg_y_min < y_min - tol or seg_y_max > y_max + tol:
            classification.append({"included": False, "side": None, "elements": None})
            continue

        # ---- x classification ----
        seg_x_min = min(x0, x1)
        seg_x_max = max(x0, x1)

        if abs(seg_x_min - midline_x) < tol and abs(seg_x_max - midline_x) < tol:
            # Vertical segment sitting exactly on the midline
            classification.append({"included": True, "side": "midline", "elements": None})
        elif seg_x_max < midline_x - tol:
            classification.append({"included": True, "side": "left", "elements": None})
        elif seg_x_min > midline_x + tol:
            classification.append({"included": True, "side": "right", "elements": None})
        else:
            # Horizontal segment crossing the midline – classify per element
            disc = setup.global_discretization.spatial_discretizations[d]
            dom_len = prob.domain_length
            elem_info: List[Dict] = []
            for k in range(disc.n_elements):
                s_left = disc.nodes[k]
                s_right = disc.nodes[k + 1]
                t_left = s_left / dom_len
                t_right = s_right / dom_len
                ex_left = x0 + t_left * (x1 - x0)
                ex_right = x0 + t_right * (x1 - x0)

                elem_x_min = min(ex_left, ex_right)
                elem_x_max = max(ex_left, ex_right)

                if elem_x_max <= midline_x + tol and elem_x_min < midline_x - tol:
                    elem_info.append({"side": "left", "frac_left": 1.0})
                elif elem_x_min >= midline_x - tol and elem_x_max > midline_x + tol:
                    elem_info.append({"side": "right", "frac_left": 0.0})
                else:
                    # Element truly straddles the midline
                    h_phys = abs(ex_right - ex_left)
                    frac = abs(midline_x - min(ex_left, ex_right)) / h_phys
                    elem_info.append({"side": "split", "frac_left": frac})

            classification.append({"included": True, "side": "straddle", "elements": elem_info})

    return classification


def compute_left_right_mass(
    bulk_data_list: list,
    classification: List[Dict],
    setup: "SolverSetup",
) -> Tuple[float, float]:
    """Compute the integral of u (equation 0) over the left and right halves.

    Uses the trapezoidal rule element-by-element, which is exact for
    piecewise linears:
        integral_k = (h / 2) * (u_left + u_right)

    For domains classified as 'midline' the integral is split 50/50.
    For straddling domains with 'split' elements, u is interpolated at
    the midline and each sub-interval is integrated separately.

    Args:
        bulk_data_list: List of BulkData objects, one per domain.
        classification: Output of :func:`classify_domains_left_right`.
        setup: The SolverSetup object.

    Returns:
        (mass_left, mass_right) – integrals of u over each half.
    """
    mass_left = 0.0
    mass_right = 0.0

    for d, info in enumerate(classification):
        if not info["included"]:
            continue

        bulk = bulk_data_list[d]
        disc = setup.global_discretization.spatial_discretizations[d]
        h = disc.element_length  # parameter-space element length = physical length
        n_elem = disc.n_elements

        side = info["side"]

        if side in ("left", "right"):
            # Whole domain belongs to one side
            domain_mass = 0.0
            for k in range(n_elem):
                u_L = bulk.data[0, k]
                u_R = bulk.data[1, k]
                domain_mass += 0.5 * h * (u_L + u_R)
            if side == "left":
                mass_left += domain_mass
            else:
                mass_right += domain_mass

        elif side == "midline":
            # Split 50 / 50
            domain_mass = 0.0
            for k in range(n_elem):
                u_L = bulk.data[0, k]
                u_R = bulk.data[1, k]
                domain_mass += 0.5 * h * (u_L + u_R)
            mass_left += 0.5 * domain_mass
            mass_right += 0.5 * domain_mass

        elif side == "straddle":
            elem_info = info["elements"]
            for k in range(n_elem):
                u_L = bulk.data[0, k]
                u_R = bulk.data[1, k]
                e = elem_info[k]
                if e["side"] == "left":
                    mass_left += 0.5 * h * (u_L + u_R)
                elif e["side"] == "right":
                    mass_right += 0.5 * h * (u_L + u_R)
                else:
                    # Element straddles – interpolate u at the midline
                    frac = e["frac_left"]  # fraction of h on the left
                    h_left = frac * h
                    h_right = (1.0 - frac) * h
                    u_mid = u_L + frac * (u_R - u_L)
                    mass_left += 0.5 * h_left * (u_L + u_mid)
                    mass_right += 0.5 * h_right * (u_mid + u_R)

    return mass_left, mass_right


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

    # --- Create per-run output directory ---
    run_dir = _create_run_output_dir(config_file)
    print(f"Run output directory: {run_dir}")
    
    # ============================================================================
    # STEP 1: SOLVER SETUP (Enhanced with config file support and error handling)
    # ============================================================================
    
    print("Step 1: Setting up solver...")
    
    # geometry = build_grid_geometry(N=2)
    _geom_dir = os.path.dirname(_geom_module.__file__)
    geometry = create_maze_geometry(
        data_dir=os.path.join(_geom_dir, "maze_3_data"),
        length=50.0,
    )   


    try:
        # Use quick_setup with both geometry and config file support
        setup = quick_setup(
            problem_module="bionetflux.problems.ooc_problem",
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
    
    # Initialize plotter (figures saved into the per-run output directory)
    plotter = LeanMatplotlibPlotter(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        equation_names=None,  # Auto-detect
        figsize=(15, 10),
        output_dir=run_dir
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
    
    # --- Mass monitoring setup (left / right halves of the maze) ---
    maze_length = 50.0  # must match the length used in create_maze_geometry
    domain_classification = classify_domains_left_right(setup, length=maze_length)
    
    mass_left_0, mass_right_0 = compute_left_right_mass(
        current_bulk_data, domain_classification, setup)
    mass_left_history = [mass_left_0]
    mass_right_history = [mass_right_0]
    print(f"Initial mass — left: {mass_left_0:.6e}, right: {mass_right_0:.6e}, "
          f"total: {mass_left_0 + mass_right_0:.6e}")
    print()
    
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
            
            # --- Mass monitoring ---
            ml, mr = compute_left_right_mass(
                current_bulk_data, domain_classification, setup)
            mass_left_history.append(ml)
            mass_right_history.append(mr)
            print(f"    Mass — left: {ml:.6e}, right: {mr:.6e}, "
                  f"total: {ml + mr:.6e}")
            
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
    
    print(f"\nStep 6: Creating visualizations (3 intermediate + final)...")

    # --- Intermediate birdview snapshots at T/4, T/2, 3T/4 ---
    snapshot_fractions = [0.25, 0.50, 0.75]
    t_arr_hist = np.array(time_history)

    for frac in snapshot_fractions:
        t_target = frac * T
        idx = int(np.argmin(np.abs(t_arr_hist - t_target)))
        t_snap = time_history[idx]
        snap_traces, _ = setup.extract_domain_solutions(solution_history[idx])
        label = f"{frac:.0%}".replace("%", "pct")
        for eq_idx in range(plotter.neq):
            plotter.plot_birdview(
                snap_traces,
                equation_idx=eq_idx,
                time=t_snap,
                save_filename=f"birdview_eq{eq_idx}_t{label}.png",
            )
        print(f"  ✓ Snapshot at t = {t_snap:.2f} (target {t_target:.2f})")

    # --- Final birdview ---
    for eq_idx in range(plotter.neq):
        plotter.plot_birdview(
            final_traces,
            equation_idx=eq_idx,
            time=current_time,
            save_filename=f"final_birdview_eq{eq_idx}.png",
        )
    print(f"  ✓ Final snapshot at t = {current_time:.2f}")
    
    # ============================================================================
    # STEP 7: MASS EVOLUTION PLOT
    # ============================================================================
    
    print(f"\nStep 7: Plotting mass evolution (left / right)...")
    
    t_arr = np.array(time_history)
    ml_arr = np.array(mass_left_history)
    mr_arr = np.array(mass_right_history)
    
    fig_mass, (ax_mass, ax_ratio) = plt.subplots(2, 1, sharex=True,
                                                  figsize=(10, 7))
    
    # -- Top subplot: absolute masses --
    ax_mass.plot(t_arr, ml_arr, label="Left half")
    ax_mass.plot(t_arr, mr_arr, label="Right half")
    ax_mass.plot(t_arr, ml_arr + mr_arr, '--', label="Total", color='gray')
    ax_mass.set_ylabel("Mass (integral of u)")
    ax_mass.set_title("Mass evolution — left vs right half of the maze")
    ax_mass.legend()
    ax_mass.grid(True, alpha=0.3)
    
    # -- Bottom subplot: asymmetry ratio (M_R - M_L) / (M_R + M_L) --
    total = ml_arr + mr_arr
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(total) > 1e-30,
                         (mr_arr - ml_arr) / total,
                         0.0)
    ax_ratio.plot(t_arr, ratio, color='tab:red')
    ax_ratio.axhline(0, color='black', linewidth=0.5)
    ax_ratio.set_xlabel("Time")
    ax_ratio.set_ylabel("(M_right − M_left) / M_total")
    ax_ratio.set_title("Left–right asymmetry ratio")
    ax_ratio.grid(True, alpha=0.3)
    
    fig_mass.tight_layout()
    mass_plot_path = os.path.join(run_dir, "mass_evolution.png")
    fig_mass.savefig(mass_plot_path, dpi=150)
    print(f"✓ Mass evolution plot saved to {mass_plot_path}")
    
    # Close all figures (non-blocking — no GUI wait)
    plotter.close_all()
    plt.close('all')
    print(f"\n✓ All figures saved to {run_dir} and closed.")
    
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

    _geom_dir = os.path.dirname(_geom_module.__file__)
    geometry = create_maze_geometry(
        data_dir=os.path.join(_geom_dir, "maze_3_data"),
        length=50.0,
    )

    try:
        setup = quick_setup(
            problem_module="bionetflux.problems.ooc_problem",
            validate=True,
            config_file=config_file,
            geometry=geometry,
        )
    except ValueError as e:
        if "not compatible with" in str(e) or "problem type" in str(e):
            print(f"\n❌ Configuration Error: {e}")
            return None, None, None, None, None
        raise
    except Exception as e:
        print(f"\n❌ Setup Error: {e}")
        return None, None, None, None, None

    info = setup.get_problem_info()
    print(f"✓ Problem loaded: {info['problem_name']}")
    print(f"  Domains: {info['num_domains']}")
    print(f"  Total DOFs: {info['total_trace_dofs'] + info['num_constraints']}")
    print(f"  Time discretization: dt={info['time_discretization']['dt']}, "
          f"T={info['time_discretization']['T']}")

    # ====================================================================
    # STEP 2: ADAPTIVE TIME STEPPER INITIALIZATION
    # ====================================================================

    print("\nStep 2: Initializing adaptive time stepper...")

    time_stepper = AdaptiveTimeStepper(
        setup,
        verbose=True,
        dt_min=dt_min,
        dt_max=dt_max,
        safety_factor=safety_factor,
    )

    current_solution, current_bulk_data = time_stepper.initialize_solution()
    print(f"✓ Adaptive time stepper initialized "
          f"(dt_min={time_stepper.dt_min:.6e}, dt_max={time_stepper.dt_max:.6e})")
    print(f"✓ Initial solution: shape {current_solution.shape}")
    print(f"✓ Initial bulk data: {len(current_bulk_data)} domains")

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
    print(f"✓ Plotter initialized for {plotter.ndom} domains, {plotter.neq} equations")

    setup.compute_geometry_from_problems()
    plotter.plot_geometry_with_indices(
        geometry=setup.geometry, save_filename="geometry_with_indices.png"
    )
    print("✓ Geometry plot created")

    # ====================================================================
    # STEP 4: TIME EVOLUTION (ADAPTIVE)
    # ====================================================================

    print("\nStep 4: Starting adaptive time evolution...")

    current_time = 0.0
    dt = setup.global_discretization.dt  # initial dt from config
    T = setup.global_discretization.T
    # Safety cap: independent of dt so adaptive shrinking cannot exhaust it.
    # With dt_min from the config this allows T/dt_min steps (far more than needed).
    max_time_steps = int(T / time_stepper.dt_min) + 100

    solution_history = [current_solution.copy()]
    time_history = [current_time]
    dt_history: List[float] = []  # dt actually used for each step

    # --- Mass monitoring setup ---
    maze_length = 50.0
    domain_classification = classify_domains_left_right(setup, length=maze_length)

    mass_left_0, mass_right_0 = compute_left_right_mass(
        current_bulk_data, domain_classification, setup
    )
    mass_left_history = [mass_left_0]
    mass_right_history = [mass_right_0]
    print(f"Initial mass — left: {mass_left_0:.6e}, right: {mass_right_0:.6e}, "
          f"total: {mass_left_0 + mass_right_0:.6e}")
    print(f"Time evolution: t ∈ [0, {T}], initial dt = {dt}")
    print()

    time_step = 0
    dt_suggested = dt

    while current_time < T - 1e-14 and time_step < max_time_steps:
        time_step += 1
        # Clip so we land exactly on T
        dt_try = min(dt_suggested, T - current_time)

        print(f"\n--- Time Step {time_step}: t = {current_time:.6f} → "
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

            print(f"  ✓ Time step successful!")
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

            # --- Mass monitoring ---
            ml, mr = compute_left_right_mass(
                current_bulk_data, domain_classification, setup
            )
            mass_left_history.append(ml)
            mass_right_history.append(mr)
            print(f"    Mass — left: {ml:.6e}, right: {mr:.6e}, "
                  f"total: {ml + mr:.6e}")
        else:
            print(f"  ✗ Time step failed after retries!")
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
        print(f"  Multipliers: ||λ|| = {np.linalg.norm(final_multipliers):.6e}")

    # ====================================================================
    # STEP 6: FINAL VISUALIZATION
    # ====================================================================

    print(f"\nStep 6: Creating visualizations (3 intermediate + final)...")

    # --- Intermediate birdview snapshots at T/4, T/2, 3T/4 ---
    snapshot_fractions = [0.25, 0.50, 0.75]
    t_arr_hist = np.array(time_history)

    for frac in snapshot_fractions:
        t_target = frac * T
        # Find the index of the closest saved time point
        idx = int(np.argmin(np.abs(t_arr_hist - t_target)))
        t_snap = time_history[idx]
        snap_traces, _ = setup.extract_domain_solutions(solution_history[idx])
        label = f"{frac:.0%}".replace("%", "pct")  # e.g. "25pct"
        for eq_idx in range(plotter.neq):
            plotter.plot_birdview(
                snap_traces,
                equation_idx=eq_idx,
                time=t_snap,
                save_filename=f"birdview_eq{eq_idx}_t{label}.png",
            )
        print(f"  ✓ Snapshot at t = {t_snap:.2f} (target {t_target:.2f})")

    # --- Final birdview ---
    for eq_idx in range(plotter.neq):
        plotter.plot_birdview(
            final_traces,
            equation_idx=eq_idx,
            time=current_time,
            save_filename=f"final_birdview_eq{eq_idx}.png",
        )
    print(f"  ✓ Final snapshot at t = {current_time:.2f}")

    # ====================================================================
    # STEP 7: MASS EVOLUTION PLOT
    # ====================================================================

    print(f"\nStep 7: Plotting mass evolution (left / right)...")

    t_arr = np.array(time_history)
    ml_arr = np.array(mass_left_history)
    mr_arr = np.array(mass_right_history)

    fig_mass, (ax_mass, ax_ratio) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    ax_mass.plot(t_arr, ml_arr, label="Left half")
    ax_mass.plot(t_arr, mr_arr, label="Right half")
    ax_mass.plot(t_arr, ml_arr + mr_arr, '--', label="Total", color='gray')
    ax_mass.set_ylabel("Mass (integral of u)")
    ax_mass.set_title("Mass evolution — left vs right (adaptive dt)")
    ax_mass.legend()
    ax_mass.grid(True, alpha=0.3)

    total = ml_arr + mr_arr
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(total) > 1e-30,
                         (mr_arr - ml_arr) / total, 0.0)
    ax_ratio.plot(t_arr, ratio, color='tab:red')
    ax_ratio.axhline(0, color='black', linewidth=0.5)
    ax_ratio.set_xlabel("Time")
    ax_ratio.set_ylabel("(M_right − M_left) / M_total")
    ax_ratio.set_title("Left–right asymmetry ratio")
    ax_ratio.grid(True, alpha=0.3)

    fig_mass.tight_layout()
    mass_plot_path = os.path.join(run_dir, "mass_evolution.png")
    fig_mass.savefig(mass_plot_path, dpi=150)
    print(f"✓ Mass evolution plot saved to {mass_plot_path}")

    # ====================================================================
    # STEP 8: DT EVOLUTION PLOT
    # ====================================================================

    if dt_history:
        print("\nStep 8: Plotting dt evolution...")
        fig_dt, ax_dt = plt.subplots(figsize=(10, 4))
        # Each dt_history[i] was used for the step ending at time_history[i+1]
        ax_dt.step(time_history[1:], dt_history, where='pre', color='tab:blue')
        ax_dt.set_xlabel("Time")
        ax_dt.set_ylabel("dt")
        ax_dt.set_yscale("log")
        ax_dt.set_title("Adaptive time step size")
        ax_dt.grid(True, alpha=0.3)
        fig_dt.tight_layout()
        dt_plot_path = os.path.join(run_dir, "dt_evolution.png")
        fig_dt.savefig(dt_plot_path, dpi=150)
        print(f"✓ dt evolution plot saved to {dt_plot_path}")

    # Close all figures
    plotter.close_all()
    plt.close('all')
    print(f"\n✓ All figures saved to {run_dir} and closed.")
    print(f"\n🎉 Adaptive evolution completed successfully!")

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
        python evolution_maze_ooc.py [config.toml]              # fixed dt
        python evolution_maze_ooc.py --adaptive [config.toml]   # adaptive dt
    """

    # Parse a simple --adaptive flag
    args = sys.argv[1:]
    use_adaptive = False
    if "--adaptive" in args:
        use_adaptive = True
        args.remove("--adaptive")

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
            result = run_evolution_with_adaptive_time_stepper(config_file)

            if result[0] is None:
                print(f"\n🛑 Stopping execution due to configuration error")
                sys.exit(1)

            setup, time_stepper, sol_history, time_hist, dt_hist = result
        else:
            result = run_evolution_with_time_stepper(config_file)

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
