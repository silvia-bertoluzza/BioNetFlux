#!/usr/bin/env python3
"""
Lean script for testing real initialization against MATLAB implementation.
Shows how to initialize BioNetFlux problem step-by-step using test_problem2.
Simple linear script for easy interpretation and modification.
"""

# TODO: The integration of constraints in the whole process is not 100% clean. Check and improve
#       - Constraints attribute access is inconsistent
#       - Constraint handling in global assembler needs review
#       - Interface between setup.constraints and solver components unclear
#       - Need unified constraint management system

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Add the python_port directory to path for absolute imports
sys.path.insert(0, os.path.dirname(__file__))

from setup_solver import quick_setup
from ooc1d.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

filename = "ooc1d.problems.KS_grid_geometry"  # New geometry-based problem

print("="*60)
print("BIONETFLUX REAL INITIALIZATION TEST")
print("="*60)
print("Testing initialization with test_problem2 for MATLAB comparison")

# =============================================================================
# STEP 1: Initialize the solver setup
# =============================================================================
print("\nStep 1: Initializing solver setup...")
setup = quick_setup(filename, validate=True)
print("✓ Setup initialized and validated")

# Get problem information
info = setup.get_problem_info()
print(f"✓ Problem: {info['problem_name']}")
print(f"  Domains: {info['num_domains']}")
print(f"  Total elements: {info['total_elements']}")
print(f"  Total trace DOFs: {info['total_trace_dofs']}")

# Check if constraints attribute exists before accessing it
if hasattr(setup, 'constraints') and setup.constraints is not None:
    print(f"  Constraints: {info['num_constraints']}")
else:
    print(f"  Constraints: Not available (attribute missing)")

print(f"  Time discretization: dt={info['time_discretization']['dt']}, T={info['time_discretization']['T']}")


# =============================================================================
# STEP 2: Create initial conditions
# =============================================================================
print("\nStep 2: Creating initial conditions...")
trace_solutions, multipliers = setup.create_initial_conditions()

print("✓ Initial trace solutions created:")
for i, trace in enumerate(trace_solutions):
    print(f"  Domain {i+1}: shape {trace.shape}, range [{np.min(trace):.6e}, {np.max(trace):.6e}]")
    
    # Debug: Print solution values for each equation
    discretization = setup.global_discretization.spatial_discretizations[i]
    n_nodes = len(discretization.nodes)
    for eq_idx in range(setup.problems[0].neq):
        eq_start = eq_idx * n_nodes
        eq_end = eq_start + n_nodes
        eq_values = trace[eq_start:eq_end]
        eq_name = plotter.equation_names[eq_idx] if 'plotter' in locals() else f'Eq{eq_idx}'
        print(f"    {eq_name}: range [{np.min(eq_values):.6f}, {np.max(eq_values):.6f}]")
        if eq_idx == 1:  # omega should be sinusoidal
            print(f"    {eq_name} values (first 10): {eq_values[:10]}")

# Initialize the lean matplotlib plotter
print("\nInitializing LeanMatplotlibPlotter...")

plotter = LeanMatplotlibPlotter(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations,
    equation_names=None,  # Will auto-detect based on problem type
    figsize=(12, 8)
)

# Plot initial trace solutions

print("Plotting initial trace solutions...")

# 2D curve visualization (all equations together)
print("Creating 2D curve visualization...")
curves_2d_fig = plotter.plot_2d_curves(
    trace_solutions=trace_solutions,
    title="Initial Solutions - 2D Curves",
    show_bounding_box=True,
    show_mesh_points=True,
    save_filename="bionetflux_initial_2d_curves.png"
)

# Flat 3D visualization for each equation
for eq_idx in range(setup.problems[0].neq):
    flat_3d_fig = plotter.plot_flat_3d(
        trace_solutions=trace_solutions,
        equation_idx=eq_idx,
        title=f"Initial {plotter.equation_names[eq_idx]} Solution - Flat 3D",
        segment_width=0.1,
        save_filename=f"bionetflux_initial_{plotter.equation_names[eq_idx]}_flat3d.png",
        view_angle=(30, 45)
    )
    
    # Bird's eye view visualization
    birdview_fig = plotter.plot_birdview(
        trace_solutions=trace_solutions,
        equation_idx=eq_idx,
        segment_width=0.15,
        save_filename=f"bionetflux_initial_{plotter.equation_names[eq_idx]}_birdview.png",
        show_colorbar=True,
        time=0.0
    )


# =============================================================================
# STEP 3: Create global solution vector
# =============================================================================
print("\nStep 3: Assembling global solution vector...")
global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
print(f"✓ Global solution vector: shape {global_solution.shape}")
print(f"  Range: [{np.min(global_solution):.6e}, {np.max(global_solution):.6e}]")

# Test round-trip extraction
extracted_traces, extracted_multipliers = setup.extract_domain_solutions(global_solution)
print("✓ Round-trip extraction verified")
for i, (orig, ext) in enumerate(zip(trace_solutions, extracted_traces)):
    if np.allclose(orig, ext, rtol=1e-14):
        print(f"  Domain {i+1} trace extraction matches original")
    else:
        print(f"  ✗ Domain {i+1} trace extraction does NOT match original")
        

# =============================================================================
# STEP 4.0: Initialize bulk data U(t=0.0)
# =============================================================================
print("\nStep 4: Creating bulk data and forcing terms...")
bulk_manager = setup.bulk_data_manager
bulk_solutions = []

bulk_guess = bulk_manager.initialize_all_bulk_data(problems=setup.problems,
                                                  discretizations=setup.global_discretization.spatial_discretizations,
                                                  time=0.0)

for i, bulk in enumerate(bulk_guess):
    print(f"  Domain {i+1} bulk guess: shape {bulk.data.shape}, range [{np.min(bulk.data):.6e}, {np.max(bulk.data):.6e}]")
    
    

print("\nStep 5: Initializing global assembler...")
global_assembler = setup.global_assembler
time = 0.0

# =============================================================================
# STEP 6.5: Time Evolution Loop
# =============================================================================
print("\nStep 6.5: Starting time evolution...")

# Get time parameters
dt = setup.global_discretization.dt
T = info['time_discretization']['T']

print(f"    Time evolution parameters:")
print(f"    Time step dt: {dt}")
print(f"    Final time T: {T}")
print(f"    Number of time steps: {int(T/dt)}")

# Initialize time evolution variables
time_step = 1
max_time_steps = int(T/dt) + 1  # Safety limit
solution_history = [global_solution.copy()]  # Store solution history
time_history = [0.0]  # Store time history

current_time = 0.0

# Newton method parameters
max_newton_iterations = 20  # Limit to 20 iterations for debugging
newton_tolerance = 1e-10
newton_solution = global_solution.copy()  # Start with initial guess

# Initialize multiplier section with constraint data at current time
if hasattr(setup, 'constraint_manager') and setup.constraint_manager is not None:
    n_trace_dofs = setup.global_assembler.total_trace_dofs
    n_multipliers = setup.constraint_manager.n_multipliers
    if n_multipliers > 0:
        # Get constraint data at current time and initialize multipliers
        constraint_data = setup.constraint_manager.get_multiplier_data(current_time)
        newton_solution[n_trace_dofs:] = constraint_data

print(f"  Newton method parameters:")
print(f"    Max iterations: {max_newton_iterations}")
print(f"    Tolerance: {newton_tolerance:.1e}")
# Note: residual variable not defined in this scope, would need to use final_residual if available


# Time evolution loop
while current_time+dt <= T and time_step <= max_time_steps:
    print(f"\n--- Time Step {time_step}: t = {current_time+dt:.6f} ---")


    current_time += dt
    time_step += 1

    # Compute source terms at current time
    source_terms = bulk_manager.compute_source_terms(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        time=current_time
    )
    
    
    
    
    
    # Assemble right-hand side for static condensation
    right_hand_side = []  # For clarity in this step
    for i, (bulk_sol, source, static_cond) in enumerate(zip(bulk_guess, source_terms, setup.static_condensations)):
        rhs = static_cond.assemble_forcing_term(previous_bulk_solution=bulk_sol.data,
                                                external_force=source.data)
        right_hand_side.append(rhs)
        
    print("  ✓ Right-hand side assembled for static condensation")
    

   
    # Newton iteration loop
    newton_converged = False
    
    
    
    
    for newton_iter in range(max_newton_iterations):
        
        # Compute residual and Jacobian at current solution
        current_residual, current_jacobian = global_assembler.assemble_residual_and_jacobian(
            global_solution=newton_solution,
            forcing_terms=right_hand_side,
            static_condensations=setup.static_condensations,
            time=current_time
        )
    
        
        # Check convergence
        residual_norm = np.linalg.norm(current_residual)
        
        if residual_norm < newton_tolerance:
            print(f"  ✓ Newton method converged in {newton_iter + 1} iterations")
            newton_converged = True
            break
        
        # Check for singular Jacobian
        jacobian_cond = np.linalg.cond(current_jacobian)
        if jacobian_cond > 1e12:
            print(f"  ⚠ Warning: Jacobian poorly conditioned (cond = {jacobian_cond:.2e})")
        
        # Solve linear system: J * delta_x = -F
        try:
            delta_x = np.linalg.solve(current_jacobian, -current_residual)
        except np.linalg.LinAlgError as e:
            print(f"  ✗ Newton method failed: Linear system singular ({e})")
            break

        # Update solution: x_{k+1} = x_k + delta_x
        newton_solution = newton_solution + delta_x

    if not newton_converged:
        print(f"  ✗ Newton method did not converge after {max_newton_iterations} iterations")
        print(f"    Final residual norm: {np.linalg.norm(current_residual):.6e}")
    else:
        # Final verification
        final_residual, final_jacobian = global_assembler.assemble_residual_and_jacobian(
            global_solution=newton_solution,
            forcing_terms=right_hand_side,
            static_condensations=setup.static_condensations,
            time=current_time
        )
        final_residual_norm = np.linalg.norm(final_residual)
        print(f"  ✓ Final verification: residual norm = {final_residual_norm:.6e}")
    
    # Update variables for subsequent steps
    global_solution = newton_solution

    # Update bulk solutions by static condensation for next time step
    bulk_sol = global_assembler.bulk_by_static_condensation(
        global_solution=newton_solution,
        forcing_terms=right_hand_side,
        static_condensations=setup.static_condensations,
        time=current_time
    )

    # Update bulk_guess with new bulk solution data for next time step
    # bulk_sol contains the actual bulk solution arrays, not BulkData objects
    for i, new_bulk_data in enumerate(bulk_sol):
        # new_bulk_data should be a numpy array with the correct shape
    
        # Extract only the first 2*neq rows (bulk solution part)
        # neq = setup.problems[i].neq
        # bulk_data_only = new_bulk_data[:2*neq, :]
         
        # Directly set the data array (bypass BulkData.set_data validation)
        bulk_guess[i].data = new_bulk_data.copy()
    
    print(f"✓ Bulk solutions updated for next time step")
    for i, bulk in enumerate(bulk_guess):
        print(f"  Domain {i+1} updated bulk: shape {bulk.data.shape}, range [{np.min(bulk.data):.6e}, {np.max(bulk.data):.6e}]")

    print(f"✓ Newton solver completed")
    print(f"  Solution range: [{np.min(global_solution):.6e}, {np.max(global_solution):.6e}]")



print("  Time evolution completed.")
# Note: residual variable not defined in this scope, would need to use final_residual if available

# =============================================================================
# STEP 6.6: Extract Final Trace Solutions and Create Plots
# =============================================================================
print("\nStep 6.6: Extracting final trace solutions and creating plots...")

# Extract final trace solutions from global solution
final_traces, final_multipliers = setup.extract_domain_solutions(global_solution)

print("✓ Final trace solutions extracted:")
for i, trace in enumerate(final_traces):
    print(f"  Domain {i+1}: shape {trace.shape}, range [{np.min(trace):.6e}, {np.max(trace):.6e}]")

print(f"✓ Final multipliers: shape {final_multipliers.shape}, range [{np.min(final_multipliers):.6e}, {np.max(final_multipliers):.6e}]")

# Create plots of the final solution using LeanMatplotlibPlotter
print("\nCreating plots of final solution...")
n_equations = setup.problems[0].neq

# 2D curve visualization for final solutions
print("Creating 2D curve visualization for final solutions...")
final_curves_2d_fig = plotter.plot_2d_curves(
    trace_solutions=final_traces,
    title=f"Final Solutions - 2D Curves at t={current_time:.4f}",
    show_bounding_box=True,
    show_mesh_points=True,
    save_filename=f"bionetflux_final_2d_curves_t{current_time-dt:.4f}.png"
)

# Flat 3D visualization for final solutions
for eq_idx in range(n_equations):
    final_flat_3d_fig = plotter.plot_flat_3d(
        trace_solutions=final_traces,
        equation_idx=eq_idx,
        title=f"Final {plotter.equation_names[eq_idx]} Solution - Flat 3D at t={current_time:.4f}",
        segment_width=0.1,
        save_filename=f"bionetflux_final_{plotter.equation_names[eq_idx]}_flat3d_t{current_time-dt:.4f}.png",
        view_angle=(30, 45)
    )
    
    # Bird's eye view visualization for final solutions
    final_birdview_fig = plotter.plot_birdview(
        trace_solutions=final_traces,
        equation_idx=eq_idx,
        segment_width=0.15,
        save_filename=f"bionetflux_final_{plotter.equation_names[eq_idx]}_birdview_t{current_time-dt:.4f}.png",
        show_colorbar=True,
        time=current_time-dt
    )

# Solution evolution comparison
print("Creating solution evolution comparison...")
comparison_fig = plotter.plot_comparison(
    initial_traces=trace_solutions,
    final_traces=final_traces,
    initial_time=0.0,
    final_time=current_time-dt,
    save_filename=f"bionetflux_solution_comparison_t{current_time-dt:.4f}.png"
)

# Show all plots
plotter.show_all()

# =============================================================================
# STEP 6.7: Solution Summary (removed old matplotlib plotting)
# =============================================================================
print("\nStep 6.7: Solution analysis summary...")

# Print summary statistics
print("\nSolution Evolution Summary:")
print(f"  Time evolution: t = 0 → {current_time-dt:.4f}")
print(f"  Number of time steps completed: {time_step-1}")
print(f"  Final global solution norm: {np.linalg.norm(global_solution):.6e}")

n_domains = len(setup.problems)  # Define n_domains here
for domain_idx in range(n_domains):
    print(f"\n  Domain {domain_idx + 1}:")
    n_nodes = len(setup.global_discretization.spatial_discretizations[domain_idx].nodes)
    
    for eq_idx in range(n_equations):
        eq_start = eq_idx * n_nodes
        eq_end = eq_start + n_nodes
        initial_values = trace_solutions[domain_idx][eq_start:eq_end]
        final_values = final_traces[domain_idx][eq_start:eq_end]
        
        initial_norm = np.linalg.norm(initial_values)
        final_norm = np.linalg.norm(final_values)
        max_change = np.max(np.abs(final_values - initial_values))
        relative_change = max_change / (initial_norm + 1e-12)  # Avoid division by zero
        
        print(f"    Equation {eq_idx + 1}: ||u_initial||={initial_norm:.6e}, ||u_final||={final_norm:.6e}")
        print(f"                      Max change: {max_change:.6e}, Relative change: {relative_change:.6e}")

print(f"\n✓ Final solution analysis completed!")
print(f"✓ Matplotlib plots saved and displayed")


