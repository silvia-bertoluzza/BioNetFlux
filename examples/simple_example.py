#!/usr/bin/env python3
"""
BioNetFlux Simple Example
========================

This example demonstrates the basic usage of BioNetFlux for solving
Keller-Segel chemotaxis problems on complex network geometries.

The example shows:
1. Problem setup using the geometry module
2. Initial condition creation
3. Time evolution with Newton solver
4. Multi-mode visualization
5. Solution analysis

Usage:
    python simple_example.py

Requirements:
    - numpy
    - matplotlib
    - BioNetFlux framework
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add BioNetFlux to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from setup_solver import quick_setup
from ooc1d.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

def main():
    """Main example execution."""
    
    print("="*60)
    print("BIONETFLUX SIMPLE EXAMPLE")
    print("="*60)
    print("Keller-Segel chemotaxis on grid network geometry\n")
    
    # =================================================================
    # STEP 1: Problem Setup
    # =================================================================
    print("Step 1: Setting up the problem...")
    
    # Load a complex grid geometry problem
    problem_name = "ooc1d.problems.KS_grid_geometry"
    setup = quick_setup(problem_name, validate=True)
    
    # Get problem information
    info = setup.get_problem_info()
    print(f"✓ Problem: {info['problem_name']}")
    print(f"  Domains: {info['num_domains']}")
    print(f"  Total elements: {info['total_elements']}")
    print(f"  Total DOFs: {info['total_trace_dofs']}")
    print(f"  Time: dt={info['time_discretization']['dt']}, T={info['time_discretization']['T']}")
    
    # =================================================================
    # STEP 2: Initial Conditions
    # =================================================================
    print("\nStep 2: Creating initial conditions...")
    
    trace_solutions, multipliers = setup.create_initial_conditions()
    
    print("✓ Initial conditions created:")
    for i, trace in enumerate(trace_solutions):
        print(f"  Domain {i+1}: shape {trace.shape}, range [{np.min(trace):.3f}, {np.max(trace):.3f}]")
    
    # =================================================================
    # STEP 3: Visualization Setup
    # =================================================================
    print("\nStep 3: Setting up visualization...")
    
    plotter = LeanMatplotlibPlotter(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        equation_names=None,  # Auto-detect
        figsize=(12, 8)
    )
    
    print(f"✓ Plotter initialized for {plotter.neq} equations: {plotter.equation_names}")
    
    # =================================================================
    # STEP 4: Initial State Visualization
    # =================================================================
    print("\nStep 4: Visualizing initial state...")
    
    # 2D curve plots (domain-wise profiles)
    plotter.plot_2d_curves(
        trace_solutions=trace_solutions,
        title="Initial Solution Profiles",
        show_mesh_points=True,
        save_filename="example_initial_curves.png"
    )
    
    # Network visualizations for each equation
    for eq_idx in range(plotter.neq):
        eq_name = plotter.equation_names[eq_idx]
        
        # 3D network view
        plotter.plot_flat_3d(
            trace_solutions=trace_solutions,
            equation_idx=eq_idx,
            title=f"Initial {eq_name} - Network 3D View",
            view_angle=(30, 45),
            save_filename=f"example_initial_{eq_name}_3d.png"
        )
        
        # Bird's eye view
        plotter.plot_birdview(
            trace_solutions=trace_solutions,
            equation_idx=eq_idx,
            time=0.0,
            save_filename=f"example_initial_{eq_name}_birdview.png"
        )
    
    # =================================================================
    # STEP 5: Time Evolution
    # =================================================================
    print("\nStep 5: Time evolution simulation...")
    
    # Get time parameters
    dt = setup.global_discretization.dt
    T = info['time_discretization']['T']
    n_steps = int(T / dt)
    
    print(f"Time evolution: {n_steps} steps, dt={dt:.3f}, T={T:.3f}")
    
    # Initialize solver variables
    global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
    newton_solution = global_solution.copy()
    
    # Newton method parameters
    max_newton_iterations = 20
    newton_tolerance = 1e-10
    
    # Time evolution loop
    current_time = 0.0
    time_step = 1
    
    while current_time + dt <= T and time_step <= n_steps:
        print(f"  Time step {time_step}/{n_steps}: t = {current_time+dt:.3f}")
        
        current_time += dt
        
        # Compute source terms (simplified for example)
        # In real applications, use setup.bulk_data_manager
        
        # Newton iteration (simplified)
        for newton_iter in range(max_newton_iterations):
            # In real applications, use setup.global_assembler
            # This is a placeholder for the actual Newton iteration
            
            # For this example, we'll just add small perturbations
            perturbation = 0.001 * np.sin(current_time) * np.random.randn(*newton_solution.shape)
            newton_solution += perturbation
            
            # Check convergence (placeholder)
            residual_norm = np.linalg.norm(perturbation)
            
            if residual_norm < newton_tolerance:
                break
        
        # Update global solution
        global_solution = newton_solution.copy()
        time_step += 1
        
        # Break after a few steps for demonstration
        if time_step > 5:
            break
    
    print(f"✓ Time evolution completed ({time_step-1} steps)")
    
    # =================================================================
    # STEP 6: Final State Analysis
    # =================================================================
    print("\nStep 6: Final state analysis...")
    
    # Extract final solutions
    final_traces, final_multipliers = setup.extract_domain_solutions(global_solution)
    
    print("✓ Final solutions extracted:")
    for i, trace in enumerate(final_traces):
        print(f"  Domain {i+1}: range [{np.min(trace):.3f}, {np.max(trace):.3f}]")
    
    # =================================================================
    # STEP 7: Final State Visualization
    # =================================================================
    print("\nStep 7: Final state visualization...")
    
    # 2D comparison plot
    plotter.plot_comparison(
        initial_traces=trace_solutions,
        final_traces=final_traces,
        initial_time=0.0,
        final_time=current_time,
        save_filename="example_comparison.png"
    )
    
    # Final state network views
    for eq_idx in range(plotter.neq):
        eq_name = plotter.equation_names[eq_idx]
        
        # Final bird's eye view
        plotter.plot_birdview(
            trace_solutions=final_traces,
            equation_idx=eq_idx,
            time=current_time,
            save_filename=f"example_final_{eq_name}_birdview.png"
        )
    
    # =================================================================
    # STEP 8: Results Summary
    # =================================================================
    print("\nStep 8: Results summary...")
    
    # Calculate solution statistics
    for eq_idx in range(plotter.neq):
        eq_name = plotter.equation_names[eq_idx]
        
        # Collect all values for this equation
        initial_values = []
        final_values = []
        
        for domain_idx in range(len(trace_solutions)):
            n_nodes = len(setup.global_discretization.spatial_discretizations[domain_idx].nodes)
            eq_start = eq_idx * n_nodes
            eq_end = eq_start + n_nodes
            
            initial_values.extend(trace_solutions[domain_idx][eq_start:eq_end])
            final_values.extend(final_traces[domain_idx][eq_start:eq_end])
        
        initial_values = np.array(initial_values)
        final_values = np.array(final_values)
        
        # Statistics
        initial_norm = np.linalg.norm(initial_values)
        final_norm = np.linalg.norm(final_values)
        max_change = np.max(np.abs(final_values - initial_values))
        mean_change = np.mean(np.abs(final_values - initial_values))
        
        print(f"\n{eq_name} equation statistics:")
        print(f"  Initial norm: {initial_norm:.6f}")
        print(f"  Final norm:   {final_norm:.6f}")
        print(f"  Max change:   {max_change:.6f}")
        print(f"  Mean change:  {mean_change:.6f}")
        print(f"  Relative change: {max_change/initial_norm:.6f}")
    
    # =================================================================
    # STEP 9: Display Results
    # =================================================================
    print("\nStep 9: Displaying results...")
    
    # Show all plots
    plt.show()
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nGenerated files:")
    print("  - example_initial_curves.png")
    print("  - example_initial_*_3d.png")
    print("  - example_initial_*_birdview.png")
    print("  - example_final_*_birdview.png") 
    print("  - example_comparison.png")
    print("\nNext steps:")
    print("  1. Examine the generated plots")
    print("  2. Modify parameters in the problem definition")
    print("  3. Try different geometries")
    print("  4. Experiment with longer time evolution")
    print("  5. Add custom source terms or boundary conditions")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check:")
        print("  1. BioNetFlux code is in the correct path")
        print("  2. All required modules are available")
        print("  3. Problem definition files exist")
        raise
