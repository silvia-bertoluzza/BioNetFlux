#!/usr/bin/env python3
"""
Lean script for testing real initialization against MATLAB implementation using TimeStepper.
Shows how to use the new TimeStepper module for clean time evolution with HDG trace error evaluation.
Demonstrates dramatic simplification from ~50 lines of Newton iteration to single function calls.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from setup_solver import quick_setup
from bionetflux.time_integration import TimeStepper
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
from bionetflux.analysis.error_evaluation import ErrorEvaluator, create_analytical_solutions_example

# filename = "bionetflux.problems.reduced_ooc_problem"  # New geometry-based problem
filename = "bionetflux.problems.KS_traveling_wave"  # Original test_problem2 for MATLAB
    
print("="*60)
print("BIONETFLUX TIME STEPPER WITH HDG ERROR EVALUATION")
print("="*60)
print("Testing TimeStepper module with HDG trace error evaluation")

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
# STEP 2: Initialize TimeStepper (NEW!)
# =============================================================================
print("\nStep 2: Initializing TimeStepper...")

# Create TimeStepper with custom Newton solver configuration
time_stepper = TimeStepper(setup, verbose=True)

# Initialize solution at t=0 - REPLACES STEPS 2-4 from original!
current_solution, current_bulk_data = time_stepper.initialize_solution()

print("✓ TimeStepper initialized with solution at t=0")
print(f"  Initial solution: shape {current_solution.shape}")
print(f"  Initial bulk data: {len(current_bulk_data)} domains")

# Extract initial trace solutions for analysis
initial_traces, multipliers = setup.extract_domain_solutions(current_solution)
print("✓ Initial trace solutions extracted:")
for i, trace in enumerate(initial_traces):
    print(f"  Domain {i+1}: shape {trace.shape}, range [{np.min(trace):.6e}, {np.max(trace):.6e}]")

# =============================================================================
# STEP 3: Initialize visualization and error evaluation
# =============================================================================
print("\nStep 3: Initializing visualization and error evaluation...")

# Initialize the lean matplotlib plotter
plotter = LeanMatplotlibPlotter(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations,
    equation_names=None,  # Will auto-detect based on problem type
    figsize=(12, 8),
    output_dir="outputs/plots"  # Set directory for saving figures
)

# Initialize the L2 error evaluator
error_evaluator = ErrorEvaluator(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations
)

# HDG trace error configuration
alpha_hdg = 0.5  # HDG scaling parameter (h^0.5 scaling)
use_hdg_formulation = True  # Enable HDG trace error formulation

analytical_solutions = error_evaluator.get_analytical_solutions()
print("✓ HDG Error Evaluator initialized with automatically extracted analytical solutions")
print(f"  HDG formulation: {'Enabled' if use_hdg_formulation else 'Disabled'}")
print(f"  Scaling parameter α: {alpha_hdg}")

# Compute initial HDG trace error
print("\nComputing initial HDG trace error...")
initial_error_results = error_evaluator.compute_trace_error(
    numerical_solutions=initial_traces,
    time=0.0,
    analytical_functions=None,  # Use auto-extracted analytical solutions
    alpha=alpha_hdg,
    use_hdg_formulation=use_hdg_formulation
)

print("✓ Initial HDG trace error computed:")
print(f"  Global HDG Error: {initial_error_results['global_error']:.6e}")
print(f"  Relative Global Error: {initial_error_results.get('relative_global_error', 'N/A'):.6e}")

# Store error history for convergence analysis
error_history = [initial_error_results]
time_history_error = [0.0]

# =============================================================================
# STEP 4: Time Evolution using TimeStepper (MASSIVELY SIMPLIFIED!)
# =============================================================================
print("\nStep 4: Starting time evolution with TimeStepper...")

# Get time parameters
dt = setup.global_discretization.dt
T = info['time_discretization']['T']

print(f"Time evolution parameters:")
print(f"  Time step dt: {dt}")
print(f"  Final time T: {T}")
print(f"  Number of time steps: {int(T/dt)}")

# Initialize time evolution variables
time_step = 0
max_time_steps = int(T/dt) + 1  # Safety limit
solution_history = [current_solution.copy()]  # Store solution history
time_history = [0.0]  # Store time history
current_time = 0.0

print("✓ Starting time evolution loop - each step is ONE function call!")

# TIME EVOLUTION LOOP - SIMPLIFIED FROM ~50 LINES TO 1 LINE PER STEP!
while current_time + dt <= T and time_step < max_time_steps:
    time_step += 1
    print(f"\n--- Time Step {time_step}: t = {current_time:.6f} → {current_time + dt:.6f} ---")
    
    # SINGLE CALL REPLACES THE ENTIRE NEWTON ITERATION SECTION!
    result = time_stepper.advance_time_step(
        current_solution=current_solution,
        current_bulk_data=current_bulk_data,
        current_time=current_time,
        dt=dt
    )
    
    # Handle TimeStepper result
    if result.converged:
        print(f"  ✓ TimeStepper success: {result.iterations} Newton its, "
              f"||R|| = {result.final_residual_norm:.6e}")
        
        # Update state for next iteration
        current_time += dt
        current_solution = result.updated_solution
        current_bulk_data = result.updated_bulk_data
        
        # Store history
        solution_history.append(current_solution.copy())
        time_history.append(current_time)
        
    else:
        print(f"  ✗ TimeStepper failed: {result.iterations} Newton its, "
              f"||R|| = {result.final_residual_norm:.6e}")
        print("  Stopping time evolution due to convergence failure")
        print(result.summary())  # Detailed error information
        break

    # Compute HDG trace error at current time step
    current_traces, current_multipliers = setup.extract_domain_solutions(current_solution)
    current_error_results = error_evaluator.compute_trace_error(
        numerical_solutions=current_traces,
        time=current_time,
        analytical_functions=None,  # Use auto-extracted analytical solutions
        alpha=alpha_hdg,
        use_hdg_formulation=use_hdg_formulation
    )
    
    # Compute bulk error at current time step
    current_bulk_error_results = error_evaluator.compute_bulk_error(
        bulk_solutions=current_bulk_data,
        time=current_time
    )
    
    # Store error history
    error_history.append(current_error_results)
    time_history_error.append(current_time)
    
    # Print error information with HDG-specific details
    error_type = 'HDG' if use_hdg_formulation else 'L2'
    print(f"  {error_type} Trace Error: {current_error_results['global_error']:.6e} "
          f"(relative: {current_error_results.get('relative_global_error', 'N/A'):.6e})")
    print(f"  Bulk L2 Error: {current_bulk_error_results['global_error']:.6e} "
          f"(relative: {current_bulk_error_results.get('relative_global_error', 'N/A'):.6e})")
    
    # Print equation-specific HDG errors
    if use_hdg_formulation and len(current_error_results['global_error_per_equation']) > 0:
        print(f"  Per-equation {error_type} errors:")
        for eq_result in current_error_results['global_error_per_equation']:
            eq_idx = eq_result['equation_idx']
            error_key = 'global_hdg_error' if use_hdg_formulation else 'global_l2_error'
            print(f"    Eq {eq_idx+1}: {eq_result[error_key]:.6e}")

print("\n✓ Time evolution completed with TimeStepper!")
print(f"📊 TIMESTEPPER PERFORMANCE SUMMARY:")
print(f"   - Total time steps: {time_step}")
print(f"   - All Newton iterations handled automatically")
print(f"   - Automatic error handling and reporting")
print(f"   - Clean separation of concerns")
print(f"   - Detailed convergence information available")

# =============================================================================
# STEP 5: Extract Final Solutions and Analysis
# =============================================================================
print("\nStep 5: Extracting final trace solutions and creating analysis...")

# Extract final trace solutions from global solution
final_traces, final_multipliers = setup.extract_domain_solutions(current_solution)

print("✓ Final trace solutions extracted:")
for i, trace in enumerate(final_traces):
    print(f"  Domain {i+1}: shape {trace.shape}, range [{np.min(trace):.6e}, {np.max(trace):.6e}]")

print(f"✓ Final multipliers: shape {final_multipliers.shape}, range [{np.min(final_multipliers):.6e}, {np.max(final_multipliers):.6e}]")

# Show all plots
plotter.show_all()

# =============================================================================
# STEP 6: Solution Summary
# =============================================================================
print("\nStep 6: Solution analysis summary...")

# Print summary statistics
print("\nTimeStepper Evolution Summary:")
print(f"  Time evolution: t = 0 → {current_time:.4f}")
print(f"  Number of time steps completed: {time_step}")
print(f"  Final global solution norm: {np.linalg.norm(current_solution):.6e}")

n_domains = len(setup.problems)
n_equations = setup.problems[0].neq

for domain_idx in range(n_domains):
    print(f"\n  Domain {domain_idx + 1}:")
    n_nodes = len(setup.global_discretization.spatial_discretizations[domain_idx].nodes)
    
    for eq_idx in range(n_equations):
        eq_start = eq_idx * n_nodes
        eq_end = eq_start + n_nodes
        initial_values = initial_traces[domain_idx][eq_start:eq_end]
        final_values = final_traces[domain_idx][eq_start:eq_end]
        
        initial_norm = np.linalg.norm(initial_values)
        final_norm = np.linalg.norm(final_values)
        max_change = np.max(np.abs(final_values - initial_values))
        relative_change = max_change / (initial_norm + 1e-12)  # Avoid division by zero
        
        print(f"    Equation {eq_idx + 1}: ||u_initial||={initial_norm:.6e}, ||u_final||={final_norm:.6e}")
        print(f"                      Max change: {max_change:.6e}, Relative change: {relative_change:.6e}")

print(f"\n✓ Final solution analysis completed!")
print(f"✓ Matplotlib plots saved and displayed")

# =============================================================================
# STEP 7: Enhanced Error Analysis and Reporting with HDG
# =============================================================================
print("\nStep 7: Enhanced HDG error analysis and reporting...")

# Generate final HDG trace error report
final_error_report = error_evaluator.generate_error_report(error_history[-1])
print("HDG TRACE ERROR REPORT:")
print(final_error_report)

# Generate final bulk error report
final_bulk_error_results = error_evaluator.compute_bulk_error(
    bulk_solutions=current_bulk_data,
    time=current_time
)
final_bulk_error_report = error_evaluator.generate_error_report(final_bulk_error_results)
print("\nBULK ERROR REPORT:")
print(final_bulk_error_report)

# Enhanced error evolution plotting with HDG information
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Extract error data
global_errors = [err['global_error'] for err in error_history]
relative_errors = [err.get('relative_global_error', np.nan) for err in error_history]

# Plot 1: Global error evolution
axes[0].semilogy(time_history_error, global_errors, 'b-o', markersize=4)
axes[0].set_xlabel('Time')
error_label = f'Global {"HDG" if use_hdg_formulation else "L2"} Error'
if use_hdg_formulation:
    error_label += f' (h^{alpha_hdg} scaling)'
axes[0].set_ylabel(error_label)
axes[0].set_title('TimeStepper: HDG Trace Error Evolution' if use_hdg_formulation else 'TimeStepper: L2 Trace Error Evolution')
axes[0].grid(True, alpha=0.3)

# Plot 2: Relative error evolution
valid_relative = [err for err in relative_errors if not np.isnan(err) and not np.isinf(err)]
if valid_relative:
    axes[1].semilogy(time_history_error[:len(valid_relative)], valid_relative, 'r-s', markersize=4)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_title('TimeStepper: Relative Error Evolution')
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No valid relative errors', ha='center', va='center', transform=axes[1].transAxes)

# Plot 3: Per-equation error evolution
if use_hdg_formulation and len(error_history) > 1:
    n_equations = len(error_history[0]['global_error_per_equation'])
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for eq_idx in range(n_equations):
        eq_errors = []
        for err_result in error_history:
            if eq_idx < len(err_result['global_error_per_equation']):
                error_key = 'global_hdg_error' if use_hdg_formulation else 'global_l2_error'
                eq_errors.append(err_result['global_error_per_equation'][eq_idx][error_key])
            else:
                eq_errors.append(np.nan)
        
        valid_eq_errors = [err for err in eq_errors if not np.isnan(err)]
        if valid_eq_errors:
            color = colors[eq_idx % len(colors)]
            axes[2].semilogy(time_history_error[:len(valid_eq_errors)], valid_eq_errors, 
                           f'{color[0]}-', marker='o', markersize=3, label=f'Equation {eq_idx+1}')
    
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel(f'Per-Equation {"HDG" if use_hdg_formulation else "L2"} Error')
    axes[2].set_title('TimeStepper: Per-Equation Error Evolution')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
else:
    axes[2].text(0.5, 0.5, 'Per-equation data not available', ha='center', va='center', transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig("outputs/plots/timestepper_hdg_error_evolution.png", dpi=300, bbox_inches='tight')
print("✓ TimeStepper HDG error evolution plot saved")

print(f"\n🎉 TimeStepper example completed successfully!")
print(f"📊 KEY IMPROVEMENTS WITH TIMESTEPPER:")
print(f"   - Time advancement: 1 line instead of ~50 lines of Newton code")
print(f"   - Automatic error handling and detailed reporting")  
print(f"   - Clean separation of concerns between time stepping and error analysis")
print(f"   - Comprehensive convergence information in TimeStepResult")
print(f"   - Easy to extend with adaptive time stepping capabilities")
print(f"   - Maintainable and testable code structure")


