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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from setup_solver import quick_setup
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
from bionetflux.analysis.error_evaluation import ErrorEvaluator, create_analytical_solutions_example

# This example uses the KS traveling wave problem as test case
filename = "bionetflux.problems.KS_traveling_wave"  # Original test_problem2 for MATL
    
print("="*60)
print("BIONETFLUX REAL INITIALIZATION TEST")
print("="*60)
print("Testing initialization with HDG trace error evaluation")

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

# Newton method parameters (moved after global_solution creation)
max_newton_iterations = 20  # Limit to 20 iterations for debugging
newton_tolerance = 1e-10

print(f"  Newton method parameters:")
print(f"    Max iterations: {max_newton_iterations}")
print(f"    Tolerance: {newton_tolerance:.1e}")

# =============================================================================
# STEP 2: Create initial conditions
# =============================================================================
print("\nStep 2: Creating initial conditions...")
trace_solutions, multipliers = setup.create_initial_conditions()

print("✓ Initial trace solutions created:")

# Initialize the lean matplotlib plotter
print("\nInitializing LeanMatplotlibPlotter...")

plotter = LeanMatplotlibPlotter(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations,
    equation_names=None,  # Will auto-detect based on problem type
    figsize=(12, 8),
    output_dir="outputs/plots"  # Set directory for saving figures
)

# Initialize the L2 error evaluator
print("\nInitializing HDG-compatible L2 Error Evaluator...")
error_evaluator = ErrorEvaluator(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations
)

# HDG trace error configuration
alpha_hdg = np.array([-0.5, 0])  # HDG scaling parameter, P0-P1 KS discretization as in paper
use_hdg_formulation = True  # Enable HDG trace error formulation

# The analytical solutions are now automatically extracted from problems
analytical_solutions = error_evaluator.get_analytical_solutions()
print("✓ HDG Error Evaluator initialized with automatically extracted analytical solutions")
print(f"  HDG formulation: {'Enabled' if use_hdg_formulation else 'Disabled'}")


# Compute initial HDG trace error
print("\nComputing initial HDG trace error...")
initial_error_results = error_evaluator.compute_trace_error(
    numerical_solutions=trace_solutions,
    time=0.0,
    analytical_functions=None,  # Use auto-extracted analytical solutions
    alpha=alpha_hdg,
    use_hdg_formulation=use_hdg_formulation
)

print("✓ Initial HDG trace error computed:")
print(f"  Global HDG Error: {initial_error_results['global_error']:.6e}")
print(f"  Relative Global Error: {initial_error_results.get('relative_global_error', 'N/A'):.6e}")
print(f"  Error formulation: {initial_error_results.get('error_formulation', 'Standard L2')}")

# Store error history for convergence analysis
error_history = [initial_error_results]
bulk_error_history = []  # Add storage for bulk errors
time_history_error = [0.0]


# Print mesh information for HDG analysis

# =============================================================================
# STEP 3: Create global solution vector
# =============================================================================
print("\nStep 3: Assembling global solution vector...")
global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
print(f"✓ Global solution vector: shape {global_solution.shape}")
print(f"  Range: [{np.min(global_solution):.6e}, {np.max(global_solution):.6e}]")

newton_solution = global_solution.copy()  # Start with initial guess




# =============================================================================
# STEP 4.0: Initialize bulk data U(t=0.0)
# =============================================================================
print("\nStep 4: Creating bulk data and forcing terms...")
bulk_manager = setup.bulk_data_manager
bulk_solutions = []

bulk_guess = bulk_manager.initialize_all_bulk_data(problems=setup.problems,
                                                  discretizations=setup.global_discretization.spatial_discretizations,
                                                  time=0.0)

# Compute initial bulk error and store it
initial_bulk_error_results = error_evaluator.compute_bulk_error(
    bulk_solutions=bulk_guess,
    time=0.0
)
bulk_error_history.append(initial_bulk_error_results)


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



# Initialize multiplier section with constraint data at current time
if hasattr(setup, 'constraint_manager') and setup.constraint_manager is not None:
    n_trace_dofs = setup.global_assembler.total_trace_dofs
    n_multipliers = setup.constraint_manager.n_multipliers
    if n_multipliers > 0:
        # Get constraint data at current time and initialize multipliers
        constraint_data = setup.constraint_manager.get_multiplier_data(current_time)
        newton_solution[n_trace_dofs:] = constraint_data



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
    
    
    
    # Compute HDG trace error at current time step
    current_traces, current_multipliers = setup.extract_domain_solutions(global_solution)
    current_error_results = error_evaluator.compute_trace_error(
        numerical_solutions=current_traces,
        time=current_time,
        analytical_functions=None,  # Use auto-extracted analytical solutions
        alpha=alpha_hdg,
        use_hdg_formulation=use_hdg_formulation
    )
    
    # Compute bulk error at current time step
    current_bulk_error_results = error_evaluator.compute_bulk_error(
        bulk_solutions=bulk_guess,
        time=current_time
    )
    
    # Store error history
    error_history.append(current_error_results)
    bulk_error_history.append(current_bulk_error_results)  # Store bulk errors too
    time_history_error.append(current_time)

    # Print error information with HDG-specific details
    error_type = 'HDG' if use_hdg_formulation else 'L2'
    print(f"  {error_type} Trace Error: {current_error_results['global_error']:.6e} (relative: {current_error_results.get('relative_global_error', 'N/A'):.6e})")
    print(f"  Bulk L2 Error: {current_bulk_error_results['global_error']:.6e} (relative: {current_bulk_error_results.get('relative_global_error', 'N/A'):.6e})")
    
    # Print equation-specific HDG errors
    if use_hdg_formulation and len(current_error_results['global_error_per_equation']) > 0:
        print(f"  Per-equation {error_type} errors:")
        for eq_result in current_error_results['global_error_per_equation']:
            eq_idx = eq_result['equation_idx']
            error_key = 'global_hdg_error' if use_hdg_formulation else 'global_l2_error'
            print(f"    Eq {eq_idx+1}: {eq_result[error_key]:.6e}")

    print(f"✓ Newton solver completed")
    print(f"  Solution range: [{np.min(global_solution):.6e}, {np.max(global_solution):.6e}]")

print("  Time evolution completed.")

# =============================================================================
# STEP 6.6: Compute Equation-Dependent Global Error
# =============================================================================
print("\nStep 6.6: Computing equation-dependent global error...")

print("  Using stored trace and bulk errors from time evolution...")

# Get number of equations
n_equations = setup.problems[0].neq
equation_global_errors = []

for eq_idx in range(n_equations):
    print(f"  Processing equation {eq_idx + 1}/{n_equations}...")
    
    # Extract trace errors for this equation from stored error_history
    trace_errors_eq = []
    for trace_error_result in error_history:
        eq_trace_error = 0.0
        # Sum trace errors across all domains for this equation
        for domain_idx in range(len(setup.problems)):
            eq_error_data = error_evaluator.get_equation_error(trace_error_result, domain_idx, eq_idx)
            if eq_error_data:
                if use_hdg_formulation:
                    eq_trace_error += eq_error_data.get('hdg_error', eq_error_data['l2_error'])**2
                else:
                    eq_trace_error += eq_error_data['l2_error']**2
        trace_errors_eq.append(np.sqrt(eq_trace_error))
    
    # Extract bulk errors for this equation from stored bulk_error_history
    bulk_errors_eq = []
    for bulk_error_result in bulk_error_history:
        eq_bulk_error = 0.0
        # Sum bulk errors across all domains for this equation
        for domain_idx in range(len(setup.problems)):
            eq_error_data = error_evaluator.get_equation_error(bulk_error_result, domain_idx, eq_idx)
            if eq_error_data:
                eq_bulk_error += eq_error_data['l2_error']**2
        bulk_errors_eq.append(np.sqrt(eq_bulk_error))
    
    # Compute global error for this equation
    # Global error = max_t(trace_error_t) + ||dt * bulk_error||_2
    max_trace_error = max(trace_errors_eq) if trace_errors_eq else 0.0
    
    # Compute Euclidean norm of dt * bulk_errors
    dt_bulk_errors = np.array(bulk_errors_eq) * np.sqrt(dt)
    euclidean_bulk_norm = np.linalg.norm(dt_bulk_errors)
    
    # Total global error for this equation
    global_error_eq = max_trace_error + euclidean_bulk_norm
    
    equation_global_errors.append({
        'equation_idx': eq_idx,
        'max_trace_error': max_trace_error,
        'euclidean_bulk_norm': euclidean_bulk_norm,
        'global_error': global_error_eq,
        'trace_errors': trace_errors_eq,
        'bulk_errors': bulk_errors_eq
    })
    
    print(f"    Equation {eq_idx + 1}: max_trace = {max_trace_error:.6e}, ||dt*bulk||₂ = {euclidean_bulk_norm:.6e}")
    print(f"                       global_error = {global_error_eq:.6e}")

# Overall global error (sum over all equations)
total_global_error = sum(eq_data['global_error'] for eq_data in equation_global_errors)

print(f"\n✓ Equation-dependent global error computed:")
print(f"  Total global error: {total_global_error:.6e}")

# =============================================================================
# STEP 6.7: Enhanced Error Analysis and Reporting
# =============================================================================
print("\nStep 6.7: Enhanced error analysis and reporting...")

# Print detailed global error report
print("\n" + "="*60)
print("EQUATION-DEPENDENT GLOBAL ERROR REPORT")
print("="*60)
error_type = 'HDG' if use_hdg_formulation else 'L2'
print(f"Error formulation: {error_type} trace + L2 bulk")
print(f"Global error definition: max_t(trace_error_t) + ||dt * bulk_error||₂")
print(f"Time step dt: {dt:.6e}")
print(f"Total time steps: {len(time_history_error)}")
print("")

for eq_data in equation_global_errors:
    eq_idx = eq_data['equation_idx']
    print(f"Equation {eq_idx + 1}:")
    print(f"  Max trace error: {eq_data['max_trace_error']:.6e}")
    print(f"  Euclidean bulk norm: {eq_data['euclidean_bulk_norm']:.6e}")
    print(f"  Global error: {eq_data['global_error']:.6e}")
    if eq_data['global_error'] > 0:
        print(f"  Relative contribution - Trace: {eq_data['max_trace_error']/eq_data['global_error']*100:.1f}%, Bulk: {eq_data['euclidean_bulk_norm']/eq_data['global_error']*100:.1f}%")
    print("")

print(f"Total Global Error (sum): {total_global_error:.6e}")
print("="*60)

# Save equation-dependent global error data
global_error_data = []
for eq_data in equation_global_errors:
    row = [
        eq_data['equation_idx'],
        eq_data['max_trace_error'],
        eq_data['euclidean_bulk_norm'],
        eq_data['global_error']
    ]
    global_error_data.append(row)

global_error_array = np.array(global_error_data)
np.savetxt("outputs/plots/equation_global_errors.txt", global_error_array,
           header="Equation_Index\tMax_Trace_Error\tEuclidean_Bulk_Norm\tGlobal_Error",
           delimiter='\t', fmt='%.6e')
print("✓ Equation-dependent global errors saved to outputs/plots/equation_global_errors.txt")

# Plot global error components
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Global error components by equation
eq_indices = [eq_data['equation_idx'] + 1 for eq_data in equation_global_errors]
max_trace_errors = [eq_data['max_trace_error'] for eq_data in equation_global_errors]
euclidean_bulk_norms = [eq_data['euclidean_bulk_norm'] for eq_data in equation_global_errors]
global_errors = [eq_data['global_error'] for eq_data in equation_global_errors]

width = 0.25
x = np.arange(len(eq_indices))
axes[0,0].bar(x - width, max_trace_errors, width, label='Max Trace Error', alpha=0.8)
axes[0,0].bar(x, euclidean_bulk_norms, width, label='Euclidean Bulk Norm', alpha=0.8)
axes[0,0].bar(x + width, global_errors, width, label='Total Global Error', alpha=0.8)
axes[0,0].set_xlabel('Equation Index')
axes[0,0].set_ylabel('Error')
axes[0,0].set_title('Global Error Components by Equation')
axes[0,0].set_yscale('log')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(eq_indices)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Trace error evolution for all equations
for eq_data in equation_global_errors:
    eq_idx = eq_data['equation_idx']
    axes[0,1].semilogy(time_history_error, eq_data['trace_errors'], 
                      marker='o', markersize=3, label=f'Equation {eq_idx+1}')
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel(f'{error_type} Trace Error')
axes[0,1].set_title('Trace Error Evolution')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Bulk error evolution for all equations
for eq_data in equation_global_errors:
    eq_idx = eq_data['equation_idx']
    axes[1,0].semilogy(time_history_error, eq_data['bulk_errors'], 
                      marker='s', markersize=3, label=f'Equation {eq_idx+1}')
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('L2 Bulk Error')
axes[1,0].set_title('Bulk Error Evolution')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Global error comparison
axes[1,1].bar(eq_indices, global_errors, alpha=0.8, color='purple')
axes[1,1].set_xlabel('Equation Index')
axes[1,1].set_ylabel('Global Error')
axes[1,1].set_title('Total Global Error by Equation')
axes[1,1].set_yscale('log')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/plots/equation_global_error_analysis.png", dpi=300, bbox_inches='tight')
print("✓ Global error analysis plot saved")

# =============================================================================
# STEP 6.8: Enhanced Error Analysis and Reporting with HDG
# =============================================================================
# print("\nStep 6.8: Enhanced HDG error analysis and reporting...")

# # Generate final HDG trace error report
# final_error_report = error_evaluator.generate_error_report(error_history[-1])
# print("HDG TRACE ERROR REPORT:")
# print(final_error_report)

# # Generate final bulk error report
# final_bulk_error_results = error_evaluator.compute_bulk_error(
#     bulk_solutions=bulk_guess,
#     time=current_time-dt
# )
# final_bulk_error_report = error_evaluator.generate_error_report(final_bulk_error_results)
# print("\nBULK ERROR REPORT:")
# print(final_bulk_error_report)

# # Enhanced error evolution plotting with HDG information
# fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# # Extract error data
# global_errors = [err['global_error'] for err in error_history]
# relative_errors = [err.get('relative_global_error', np.nan) for err in error_history]

# # Plot 1: Global error evolution
# axes[0].semilogy(time_history_error, global_errors, 'b-o', markersize=4)
# axes[0].set_xlabel('Time')
# error_label = f'Global {"HDG" if use_hdg_formulation else "L2"} Error'
# if use_hdg_formulation:
#     error_label += f' (h^{alpha_hdg} scaling)'
# axes[0].set_ylabel(error_label)
# axes[0].set_title('HDG Trace Error Evolution' if use_hdg_formulation else 'L2 Trace Error Evolution')
# axes[0].grid(True, alpha=0.3)

# # Plot 2: Relative error evolution
# valid_relative = [err for err in relative_errors if not np.isnan(err) and not np.isinf(err)]
# if valid_relative:
#     axes[1].semilogy(time_history_error[:len(valid_relative)], valid_relative, 'r-s', markersize=4)
#     axes[1].set_xlabel('Time')
#     axes[1].set_ylabel('Relative Error')
#     axes[1].set_title('Relative Error Evolution')
#     axes[1].grid(True, alpha=0.3)
# else:
#     axes[1].text(0.5, 0.5, 'No valid relative errors', ha='center', va='center', transform=axes[1].transAxes)

# # Plot 3: Per-equation error evolution
# if use_hdg_formulation and len(error_history) > 1:
#     n_equations = len(error_history[0]['global_error_per_equation'])
#     colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
#     for eq_idx in range(n_equations):
#         eq_errors = []
#         for err_result in error_history:
#             if eq_idx < len(err_result['global_error_per_equation']):
#                 error_key = 'global_hdg_error' if use_hdg_formulation else 'global_l2_error'
#                 eq_errors.append(err_result['global_error_per_equation'][eq_idx][error_key])
#             else:
#                 eq_errors.append(np.nan)
        
#         valid_eq_errors = [err for err in eq_errors if not np.isnan(err)]
#         if valid_eq_errors:
#             color = colors[eq_idx % len(colors)]
#             axes[2].semilogy(time_history_error[:len(valid_eq_errors)], valid_eq_errors, 
#                            f'{color[0]}-', marker='o', markersize=3, label=f'Equation {eq_idx+1}')
    
#     axes[2].set_xlabel('Time')
#     axes[2].set_ylabel(f'Per-Equation {"HDG" if use_hdg_formulation else "L2"} Error')
#     axes[2].set_title('Per-Equation Error Evolution')
#     axes[2].grid(True, alpha=0.3)
#     axes[2].legend()
# else:
#     axes[2].text(0.5, 0.5, 'Per-equation data not available', ha='center', va='center', transform=axes[2].transAxes)

# plt.tight_layout()
# plt.savefig("outputs/plots/hdg_error_evolution.png", dpi=300, bbox_inches='tight')
# print("✓ HDG error evolution plot saved")

# # Enhanced error statistics summary
# print(f"\nEnhanced Error Statistics Summary:")
# print(f"  Error formulation: {error_history[0].get('error_formulation', 'Standard L2')}")
# if use_hdg_formulation:
#     print(f"  HDG scaling parameter α: {alpha_hdg}")
# print(f"  Initial global error: {global_errors[0]:.6e}")
# print(f"  Final global error: {global_errors[-1]:.6e}")
# print(f"  Maximum error during evolution: {max(global_errors):.6e}")
# print(f"  Minimum error during evolution: {min(global_errors):.6e}")

# if len(global_errors) > 1:
#     error_trend = (global_errors[-1] - global_errors[0]) / global_errors[0]
#     print(f"  Error trend (relative change): {error_trend:.2%}")

# # HDG convergence analysis (if multiple mesh sizes were available)
# print(f"\nMesh size analysis for HDG theory:")
# for i, discretization in enumerate(setup.global_discretization.spatial_discretizations):
#     h = setup.problems[i].domain_length / discretization.n_elements
#     final_domain_error = 0.0
    
#     # Get final error for this domain
#     if len(error_history) > 0:
#         final_result = error_history[-1]
#         for domain_result in final_result['domain_errors']:
#             if domain_result['domain_idx'] == i:
#                 # Sum errors over all equations in this domain
#                 for eq_error in domain_result['equation_errors']:
#                     if use_hdg_formulation:
#                         final_domain_error += eq_error.get('hdg_error', eq_error['l2_error'])**2
#                     else:
#                         final_domain_error += eq_error['l2_error']**2
#                 final_domain_error = np.sqrt(final_domain_error)
#                 break
    
#     print(f"  Domain {i+1}: h = {h:.6f}, final error = {final_domain_error:.6e}")


# # Save enhanced error history to file
# error_data_columns = [time_history_error, global_errors, relative_errors]
# column_headers = ["Time", f"Global_{'HDG' if use_hdg_formulation else 'L2'}_Error", "Relative_Error"]

# # Add per-equation errors if available
# if use_hdg_formulation and len(error_history) > 0:
#     n_equations = len(error_history[0]['global_error_per_equation'])
#     for eq_idx in range(n_equations):
#         eq_errors = []
#         for err_result in error_history:
#             if eq_idx < len(err_result['global_error_per_equation']):
#                 error_key = 'global_hdg_error' if use_hdg_formulation else 'global_l2_error'
#                 eq_errors.append(err_result['global_error_per_equation'][eq_idx][error_key])
#             else:
#                 eq_errors.append(np.nan)
#         error_data_columns.append(eq_errors)
#         column_headers.append(f"Equation_{eq_idx+1}_Error")

# error_data = np.column_stack(error_data_columns)
# np.savetxt("outputs/plots/hdg_error_history.txt", error_data, 
#            header="\t".join(column_headers), 
#            delimiter='\t', fmt='%.6e')
# print("✓ Enhanced error history saved to outputs/plots/hdg_error_history.txt")

# # Print final HDG-specific information
# if use_hdg_formulation:
#     print(f"\n📊 HDG TRACE ERROR SUMMARY:")
#     print(f"  Scaling formulation: error = h^{alpha_hdg} * ||pointwise_errors||₂")
#     print(f"  Theoretical convergence: error ∼ h^(α + p) where p is convergence order")
#     print(f"  Current α parameter: {alpha_hdg}")
#     print(f"  Expected optimal convergence for HDG: p ≈ k+1 (k = polynomial degree)")


