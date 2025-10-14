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

# Add the python_port directory to path for absolute imports
sys.path.insert(0, os.path.dirname(__file__))

from setup_solver import quick_setup

print("="*60)
print("BIONETFLUX REAL INITIALIZATION TEST")
print("="*60)
print("Testing initialization with test_problem2 for MATLAB comparison")

# =============================================================================
# STEP 1: Initialize the solver setup
# =============================================================================
print("\nStep 1: Initializing solver setup...")
setup = quick_setup("ooc1d.problems.ooc_test_problem", validate=True)
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
    print(f" DEBUG: Domain {i+1}: shape {trace.shape}, range [{np.min(trace):.6e}, {np.max(trace):.6e}]")

print(f"✓ Initial multipliers: shape {multipliers.shape}, values {multipliers}")

# =============================================================================
# DEBUG: Print static condensation matrices
# =============================================================================
print("\nDEBUG: Static condensation matrices:")
for i, static_cond in enumerate(setup.static_condensations):
    print(f"\nDomain {i+1} static condensation matrices:")
    if hasattr(static_cond, 'sc_matrices') and static_cond.sc_matrices:
        for matrix_name, matrix_value in static_cond.sc_matrices.items():
            print(f"  {matrix_name}: shape {matrix_value.shape}")
            print(f"    Range: [{np.min(matrix_value):.6e}, {np.max(matrix_value):.6e}]")
            print(f"    Values:\n{matrix_value}")
    else:
        print(f"  No sc_matrices found for domain {i+1}")

exit(0)  # Remove or comment out this line to proceed with the rest of the script

# =============================================================================
# STEP 3: Create global solution vector
# =============================================================================
print("\nStep 3: Assembling global solution vector...")
global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
print(f"✓ Global solution vector: shape {global_solution.shape}")
print(f"  Range: [{np.min(global_solution):.6e}, {np.max(global_solution):.6e}]")
print(f"  Values: {global_solution}")

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
    # print(f"  Domain {i+1} bulk guess values: {bulk.data}")


# =============================================================================
# STEP 4.1: Initialize source terms
# =============================================================================
source_terms = bulk_manager.compute_source_terms(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations,
    time=setup.global_discretization.dt # First time step - source for implicit Euler
)
print("✓ Source terms extracted:")

for i, ft in enumerate(source_terms):
    print(f"  Domain {i+1}: shape {ft.data.shape}, range [{np.min(ft.data):.6e}, {np.max(ft.data):.6e}]")
    # print(f"  Domain {i+1}: forcing term {ft.data}")



# =============================================================================
# STEP 4.2: Assembly of the right hand side for the static condensation
# =============================================================================
print("\nStep 4.2: Assembling right-hand side for static condensation...")
right_hand_side = []  # For clarity in this step
for i, (bulk_sol, source, static_cond) in enumerate(zip(bulk_guess, source_terms, setup.static_condensations)):
    rhs = static_cond.assemble_forcing_term(previous_bulk_solution=bulk_sol.data,
                                            external_force=source.data)
    print(f"  Domain {i+1} RHS: shape {rhs.shape}, range [{np.min(rhs):.6e}, {np.max(rhs):.6e}]")
    # print(f"  Domain {i+1} RHS values: {rhs}")
    right_hand_side.append(rhs)

print("✓ Right-hand sides assembled")

# =============================================================================
# STEP 5: Compute initial residual and Jacobian
# =============================================================================
print("\nStep 5: Computing initial residual and Jacobian...")
global_assembler = setup.global_assembler
time = 0.0

residual, jacobian = global_assembler.assemble_residual_and_jacobian(
    global_solution=global_solution,
    forcing_terms=right_hand_side,
    static_condensations=setup.static_condensations,
    time=time
)

print(f"✓ Residual computed: shape {residual.shape}")
print(f"  Residual norm: {np.linalg.norm(residual):.6e}")
print(f"  Residual range: [{np.min(residual):.6e}, {np.max(residual):.6e}]")
# print(f"  Residual values: {residual}")

print(f"✓ Jacobian computed: shape {jacobian.shape}")
print(f"  Jacobian condition number: {np.linalg.cond(jacobian):.6e}")
print(f"  Jacobian density: {np.count_nonzero(jacobian) / jacobian.size:.4f}")
print(f"  Jacobian range: [{np.min(jacobian):.6e}, {np.max(jacobian):.6e}]")
# print(f"  Jacobian values:\n{jacobian}")

# =============================================================================
# STEP 6: Solve nonlinear system F(x, b) = 0 using Newton method
# =============================================================================
print("\nStep 6: Solving nonlinear system F(x, b) = 0 using Newton method...")

# Newton method parameters
max_newton_iterations = 20
newton_tolerance = 1e-10
newton_solution = global_solution.copy()  # Start with initial guess

print(f"  Newton method parameters:")
print(f"    Max iterations: {max_newton_iterations}")
print(f"    Tolerance: {newton_tolerance:.1e}")
print(f"    Initial residual norm: {np.linalg.norm(residual):.6e}")

# Newton iteration loop
newton_converged = False
for newton_iter in range(max_newton_iterations):
    # Compute residual and Jacobian at current solution
    current_residual, current_jacobian = global_assembler.assemble_residual_and_jacobian(
        global_solution=newton_solution,
        forcing_terms=right_hand_side,
        static_condensations=setup.static_condensations,
        time=time
    )
    
    # Check convergence
    residual_norm = np.linalg.norm(current_residual)
    print(f"    Iteration {newton_iter + 1}: residual norm = {residual_norm:.6e}")
    
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
    
    print(f"      delta_x: {delta_x.shape}") 
    print(f"      newton_solution: {newton_solution.shape}") 



    # Update solution: x_{k+1} = x_k + delta_x
    newton_solution = newton_solution + delta_x
    
    # Optional: Print step size info
    step_norm = np.linalg.norm(delta_x)
    print(f"      Step norm: {step_norm:.6e}, Jacobian cond: {jacobian_cond:.2e}")

if not newton_converged:
    print(f"  ✗ Newton method did not converge after {max_newton_iterations} iterations")
    print(f"    Final residual norm: {np.linalg.norm(current_residual):.6e}")
else:
    # Final verification
    final_residual, final_jacobian = global_assembler.assemble_residual_and_jacobian(
        global_solution=newton_solution,
        forcing_terms=right_hand_side,
        static_condensations=setup.static_condensations,
        time=time
    )
    final_residual_norm = np.linalg.norm(final_residual)
    print(f"  ✓ Final verification: residual norm = {final_residual_norm:.6e}")
    
    # Update variables for subsequent steps
    global_solution = newton_solution
    residual = final_residual
    jacobian = final_jacobian

print(f"✓ Newton solver completed")
print(f"  Solution range: [{np.min(global_solution):.6e}, {np.max(global_solution):.6e}]")
print(f"  Final residual norm: {np.linalg.norm(residual):.6e}")



# =============================================================================
# STEP 7: Extract domain-specific data for MATLAB comparison
# =============================================================================
print("\nStep 7: Extracting domain-specific data...")

for i, (problem, discretization) in enumerate(zip(setup.problems, setup.global_discretization.spatial_discretizations)):
    print(f"\nDomain {i+1} data:")
    print(f"  Problem type: {problem.type}")
    print(f"  Number of equations: {problem.neq}")
    print(f"  Number of elements: {discretization.n_elements}")
    print(f"  Number of nodes: {discretization.n_elements + 1}")
    print(f"  Domain: [{problem.domain_start}, {problem.domain_end}]")
    print(f"  Node coordinates: {discretization.nodes}")
    print(f"  Trace solution: {trace_solutions[i]}")
    print(f"  Forcing term shape: {right_hand_side[i].shape}")

# =============================================================================
# STEP 8: Mathematical property checks
# =============================================================================
print("\nStep 8: Mathematical property checks...")

# Check 1: Round-trip consistency
round_trip_ok = True
for i, (orig, extracted) in enumerate(zip(trace_solutions, extracted_traces)):
    if not np.allclose(orig, extracted, rtol=1e-14):
        print(f"  ✗ Round-trip failed for domain {i+1}")
        round_trip_ok = False

if not np.allclose(multipliers, extracted_multipliers, rtol=1e-14):
    print(f"  ✗ Round-trip failed for multipliers")
    round_trip_ok = False

if round_trip_ok:
    print("  ✓ Round-trip consistency verified")


# Check 3: Jacobian properties
jacobian_symmetric = np.allclose(jacobian, jacobian.T, rtol=1e-12)
if jacobian_symmetric:
    print("  ✓ Jacobian is symmetric")
else:
    print("  ○ Jacobian is non-symmetric (expected for HDG)")

# =============================================================================
# STEP 9: Summary for MATLAB comparison
# =============================================================================
print("\n" + "="*60)
print("MATLAB COMPARISON SUMMARY")
print("="*60)

print(f"\nProblem configuration:")
print(f"  Name: {info['problem_name']}")
print(f"  Domains: {info['num_domains']}")
print(f"  Total elements: {info['total_elements']}")
print(f"  Total trace DOFs: {info['total_trace_dofs']}")
print(f"  Total system DOFs: {len(global_solution)}")

print(f"\nNumerical results:")
print(f"  Initial residual norm: {np.linalg.norm(residual):.10e}")
print(f"  Jacobian condition number: {np.linalg.cond(jacobian):.6e}")

print(f"\nData arrays for comparison:")
print(f"  Global solution: shape {global_solution.shape}")
print(f"  Residual: shape {residual.shape}")
print(f"  Jacobian: shape {jacobian.shape}")

# =============================================================================
# STEP 10: Export key arrays for external comparison
# =============================================================================
print("\nStep 10: Key arrays ready for export...")

# You can access these variables for MATLAB comparison:
print(f"\nKey variables available for inspection:")
print(f"  - global_solution: {global_solution.shape}")
print(f"  - residual: {residual.shape}")  
print(f"  - jacobian: {jacobian.shape}")
print(f"  - trace_solutions: {len(trace_solutions)} domains")
print(f"  - forcing_terms: {len(right_hand_side)} domains")
print(f"  - bulk_solutions: {len(bulk_solutions)} BulkData objects")

print(f"\n✓ Real initialization test completed successfully!")
print(f"✓ All data structures ready for MATLAB comparison")
print(f"✓ Use variables above to extract specific values for validation")

# Optional: Print first few values for quick verification
print(f"\nQuick verification (first 5 values):")
print(f"  global_solution[0:5] = {global_solution[:5]}")
print(f"  residual[0:5] = {residual[:5]}")
if len(trace_solutions) > 0:
    print(f"  trace_solutions[0][0:5] = {trace_solutions[0][:5]}")
