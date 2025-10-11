#!/usr/bin/env python3
"""
Debug comparison tests between Python and MATLAB implementations.
Exports intermediate values at each step for systematic comparison.
"""

import sys
import os
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from setup_solver import quick_setup

def debug_initialization_comparison():
    """Test 1: Compare initialization between Python and MATLAB"""
    
    print("="*60)
    print("DEBUG TEST 1: INITIALIZATION COMPARISON")
    print("="*60)
    
    setup = quick_setup("ooc1d.problems.pure_parabolic", validate=True)
    
    # Export initialization data
    init_data = {}
    
    # Problem parameters
    problem = setup.problems[0]
    init_data['problem_parameters'] = problem.parameters
    init_data['problem_neq'] = problem.neq
    
    # Discretization parameters
    disc = setup.global_discretization.spatial_discretizations[0]
    init_data['n_elements'] = disc.n_elements
    init_data['element_length'] = disc.element_length
    init_data['domain_start'] = disc.domain_start
    init_data['domain_length'] = disc.domain_length
    init_data['nodes'] = disc.nodes
    init_data['tau'] = disc.tau
    
    # Time parameters
    init_data['dt'] = setup.global_discretization.dt
    init_data['T'] = setup.global_discretization.T
    
    # Elementary matrices
    elem_matrices = setup.static_condensations[0].elementary_matrices
    init_data['eM'] = elem_matrices.get_matrix('M')
    init_data['eD'] = elem_matrices.get_matrix('D')
    init_data['eT'] = elem_matrices.get_matrix('T')
    init_data['eIM'] = elem_matrices.get_matrix('IM')
    init_data['eMb'] = elem_matrices.get_matrix('Mb')
    init_data['eGb'] = elem_matrices.get_matrix('Gb')
    init_data['eNtil'] = elem_matrices.get_matrix('Ntil')
    init_data['eNhat'] = elem_matrices.get_matrix('Nhat')
    init_data['eQUAD'] = elem_matrices.get_matrix('QUAD')
    
    # Static condensation matrices
    sc_matrices = setup.static_condensations[0].sc_matrices
    for key, matrix in sc_matrices.items():
        init_data[f'sc_{key}'] = matrix
    
    # Initial conditions
    trace_solutions, multipliers = setup.create_initial_conditions()
    init_data['initial_trace_solutions'] = trace_solutions[0]
    init_data['initial_multipliers'] = multipliers
    
    # Global solution
    global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
    init_data['initial_global_solution'] = global_solution
    
    # Constraint data
    if hasattr(setup, 'constraint_manager') and setup.constraint_manager is not None:
        init_data['n_constraints'] = setup.constraint_manager.n_constraints
        init_data['n_multipliers'] = setup.constraint_manager.n_multipliers
        init_data['constraint_data_t0'] = setup.constraint_manager.get_multiplier_data(0.0)
    
    savemat('debug_test1_initialization.mat', init_data)
    print("✓ Initialization data exported to debug_test1_initialization.mat")
    
    return setup, global_solution, trace_solutions, multipliers

def debug_bulk_data_comparison(setup, time=0.0):
    """Test 2: Compare bulk data initialization and source terms"""
    
    print("\n" + "="*60)
    print("DEBUG TEST 2: BULK DATA AND SOURCE TERMS")
    print("="*60)
    
    bulk_data = {}
    
    # Initialize bulk data
    bulk_manager = setup.bulk_data_manager
    bulk_guess = bulk_manager.initialize_all_bulk_data(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        time=time
    )
    
    bulk_data['bulk_solution_data'] = bulk_guess[0].data
    bulk_data['bulk_solution_shape'] = bulk_guess[0].data.shape
    
    # Source terms at different times
    times = [0.0, 0.05, 0.1]
    for i, t in enumerate(times):
        source_terms = bulk_manager.compute_source_terms(
            problems=setup.problems,
            discretizations=setup.global_discretization.spatial_discretizations,
            time=t
        )
            
        bulk_data[f'source_terms_t{i}'] = source_terms[0].data
        
        # Right-hand side for static condensation
        rhs = setup.static_condensations[0].assemble_forcing_term(
            previous_bulk_solution=bulk_guess[0].data,
            external_force=source_terms[0].data
        )
        bulk_data[f'rhs_static_cond_t{i}'] = rhs
    
    savemat('debug_test2_bulk_data.mat', bulk_data)
    print("✓ Bulk data exported to debug_test2_bulk_data.mat")
    
    return bulk_guess

def debug_static_condensation_comparison(setup, global_solution, bulk_guess, time=0.05):
    """Test 3: Compare static condensation step by step"""
    
    print("\n" + "="*60)
    print("DEBUG TEST 3: STATIC CONDENSATION STEP-BY-STEP")
    print("="*60)
    
    sc_data = {}
    
    # Extract trace solution for first domain
    trace_solutions = setup.global_assembler.get_domain_solutions(global_solution)
    multipliers = setup.global_assembler.get_multipliers(global_solution)
    
    sc_data['input_trace_solution'] = trace_solutions[0]
    sc_data['input_multipliers'] = multipliers
    sc_data['time'] = time
    
    # Compute source terms and RHS
    source_terms = setup.bulk_data_manager.compute_source_terms(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        time=time
    )
    
    rhs = setup.static_condensations[0].assemble_forcing_term(
        previous_bulk_solution=bulk_guess[0].data,
        external_force=source_terms[0].data
    )
    
    sc_data['rhs_forcing_term'] = rhs
    
    # Get discretization info
    discretization = setup.global_discretization.spatial_discretizations[0]
    n_elements = discretization.n_elements
    neq = setup.problems[0].neq
    
    sc_data['n_elements'] = n_elements
    sc_data['neq'] = neq
    
    # Use flux_jump function (correct approach)
    from ooc1d.core.flux_jump import domain_flux_jump
    
    # Call flux_jump as done in global assembler
    global_trace = trace_solutions[0].reshape(-1, 1)
    U_flux_jump, F_flux_jump, JF_flux_jump = domain_flux_jump(
        global_trace,
        rhs,
        None, None,  # No constraints for this test
        setup.static_condensations[0]
    )
    
    sc_data['flux_jump_U'] = U_flux_jump
    sc_data['flux_jump_F'] = F_flux_jump
    sc_data['flux_jump_JF'] = JF_flux_jump
    
    # Also debug element-by-element static condensation
    static_cond = setup.static_condensations[0]
    
    # Store element-wise data
    element_data = {}
    
    for elem in range(n_elements):
        print(f"  Processing element {elem+1}/{n_elements}")
        
        # Extract local trace for this element
        # Element k has nodes k and k+1, so trace indices are [k, k+1] for each equation
        local_trace_indices = []
        for eq in range(neq):
            # For equation eq, nodes k and k+1 have global indices: eq*n_nodes + k, eq*n_nodes + (k+1)
            n_nodes = n_elements + 1
            local_trace_indices.extend([eq * n_nodes + elem, eq * n_nodes + (elem + 1)])
        
        # Extract local trace: [u_left, u_right, phi_left, phi_right] for element k
        local_trace = trace_solutions[0][local_trace_indices].reshape(-1, 1)
        
        # Extract local source (RHS) for this element 
        # RHS has shape (4, n_elements) - each column is for one element
        local_source = rhs[:, elem].reshape(-1, 1)
        
        # Store element input data
        element_data[f'elem_{elem}_local_trace'] = local_trace
        element_data[f'elem_{elem}_local_source'] = local_source
        element_data[f'elem_{elem}_trace_indices'] = np.array(local_trace_indices)
        
        # Manual step-by-step computation for first element only (for debugging)
        if elem == 0:
            # Extract g components as in MATLAB
            gu = local_source[[0, 1]]
            gp = local_source[[2, 3]]
            sc_data['elem0_gu'] = gu
            sc_data['elem0_gp'] = gp
            
            # Step-by-step computation (matching MATLAB StaticC.m)
            L1 = static_cond.sc_matrices['L1']
            L2 = static_cond.sc_matrices['L2']
            M = static_cond.sc_matrices['M']
            IM = static_cond.sc_matrices['IM']
            D = static_cond.sc_matrices['D']
            
            # g1 = L1 * gu
            g1 = L1 @ gu
            sc_data['elem0_g1'] = g1
            
            # CORRECTED: tgp = gp + dt * b * M * g1  (was missing dt)
            dt = setup.global_discretization.dt
            b = setup.problems[0].parameters[3]
            tgp = gp + dt * b * M @ g1
            sc_data['elem0_tgp'] = tgp
            sc_data['elem0_dt_b_term'] = dt * b * M @ g1
            
            # g2 = L2 * tgp
            g2 = L2 @ tgp
            sc_data['elem0_g2'] = g2
            
            # g3 = mu * IM * D * g2
            mu = setup.problems[0].parameters[0]
            g3 = mu * IM @ D @ g2
            sc_data['elem0_g3'] = g3
            
            # Complete static condensation for element 0
            local_solution, flux, flux_trace, jacobian = static_cond.static_condensation(
                local_trace, local_source
            )
            
            sc_data['elem0_local_solution'] = local_solution
            sc_data['elem0_flux'] = flux
            sc_data['elem0_flux_trace'] = flux_trace
            sc_data['elem0_jacobian'] = jacobian
            
            # Average phi and chi computation
            Av = static_cond.sc_matrices['Av']
            phi_avg = float(Av @ local_solution)
            chi_val = setup.problems[0].chi(phi_avg) if hasattr(setup.problems[0], 'chi') else 1.0;
            
            sc_data['elem0_phi_avg'] = phi_avg
            sc_data['elem0_chi_val'] = chi_val
        
        # Call static condensation for this element
        try:
            local_solution, flux, flux_trace, jacobian = static_cond.static_condensation(
                local_trace, local_source
            )
            
            element_data[f'elem_{elem}_solution'] = local_solution
            element_data[f'elem_{elem}_flux'] = flux
            element_data[f'elem_{elem}_flux_trace'] = flux_trace
            element_data[f'elem_{elem}_jacobian'] = jacobian
            
        except Exception as e:
            print(f"    Error in element {elem}: {e}")
            element_data[f'elem_{elem}_error'] = str(e)
    
    # Store all element data
    sc_data['element_data'] = element_data
    
    savemat('debug_test3_static_condensation.mat', sc_data)
    print("✓ Static condensation data exported to debug_test3_static_condensation.mat")
    
    return U_flux_jump, F_flux_jump, JF_flux_jump

def debug_newton_iteration_comparison(setup, global_solution, bulk_guess, time=0.05):
    """Test 4: Compare Newton iteration step by step"""
    
    print("\n" + "="*60)
    print("DEBUG TEST 4: NEWTON ITERATION COMPARISON")
    print("="*60)
    
    newton_data = {}
    
    # Compute right-hand side
    source_terms = setup.bulk_data_manager.compute_source_terms(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        time=time
    )
    
    rhs = setup.static_condensations[0].assemble_forcing_term(
        previous_bulk_solution=bulk_guess[0].data,
        external_force=source_terms[0].data
    )
    
    newton_data['newton_input_solution'] = global_solution
    newton_data['newton_rhs'] = rhs
    newton_data['newton_time'] = time
    
    # Compute residual and Jacobian
    residual, jacobian = setup.global_assembler.assemble_residual_and_jacobian(
        global_solution=global_solution,
        forcing_terms=[rhs],
        static_condensations=setup.static_condensations,
        time=time
    )
    
    # Construct zero trace solution with ones for multipliers for debugging
    zero_trace_ones_multiplier = global_solution.copy()
    n_trace_dofs = setup.global_assembler.total_trace_dofs
    n_multipliers = setup.global_assembler.n_multipliers
    
    # Set trace part to zero
    zero_trace_ones_multiplier[:n_trace_dofs] = 0.0
    # Set multiplier part to ones
    if n_multipliers > 0:
        zero_trace_ones_multiplier[n_trace_dofs:] = 1.0
    
    test_forcing_term = 0.0 * rhs
    test_solution = zero_trace_ones_multiplier
    
    test_residual, jacobian_copy = setup.global_assembler.assemble_residual_and_jacobian(
        global_solution=test_solution,
        forcing_terms=[test_forcing_term],
        static_condensations=setup.static_condensations,
        time=time
    )
    
    newton_data['test_residual'] = test_residual
    newton_data['test_solution'] = test_solution
    newton_data['test_forcing_term'] = test_forcing_term

    newton_data['newton_residual'] = residual
    newton_data['newton_jacobian'] = jacobian
    newton_data['residual_norm'] = np.linalg.norm(residual)
    newton_data['jacobian_cond'] = np.linalg.cond(jacobian)
    
    # Newton step
    try:
        delta_x = np.linalg.solve(jacobian, -residual)
        newton_data['newton_delta'] = delta_x
        newton_data['newton_step_success'] = True
    except:
        newton_data['newton_step_success'] = False
    
    savemat('debug_test4_newton_iteration.mat', newton_data)
    print("✓ Newton iteration data exported to debug_test4_newton_iteration.mat")

def debug_time_evolution_comparison(setup):
    """Test 5: Compare multiple time steps"""
    
    print("\n" + "="*60)
    print("DEBUG TEST 5: TIME EVOLUTION COMPARISON")
    print("="*60)
    
    # Initialize
    trace_solutions, multipliers = setup.create_initial_conditions()
    global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
    
    bulk_guess = setup.bulk_data_manager.initialize_all_bulk_data(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        time=0.0
    )
    
    dt = setup.global_discretization.dt
    times = [dt, 2*dt, 3*dt]  # First few time steps
    
    evolution_data = {
        'dt': dt,
        'times': np.array(times),
        'initial_solution': global_solution.copy()
    }
    
    current_solution = global_solution.copy()
    
    for step, time in enumerate(times):
        print(f"  Time step {step+1}: t = {time:.6f}")
        
        # Update multipliers with constraint data
        if hasattr(setup, 'constraint_manager') and setup.constraint_manager is not None:
            n_trace_dofs = setup.global_assembler.total_trace_dofs
            constraint_data = setup.constraint_manager.get_multiplier_data(time)
            current_solution[n_trace_dofs:] = constraint_data
        
        # Compute source and RHS
        source_terms = setup.bulk_data_manager.compute_source_terms(
            problems=setup.problems,
            discretizations=setup.global_discretization.spatial_discretizations,
            time=time
        )
        
        rhs = setup.static_condensations[0].assemble_forcing_term(
            previous_bulk_solution=bulk_guess[0].data,
            external_force=source_terms[0].data
        )
        
        # One Newton iteration
        residual, jacobian = setup.global_assembler.assemble_residual_and_jacobian(
            global_solution=current_solution,
            forcing_terms=[rhs],
            static_condensations=setup.static_condensations,
            time=time
        )
        
        # Store step data
        evolution_data[f'step_{step+1}_time'] = time
        evolution_data[f'step_{step+1}_solution_before'] = current_solution.copy()
        evolution_data[f'step_{step+1}_rhs'] = rhs
        evolution_data[f'step_{step+1}_residual'] = residual
        evolution_data[f'step_{step+1}_jacobian'] = jacobian
        evolution_data[f'step_{step+1}_residual_norm'] = np.linalg.norm(residual)
        
        # Newton update
        try:
            delta_x = np.linalg.solve(jacobian, -residual)
            current_solution = current_solution + delta_x
            evolution_data[f'step_{step+1}_delta'] = delta_x
            evolution_data[f'step_{step+1}_solution_after'] = current_solution.copy()
        except:
            evolution_data[f'step_{step+1}_newton_failed'] = True
            break
    
    savemat('debug_test5_time_evolution.mat', evolution_data)
    print("✓ Time evolution data exported to debug_test5_time_evolution.mat")

def main():
    """Run all debug tests"""
    
    print("Starting comprehensive debug comparison tests...")
    print("These will generate .mat files for comparison with MATLAB main.m")
    
    # Test 1: Initialization
    setup, global_solution, trace_solutions, multipliers = debug_initialization_comparison()
    
    # Test 2: Bulk data
    bulk_guess = debug_bulk_data_comparison(setup)
    
    # Test 3: Static condensation
    debug_static_condensation_comparison(setup, global_solution, bulk_guess)
    
    # Test 4: Newton iteration
    debug_newton_iteration_comparison(setup, global_solution, bulk_guess)
    
    # Test 5: Time evolution
    debug_time_evolution_comparison(setup)
    
    print("\n" + "="*60)
    print("DEBUG TESTS COMPLETED")
    print("="*60)
    print("Generated files:")
    print("  - debug_test1_initialization.mat")
    print("  - debug_test2_bulk_data.mat") 
    print("  - debug_test3_static_condensation.mat")
    print("  - debug_test4_newton_iteration.mat")
    print("  - debug_test5_time_evolution.mat")
    print("\nLoad these in MATLAB and compare with your main.m outputs.")
    print("Focus on the static condensation step (test 3) - the dt*b term was missing!")

if __name__ == "__main__":
    main()
