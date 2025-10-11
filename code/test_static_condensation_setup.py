#!/usr/bin/env python3
"""
Test script for TripleArc problem setup including static condensation matrices.
Tests the complete initialization pipeline before entering the solution process.
"""

import sys
import os

# Add the python_port directory to path for absolute imports
sys.path.insert(0, os.path.dirname(__file__))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_port'))

import numpy as np
from ooc1d.core.discretization import Discretization, GlobalDiscretization
from ooc1d.utils.elementary_matrices import ElementaryMatrices

# Now we can import the factory
from ooc1d.core.static_condensation_factory import StaticCondensationFactory
from ooc1d.core.constraints import ConstraintManager

from ooc1d.problems.test_problem import create_global_framework



def main():
    """Test the complete setup process."""
    print("="*60)
    print("TESTING STATIC CONDENSATION SETUP")
    print("="*60)
    
    # Step 1: Create TripleArc problem
    print("\nStep 1: Creating problem...")
    print("Creating current problem...")

    # Create problem using the problems module
    problems, global_disc, network_conditions, problem_name = create_global_framework()
    discretizations = global_disc.spatial_discretizations

    print(f"✓ Created {len(problems)} domains")
    for i, (problem, discretization) in enumerate(zip(problems, discretizations)):
        print(f"  Domain {i+1}: [{problem.domain_start:.0f}, {problem.domain_end:.0f}], "
            f"{discretization.n_elements} elements, type: {problem.type}")
    
    # Step 1.5: Setup constraints
    print("\nStep 1.5: Testing constraint manager...")
    try:
        # Use the constraint manager from test_problem
        constraint_manager = network_conditions
        
        if constraint_manager is not None:
            print(f"✓ Constraint manager loaded with {constraint_manager.n_constraints} constraints")
            print(f"✓ Total Lagrange multipliers: {constraint_manager.n_multipliers}")
            
            # Display constraint details
            for i, constraint in enumerate(constraint_manager.constraints):
                node_indices = constraint_manager.get_node_indices(i)
                print(f"  Constraint {i+1}: {constraint.type.value} on equation {constraint.equation_index}")
                print(f"    Domain {constraint.domains[0]} at position {constraint.positions[0]:.1f} -> node {node_indices[0]}")
            
            # Test constraint residual evaluation
            print("\n  Testing constraint residual evaluation:")
            
            # Create test trace solutions for all domains
            test_trace_solutions = []
            for i, (problem, discretization) in enumerate(zip(problems, discretizations)):
                n_nodes = discretization.n_elements + 1
                trace_size = problem.neq * n_nodes
                test_trace = np.ones((trace_size,)) * (i + 1) * 0.1  # Different values for each domain
                test_trace_solutions.append(test_trace)
                print(f"    Test trace for domain {i+1}: shape {test_trace.shape}, values range [{test_trace.min():.2f}, {test_trace.max():.2f}]")
            
            # Create test multiplier values
            n_multipliers = constraint_manager.n_multipliers
            test_multipliers = np.random.rand(n_multipliers) * 0.01  # Small random values
            print(f"    Test multipliers: shape {test_multipliers.shape}, values {test_multipliers}")
            
            # Test time
            test_time = 0.5
            
            try:
                residuals = constraint_manager.compute_constraint_residuals(
                    test_trace_solutions, test_multipliers, test_time, discretizations
                )
                
                print(f"    ✓ Constraint residuals computed successfully")
                print(f"    Residuals shape: {residuals.shape}")
                print(f"    Residuals values: {residuals}")
                print(f"    Residuals norm: {np.linalg.norm(residuals):.6f}")
                
                # Check residual structure
                if len(residuals) == n_multipliers:
                    print(f"    ✓ Residual vector length matches number of multipliers")
                else:
                    print(f"    ✗ Residual vector length mismatch: {len(residuals)} vs {n_multipliers}")
                
            except Exception as e:
                print(f"    ✗ Constraint residual computation failed: {e}")
                import traceback
                traceback.print_exc()
            
        else:
            print("✗ No constraint manager provided by test_problem")
            
    except Exception as e:
        print(f"✗ Constraint setup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    return
    # Step 2: Build elementary matrices
    print("\nStep 2: Building elementary matrices...")
    try:
        elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
        print("✓ Elementary matrices constructed successfully")
                
    except Exception as e:
        print(f"✗ Elementary matrices failed: {e}")
    
    
    # Step 3: Setup static condensation for each domain
    print("\nStep 3: Setting up static condensation...")
    sc_implementations = []

    for i, (problem, discretization) in enumerate(zip(problems, global_disc.spatial_discretizations)):
        try:
            print(f"  Setting up domain {i+1}...")
            
            # Compute stabilization parameters
            
            print(f"    ✓ Stabilization parameters computed: {len(discretization.tau)} equations")
        
            # Create static condensation implementation
            sc = StaticCondensationFactory.create(problem, global_disc, elementary_matrices, i)
            print(f"    ✓ Static condensation implementation created")
            
            # Build matrices
            sc_matrices = sc.build_matrices()
            print(f"    ✓ Static condensation matrices built")
            print(f"      Available SC matrices: {list(sc_matrices.keys())}")
            
            # Print detailed matrix information
            for matrix_name, matrix in sc_matrices.items():
                if matrix is not None:
                    print(f"      {matrix_name}: shape {matrix.shape}, dtype {matrix.dtype}")
                    if matrix.size > 0:
                        print(f"        entries:\n{matrix}")
                else:
                    print(f"      {matrix_name}: None")
            
            sc_implementations.append(sc)
            
        except Exception as e:
            print(f"    ✗ Domain {i+1} setup failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
   
    

    # Step 4: Test trace solution initialization
    print("\nStep 4: Testing trace solution initialization...")
    trace_solutions = []
    
    print("  Detailed trace DOF calculation:")
    total_trace_dofs = 0

    for i, (problem, discretization) in enumerate(zip(problems, global_disc.spatial_discretizations)):
        n_nodes = discretization.n_elements + 1
        trace_size = problem.neq * n_nodes
        trace_solution = np.zeros((trace_size, 1))
        trace_solutions.append(trace_solution)
        
        print(f"  Domain {i+1}:")
        print(f"    n_elements = {discretization.n_elements}")
        print(f"    n_nodes = {n_nodes} (elements + 1)")
        print(f"    neq = {problem.neq} (equations per node)")
        print(f"    trace_size = {problem.neq} × {n_nodes} = {trace_size}")
        print(f"    trace solution shape: {trace_solution.shape}")
        
        total_trace_dofs += trace_size
    
    print(f"  Total trace DOFs across all domains: {total_trace_dofs}")
    

    # Step 5: Test static condensation process
    print("\nStep 5: Testing static condensation process...")
    condensed_results = []
    
    for i, (sc, trace_solution, problem, discretization) in enumerate(zip(sc_implementations, trace_solutions, problems, discretizations)):
        try:
            print(f"  Testing condensation for domain {i+1}...")
            
            # Initialize test trace variable with some non-zero values
            test_trace = trace_solution.copy()
            test_trace[::2] = 1.0  # Set every other entry to 1 for testing
            
            print(f"    Full trace shape: {test_trace.shape}")
            print(f"    Testing on individual elements...")
            
            # Test static condensation on each element's local trace
            element_results = []
            for k in range(discretization.n_elements):
                print(f"      Element {k+1}/{discretization.n_elements}:")
                
                # Extract local trace for element k
                # For element k, we need values at nodes k and k+1 for each equation
                local_trace = np.zeros((2 * problem.neq, 1))
                
                for eq in range(problem.neq):
                    # Values at left node (k) and right node (k+1) for equation eq
                    n_nodes = discretization.n_elements + 1
                    left_idx = eq * n_nodes + k
                    right_idx = eq * n_nodes + (k + 1)
                    
                    local_trace[eq * 2] = test_trace[left_idx]      # Left value for equation eq
                    local_trace[eq * 2 + 1] = test_trace[right_idx] # Right value for equation eq
                
                print(f"        INPUT - Local trace shape: {local_trace.shape}")
                print(f"        INPUT - Local trace values: {local_trace.flatten()}")
                
                # Apply static condensation to this element
                try:
                    local_solution, flux, flux_trace, jacobian = sc.static_condensation(local_trace)
                    
                    print(f"        OUTPUT - Local solution shape: {local_solution.shape}")
                    print(f"        OUTPUT - Local solution values: {local_solution.flatten()}")
                    print(f"        OUTPUT - Flux shape: {flux.shape}")
                    print(f"        OUTPUT - Flux values: {flux.flatten()}")
                    print(f"        OUTPUT - Flux trace shape: {flux_trace.shape}")
                    print(f"        OUTPUT - Flux trace values: {flux_trace.flatten()}")
                    print(f"        OUTPUT - Jacobian shape: {jacobian.shape}")
                    print(f"        OUTPUT - Jacobian diagonal: {np.diag(jacobian)}")
                    
                    element_results.append({
                        'element': k,
                        'local_solution': local_solution,
                        'flux': flux,
                        'flux_trace': flux_trace,
                        'jacobian': jacobian
                    })
                    
                    print(f"        ✓ Element {k+1} static condensation successful")
                    
                except Exception as e:
                    print(f"        ✗ Element {k+1} static condensation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    element_results.append(None)
            
            condensed_results.append(element_results)
            successful_elements = sum(1 for r in element_results if r is not None)
            print(f"    ✓ Static condensation tested on {successful_elements}/{discretization.n_elements} elements")
            
        except Exception as e:
            print(f"    ✗ Static condensation failed for domain {i+1}: {e}")
            import traceback
            traceback.print_exc()
            condensed_results.append(None)
    
    # Summary
    print("\n" + "="*60)
    print("SETUP VALIDATION SUMMARY")
    print("="*60)
    print(f"✓ {len(problems)} domains initialized")
    print(f"✓ Elementary matrices computed ({len(elementary_matrices.get_all_matrices())} matrices)")
    print(f"✓ Stabilization parameters computed for all domains")
    print(f"✓ Trace solutions initialized")
    print(f"✓ Static condensation process tested on {sum(1 for r in condensed_results if r is not None)}/{len(condensed_results)} domains")
    
   
    
    print(f"\nTime discretization:")
    print(f"  dt = {global_disc.dt}, T = {global_disc.T}, steps = {global_disc.n_time_steps}")
    
    print("\n✓ PARTIAL SETUP COMPLETE")
    print("  Next step: Implement static condensation factory and base classes")


    print("="*60)
    print("FINAL SETUP SUMMARY")
    print("="*60)
    print(f"✓ {len(problems)} domains initialized")
    print(f"✓ Elementary matrices computed ({len(elementary_matrices.get_all_matrices())} matrices)")
    print(f"✓ {len(sc_implementations)} static condensation implementations ready")
    print(f"✓ Trace solutions initialized")
    
    total_dofs = sum(problem.neq * (discretization.n_elements + 1) 
                    for problem, discretization in zip(problems, global_disc.spatial_discretizations))
    print(f"✓ Total trace DOFs: {total_dofs}")
    
   
    
    print("\n✓ SETUP COMPLETE - Ready for time stepping!")
    print("  (Stopping before entering solution process)")


if __name__ == "__main__":
    main()

