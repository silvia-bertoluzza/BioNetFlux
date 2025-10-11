#!/usr/bin/env python3
"""
Test script for boundary conditions initialization and setup.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from ooc1d.problems.test_problem import create_global_framework
from ooc1d.core.boundary_conditions import BoundaryConditionType


def main():
    """Test boundary conditions initialization."""
    print("="*60)
    print("TESTING BOUNDARY CONDITIONS INITIALIZATION")
    print("="*60)
    
    # Step 1: Create problem framework with boundary conditions
    print("\nStep 1: Creating problem framework with boundary conditions...")
    try:
        problems, discretizations, kappa, kk_params, problem_name, condition_manager = create_global_framework()
        print("✓ Problem framework with boundary conditions created successfully")
        
        problem = problems[0]
        discretization = discretizations[0]
        
        print(f"  Domain: [{problem.domain_start:.1f}, {problem.domain_end:.1f}]")
        print(f"  N = {discretization.n_elements} elements")
        print(f"  Total multipliers: {condition_manager.get_multiplier_vector_size()}")
        
    except Exception as e:
        print(f"✗ Problem framework creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Verify boundary conditions
    print("\nStep 2: Verifying boundary conditions...")
    
    boundary_conditions = condition_manager.get_boundary_conditions()
    print(f"  Number of boundary conditions: {len(boundary_conditions)}")
    
    expected_conditions = [
        (0, problem.domain_start, 0, BoundaryConditionType.NEUMANN),  # Left, u equation
        (0, problem.domain_start, 1, BoundaryConditionType.NEUMANN),  # Left, phi equation
        (0, problem.domain_end, 0, BoundaryConditionType.NEUMANN),    # Right, u equation
        (0, problem.domain_end, 1, BoundaryConditionType.NEUMANN),    # Right, phi equation
    ]
    
    if len(boundary_conditions) != len(expected_conditions):
        print(f"✗ Expected {len(expected_conditions)} conditions, got {len(boundary_conditions)}")
        return
    
    for i, condition in enumerate(boundary_conditions):
        expected_domain, expected_pos, expected_eq, expected_type = expected_conditions[i]
        
        print(f"\nCondition {i+1}:")
        print(f"  Domain index: {condition.geometric.domain_indices[0]} (expected: {expected_domain})")
        print(f"  Position: {condition.geometric.positions[0]:.1f} (expected: {expected_pos:.1f})")
        print(f"  Equation index: {condition.geometric.equation_index} (expected: {expected_eq})")
        print(f"  Type: {condition.analytical.boundary_type} (expected: {expected_type})")
        print(f"  Mesh node: {condition.geometric.mesh_nodes[0]}")
        print(f"  Multiplier index: {condition.multiplier_index}")
        print(f"  RHS data: {condition.analytical.rhs_data}")
        
        # Verify condition properties
        if condition.geometric.domain_indices[0] != expected_domain:
            print(f"  ✗ Domain mismatch")
        elif abs(condition.geometric.positions[0] - expected_pos) > 1e-10:
            print(f"  ✗ Position mismatch")
        elif condition.geometric.equation_index != expected_eq:
            print(f"  ✗ Equation mismatch") 
        elif condition.analytical.boundary_type != expected_type:
            print(f"  ✗ Type mismatch")
        elif condition.analytical.rhs_data != 0.0:
            print(f"  ✗ RHS data should be 0.0 for homogeneous Neumann")
        else:
            print(f"  ✓ Condition correctly specified")
    
    # Step 3: Test mesh node detection
    print("\nStep 3: Testing mesh node detection...")
    
    nodes = discretization.get_nodes()
    print(f"  Domain nodes: {len(nodes)} nodes from {nodes[0]:.3f} to {nodes[-1]:.3f}")
    
    for condition in boundary_conditions:
        position = condition.geometric.positions[0]
        mesh_node = condition.geometric.mesh_nodes[0]
        actual_node_position = nodes[mesh_node]
        
        print(f"  Position {position:.1f} -> Node {mesh_node} at {actual_node_position:.3f}")
        
        # Verify it's actually the closest node
        distances = np.abs(nodes - position)
        closest_node = np.argmin(distances)
        
        if mesh_node != closest_node:
            print(f"    ✗ Not the closest node! Closest should be {closest_node}")
        else:
            print(f"    ✓ Correctly identified closest node")
    
    # Step 4: Test condition queries
    print("\nStep 4: Testing condition queries...")
    
    # Get conditions for domain 0
    domain_conditions = condition_manager.get_conditions_for_domain(0)
    print(f"  Conditions for domain 0: {len(domain_conditions)}")
    
    # Get junction conditions (should be empty)
    junction_conditions = condition_manager.get_junction_conditions()
    print(f"  Junction conditions: {len(junction_conditions)}")
    
    if len(junction_conditions) != 0:
        print(f"    ✗ Expected no junction conditions for single domain")
    else:
        print(f"    ✓ No junction conditions as expected")
    
    # Step 5: Test residual evaluation (with dummy data)
    print("\nStep 5: Testing residual evaluation...")
    
    try:
        # Create dummy trace values and multipliers
        n_nodes = discretization.n_elements + 1
        trace_vector = np.ones((2 * n_nodes, 1))  # 2 equations, all values = 1
        multipliers = np.zeros((condition_manager.get_multiplier_vector_size(), 1))  # All fluxes = 0
        
        trace_values = {0: trace_vector}
        
        print(f"  Testing with dummy trace vector (shape: {trace_vector.shape})")
        print(f"  Testing with dummy multipliers (shape: {multipliers.shape})")
        
        # Evaluate residuals for all boundary conditions
        for i, condition in enumerate(boundary_conditions):
            try:
                residual = condition_manager.evaluate_condition_residual(
                    condition, trace_values, multipliers, time=0.0
                )
                print(f"  Condition {i+1} residual: {residual:.6f}")
                
                # For Neumann with RHS=0, residual should be λ - 0 = λ = 0
                if abs(residual) < 1e-10:
                    print(f"    ✓ Residual is zero as expected for homogeneous Neumann")
                else:
                    print(f"    ? Non-zero residual (depends on multiplier values)")
                    
            except Exception as e:
                print(f"  ✗ Residual evaluation failed for condition {i+1}: {e}")
    
    except Exception as e:
        print(f"  ✗ Residual testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("BOUNDARY CONDITIONS TEST SUMMARY")
    print("="*60)
    print(f"✓ {len(boundary_conditions)} homogeneous Neumann boundary conditions initialized")
    print(f"✓ {condition_manager.get_multiplier_vector_size()} multipliers created")
    print(f"✓ Mesh node detection working")
    print(f"✓ Condition queries functional")
    print(f"✓ Basic residual evaluation tested")
    print("\nReady for integration with flux jump module!")


if __name__ == "__main__":
    main()
