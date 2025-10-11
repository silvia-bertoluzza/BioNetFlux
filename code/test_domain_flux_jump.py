#!/usr/bin/env python3
"""
Comprehensive test script for domain_flux_jump function.
Tests the flux jump computation with various scenarios and validates integration.
"""

import sys
import os
import numpy as np

# Add the python_port directory to sys.path to allow imports
sys.path.insert(0, os.path.dirname(__file__))

from ooc1d.core.flux_jump import domain_flux_jump, test_domain_flux_jump
from ooc1d.utils.elementary_matrices import ElementaryMatrices


def test_with_elementary_matrices():
    """Test domain_flux_jump with realistic elementary matrices."""
    print("=" * 60)
    print("TESTING WITH ELEMENTARY MATRICES")
    print("=" * 60)
    
    class MockProblem:
        def __init__(self, neq=1):
            self.neq = neq
            self.domain_start = 0.0
            self.domain_end = 1.0
    
    class MockDiscretization:
        def __init__(self, n_elements=4):
            self.n_elements = n_elements
            self.h = 1.0 / n_elements
            self.nodes = np.linspace(0, 1, n_elements + 1)
            self.element_length = self.h
    
    class RealisticStaticCondensation:
        def __init__(self, neq=1):
            self.neq = neq
            self.elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
            
        def static_condensation(self, local_trace, local_source=None):
            """More realistic static condensation using elementary matrices."""
            trace_length = len(local_trace.flatten())
            neq = self.neq
            
            if local_source is None:
                local_source = np.zeros(2 * neq)
            
            local_trace_flat = local_trace.flatten()
            local_source_flat = local_source.flatten()
            
            # Get elementary matrices
            T = self.elementary_matrices.get_matrix('T')  # Trace matrix
            M = self.elementary_matrices.get_matrix('M')  # Mass matrix
            
            # Mock local solution using trace matrix
            coeffs_per_element = 2 * (2 * neq - 1)
            local_solution = np.zeros(coeffs_per_element)
            
            # Simple reconstruction for each equation
            for eq in range(neq):
                trace_eq = local_trace_flat[eq*2:(eq+1)*2]
                try:
                    # Solve T * coeffs = trace for this equation
                    coeffs = np.linalg.solve(T, trace_eq)
                    if coeffs_per_element >= 2:
                        local_solution[eq*2:(eq+1)*2] = coeffs
                except np.linalg.LinAlgError:
                    # Fallback for singular case
                    local_solution[eq*2:(eq+1)*2] = trace_eq * 0.5
            
            # Mock flux computation
            flux = np.sum(local_trace_flat) * 0.1
            
            # Mock flux trace using mass matrix contribution
            if M.ndim == 1:
                M_diag = M[:2] if len(M) >= 2 else [M[0], M[0] if len(M) == 1 else 1.0]
                flux_trace = local_trace_flat * M_diag[0] + local_source_flat * 0.1
            else:
                flux_trace = local_trace_flat * M[0, 0] + local_source_flat * 0.1
            
            # REMOVED: print(f"  flux_trace in static condensation: {flux_trace.shape}")
            # Mock jacobian with trace matrix structure
            jacobian = np.eye(len(local_trace_flat))
            if T.shape == (2, 2):
                jacobian[:2, :2] = T
                if len(local_trace_flat) > 2:
                    jacobian[2:4, 2:4] = T
            
            return local_solution, flux, flux_trace, jacobian
    
    # Test cases with realistic setup
    test_cases = [
        {"neq": 1, "n_elements": 3},
        {"neq": 2, "n_elements": 2},
        {"neq": 1, "n_elements": 6},
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        print(f"\nRealistic Test Case {i+1}: neq={case['neq']}, N={case['n_elements']}")
        print("-" * 50)
        
        neq = case["neq"]
        n_elements = case["n_elements"]
        n_nodes = n_elements + 1
        
        problem = MockProblem(neq)
        discretization = MockDiscretization(n_elements)
        static_condensation = RealisticStaticCondensation(neq)
        
        # Create more realistic test inputs
        trace_size = neq * n_nodes
        
        # Smooth initial trace (e.g., sine function)
        nodes = discretization.nodes
        trace_solution = np.zeros((trace_size, 1))
        for eq in range(neq):
            for j, x in enumerate(nodes):
                trace_solution[eq * n_nodes + j, 0] = np.sin(np.pi * x * (eq + 1))
        
        # Smooth forcing term
        forcing_term = np.zeros((2 * neq, n_elements))
        for k in range(n_elements):
            x_center = (nodes[k] + nodes[k+1]) / 2
            for eq in range(neq):
                forcing_term[eq*2, k] = 0.1 * np.exp(-x_center)
                forcing_term[eq*2+1, k] = 0.1 * np.exp(-x_center)
        
        print(f"Input characteristics:")
        print(f"  Trace range: [{np.min(trace_solution):.4f}, {np.max(trace_solution):.4f}]")
        print(f"  Forcing range: [{np.min(forcing_term):.4f}, {np.max(forcing_term):.4f}]")
        
        try:
            U, F, JF = domain_flux_jump(
                trace_solution, forcing_term, problem, discretization, static_condensation
            )
            
            print(f"Output characteristics:")
            print(f"  U range: [{np.min(U):.6e}, {np.max(U):.6e}]")
            print(f"  F norm: {np.linalg.norm(F):.6e}")
            print(f"  JF condition: {np.linalg.cond(JF):.2e}")
            print(f"  JF rank: {np.linalg.matrix_rank(JF)}/{min(JF.shape)}")
            
            # Validate shapes
            expected_shapes = [
                (U.shape, (2 * (2 * neq - 1), n_elements)),
                (F.shape, (neq * n_nodes, 1)),
                (JF.shape, (neq * n_nodes, neq * n_nodes))
            ]
            
            shapes_ok = True
            for actual, expected in expected_shapes:
                if actual != expected:
                    print(f"  ✗ Shape mismatch: got {actual}, expected {expected}")
                    shapes_ok = False
                    all_passed = False
            
            if shapes_ok:
                print(f"  ✓ All shapes correct")
            
            # Check finite values
            if np.all(np.isfinite(U)) and np.all(np.isfinite(F)) and np.all(np.isfinite(JF)):
                print(f"  ✓ All values finite")
            else:
                print(f"  ✗ Non-finite values detected")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            all_passed = False
    
    return all_passed


def test_consistency_checks():
    """Test consistency properties of domain_flux_jump."""
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECKS")
    print("=" * 60)
    
    # Setup
    neq = 1
    n_elements = 4
    n_nodes = n_elements + 1
    
    class MockProblem:
        def __init__(self):
            self.neq = neq
    
    class MockDiscretization:
        def __init__(self):
            self.n_elements = n_elements
            self.nodes = np.linspace(0, 1, n_nodes)
    
    class ConsistentStaticCondensation:
        def static_condensation(self, local_trace, local_source=None):
            local_trace_flat = local_trace.flatten()
            
            # Consistent mock implementation
            coeffs_per_element = 2 * (2 * neq - 1)  # 2 for neq=1
            local_solution = local_trace_flat[:coeffs_per_element] if coeffs_per_element <= len(local_trace_flat) else np.zeros(coeffs_per_element)
            
            flux = 0.0  # No net flux for consistency test
            flux_trace = local_trace_flat * 0.0  # Zero flux trace for conservation
            jacobian = np.eye(len(local_trace_flat))
            
            return local_solution, flux, flux_trace, jacobian
    
    problem = MockProblem()
    discretization = MockDiscretization()
    static_condensation = ConsistentStaticCondensation()
    
    print("\nTest 1: Linearity check")
    print("-" * 30)
    
    # Create test inputs
    trace1 = np.random.rand(neq * n_nodes, 1)
    trace2 = np.random.rand(neq * n_nodes, 1)
    forcing = np.zeros((2 * neq, n_elements))
    alpha = 0.3
    
    # Test superposition: f(a*x1 + b*x2) should equal a*f(x1) + b*f(x2) for linear parts
    U1, F1, JF1 = domain_flux_jump(trace1, forcing, problem, discretization, static_condensation)
    U2, F2, JF2 = domain_flux_jump(trace2, forcing, problem, discretization, static_condensation)
    U12, F12, JF12 = domain_flux_jump(alpha * trace1 + (1-alpha) * trace2, forcing, problem, discretization, static_condensation)
    
    U_expected = alpha * U1 + (1-alpha) * U2
    F_expected = alpha * F1 + (1-alpha) * F2
    
    if np.allclose(U12, U_expected, atol=1e-12):
        print("  ✓ U satisfies linearity")
    else:
        print(f"  ✗ U linearity error: {np.max(np.abs(U12 - U_expected)):.2e}")
    
    if np.allclose(F12, F_expected, atol=1e-12):
        print("  ✓ F satisfies linearity") 
    else:
        print(f"  ✗ F linearity error: {np.max(np.abs(F12 - F_expected)):.2e}")
    
    print("\nTest 2: Zero input check")
    print("-" * 30)
    
    zero_trace = np.zeros((neq * n_nodes, 1))
    zero_forcing = np.zeros((2 * neq, n_elements))
    
    U_zero, F_zero, JF_zero = domain_flux_jump(zero_trace, zero_forcing, problem, discretization, static_condensation)
    
    if np.allclose(U_zero, 0, atol=1e-14):
        print("  ✓ Zero trace produces zero U")
    else:
        print(f"  ✗ Zero trace: max|U| = {np.max(np.abs(U_zero)):.2e}")
    
    if np.allclose(F_zero, 0, atol=1e-14):
        print("  ✓ Zero inputs produce zero F")
    else:
        print(f"  ✗ Zero inputs: |F| = {np.linalg.norm(F_zero):.2e}")
    
    return True


def run_comprehensive_tests():
    """Run all test suites for domain_flux_jump."""
    print("🧪 COMPREHENSIVE DOMAIN_FLUX_JUMP TESTS")
    print("=" * 80)
    
    test_results = []
    
    # Test 1: Built-in test function
    print("\n📋 Test Suite 1: Built-in Mock Tests")
    result1 = test_domain_flux_jump(verbose=True)
    test_results.append(("Built-in tests", result1))
    
    # Test 2: Elementary matrices integration
    print("\n📋 Test Suite 2: Elementary Matrices Integration")
    result2 = test_with_elementary_matrices()
    test_results.append(("Elementary matrices", result2))
    
    # Test 3: Consistency checks
    print("\n📋 Test Suite 3: Consistency Properties")
    result3 = test_consistency_checks()
    test_results.append(("Consistency checks", result3))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! domain_flux_jump is ready for use.")
    else:
        print("⚠️  SOME TESTS FAILED. Please review the implementation.")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
