import numpy as np
from typing import Tuple


def domain_flux_jump(
    trace_solution: np.ndarray,
    forcing_term: np.ndarray,
    problem, # dummy placeholder for backwards compatibility
    discretization, # dummy placeholder for backwards compatibility
    static_condensation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the local contribution to the flux balance equation for a domain.
    
    This is the Python equivalent of MATLAB fluxJump.m function.
    
    Args:
        trace_solution: Trial trace solution vector of length neq*(N+1) where N is number of elements
        forcing_term: (2*neq)×N matrix accounting for previous timestep and forcing terms
                     Format: stacked for all equations
        problem: Problem instance containing boundary conditions
        discretization: Discretization parameters
        static_condensation: Static condensation implementation
        
    Returns:
        tuple: (U, F, JF) where:
            - U: (2*(2*neq-1)×N matrix of bulk solutions for each element
            - F: neq*(N+1) vector of flux jumps at mesh points
            - JF: neq*(N+1)×neq*(N+1) Jacobian matrix
    """
    
    # Deduce N and neq from forcing_term shape
    N = forcing_term.shape[1]  # Number of elements
    neq = forcing_term.shape[0] // 2  # Number of equations
    n_nodes = N + 1
    
    # Initialize outputs
    coeffs_per_element = 2 * (2*neq - 1)  # Flexible: 2*(2*neq - 1) coefficients per element
    U = np.zeros((coeffs_per_element, N))  # 2*(2*neq - 1) × N matrix of bulk solutions
    F = np.zeros((neq * n_nodes, 1)) # Flexible: neq * (N+1) vector
    JF = np.zeros((neq * n_nodes, neq * n_nodes))
    
    # Set boundary conditions in F
    # F([1; N+1; N+2; 2*(N+1)]) = problem.NeumannData in MATLAB (1-indexed)
    # Convert to 0-indexed: [0; N; N+1; 2*N+1]
    
    # Cycle over elements
    for k in range(N):
        # Create logical indexing for element k
        # This selects nodes k and k+1 for all equations
        
        # Initialize local_indices as vector of length 2*neq
        local_indices = np.zeros(2 * neq, dtype=int)
        
        # Fill indices for each equation
        for ieq in range(neq):
            # For equation ieq: indices k and k+1 in the corresponding block
            local_indices[ieq * 2] = ieq * n_nodes + k      # Left node for equation ieq
            local_indices[ieq * 2 + 1] = ieq * n_nodes + (k + 1)  # Right node for equation ieq
        
        # Extract local forcing term for element k
        gk = forcing_term[:, k].reshape(-1, 1)  # (2*neq)×1 vector
        
        # Extract local trace values for element k
        local_trace = trace_solution[local_indices].reshape(-1, 1)  # (2*neq)×1 vector
        
        # Apply static condensation
        try:
            local_solution, flux, flux_trace, jacobian = static_condensation.static_condensation(
                local_trace, gk)
            flux_trace = flux_trace.reshape(-1, 1)  # Ensure column vector
            local_solution = local_solution.reshape(-1,)  # Ensure column vector
            
            # Store bulk solution for element k
            U[:, k] = local_solution
            
        
            # Update flux jump vector F
            # Add flux trace contributions to the global residual
            F[local_indices] += flux_trace
            
            
            # Update Jacobian JF
            # Add local Jacobian contributions to global Jacobian
            np.add.at(JF, (local_indices[:, None], local_indices[None, :]), jacobian)
            
        except Exception as e:
            raise RuntimeError(f"Static condensation failed for element {k+1}: {e}")
    
    return U, F, JF


def test_domain_flux_jump(verbose=True):
    """
    Test function for domain_flux_jump with mock objects.
    
    Args:
        verbose: If True, print detailed test information
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    if verbose:
        print("=" * 60)
        print("TESTING domain_flux_jump FUNCTION")
        print("=" * 60)
    
    try:
        # Create mock objects for testing
        class MockProblem:
            def __init__(self, neq=1):
                self.neq = neq
                self.domain_start = 0.0
                self.domain_end = 1.0
        
        class MockDiscretization:
            def __init__(self, n_elements=4):
                self.n_elements = n_elements
                self.nodes = np.linspace(0, 1, n_elements + 1)
                self.element_length = 1.0 / n_elements
        
        class MockStaticCondensation:
            def __init__(self, neq=1):
                self.neq = neq
                
            def static_condensation(self, local_trace, local_source=None):
                """Mock static condensation that returns predictable results."""
                trace_length = len(local_trace.flatten())
                neq = trace_length // 2
                
                if local_source is None:
                    local_source = np.zeros(2 * neq)
                
                local_trace_flat = local_trace.flatten()
                local_source_flat = local_source.flatten()
                
                # Mock local solution: 2*(2*neq-1) coefficients
                coeffs_per_element = 2 * (2 * neq - 1)
                local_solution = np.zeros(coeffs_per_element)
                
                # Fill with simple pattern for testing
                for i in range(min(coeffs_per_element, len(local_trace_flat))):
                    local_solution[i] = 0.8 * local_trace_flat[i] + 0.1 * (i + 1)
                
                # Mock flux: scalar for single equation case
                flux = np.sum(local_trace_flat) * 0.1
                
                # Mock flux_trace: same length as local_trace
                flux_trace = local_trace_flat * 0.9 + local_source_flat * 0.1
                
                # Mock jacobian
                jacobian = np.eye(len(local_trace_flat)) * 1.1 + 0.1
                
                return local_solution, flux, flux_trace, jacobian
        
        # Test parameters
        test_cases = [
            {"neq": 1, "n_elements": 3, "name": "Single equation, 3 elements"},
            {"neq": 2, "n_elements": 4, "name": "Two equations, 4 elements"},
            {"neq": 1, "n_elements": 5, "name": "Single equation, 5 elements"},
        ]
        
        all_passed = True
        
        for i, case in enumerate(test_cases):
            if verbose:
                print(f"\nTest Case {i+1}: {case['name']}")
                print("-" * 40)
            
            # Setup test case
            neq = case["neq"]
            n_elements = case["n_elements"]
            n_nodes = n_elements + 1
            
            problem = MockProblem(neq)
            discretization = MockDiscretization(n_elements)
            static_condensation = MockStaticCondensation(neq)
            
            # Create test inputs
            trace_size = neq * n_nodes
            trace_solution = np.random.rand(trace_size, 1) * 0.5
            forcing_term = np.random.rand(2 * neq, n_elements) * 0.2
            
            if verbose:
                print(f"  Input shapes:")
                print(f"    trace_solution: {trace_solution.shape}")
                print(f"    forcing_term: {forcing_term.shape}")
            
            # Test domain_flux_jump
            try:
                U, F, JF = domain_flux_jump(
                    trace_solution, forcing_term, problem, discretization, static_condensation
                )
                
                # Check output shapes
                expected_u_shape = (2 * (2 * neq - 1), n_elements)
                expected_f_shape = (neq * n_nodes, 1)
                expected_jf_shape = (neq * n_nodes, neq * n_nodes)
                
                shape_tests = [
                    (U.shape, expected_u_shape, "U shape"),
                    (F.shape, expected_f_shape, "F shape"),
                    (JF.shape, expected_jf_shape, "JF shape")
                ]
                
                case_passed = True
                for actual, expected, name in shape_tests:
                    if actual != expected:
                        if verbose:
                            print(f"    ✗ {name}: got {actual}, expected {expected}")
                        case_passed = False
                        all_passed = False
                    elif verbose:
                        print(f"    ✓ {name}: {actual}")
                
                # Check for NaN or infinite values
                arrays_to_check = [
                    (U, "U"),
                    (F, "F"), 
                    (JF, "JF")
                ]
                
                for arr, name in arrays_to_check:
                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                        if verbose:
                            print(f"    ✗ {name} contains NaN or infinite values")
                        case_passed = False
                        all_passed = False
                    elif verbose:
                        print(f"    ✓ {name} contains only finite values")
                
                # Check Jacobian properties
                if np.allclose(JF, 0):
                    if verbose:
                        print(f"    ⚠ JF is all zeros (may be expected for linear case)")
                elif verbose:
                    print(f"    ✓ JF has non-zero entries")
                
                if verbose:
                    if case_passed:
                        print(f"    ✓ Test case passed")
                    else:
                        print(f"    ✗ Test case failed")
                        
                    print(f"  Results summary:")
                    print(f"    |U|_max = {np.max(np.abs(U)):.6e}")
                    print(f"    |F|_norm = {np.linalg.norm(F):.6e}")
                    print(f"    JF condition = {np.linalg.cond(JF):.2e}")
                
            except Exception as e:
                if verbose:
                    print(f"    ✗ Exception during computation: {e}")
                case_passed = False
                all_passed = False
        
        # Test edge cases
        if verbose:
            print(f"\nEdge Case Tests:")
            print("-" * 40)
        
        # Test with zero trace
        try:
            zero_trace = np.zeros((neq * n_nodes, 1))
            U0, F0, JF0 = domain_flux_jump(
                zero_trace, forcing_term, problem, discretization, static_condensation
            )
            if verbose:
                print(f"  ✓ Zero trace test passed: |F| = {np.linalg.norm(F0):.6e}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Zero trace test failed: {e}")
            all_passed = False
        
        # Test with zero forcing
        try:
            zero_forcing = np.zeros((2 * neq, n_elements))
            Uzf, Fzf, JFzf = domain_flux_jump(
                trace_solution, zero_forcing, problem, discretization, static_condensation
            )
            if verbose:
                print(f"  ✓ Zero forcing test passed: |F| = {np.linalg.norm(Fzf):.6e}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Zero forcing test failed: {e}")
            all_passed = False
        
        if verbose:
            print("\n" + "=" * 60)
            if all_passed:
                print("✅ ALL TESTS PASSED")
            else:
                print("❌ SOME TESTS FAILED")
            print("=" * 60)
        
        return all_passed
        
    except Exception as e:
        if verbose:
            print(f"❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests when module is executed directly
    success = test_domain_flux_jump(verbose=True)
    exit(0 if success else 1)


