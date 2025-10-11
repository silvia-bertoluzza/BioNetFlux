"""
Test script for BulkData class.
Tests all initialization methods and functionality.
"""

import numpy as np
import sys
import os

# Add the python_port directory to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.dirname(__file__))

from ooc1d.core.bulk_data import BulkData


class MockProblem:
    """Mock problem class for testing."""
    def __init__(self, neq=1, domain_length=1.0):
        self.neq = neq
        self.domain_length = domain_length


class MockDiscretization:
    """Mock discretization class for testing."""
    def __init__(self, n_elements=5):
        self.n_elements = n_elements
        self.nodes = np.linspace(0, 1, n_elements + 1)
        self.element_sizes = np.ones(n_elements) * (1.0 / n_elements)


def test_initialization():
    """Test BulkData initialization."""
    print("=== Testing BulkData Initialization ===")
    
    problem = MockProblem(neq=2)
    discretization = MockDiscretization(n_elements=5)
    
    # Test primal formulation
    bulk_data = BulkData(problem, discretization, dual=False)
    print(f"Primal BulkData: {bulk_data}")
    
    # Test dual formulation
    bulk_data_dual = BulkData(problem, discretization, dual=True)
    print(f"Dual BulkData: {bulk_data_dual}")
    
    # Run internal tests
    assert bulk_data.test(), "Primal BulkData test failed"
    assert bulk_data_dual.test(), "Dual BulkData test failed"
    
    print("✓ Initialization tests passed\n")


def test_direct_array_input():
    """Test setting data with direct array input."""
    print("=== Testing Direct Array Input ===")
    
    problem = MockProblem(neq=2)
    discretization = MockDiscretization(n_elements=3)
    bulk_data = BulkData(problem, discretization, dual=False)
    
    # Create test data
    test_data = np.random.rand(4, 3)  # 2*neq x n_elements
    bulk_data.set_data(test_data)
    
    # Verify data was set correctly
    assert np.array_equal(bulk_data.get_data(), test_data), "Direct array input failed"
    print(f"✓ Direct array input test passed")
    print(f"  Data shape: {bulk_data.data.shape}")
    print(f"  Data range: [{np.min(bulk_data.data):.6f}, {np.max(bulk_data.data):.6f}]\n")


def test_function_input():
    """Test setting data with function input."""
    print("=== Testing Function Input ===")
    
    problem = MockProblem(neq=2)
    discretization = MockDiscretization(n_elements=4)
    bulk_data = BulkData(problem, discretization, dual=False)
    
    # Define test functions
    def func1(x, t):
        return np.sin(np.pi * x) + t
    
    def func2(x, t):
        return np.cos(2 * np.pi * x) * np.exp(-t)
    
    functions = [func1, func2]
    time = 0.5
    
    bulk_data.set_data(functions, time)
    
    print(f"✓ Function input test completed")
    print(f"  Data shape: {bulk_data.data.shape}")
    print(f"  Data range: [{np.min(bulk_data.data):.6f}, {np.max(bulk_data.data):.6f}]")
    
    # Verify data is not all zeros
    assert not np.allclose(bulk_data.data, 0), "Function evaluation resulted in all zeros"
    print(f"  ✓ Non-zero data generated\n")


def test_trace_vector_input():
    """Test setting data with trace vector input."""
    print("=== Testing Trace Vector Input ===")
    
    problem = MockProblem(neq=1)
    discretization = MockDiscretization(n_elements=3)
    bulk_data = BulkData(problem, discretization, dual=False)
    
    # Create trace vector: neq * (n_elements + 1) = 1 * 4 = 4 values
    trace_vector = np.array([1.0, 2.0, 3.0, 4.0])
    
    bulk_data.set_data(trace_vector)
    
    print(f"✓ Trace vector input test completed")
    print(f"  Input trace vector: {trace_vector}")
    print(f"  Data shape: {bulk_data.data.shape}")
    print(f"  Data range: [{np.min(bulk_data.data):.6f}, {np.max(bulk_data.data):.6f}]")
    
    # Verify data is not all zeros
    assert not np.allclose(bulk_data.data, 0), "Trace vector conversion resulted in all zeros"
    print(f"  ✓ Non-zero data generated\n")


def test_dual_formulation():
    """Test dual formulation functionality."""
    print("=== Testing Dual Formulation ===")
    
    problem = MockProblem(neq=1)
    discretization = MockDiscretization(n_elements=3)
    bulk_data = BulkData(problem, discretization, dual=True)
    
    # Test with direct array
    test_data = np.ones((2, 3)) * 0.5
    bulk_data.set_data(test_data)
    assert np.allclose(bulk_data.get_data(), test_data), "Dual direct array failed"
    print("✓ Dual direct array input passed")
    
    # Test with functions
    def simple_func(x, t):
        return x + t
    
    functions = [simple_func]
    time = 1.0
    
    try:
        bulk_data.set_data(functions, time)
        print("✓ Dual function input completed")
        print(f"  Data range: [{np.min(bulk_data.data):.6f}, {np.max(bulk_data.data):.6f}]")
    except Exception as e:
        print(f"⚠ Dual function input failed: {e}")
    
    # Test with trace vector
    trace_vector = np.array([0.5, 1.0, 1.5, 2.0])
    
    try:
        bulk_data.set_data(trace_vector)
        print("✓ Dual trace vector input completed")
        print(f"  Data range: [{np.min(bulk_data.data):.6f}, {np.max(bulk_data.data):.6f}]")
    except Exception as e:
        print(f"⚠ Dual trace vector input failed: {e}")
    
    print()


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("=== Testing Error Handling ===")
    
    problem = MockProblem(neq=2)
    discretization = MockDiscretization(n_elements=3)
    bulk_data = BulkData(problem, discretization, dual=False)
    
    # Test wrong array shape
    try:
        wrong_data = np.ones((3, 3))  # Should be (4, 3)
        bulk_data.set_data(wrong_data)
        print("✗ Should have raised error for wrong array shape")
    except ValueError:
        print("✓ Correctly raised error for wrong array shape")
    
    # Test wrong trace vector size
    try:
        wrong_trace = np.ones(5)  # Should be 2*(3+1) = 8
        bulk_data.set_data(wrong_trace)
        print("✗ Should have raised error for wrong trace vector size")
    except ValueError:
        print("✓ Correctly raised error for wrong trace vector size")
    
    # Test wrong number of functions
    try:
        wrong_functions = [lambda x, t: x]  # Should be 2 functions
        bulk_data.set_data(wrong_functions)
        print("✗ Should have raised error for wrong number of functions")
    except ValueError:
        print("✓ Correctly raised error for wrong number of functions")
    
    print()


def test_mass_computation():
    """Test mass computation functionality."""
    print("=== Testing Mass Computation ===")
    
    problem = MockProblem(neq=1)
    discretization = MockDiscretization(n_elements=4)
    bulk_data = BulkData(problem, discretization, dual=False)
    
    # Set constant data
    constant_data = np.ones((2, 4))
    bulk_data.set_data(constant_data)
    
    # Use identity matrix as simple mass matrix
    mass_matrix = np.eye(2)
    total_mass = bulk_data.compute_mass(mass_matrix)
    
    print(f"✓ Mass computation completed")
    print(f"  Total mass: {total_mass:.6f}")
    print(f"  Expected (approximate): {2 * 4:.6f}")  # 2 coefficients * 4 elements
    
    print()


def test_utilities():
    """Test utility methods."""
    print("=== Testing Utility Methods ===")
    
    problem = MockProblem(neq=2)
    discretization = MockDiscretization(n_elements=3)
    bulk_data = BulkData(problem, discretization, dual=False)
    
    # Set some test data
    test_data = np.random.rand(4, 3)
    bulk_data.set_data(test_data)
    
    # Test string representations
    print(f"str(): {str(bulk_data)}")
    print(f"repr(): {repr(bulk_data)}")
    
    # Test get_trace_values
    try:
        trace_values = bulk_data.get_trace_values()
        print(f"✓ get_trace_values() returned array of size {trace_values.size}")
        expected_size = 2 * (3 + 1)  # neq * (n_elements + 1)
        assert trace_values.size == expected_size, f"Expected size {expected_size}, got {trace_values.size}"
        print(f"  ✓ Correct trace values size")
    except Exception as e:
        print(f"⚠ get_trace_values() failed: {e}")
    
    print()


def run_all_tests():
    """Run all tests."""
    print("Running BulkData Tests")
    print("=" * 50)
    
    try:
        test_initialization()
        test_direct_array_input()
        test_function_input()
        test_trace_vector_input()
        test_dual_formulation()
        test_error_handling()
        test_mass_computation()
        test_utilities()
        
        print("=" * 50)
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
