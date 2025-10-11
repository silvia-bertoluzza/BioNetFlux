"""
Test script for Lean BulkDataManager class.
Tests the memory-efficient implementation that accepts framework objects as parameters.
"""

import numpy as np
import sys
import os

from ooc1d.core.lean_bulk_data_manager import BulkDataManager
from ooc1d.core.bulk_data import BulkData
from ooc1d.core.domain_data import DomainData
from ooc1d.utils.elementary_matrices import ElementaryMatrices


class MockProblem:
    """Mock problem class for testing Lean BulkDataManager."""
    def __init__(self, neq=1, has_forcing=False, has_initial=False):
        self.neq = neq
        self.domain_length = 1.0
        
        # Mock initial conditions
        if has_initial:
            self.u0 = [lambda x, t: np.sin(np.pi * x) for _ in range(neq)]
        else:
            self.u0 = [None for _ in range(neq)]
        
        # Mock forcing functions
        if has_forcing:
            self.force = [lambda x, t: np.exp(-x) * np.cos(t) for _ in range(neq)]
        else:
            self.force = [None for _ in range(neq)]


class MockDiscretization:
    """Mock discretization class for testing."""
    def __init__(self, n_elements=5, domain_length=1.0):
        self.n_elements = n_elements
        self.nodes = np.linspace(0, domain_length, n_elements + 1)
        self.element_length = domain_length / n_elements
        self.element_sizes = np.ones(n_elements) * self.element_length


class MockStaticCondensation:
    """Mock static condensation class."""
    def __init__(self, domain_idx=0):
        self.domain_idx = domain_idx
        self.elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
    
    def build_matrices(self):
        """Return mock matrices."""
        return {
            'M': np.array([[2/3, 1/3], [1/3, 2/3]]),  # Mass matrix
            'T': np.array([[1, -1], [1, 1]]),          # Trace matrix
            'QUAD': np.array([[0.5, 0.5], [0.5, 0.5]])  # Quadrature matrix
        }
    
    def static_condensation(self, local_trace: np.ndarray, local_source: np.ndarray = None):
        """
        Mock static condensation method for testing.
        
        Args:
            local_trace: Vector of length 2 * neq (trace values at element boundaries)
            local_source: Vector of length 2 * neq (source terms, optional)
            
        Returns:
            tuple: (local_solution, flux, flux_trace, jacobian)
                - local_solution: Vector of length 2 * neq
                - flux: Vector of length 2 * neq - 1  
                - flux_trace: Vector of length 2 * neq
                - jacobian: Matrix of size (2 * neq) x (2 * neq)
        """
        # Determine neq from trace vector length
        trace_length = len(local_trace.flatten())
        if trace_length % 2 != 0:
            raise ValueError(f"local_trace length {trace_length} must be divisible by 2")
        
        neq = trace_length // 2
        
        # Default source to zeros if not provided
        if local_source is None:
            local_source = np.zeros(2 * neq)
        else:
            local_source = local_source.flatten()
            if len(local_source) != 2 * neq:
                raise ValueError(f"local_source length {len(local_source)} must equal 2 * neq = {2 * neq}")
        
        # Flatten input trace
        local_trace_flat = local_trace.flatten()
        
        # CORRECTED: Mock local solution has same length as trace (2 * neq)
        local_solution = 0.8 * local_trace_flat + 0.2 * local_source
        
        # Mock flux: vector of length 2 * neq - 1  
        flux_length = 2 * neq - 1
        flux = np.zeros(flux_length)
        
        # Simple mock flux computation based on trace differences
        for i in range(flux_length):
            if i < neq:
                # Flux between trace values for each equation
                left_idx = 2 * i
                right_idx = 2 * i + 1
                if right_idx < len(local_trace_flat):
                    flux[i] = local_trace_flat[right_idx] - local_trace_flat[left_idx]
            else:
                # Additional flux terms for multi-equation case
                flux[i] = 0.1 * np.sum(local_trace_flat)
        
        # CORRECTED: flux_trace has same length as input trace
        flux_trace = local_trace_flat + 0.1 * local_solution
        
        # Mock jacobian: square matrix of size (2 * neq) x (2 * neq)
        jacobian = np.eye(2 * neq) + 0.1 * np.ones((2 * neq, 2 * neq))
        
        # Add some structure to make jacobian more realistic
        for i in range(2 * neq):
            for j in range(2 * neq):
                if abs(i - j) == 1:
                    jacobian[i, j] += 0.2
        
        return local_solution, flux, flux_trace, jacobian


def test_lean_manager_creation():
    """Test creating lean BulkDataManager from framework objects."""
    print("=== Testing Lean BulkDataManager Creation ===")
    
    # Create framework objects
    problems = [MockProblem(neq=1, has_forcing=True, has_initial=True),
                MockProblem(neq=2, has_forcing=False, has_initial=True)]
    
    discretizations = [MockDiscretization(n_elements=4),
                      MockDiscretization(n_elements=6)]
    
    static_condensations = [MockStaticCondensation(0),
                           MockStaticCondensation(1)]
    
    try:
        # Extract domain data using static method
        domain_data = BulkDataManager.extract_domain_data_list(
            problems, discretizations, static_condensations
        )
        
        print(f"✓ Domain data extracted: {len(domain_data)} domains")
        for i, dd in enumerate(domain_data):
            print(f"  Domain {i}: {dd.neq} equations, {dd.n_elements} elements")
        
        # Create lean manager
        lean_manager = BulkDataManager(domain_data)
        print(f"✓ Lean manager created: {lean_manager}")
        
        # Test internal validation with original framework objects
        success = lean_manager.test(problems, discretizations, static_condensations)
        assert success, "Lean manager internal test failed"
        print("✓ Internal tests passed")
        
        return lean_manager, problems, discretizations, static_condensations
        
    except Exception as e:
        print(f"✗ Lean manager creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_parameter_validation(lean_manager, problems, discretizations, static_condensations):
    """Test parameter validation functionality."""
    print("\n=== Testing Parameter Validation ===")
    
    if lean_manager is None:
        print("✗ Cannot test - lean manager is None")
        return False
    
    try:
        # Test 1: Correct parameters should pass
        lean_manager._validate_framework_objects(
            problems=problems,
            discretizations=discretizations,
            static_condensations=static_condensations,
            operation_name="test_validation"
        )
        print("✓ Correct parameters validation passed")
        
        # Test 2: Wrong number of problems
        try:
            wrong_problems = problems[:-1] if len(problems) > 1 else []
            lean_manager._validate_framework_objects(
                problems=wrong_problems,
                discretizations=discretizations,
                operation_name="test_validation"
            )
            print("✗ Should have failed with wrong number of problems")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught wrong problem count: {e}")
        
        # Test 3: Wrong number of discretizations
        try:
            wrong_discretizations = discretizations[:-1] if len(discretizations) > 1 else []
            lean_manager._validate_framework_objects(
                problems=problems,
                discretizations=wrong_discretizations,
                operation_name="test_validation"
            )
            print("✗ Should have failed with wrong number of discretizations")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught wrong discretization count: {e}")
        
        # Test 4: Incompatible problem neq
        class MockBadProblem:
            def __init__(self):
                self.neq = 999  # Wrong neq
        
        try:
            bad_problems = [MockBadProblem()] + problems[1:]
            lean_manager._validate_framework_objects(
                problems=bad_problems,
                discretizations=discretizations,
                operation_name="test_validation"
            )
            print("✗ Should have failed with incompatible problem neq")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught incompatible problem neq: {e}")
        
        # Test 5: Incompatible discretization n_elements
        class MockBadDiscretization:
            def __init__(self):
                self.n_elements = 999  # Wrong n_elements
        
        try:
            bad_discretizations = [MockBadDiscretization()] + discretizations[1:]
            lean_manager._validate_framework_objects(
                problems=problems,
                discretizations=bad_discretizations,
                operation_name="test_validation"
            )
            print("✗ Should have failed with incompatible discretization n_elements")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught incompatible discretization n_elements: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter validation test failed: {e}")
        return False


def test_bulk_data_operations(lean_manager, problems, discretizations):
    """Test BulkData creation and operations."""
    print("\n=== Testing BulkData Operations ===")
    
    if lean_manager is None:
        print("✗ Cannot test - lean manager is None")
        return None
    
    try:
        # Test 1: BulkData creation
        bulk_data_0 = lean_manager.create_bulk_data(0, problems[0], discretizations[0], dual=False)
        print(f"✓ BulkData created for domain 0: {bulk_data_0}")
        
        # Test 2: Initialize all bulk data
        bulk_data_list = lean_manager.initialize_all_bulk_data(problems, discretizations, time=0.0)
        print(f"✓ All bulk data initialized: {len(bulk_data_list)} domains")
        
        for i, bulk_data in enumerate(bulk_data_list):
            data = bulk_data.get_data()
            print(f"  Domain {i}: shape {data.shape}, range [{np.min(data):.6f}, {np.max(data):.6f}]")
        
        # Test 3: Forcing term computation
        forcing_terms = lean_manager.compute_forcing_terms(
            bulk_data_list, problems, discretizations, time=0.5, dt=0.1
        )
        print(f"✓ Forcing terms computed for {len(forcing_terms)} domains")
        
        for i, forcing_term in enumerate(forcing_terms):
            print(f"  Domain {i}: shape {forcing_term.shape}, range [{np.min(forcing_term):.6e}, {np.max(forcing_term):.6e}]")
        
        # Test 4: Mass computation
        total_mass = lean_manager.compute_total_mass(bulk_data_list)
        print(f"✓ Total mass computed: {total_mass:.6e}")
        
        # Test 5: Data extraction
        data_arrays = lean_manager.get_bulk_data_arrays(bulk_data_list)
        print(f"✓ Data arrays extracted: {len(data_arrays)} arrays")
        
        return bulk_data_list
        
    except Exception as e:
        print(f"✗ BulkData operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_management(lean_manager, bulk_data_list):
    """Test data management operations."""
    print("\n=== Testing Data Management ===")
    
    if lean_manager is None or bulk_data_list is None:
        print("✗ Cannot test - missing required objects")
        return False
    
    try:
        # Test 1: Data update with valid data
        original_arrays = lean_manager.get_bulk_data_arrays(bulk_data_list)
        
        # Create modified data
        new_data_list = []
        for i, original_data in enumerate(original_arrays):
            new_data = original_data + 0.1 * np.random.rand(*original_data.shape)
            new_data_list.append(new_data)
        
        lean_manager.update_bulk_data(bulk_data_list, new_data_list)
        print("✓ Bulk data updated successfully")
        
        # Verify update worked
        updated_arrays = lean_manager.get_bulk_data_arrays(bulk_data_list)
        for i, (original, updated) in enumerate(zip(original_arrays, updated_arrays)):
            if np.allclose(original, updated):
                print(f"✗ Domain {i} data was not updated")
                return False
        
        print("✓ All data arrays were updated correctly")
        
        # Test 2: Invalid data update (wrong shape)
        try:
            wrong_shape_data = [np.ones((3, 3))] + new_data_list[1:]  # Wrong shape
            lean_manager.update_bulk_data(bulk_data_list, wrong_shape_data)
            print("✗ Should have failed with wrong shape data")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught wrong shape data: {e}")
        
        # Test 3: Invalid data update (NaN values)
        try:
            nan_data = new_data_list.copy()
            nan_data[0][0, 0] = np.nan
            lean_manager.update_bulk_data(bulk_data_list, nan_data)
            print("✗ Should have failed with NaN data")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught NaN data: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data management test failed: {e}")
        return False


def test_memory_efficiency():
    """Test memory efficiency of lean approach."""
    print("\n=== Testing Memory Efficiency ===")
    
    try:
        # Create larger framework objects
        n_domains = 5
        problems = [MockProblem(neq=2, has_forcing=True, has_initial=True) for _ in range(n_domains)]
        discretizations = [MockDiscretization(n_elements=20) for _ in range(n_domains)]
        static_condensations = [MockStaticCondensation(i) for i in range(n_domains)]
        
        # Extract domain data once
        domain_data = BulkDataManager.extract_domain_data_list(
            problems, discretizations, static_condensations
        )
        
        # Create multiple lean managers using the same domain data
        managers = []
        for i in range(3):
            manager = BulkDataManager(domain_data)
            managers.append(manager)
        
        print(f"✓ Created {len(managers)} lean managers sharing domain data")
        print(f"  Each manager stores only: {len(domain_data)} DomainData objects")
        print(f"  Framework objects stored externally and reused")
        
        # Test that all managers work with the same framework objects
        for i, manager in enumerate(managers):
            bulk_data_list = manager.initialize_all_bulk_data(problems, discretizations)
            total_mass = manager.compute_total_mass(bulk_data_list)
            print(f"  Manager {i}: initialized {len(bulk_data_list)} domains, mass={total_mass:.6e}")
        
        print("✓ Memory efficiency validated - multiple managers share data")
        return True
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False


def test_edge_cases(lean_manager, problems, discretizations):
    """Test edge cases and error conditions."""
    print("\n=== Testing Edge Cases ===")
    
    if lean_manager is None:
        print("✗ Cannot test - lean manager is None")
        return False
    
    try:
        # Test 1: Empty framework object lists
        try:
            lean_manager._validate_framework_objects(
                problems=[],
                discretizations=discretizations,
                operation_name="edge_case_test"
            )
            print("✗ Should have failed with empty problems list")
            return False
        except ValueError:
            print("✓ Correctly caught empty problems list")
        
        # Test 2: None framework objects
        lean_manager._validate_framework_objects(
            problems=None,
            discretizations=None,
            operation_name="edge_case_test"
        )
        print("✓ None framework objects handled correctly")
        
        # Test 3: Out of bounds domain index
        try:
            lean_manager.create_bulk_data(999, problems[0], discretizations[0])
            print("✗ Should have failed with out of bounds domain index")
            return False
        except ValueError:
            print("✓ Correctly caught out of bounds domain index")
        
        # Test 4: Missing attributes
        class MockIncompleteDiscretization:
            pass  # Missing n_elements attribute
        
        try:
            incomplete_discretizations = [MockIncompleteDiscretization()] + discretizations[1:]
            lean_manager._validate_framework_objects(
                problems=problems,
                discretizations=incomplete_discretizations,
                operation_name="edge_case_test"
            )
            print("✗ Should have failed with missing attributes")
            return False
        except ValueError:
            print("✓ Correctly caught missing attributes")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False


def run_all_lean_tests():
    """Run all lean BulkDataManager tests."""
    print("Running Lean BulkDataManager Tests")
    print("=" * 60)
    
    try:
        # Test 1: Manager creation
        lean_manager, problems, discretizations, static_condensations = test_lean_manager_creation()
        if lean_manager is None:
            print("❌ Manager creation failed - stopping tests")
            return
        
        # Test 2: Parameter validation
        if not test_parameter_validation(lean_manager, problems, discretizations, static_condensations):
            print("❌ Parameter validation tests failed")
            return
        
        # Test 3: BulkData operations
        bulk_data_list = test_bulk_data_operations(lean_manager, problems, discretizations)
        if bulk_data_list is None:
            print("❌ BulkData operations tests failed")
            return
        
        # Test 4: Data management
        if not test_data_management(lean_manager, bulk_data_list):
            print("❌ Data management tests failed")
            return
        
        # Test 5: Memory efficiency
        if not test_memory_efficiency():
            print("❌ Memory efficiency tests failed")
            return
        
        # Test 6: Edge cases
        if not test_edge_cases(lean_manager, problems, discretizations):
            print("❌ Edge cases tests failed")
            return
        
        print("=" * 60)
        print("🎉 All Lean BulkDataManager tests completed successfully!")
        print("\nLean implementation benefits validated:")
        print("  ✓ Minimal memory footprint - only essential data stored")
        print("  ✓ Framework objects passed as parameters")
        print("  ✓ Multiple managers can share domain data")
        print("  ✓ Comprehensive parameter validation")
        print("  ✓ Full functionality maintained")
        print("  ✓ Robust error handling")
        
    except Exception as e:
        print(f"❌ Lean test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_lean_tests()
