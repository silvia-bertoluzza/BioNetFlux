"""
Pytest-compatible test script for Lean BulkDataManager class.
Tests the memory-efficient implementation that accepts framework objects as parameters.

Usage:
    pytest test_lean_bulk_data_manager.py
    pytest test_lean_bulk_data_manager.py -v  # verbose output
    pytest test_lean_bulk_data_manager.py -s  # show print statements
"""

import numpy as np
import sys
import os
import pytest

# sys.path hack — commented out, use pip install -e . instead
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.core.lean_bulk_data_manager import BulkDataManager
from bionetflux.core.bulk_data import BulkData
from bionetflux.core.domain_data import DomainData
from bionetflux.utils.elementary_matrices import ElementaryMatrices


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
        """Mock static condensation method for testing."""
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


# Fixtures
@pytest.fixture
def mock_problems():
    """Fixture providing mock problems for testing."""
    return [
        MockProblem(neq=1, has_forcing=True, has_initial=True),
        MockProblem(neq=2, has_forcing=False, has_initial=True)
    ]


@pytest.fixture
def mock_discretizations():
    """Fixture providing mock discretizations for testing."""
    return [
        MockDiscretization(n_elements=4),
        MockDiscretization(n_elements=6)
    ]


@pytest.fixture
def mock_static_condensations():
    """Fixture providing mock static condensations for testing."""
    return [
        MockStaticCondensation(0),
        MockStaticCondensation(1)
    ]


@pytest.fixture
def lean_manager(mock_problems, mock_discretizations, mock_static_condensations):
    """Fixture providing initialized lean BulkDataManager."""
    # Extract domain data using static method
    domain_data = BulkDataManager.extract_domain_data_list(
        mock_problems, mock_discretizations, mock_static_condensations
    )
    
    # Create lean manager
    return BulkDataManager(domain_data)


@pytest.fixture
def bulk_data_list(lean_manager, mock_problems, mock_discretizations):
    """Fixture providing initialized bulk data list."""
    return lean_manager.initialize_all_bulk_data(mock_problems, mock_discretizations, time=0.0)


class TestLeanManagerCreation:
    """Test creating lean BulkDataManager from framework objects."""

    def test_domain_data_extraction(self, mock_problems, mock_discretizations, mock_static_condensations):
        """Test domain data extraction."""
        domain_data = BulkDataManager.extract_domain_data_list(
            mock_problems, mock_discretizations, mock_static_condensations
        )
        
        assert len(domain_data) == len(mock_problems)
        for i, dd in enumerate(domain_data):
            assert dd.neq == mock_problems[i].neq
            assert dd.n_elements == mock_discretizations[i].n_elements

    def test_lean_manager_creation(self, lean_manager):
        """Test lean manager creation."""
        assert lean_manager is not None
        assert isinstance(lean_manager, BulkDataManager)

    def test_internal_validation(self, lean_manager, mock_problems, mock_discretizations, mock_static_condensations):
        """Test internal validation."""
        success = lean_manager.test(mock_problems, mock_discretizations, mock_static_condensations)
        assert success, "Lean manager internal test failed"


class TestParameterValidation:
    """Test parameter validation functionality."""

    def test_correct_parameters_validation(self, lean_manager, mock_problems, mock_discretizations, mock_static_condensations):
        """Test that correct parameters pass validation."""
        # Should not raise exception
        lean_manager._validate_framework_objects(
            problems=mock_problems,
            discretizations=mock_discretizations,
            static_condensations=mock_static_condensations,
            operation_name="test_validation"
        )

    def test_wrong_number_of_problems(self, lean_manager, mock_problems, mock_discretizations):
        """Test validation with wrong number of problems."""
        wrong_problems = mock_problems[:-1] if len(mock_problems) > 1 else []
        
        with pytest.raises(ValueError, match="Number of problems"):
            lean_manager._validate_framework_objects(
                problems=wrong_problems,
                discretizations=mock_discretizations,
                operation_name="test_validation"
            )

    def test_wrong_number_of_discretizations(self, lean_manager, mock_problems, mock_discretizations):
        """Test validation with wrong number of discretizations."""
        wrong_discretizations = mock_discretizations[:-1] if len(mock_discretizations) > 1 else []
        
        with pytest.raises(ValueError, match="Number of discretizations"):
            lean_manager._validate_framework_objects(
                problems=mock_problems,
                discretizations=wrong_discretizations,
                operation_name="test_validation"
            )

    def test_incompatible_problem_neq(self, lean_manager, mock_problems, mock_discretizations):
        """Test validation with incompatible problem neq."""
        class MockBadProblem:
            def __init__(self):
                self.neq = 999  # Wrong neq

        bad_problems = [MockBadProblem()] + mock_problems[1:]
        
        with pytest.raises(ValueError, match="doesn't match domain data neq"):
            lean_manager._validate_framework_objects(
                problems=bad_problems,
                discretizations=mock_discretizations,
                operation_name="test_validation"
            )

    def test_incompatible_discretization_elements(self, lean_manager, mock_problems, mock_discretizations):
        """Test validation with incompatible discretization n_elements."""
        class MockBadDiscretization:
            def __init__(self):
                self.n_elements = 999  # Wrong n_elements

        bad_discretizations = [MockBadDiscretization()] + mock_discretizations[1:]
        
        with pytest.raises(ValueError, match="doesn't match domain data n_elements"):
            lean_manager._validate_framework_objects(
                problems=mock_problems,
                discretizations=bad_discretizations,
                operation_name="test_validation"
            )

    def test_none_framework_objects(self, lean_manager):
        """Test that None framework objects are handled correctly."""
        # Should not raise exception
        lean_manager._validate_framework_objects(
            problems=None,
            discretizations=None,
            operation_name="test_validation"
        )


class TestBulkDataOperations:
    """Test BulkData creation and operations."""

    def test_bulk_data_creation(self, lean_manager, mock_problems, mock_discretizations):
        """Test individual BulkData creation."""
        bulk_data_0 = lean_manager.create_bulk_data(0, mock_problems[0], mock_discretizations[0], dual=False)
        
        assert bulk_data_0 is not None
        assert isinstance(bulk_data_0, BulkData)

    def test_initialize_all_bulk_data(self, lean_manager, mock_problems, mock_discretizations):
        """Test initialization of all bulk data."""
        bulk_data_list = lean_manager.initialize_all_bulk_data(mock_problems, mock_discretizations, time=0.0)
        
        assert len(bulk_data_list) == len(mock_problems)
        
        for i, bulk_data in enumerate(bulk_data_list):
            assert bulk_data is not None
            data = bulk_data.get_data()
            assert data is not None
            assert data.shape[0] > 0

    def test_forcing_term_computation(self, lean_manager, bulk_data_list, mock_problems, mock_discretizations):
        """Test forcing term computation."""
        forcing_terms = lean_manager.compute_forcing_terms(
            bulk_data_list, mock_problems, mock_discretizations, time=0.5, dt=0.1
        )
        
        assert len(forcing_terms) == len(mock_problems)
        
        for i, forcing_term in enumerate(forcing_terms):
            assert forcing_term is not None
            assert isinstance(forcing_term, np.ndarray)
            assert forcing_term.shape[0] > 0

    def test_mass_computation(self, lean_manager, bulk_data_list):
        """Test mass computation."""
        total_mass = lean_manager.compute_total_mass(bulk_data_list)
        
        assert isinstance(total_mass, (int, float, np.number))
        assert not np.isnan(total_mass)

    def test_data_extraction(self, lean_manager, bulk_data_list):
        """Test data array extraction."""
        data_arrays = lean_manager.get_bulk_data_arrays(bulk_data_list)
        
        assert len(data_arrays) == len(bulk_data_list)
        
        for data_array in data_arrays:
            assert isinstance(data_array, np.ndarray)
            assert data_array.shape[0] > 0


class TestDataManagement:
    """Test data management operations."""

    def test_data_update_valid(self, lean_manager, bulk_data_list):
        """Test data update with valid data."""
        original_arrays = lean_manager.get_bulk_data_arrays(bulk_data_list)
        
        # Create modified data
        new_data_list = []
        for original_data in original_arrays:
            new_data = original_data + 0.1 * np.random.rand(*original_data.shape)
            new_data_list.append(new_data)
        
        # Should not raise exception
        lean_manager.update_bulk_data(bulk_data_list, new_data_list)
        
        # Verify update worked
        updated_arrays = lean_manager.get_bulk_data_arrays(bulk_data_list)
        for original, updated in zip(original_arrays, updated_arrays):
            assert not np.allclose(original, updated), "Data should have been updated"

    def test_data_update_wrong_shape(self, lean_manager, bulk_data_list):
        """Test data update with wrong shape."""
        original_arrays = lean_manager.get_bulk_data_arrays(bulk_data_list)
        wrong_shape_data = [np.ones((3, 3))] + original_arrays[1:]  # Wrong shape
        
        with pytest.raises(ValueError):  # Remove specific match pattern
            lean_manager.update_bulk_data(bulk_data_list, wrong_shape_data)

    def test_data_update_nan_values(self, lean_manager, bulk_data_list):
        """Test data update with NaN values."""
        original_arrays = lean_manager.get_bulk_data_arrays(bulk_data_list)
        nan_data = [arr.copy() for arr in original_arrays]
        nan_data[0].flat[0] = np.nan
        
        with pytest.raises(ValueError):  # Remove specific match pattern
            lean_manager.update_bulk_data(bulk_data_list, nan_data)


class TestMemoryEfficiency:
    """Test memory efficiency of lean approach."""

    def test_shared_domain_data(self):
        """Test that multiple managers can share domain data."""
        # Create framework objects
        n_domains = 5
        problems = [MockProblem(neq=2, has_forcing=True, has_initial=True) for _ in range(n_domains)]
        discretizations = [MockDiscretization(n_elements=20) for _ in range(n_domains)]
        static_condensations = [MockStaticCondensation(i) for i in range(n_domains)]
        
        # Extract domain data once
        domain_data = BulkDataManager.extract_domain_data_list(
            problems, discretizations, static_condensations
        )
        
        # Create multiple lean managers using the same domain data
        managers = [BulkDataManager(domain_data) for _ in range(3)]
        
        assert len(managers) == 3
        
        # Test that all managers work with the same framework objects
        for manager in managers:
            bulk_data_list = manager.initialize_all_bulk_data(problems, discretizations)
            total_mass = manager.compute_total_mass(bulk_data_list)
            
            assert len(bulk_data_list) == len(problems)
            assert isinstance(total_mass, (int, float, np.number))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_problems_list(self, lean_manager, mock_discretizations):
        """Test with empty problems list."""
        with pytest.raises(ValueError):
            lean_manager._validate_framework_objects(
                problems=[],
                discretizations=mock_discretizations,
                operation_name="edge_case_test"
            )

    def test_out_of_bounds_domain_index(self, lean_manager, mock_problems, mock_discretizations):
        """Test with out of bounds domain index."""
        with pytest.raises(ValueError, match="Domain index.*out of range"):
            lean_manager.create_bulk_data(999, mock_problems[0], mock_discretizations[0])

    def test_missing_attributes(self, lean_manager, mock_problems):
        """Test with missing attributes."""
        class MockIncompleteDiscretization:
            pass  # Missing n_elements attribute
        
        incomplete_discretizations = [MockIncompleteDiscretization()]
        
        with pytest.raises((ValueError, AttributeError)):
            lean_manager._validate_framework_objects(
                problems=mock_problems[:1],
                discretizations=incomplete_discretizations,
                operation_name="edge_case_test"
            )


class TestPerformance:
    """Performance tests for lean BulkDataManager."""

    @pytest.mark.slow
    def test_large_domain_performance(self):
        """Test performance with larger domains."""
        import time
        
        # Create larger test case
        n_domains = 10
        n_elements = 50
        
        problems = [MockProblem(neq=3, has_forcing=True, has_initial=True) for _ in range(n_domains)]
        discretizations = [MockDiscretization(n_elements=n_elements) for _ in range(n_domains)]
        static_condensations = [MockStaticCondensation(i) for i in range(n_domains)]
        
        # Time domain data extraction
        start_time = time.time()
        domain_data = BulkDataManager.extract_domain_data_list(
            problems, discretizations, static_condensations
        )
        extraction_time = time.time() - start_time
        
        # Time manager creation and operations
        start_time = time.time()
        manager = BulkDataManager(domain_data)
        bulk_data_list = manager.initialize_all_bulk_data(problems, discretizations)
        forcing_terms = manager.compute_forcing_terms(bulk_data_list, problems, discretizations, time=0.0, dt=0.01)
        operation_time = time.time() - start_time
        
        # Performance assertions
        assert extraction_time < 1.0, f"Domain data extraction too slow: {extraction_time:.3f}s"
        assert operation_time < 2.0, f"Manager operations too slow: {operation_time:.3f}s"
        
        # Verify results
        assert len(bulk_data_list) == n_domains
        assert len(forcing_terms) == n_domains


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    # Register markers to avoid warnings
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Add marker to slow tests if not already marked."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name.lower() or "large_domain" in item.name:
            if not any(mark.name == "slow" for mark in item.iter_markers()):
                item.add_marker(pytest.mark.slow)
        
        # Add unit marker to most tests
        if not any(mark.name in ["slow", "integration"] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Add pytest.ini configuration content programmatically
pytest_plugins = []

def pytest_collection_modifyitems(config, items):
    """Add marker to slow tests if not already marked."""
    for item in items:
        if "large_domain_performance" in item.name:
            item.add_marker(pytest.mark.slow)

def main():
    """Run tests using pytest."""
    print("This file is now pytest-compatible!")
    print("Usage:")
    print("  pytest test_lean_bulk_data_manager.py")
    print("  pytest test_lean_bulk_data_manager.py -v")
    print("  pytest test_lean_bulk_data_manager.py -s  # show prints")
    print("  pytest test_lean_bulk_data_manager.py -m \"not slow\"  # skip slow tests")
    
    # Run with pytest for backwards compatibility
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    main()
