"""
Pytest-compatible test script for LeanGlobalAssembler class.
Tests the lean global assembly implementation that uses parameter-passing approach.

Usage:
    pytest test_lean_global_assembly.py
    pytest test_lean_global_assembly.py -v  # verbose output
    pytest test_lean_global_assembly.py -s  # show print statements
"""

import numpy as np
import sys
import os
import pytest

# sys.path hack — commented out, use pip install -e . instead
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.core.lean_global_assembly import GlobalAssembler
from bionetflux.core.lean_bulk_data_manager import BulkDataManager
from bionetflux.core.bulk_data import BulkData
from bionetflux.core.constraints import ConstraintManager
from bionetflux.utils.elementary_matrices import ElementaryMatrices


class MockProblem:
    """Mock problem class for testing."""
    def __init__(self, neq=1, has_forcing=False, has_initial=False):
        self.neq = neq
        self.domain_length = 1.0
        
        # Mock initial conditions
        if has_initial:
            self.u0 = [lambda x, t: np.sin(np.pi * x) + 0.1 * t for _ in range(neq)]
        else:
            self.u0 = [None for _ in range(neq)]
        
        # Mock forcing functions
        if has_forcing:
            self.force = [lambda x, t: 0.1 * np.exp(-x) * np.cos(t) for _ in range(neq)]
        else:
            self.force = [None for _ in range(neq)]


class MockDiscretization:
    """Mock discretization class for testing."""
    def __init__(self, n_elements=5, domain_length=1.0):
        self.n_elements = n_elements
        self.nodes = np.linspace(0, domain_length, n_elements + 1)
        self.element_length = domain_length / n_elements
        self.element_sizes = np.ones(n_elements) * self.element_length


class MockGlobalDiscretization:
    """Mock global discretization class."""
    def __init__(self, discretizations):
        self.spatial_discretizations = discretizations


class MockStaticCondensation:  # FIXED: Removed extra indentation
    def __init__(self, neq=1):  # FIXED: Proper indentation
        self.neq = neq
        self.elementary_matrices = ElementaryMatrices(orthonormal_basis=False)

    @property
    def total_flux_dofs_per_element(self):
        return self.neq  # Minimal: 1 DOF per equation

    def build_matrices(self):  # FIXED: Proper indentation
        """Return mock matrices."""
        return {
            'M': np.array([[2/3, 1/3], [1/3, 2/3]]),  # Mass matrix
            'T': np.array([[1, -1], [1, 1]]),          # Trace matrix
            'QUAD': np.array([[0.5, 0.5], [0.5, 0.5]])  # Quadrature matrix
        }
    
    def static_condensation(self, local_trace, local_source=None):  # FIXED: Proper indentation
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
        coeffs_per_element = 2 * neq
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
        
        # Mock jacobian with trace matrix structure
        jacobian = np.eye(len(local_trace_flat))
        if T.shape == (2, 2):
            jacobian[:2, :2] = T
            if len(local_trace_flat) > 2:
                jacobian[2:4, 2:4] = T
        
        return local_solution, flux, flux_trace, jacobian
        
class OldMockStaticCondensation:
    """Mock static condensation class."""
    def __init__(self, domain_idx=0):
        self.domain_idx = domain_idx
        self.elementary_matrices = ElementaryMatrices(orthonormal_basis=False)

    @property
    def total_flux_dofs_per_element(self):
        return 1  # Minimal: 1 DOF
    
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
            local_trace: Vector of length 2 * neq
            local_source: Vector of length 2 * neq (optional)
            
        Returns:
            tuple: (local_solution, flux, flux_trace, jacobian)
        """
        trace_length = len(local_trace.flatten())
        if trace_length % 2 != 0:
            raise ValueError(f"local_trace length {trace_length} must be divisible by 2")
        
        neq = trace_length // 2
        
        if local_source is None:
            local_source = np.zeros(2 * neq)
        
        # Ensure consistent shapes - flatten both inputs
        local_trace_flat = local_trace.flatten()
        local_source_flat = local_source.flatten()
        
        # Validate input lengths
        if len(local_source_flat) != 2 * neq:
            raise ValueError(f"local_source length {len(local_source_flat)} must equal 2 * neq = {2 * neq}")
        
        # CORRECTED: local_solution should have length 2 * neq (not 2 * (2*neq-1))
        local_solution = 0.8 * local_trace_flat + 0.2 * local_source_flat
        
        # Mock flux: vector of length 2 * neq - 1  
        flux_length = 2 * neq - 1
        flux = np.zeros(flux_length)
        
        # Simple mock flux computation
        for i in range(min(flux_length, neq)):
            if 2*i+1 < len(local_trace_flat):
                flux[i] = local_trace_flat[2*i+1] - local_trace_flat[2*i]
        
        # Fill remaining flux entries if any
        for i in range(neq, flux_length):
            flux[i] = 0.1 * np.sum(local_trace_flat)
        
        # CORRECTED: flux_trace has same length as local_trace_flat
        flux_trace = local_trace_flat + 0.1 * local_solution
        
        # Mock jacobian: square matrix of size (2 * neq) x (2 * neq)
        jacobian = np.eye(2 * neq) + 0.1 * np.random.rand(2 * neq, 2 * neq)
        
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
def mock_static_condensations(mock_problems):
    """Fixture providing mock static condensations for testing."""
    return [
        MockStaticCondensation(neq=problem.neq) for problem in mock_problems
    ]

@pytest.fixture
def mock_global_discretization(mock_discretizations):
    """Fixture providing mock global discretization."""
    return MockGlobalDiscretization(mock_discretizations)

@pytest.fixture
def assembler(mock_problems, mock_global_discretization, mock_static_condensations):
    """Fixture providing initialized LeanGlobalAssembler."""
    return GlobalAssembler.from_framework_objects(
        mock_problems, mock_global_discretization, mock_static_condensations
    )

@pytest.fixture
def bulk_data_list(assembler, mock_problems, mock_discretizations):
    """Fixture providing initialized bulk data list."""
    return assembler.initialize_bulk_data(mock_problems, mock_discretizations, time=0.0)

# Test Classes
class TestLeanAssemblerCreation:
    """Test creating LeanGlobalAssembler in different ways."""

    def test_factory_method_creation(self, mock_problems, mock_global_discretization, mock_static_condensations):
        """Test factory method creation."""
        assembler = GlobalAssembler.from_framework_objects(
            mock_problems, mock_global_discretization, mock_static_condensations
        )
        
        assert assembler is not None
        assert isinstance(assembler, GlobalAssembler)

    def test_direct_creation(self, mock_problems, mock_discretizations, mock_static_condensations):
        """Test direct creation using domain data."""
        domain_data = BulkDataManager.extract_domain_data_list(
            mock_problems, mock_discretizations, mock_static_condensations
        )
        
        assembler = GlobalAssembler(domain_data)
        
        assert assembler is not None
        assert isinstance(assembler, GlobalAssembler)

    def test_creation_methods_equivalence(self, mock_problems, mock_global_discretization, mock_discretizations, mock_static_condensations):
        """Test that both creation methods produce equivalent assemblers."""
        assembler1 = GlobalAssembler.from_framework_objects(
            mock_problems, mock_global_discretization, mock_static_condensations
        )
        
        domain_data = BulkDataManager.extract_domain_data_list(
            mock_problems, mock_discretizations, mock_static_condensations
        )
        assembler2 = GlobalAssembler(domain_data)
        
        assert assembler1.n_domains == assembler2.n_domains
        assert assembler1.total_dofs == assembler2.total_dofs

    def test_internal_validation(self, assembler, mock_problems, mock_discretizations, mock_static_condensations):
        """Test internal validation."""
        success = assembler.test(mock_problems, mock_discretizations, mock_static_condensations)
        assert success, "Lean assembler internal test failed"


class TestDOFStructure:
    """Test DOF structure and indexing."""

    def test_total_dof_calculation(self, assembler, mock_problems, mock_discretizations):
        """Test total DOF calculation."""
        expected_trace_dofs = 0
        for problem, discretization in zip(mock_problems, mock_discretizations):
            n_nodes = discretization.n_elements + 1
            domain_trace_dofs = problem.neq * n_nodes
            expected_trace_dofs += domain_trace_dofs
        
        assert assembler.total_trace_dofs == expected_trace_dofs

    def test_domain_trace_sizes(self, assembler, mock_problems, mock_discretizations):
        """Test domain trace sizes."""
        for i, (problem, discretization) in enumerate(zip(mock_problems, mock_discretizations)):
            n_nodes = discretization.n_elements + 1
            expected_size = problem.neq * n_nodes
            assert assembler.domain_trace_sizes[i] == expected_size

    def test_domain_offsets(self, assembler):
        """Test domain offset calculation."""
        expected_offset = 0
        for i in range(assembler.n_domains):
            assert assembler.domain_trace_offsets[i] == expected_offset
            expected_offset += assembler.domain_trace_sizes[i]

    def test_solution_extraction(self, assembler):
        """Test solution extraction."""
        test_solution = np.random.rand(assembler.total_dofs)
        domain_solutions = assembler.get_domain_solutions(test_solution)
        
        for i, domain_sol in enumerate(domain_solutions):
            assert len(domain_sol) == assembler.domain_trace_sizes[i]
            
            # Verify extracted solution matches original
            start_idx = assembler.domain_trace_offsets[i]
            end_idx = start_idx + assembler.domain_trace_sizes[i]
            expected_sol = test_solution[start_idx:end_idx]
            
            assert np.allclose(domain_sol, expected_sol)


class TestInitialGuessMethods:
    """Test different initial guess creation methods."""

    def test_initial_guess_from_bulk_data(self, assembler, mock_problems, mock_discretizations):
        """Test initial guess from BulkData objects."""
        time = 0.5
        bulk_data_list = assembler.initialize_bulk_data(mock_problems, mock_discretizations, time)
        initial_guess = assembler.create_initial_guess_from_bulk_data(bulk_data_list)
        
        assert initial_guess.shape == (assembler.total_dofs,)
        assert not np.any(np.isnan(initial_guess))
        assert not np.any(np.isinf(initial_guess))

    def test_initial_guess_from_problems(self, assembler, mock_problems, mock_discretizations):
        """Test initial guess directly from problems."""
        time = 0.5
        initial_guess = assembler.create_initial_guess_from_problems(mock_problems, mock_discretizations, time)
        
        assert initial_guess.shape == (assembler.total_dofs,)
        assert not np.any(np.isnan(initial_guess))
        assert not np.any(np.isinf(initial_guess))

    def test_initial_guess_methods_shape_consistency(self, assembler, mock_problems, mock_discretizations):
        """Test that both initial guess methods produce same shape."""
        time = 0.5
        
        bulk_data_list = assembler.initialize_bulk_data(mock_problems, mock_discretizations, time)
        guess_bd = assembler.create_initial_guess_from_bulk_data(bulk_data_list)
        guess_prob = assembler.create_initial_guess_from_problems(mock_problems, mock_discretizations, time)
        
        assert guess_bd.shape == guess_prob.shape

    @pytest.mark.parametrize("test_time", [0.0, 1.0, 2.0])
    def test_initial_guess_different_times(self, assembler, mock_problems, mock_discretizations, test_time):
        """Test initial guess at different times."""
        guess = assembler.create_initial_guess_from_problems(mock_problems, mock_discretizations, test_time)
        assert not np.any(np.isnan(guess))
        assert not np.any(np.isinf(guess))


class TestResidualJacobianAssembly:
    """Test residual and Jacobian assembly."""

    def test_assembly_basic(self, assembler, mock_problems, mock_discretizations, mock_static_condensations):
        """Test basic residual and Jacobian assembly."""
        global_guess = np.random.rand(assembler.total_dofs) * 0.1
        
        bulk_data_list = assembler.initialize_bulk_data(mock_problems, mock_discretizations, time=0.0)
        forcing_terms = [bulk_sol.get_data() for bulk_sol in bulk_data_list]
        
        residual, jacobian = assembler.assemble_residual_and_jacobian(
            global_solution=global_guess,
            forcing_terms=forcing_terms,
            static_condensations=mock_static_condensations,
            time=0.5
        )
        
        assert residual.shape == (assembler.total_dofs,)
        assert jacobian.shape == (assembler.total_dofs, assembler.total_dofs)

    def test_assembly_no_invalid_values(self, assembler, mock_problems, mock_discretizations, mock_static_condensations):
        """Test that assembly produces no invalid values."""
        global_guess = np.random.rand(assembler.total_dofs) * 0.1
        
        bulk_data_list = assembler.initialize_bulk_data(mock_problems, mock_discretizations, time=0.0)
        forcing_terms = [bulk_sol.get_data() for bulk_sol in bulk_data_list]
        
        residual, jacobian = assembler.assemble_residual_and_jacobian(
            global_solution=global_guess,
            forcing_terms=forcing_terms,
            static_condensations=mock_static_condensations,
            time=0.5
        )
        
        assert not np.any(np.isnan(residual))
        assert not np.any(np.isinf(residual))
        assert not np.any(np.isnan(jacobian))
        assert not np.any(np.isinf(jacobian))

    def test_jacobian_not_all_zeros(self, assembler, mock_problems, mock_discretizations, mock_static_condensations):
        """Test that Jacobian is not all zeros."""
        global_guess = np.random.rand(assembler.total_dofs) * 0.1
        
        bulk_data_list = assembler.initialize_bulk_data(mock_problems, mock_discretizations, time=0.0)
        forcing_terms = [bulk_sol.get_data() for bulk_sol in bulk_data_list]
        
        _, jacobian = assembler.assemble_residual_and_jacobian(
            global_solution=global_guess,
            forcing_terms=forcing_terms,
            static_condensations=mock_static_condensations,
            time=0.5
        )
        
        assert not np.allclose(jacobian, 0)

    def test_assembly_sensitivity(self, assembler, mock_problems, mock_discretizations, mock_static_condensations):
        """Test that residual changes with solution perturbation."""
        global_guess = np.random.rand(assembler.total_dofs) * 0.1
        
        bulk_data_list = assembler.initialize_bulk_data(mock_problems, mock_discretizations, time=0.0)
        forcing_terms = [bulk_sol.get_data() for bulk_sol in bulk_data_list]
        
        residual1, _ = assembler.assemble_residual_and_jacobian(
            global_solution=global_guess,
            forcing_terms=forcing_terms,
            static_condensations=mock_static_condensations,
            time=0.5
        )
        
        # Use a larger perturbation to ensure detectable change
        perturbed_solution = global_guess + 1e-3 * np.random.rand(len(global_guess))
        residual2, _ = assembler.assemble_residual_and_jacobian(
            global_solution=perturbed_solution,
            forcing_terms=forcing_terms,
            static_condensations=mock_static_condensations,
            time=0.5
        )
        
        # Use stricter tolerance to detect small but meaningful changes
        assert not np.allclose(residual1, residual2, rtol=1e-10, atol=1e-12)


class TestMassConservation:
    """Test mass conservation computation."""

    def test_mass_conservation_basic(self, assembler, bulk_data_list):
        """Test basic mass conservation computation."""
        mass = assembler.compute_mass_conservation(bulk_data_list)
        
        assert isinstance(mass, (int, float, np.number))
        assert not np.isnan(mass)
        assert not np.isinf(mass)

    def test_mass_conservation_consistency(self, assembler, bulk_data_list):
        """Test mass conservation consistency."""
        mass1 = assembler.compute_mass_conservation(bulk_data_list)
        mass2 = assembler.compute_mass_conservation(bulk_data_list)
        
        assert np.isclose(mass1, mass2)


class TestParameterValidation:
    """Test parameter validation in assembly methods."""

    def test_wrong_number_of_problems(self, assembler, mock_problems, mock_discretizations):
        """Test validation with wrong number of problems."""
        wrong_problems = mock_problems[:-1] if len(mock_problems) > 1 else []
        
        with pytest.raises(ValueError):
            assembler.initialize_bulk_data(wrong_problems, mock_discretizations)

    def test_wrong_number_of_discretizations(self, assembler, mock_problems, mock_discretizations):
        """Test validation with wrong number of discretizations."""
        wrong_discretizations = mock_discretizations[:-1] if len(mock_discretizations) > 1 else []
        
        with pytest.raises(ValueError):
            assembler.initialize_bulk_data(mock_problems, wrong_discretizations)

    def test_incompatible_problem_neq(self, assembler, mock_problems, mock_discretizations):
        """Test validation with incompatible problem neq."""
        class MockBadProblem:
            def __init__(self):
                self.neq = 999  # Wrong neq
        
        bad_problems = [MockBadProblem()] + mock_problems[1:] if len(mock_problems) > 1 else [MockBadProblem()]
        
        with pytest.raises(ValueError):
            assembler.initialize_bulk_data(bad_problems, mock_discretizations)


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_num_domains(self, assembler):
        """Test get_num_domains method."""
        num_domains = assembler.get_num_domains()
        assert num_domains == assembler.n_domains

    def test_get_domain_info(self, assembler):
        """Test get_domain_info method."""
        for i in range(assembler.n_domains):
            domain_info = assembler.get_domain_info(i)
            assert hasattr(domain_info, 'neq')
            assert hasattr(domain_info, 'n_elements')

    def test_string_representations(self, assembler):
        """Test string representations."""
        str_repr = str(assembler)
        repr_repr = repr(assembler)
        
        assert isinstance(str_repr, str)
        assert isinstance(repr_repr, str)
        assert len(str_repr) > 0
        assert len(repr_repr) > 0


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")

def main():
    """Run tests using pytest."""
    print("This file is now pytest-compatible!")
    print("Usage:")
    print("  pytest test_lean_global_assembly.py")
    print("  pytest test_lean_global_assembly.py -v")
    print("  pytest test_lean_global_assembly.py -s  # show prints")
    
    # Run with pytest for backwards compatibility
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    main()
