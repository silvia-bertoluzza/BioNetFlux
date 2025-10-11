"""
Lean solver setup module for OOC1D problems on networks.
Minimizes data redundancy and provides clean interfaces for different problem types.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import importlib

# Core imports
from ooc1d.core.discretization import Discretization, GlobalDiscretization
from ooc1d.utils.elementary_matrices import ElementaryMatrices
from ooc1d.core.static_condensation_factory import StaticCondensationFactory
from ooc1d.core.constraints import ConstraintManager
from ooc1d.core.lean_global_assembly import GlobalAssembler
from ooc1d.core.lean_bulk_data_manager import BulkDataManager

class SolverSetup:
    """
    Lean setup class that orchestrates the initialization of all solver components.
    Maintains minimal data storage and provides efficient access to components.
    """
    
    def __init__(self, problem_module: str = "ooc1d.problems.test_problem2"):
        """
        Initialize solver setup with specified problem module.
        
        Args:
            problem_module: String path to problem module containing create_global_framework
        """
        self.problem_module = problem_module
        self._initialized = False
        
        # Core problem data (lean storage)
        self.problems = None
        self.global_discretization = None
        self.constraint_manager = None
        self.problem_name = None
        
        # Computed components (created on demand)
        self._elementary_matrices = None
        self._static_condensations = None
        self._global_assembler = None
        self._bulk_data_manager = None
        
    def initialize(self) -> None:
        """Initialize the core problem configuration."""
        if self._initialized:
            return
            
        # Import and call the problem-specific setup
        try:
            problem_mod = importlib.import_module(self.problem_module)
            create_global_framework = getattr(problem_mod, 'create_global_framework')
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import create_global_framework from {self.problem_module}: {e}")
        
        # Get core problem configuration
        self.problems, self.global_discretization, self.constraint_manager, self.problem_name = create_global_framework()
        self.constraints = self.constraint_manager  # For backward compatibility if needed  

        self._initialized = True
        
    @property
    def elementary_matrices(self) -> ElementaryMatrices:
        """Get elementary matrices (created once, cached)."""
        if self._elementary_matrices is None:
            self._elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
        return self._elementary_matrices
    
    @property
    def static_condensations(self) -> List:
        """Get static condensation implementations for all domains (created once, cached)."""
        if self._static_condensations is None:
            self._ensure_initialized()
            self._static_condensations = []
            
            for domain_idx in range(len(self.problems)):
                sc = StaticCondensationFactory.create(
                    self.problems[domain_idx],
                    self.global_discretization,
                    self.elementary_matrices,
                    domain_idx
                )
                # Build matrices immediately to cache them
                sc.build_matrices()
                self._static_condensations.append(sc)
                
        return self._static_condensations
    
    @property
    def global_assembler(self) -> GlobalAssembler:
        """Get global assembler (created once, cached)."""
        if self._global_assembler is None:
            self._ensure_initialized()
            # Check the actual constructor signature and use correct parameters
            self._global_assembler = GlobalAssembler.from_framework_objects(
                self.problems,
                self.global_discretization,
                self.static_condensations,
                self.constraint_manager
            )
        return self._global_assembler
    
    @property
    def bulk_data_manager(self) -> BulkDataManager:
        """Get bulk data manager (created once, cached)."""
        if self._bulk_data_manager is None:
            self._ensure_initialized()
            discretizations = self.global_discretization.spatial_discretizations
            self._domain_data = BulkDataManager.extract_domain_data_list(
                self.problems, discretizations, self.static_condensations
            )
            self._bulk_data_manager = BulkDataManager(
                self._domain_data
            )
        return self._bulk_data_manager
    
    def _ensure_initialized(self) -> None:
        """Ensure core problem data has been initialized."""
        if not self._initialized:
            self.initialize()
    
    def get_problem_info(self) -> Dict[str, Any]:
        """Get summary information about the problem setup."""
        self._ensure_initialized()
        
        info = {
            'problem_name': self.problem_name,
            'num_domains': len(self.problems),
            'total_elements': sum(disc.n_elements for disc in self.global_discretization.spatial_discretizations),
            'total_trace_dofs': self.global_assembler.total_trace_dofs,
            'num_constraints': self.constraint_manager.n_multipliers if self.constraint_manager else 0,
            'time_discretization': {
                'dt': self.global_discretization.dt,
                'T': self.global_discretization.T,
                'n_steps': self.global_discretization.n_time_steps
            },
            'domains': []
        }
        
        for i, (problem, discretization) in enumerate(zip(self.problems, self.global_discretization.spatial_discretizations)):
            domain_info = {
                'index': i,
                'type': problem.type,
                'domain': [problem.domain_start, problem.domain_end],
                'n_elements': discretization.n_elements,
                'n_equations': problem.neq,
                'trace_size': problem.neq * (discretization.n_elements + 1)
            }
            info['domains'].append(domain_info)
        
        return info
    
    def create_initial_conditions(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Create initial conditions for trace solutions and multipliers.
        
        Returns:
            Tuple of (trace_solutions, initial_multipliers)
        """
        self._ensure_initialized()
        
        trace_solutions = []
        
        for i, (problem, discretization) in enumerate(zip(self.problems, self.global_discretization.spatial_discretizations)):
            n_nodes = discretization.n_elements + 1
            trace_size = problem.neq * n_nodes
            nodes = discretization.nodes
            
            trace_solution = np.zeros(trace_size)
            
            # Apply initial conditions if available
            for eq in range(problem.neq):
                for j in range(n_nodes):
                    node_idx = eq * n_nodes + j
                    if hasattr(problem, 'u0') and len(problem.u0) > eq and callable(problem.u0[eq]):
                        trace_solution[node_idx] = problem.u0[eq](nodes[j])
                    elif hasattr(problem, 'initial_conditions') and len(problem.initial_conditions) > eq:
                        if callable(problem.initial_conditions[eq]):
                            trace_solution[node_idx] = problem.initial_conditions[eq](nodes[j])
            
            trace_solutions.append(trace_solution)
        
        # Initialize multipliers to zero
        n_multipliers = self.constraint_manager.n_multipliers if self.constraint_manager else 0
        initial_multipliers = np.zeros(n_multipliers)
        
        return trace_solutions, initial_multipliers
    
    def create_global_solution_vector(self, trace_solutions: List[np.ndarray], multipliers: np.ndarray) -> np.ndarray:
        """
        Assemble global solution vector from domain traces and multipliers.
        
        Args:
            trace_solutions: List of trace solution arrays for each domain
            multipliers: Array of Lagrange multiplier values
            
        Returns:
            Global solution vector
        """
        global_assembler = self.global_assembler
        
        # Calculate total size
        total_trace_size = sum(len(trace) for trace in trace_solutions)
        total_size = total_trace_size + len(multipliers)
        
        # Create global solution vector
        global_solution = np.zeros(total_size)
        
        # Fill trace solutions
        offset = 0
        for trace in trace_solutions:
            trace_flat = trace.flatten() if trace.ndim > 1 else trace
            global_solution[offset:offset+len(trace_flat)] = trace_flat
            offset += len(trace_flat)
        
        # Fill multipliers
        if len(multipliers) > 0:
            global_solution[offset:offset+len(multipliers)] = multipliers
        
        return global_solution
    
    def extract_domain_solutions(self, global_solution: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract domain trace solutions and multipliers from global solution vector.
        
        Args:
            global_solution: Global solution vector
            
        Returns:
            Tuple of (trace_solutions, multipliers)
        """
        self._ensure_initialized()
        
        trace_solutions = []
        offset = 0
        
        # Extract trace solutions for each domain
        for i, (problem, discretization) in enumerate(zip(self.problems, self.global_discretization.spatial_discretizations)):
            n_nodes = discretization.n_elements + 1
            trace_size = problem.neq * n_nodes
            
            trace_solution = global_solution[offset:offset+trace_size]
            trace_solutions.append(trace_solution)
            offset += trace_size
        
        # Extract multipliers
        n_multipliers = self.constraint_manager.n_multipliers if self.constraint_manager else 0
        if n_multipliers > 0:
            multipliers = global_solution[offset:offset+n_multipliers]
        else:
            multipliers = np.array([])
        
        return trace_solutions, multipliers
    
    def validate_setup(self, verbose: bool = False) -> bool:
        """
        Validate that all setup components are working correctly.
        
        Args:
            verbose: If True, print detailed validation information
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            self._ensure_initialized()
            
            if verbose:
                print(f"Validating setup for problem: {self.problem_name}")
                print(f"Number of domains: {len(self.problems)}")
                print(f"Total DOFs: {self.global_assembler.total_dofs}")
            
            # Test initial conditions
            trace_solutions, multipliers = self.create_initial_conditions()
            if verbose:
                print(f"✓ Initial conditions created")
            
            # Test global vector assembly/extraction
            global_solution = self.create_global_solution_vector(trace_solutions, multipliers)
            extracted_traces, extracted_multipliers = self.extract_domain_solutions(global_solution)
            
            # Verify round-trip consistency
            for i, (orig, extracted) in enumerate(zip(trace_solutions, extracted_traces)):
                if not np.allclose(orig, extracted):
                    if verbose:
                        print(f"✗ Round-trip test failed for domain {i}")
                    return False
            
            if not np.allclose(multipliers, extracted_multipliers):
                if verbose:
                    print(f"✗ Round-trip test failed for multipliers")
                return False
            
            if verbose:
                print(f"✓ Global vector round-trip test passed")
            
            # Test bulk data manager
            bulk_solutions = []
            for i in range(len(self.problems)):
                problem = self.problems[i]
                discretization = self.global_discretization.spatial_discretizations[i]
                bulk_sol = self.bulk_data_manager.create_bulk_data(i, problem, discretization)
                bulk_solutions.append(bulk_sol)
            
            if verbose:
                print(f"✓ Bulk solutions created")
            
            # Test forcing term computation
            forcing_terms = self.bulk_data_manager.compute_forcing_terms(bulk_solutions, 
                                                                         self.problems, 
                                                                         self.global_discretization.spatial_discretizations, 
                                                                         0.0, 
                                                                         self.global_discretization.dt
                                                                         )

            if verbose:
                print(f"✓ Forcing terms computed")
            
            # Test residual/jacobian computation
            global_residual, global_jacobian = self.global_assembler.assemble_residual_and_jacobian(
                global_solution=global_solution,
                forcing_terms=forcing_terms,
                static_condensations=self._static_condensations,
                time=0.0
            )
            
            if verbose:
                print(f"✓ Global residual and Jacobian assembled")
                print(f"  Residual norm: {np.linalg.norm(global_residual):.6e}")
                print(f"  Jacobian condition number: {np.linalg.cond(global_jacobian):.2e}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"✗ Validation failed: {e}")
                import traceback
                traceback.print_exc()
            return False


def create_solver_setup(problem_module: str = "ooc1d.problems.test_problem2") -> SolverSetup:
    """
    Factory function to create and initialize a solver setup.
    
    Args:
        problem_module: String path to problem module
        
    Returns:
        Initialized SolverSetup instance
    """
   
    setup = SolverSetup(problem_module)
    setup.initialize()
    return setup


# Convenience functions for common operations
def quick_setup(problem_module: str = "ooc1d.problems.test_problem2", validate: bool = True) -> SolverSetup:
    """
    Quick setup with optional validation.
    
    Args:
        problem_module: String path to problem module
        validate: If True, run validation tests
        
    Returns:
        Validated SolverSetup instance
    """
    
    setup = create_solver_setup(problem_module)
    if validate:
        if not setup.validate_setup(verbose=True):
            raise RuntimeError("Setup validation failed")
    
    return setup
