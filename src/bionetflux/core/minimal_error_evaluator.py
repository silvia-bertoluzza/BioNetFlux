"""
Minimal Error Evaluator for HDG method.

This module provides lean error computation for both trace and bulk solutions
using L2 norms with 4-point Legendre quadrature. Handles missing analytical
solutions gracefully with warning messages.
"""

import numpy as np
import warnings
from typing import List, Dict, Optional, Callable, Union

from .problem import Problem
from .discretization import Discretization
from .bulk_data import BulkData
from bionetflux.utils.elementary_matrices import ElementaryMatrices


class MinimalErrorEvaluator:
    """
    Minimal error evaluator for HDG solutions.
    
    Computes L2 errors for trace (weighted Euclidean) and bulk (standard L2) solutions
    with separate tracking per equation and lean output structure avoiding duplication.
    """
    
    def __init__(self):
        """Initialize minimal error evaluator with 4-point Legendre quadrature."""
        self.elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
        self.quad_matrix = self.elementary_matrices.get_matrix('QUAD')  # (2, 4)
        self.quad_nodes = self.elementary_matrices.get_matrix('qnodes')  # (4,)
    
    def compute_trace_error(self, 
                           trace_solutions: List[np.ndarray],
                           problems: List[Problem],
                           discretizations: List[Discretization],
                           time: float) -> Dict[str, Dict[int, Union[float, None]]]:
        """
        Compute weighted Euclidean error for trace solutions.
        
        Uses sqrt(h) scaling where h is the mesh size.
        
        Args:
            trace_solutions: List of trace solution arrays, one per domain
            problems: List of Problem objects with analytical solutions
            discretizations: List of Discretization objects
            time: Time for analytical solution evaluation
            
        Returns:
            Dict with structure {'local': {domain_idx: {eq_idx: error}}, 
                               'global': {eq_idx: global_error}}
        """
        self._validate_inputs(trace_solutions, problems, discretizations, "trace")
        
        n_domains = len(problems)
        local_errors = {}
        global_errors_squared = {}
        
        # Initialize global error accumulation
        for domain_idx in range(n_domains):
            neq = problems[domain_idx].neq
            for eq_idx in range(neq):
                if eq_idx not in global_errors_squared:
                    global_errors_squared[eq_idx] = 0.0
        
        # Compute per-domain errors
        for domain_idx in range(n_domains):
            problem = problems[domain_idx]
            discretization = discretizations[domain_idx]
            trace_solution = trace_solutions[domain_idx]
            
            domain_errors = self._compute_domain_trace_error(
                trace_solution, problem, discretization, time
            )
            local_errors[domain_idx] = domain_errors
            
            # Accumulate for global error (sum of squares)
            for eq_idx, error_val in domain_errors.items():
                if error_val is not None:
                    global_errors_squared[eq_idx] += error_val ** 2
        
        # Compute global errors (root-sum-of-squares)
        global_errors = {}
        for eq_idx, error_sq in global_errors_squared.items():
            global_errors[eq_idx] = np.sqrt(error_sq) if error_sq > 0 else None
        
        return {'local': local_errors, 'global': global_errors}
    
    def compute_bulk_error(self,
                          bulk_solutions: List[BulkData],
                          problems: List[Problem],
                          discretizations: List[Discretization],
                          time: float) -> Dict[str, Dict[int, Union[float, None]]]:
        """
        Compute L2 error for bulk solutions using 4-point Legendre quadrature.
        
        Args:
            bulk_solutions: List of BulkData objects, one per domain
            problems: List of Problem objects with analytical solutions
            discretizations: List of Discretization objects
            time: Time for analytical solution evaluation
            
        Returns:
            Dict with structure {'local': {domain_idx: {eq_idx: error}}, 
                               'global': {eq_idx: global_error}}
        """
        self._validate_inputs(bulk_solutions, problems, discretizations, "bulk")
        
        n_domains = len(problems)
        local_errors = {}
        global_errors_squared = {}
        
        # Initialize global error accumulation
        for domain_idx in range(n_domains):
            neq = problems[domain_idx].neq
            for eq_idx in range(neq):
                if eq_idx not in global_errors_squared:
                    global_errors_squared[eq_idx] = 0.0
        
        # Compute per-domain errors
        for domain_idx in range(n_domains):
            problem = problems[domain_idx]
            discretization = discretizations[domain_idx]
            bulk_solution = bulk_solutions[domain_idx]
            
            domain_errors = self._compute_domain_bulk_error(
                bulk_solution, problem, discretization, time
            )
            local_errors[domain_idx] = domain_errors
            
            # Accumulate for global error (sum of squares)
            for eq_idx, error_val in domain_errors.items():
                if error_val is not None:
                    global_errors_squared[eq_idx] += error_val ** 2
        
        # Compute global errors (root-sum-of-squares)
        global_errors = {}
        for eq_idx, error_sq in global_errors_squared.items():
            global_errors[eq_idx] = np.sqrt(error_sq) if error_sq > 0 else None
        
        return {'local': local_errors, 'global': global_errors}
    
    def compute_flux_error(self,
                          flux_data: List,
                          problems: List[Problem],
                          discretizations: List[Discretization],
                          static_condensations: List,
                          time: float) -> Dict[str, Dict[int, Union[float, None]]]:
        """
        Compute L2 error for flux solutions using 4-point Legendre quadrature.
        
        Handles both P0 (constant) and P1 (linear) flux representations
        based on ``flux_orders`` stored on each static condensation object.
        
        Args:
            flux_data: List of flux coefficient arrays (one per domain).
                       Each array has shape (total_flux_dofs_per_element, N).
                       May be None for domains that don't produce flux data.
            problems: List of Problem objects with ``flux_solution`` attributes
            discretizations: List of Discretization objects
            static_condensations: List of StaticCondensationBase objects
                (used to read ``flux_orders``)
            time: Time for analytical solution evaluation
            
        Returns:
            Dict with structure ``{'local': {domain_idx: {eq_idx: error}},
            'global': {eq_idx: global_error}}``
        """
        n_domains = len(problems)
        local_errors = {}
        global_errors_squared: Dict[int, float] = {}
        equations_computed: Dict[int, bool] = {}
        
        # Initialize global error accumulation
        for domain_idx in range(n_domains):
            neq = problems[domain_idx].neq
            for eq_idx in range(neq):
                if eq_idx not in global_errors_squared:
                    global_errors_squared[eq_idx] = 0.0
                    equations_computed[eq_idx] = False
        
        for domain_idx in range(n_domains):
            problem = problems[domain_idx]
            discretization = discretizations[domain_idx]
            sc = static_condensations[domain_idx]
            flux_array = flux_data[domain_idx] if flux_data is not None else None
            
            domain_errors = self._compute_domain_flux_error(
                flux_array, problem, discretization, sc, time
            )
            local_errors[domain_idx] = domain_errors
            
            for eq_idx, error_val in domain_errors.items():
                if error_val is not None:
                    global_errors_squared[eq_idx] += error_val ** 2
                    equations_computed[eq_idx] = True
        
        global_errors = {}
        for eq_idx, error_sq in global_errors_squared.items():
            if equations_computed.get(eq_idx, False):
                global_errors[eq_idx] = np.sqrt(max(0, error_sq))
            else:
                global_errors[eq_idx] = None
        
        return {'local': local_errors, 'global': global_errors}
    
    def _compute_domain_flux_error(self,
                                   flux_array,
                                   problem: Problem,
                                   discretization: Discretization,
                                   static_condensation,
                                   time: float) -> Dict[int, Union[float, None]]:
        """Compute flux L2 error for a single domain using 4-point quadrature."""
        neq = problem.neq
        n_elements = discretization.n_elements
        nodes = discretization.nodes
        flux_orders = static_condensation.flux_orders
        flux_dofs = static_condensation.flux_dofs_per_element  # list of DOFs per eq
        
        # Check if flux data is available
        if flux_array is None:
            return {eq_idx: None for eq_idx in range(neq)}
        
        # Check analytical flux solutions availability
        flux_funcs = self._get_flux_analytical_functions(problem)
        if flux_funcs is None:
            warnings.warn("No analytical flux solutions found for flux error computation")
            return {eq_idx: None for eq_idx in range(neq)}
        
        domain_errors = {}
        
        # Compute DOF offsets for each equation within the J column
        dof_offsets = []
        offset = 0
        for eq_idx in range(neq):
            dof_offsets.append(offset)
            offset += flux_dofs[eq_idx]
        
        for eq_idx in range(neq):
            if eq_idx >= len(flux_funcs) or flux_funcs[eq_idx] is None:
                warnings.warn(f"No analytical flux solution for equation {eq_idx}")
                domain_errors[eq_idx] = None
                continue
            
            try:
                error_squared = 0.0
                order = flux_orders[eq_idx]
                n_dofs = flux_dofs[eq_idx]
                eq_offset = dof_offsets[eq_idx]
                
                for elem_idx in range(n_elements):
                    x_left = nodes[elem_idx]
                    x_right = nodes[elem_idx + 1]
                    h_elem = x_right - x_left
                    
                    # Extract flux coefficients for this element and equation
                    coeffs = flux_array[eq_offset:eq_offset + n_dofs, elem_idx]
                    
                    # Map quadrature nodes to physical element
                    xi_01 = (self.quad_nodes + 1) / 2  # Map [-1,1] to [0,1]
                    mapped_nodes = x_left + xi_01 * h_elem
                    
                    # Evaluate numerical flux at quadrature points
                    if order == 0:
                        # P0: constant flux value
                        numerical_values = np.full_like(xi_01, coeffs[0])
                    else:
                        # P1: linear interpolation c0*(1-ξ) + c1*ξ
                        numerical_values = coeffs[0] * (1 - xi_01) + coeffs[1] * xi_01
                    
                    # Evaluate analytical flux solution at quadrature points
                    analytical_values = flux_funcs[eq_idx](mapped_nodes, time)
                    if np.isscalar(analytical_values):
                        analytical_values = np.full_like(mapped_nodes, analytical_values)
                    
                    # Integrate (numerical - analytical)^2
                    error_function = numerical_values - analytical_values
                    error_squared_values = error_function ** 2
                    
                    element_contribution = h_elem * np.dot(
                        self.quad_matrix[0, :], error_squared_values
                    )
                    error_squared += element_contribution
                
                domain_errors[eq_idx] = np.sqrt(max(0, error_squared))
                
            except Exception as e:
                warnings.warn(f"Error computing flux L2 error for equation {eq_idx}: {e}")
                domain_errors[eq_idx] = None
        
        return domain_errors
    
    def _get_flux_analytical_functions(self, problem: Problem) -> Optional[List[Callable]]:
        """Extract analytical flux functions from Problem object."""
        if hasattr(problem, 'flux_solution') and problem.flux_solution is not None:
            if isinstance(problem.flux_solution, (list, tuple)):
                return list(problem.flux_solution)
        return None
    
    def _compute_domain_trace_error(self,
                                   trace_solution: np.ndarray,
                                   problem: Problem,
                                   discretization: Discretization,
                                   time: float) -> Dict[int, Union[float, None]]:
        """Compute trace error for a single domain."""
        neq = problem.neq
        n_nodes = discretization.n_elements + 1
        nodes = discretization.nodes
        h = discretization.element_length
        
        # Check analytical solutions availability
        analytical_funcs = self._get_analytical_functions(problem)
        if analytical_funcs is None:
            warnings.warn(f"No analytical solutions found for trace error computation")
            return {eq_idx: None for eq_idx in range(neq)}
        
        domain_errors = {}
        
        for eq_idx in range(neq):
            # Check if analytical function exists for this equation
            if eq_idx >= len(analytical_funcs) or analytical_funcs[eq_idx] is None:
                warnings.warn(f"No analytical solution for equation {eq_idx}")
                domain_errors[eq_idx] = None
                continue
            
            # Extract trace values for this equation
            eq_start = eq_idx * n_nodes
            eq_end = eq_start + n_nodes
            eq_trace_values = trace_solution[eq_start:eq_end]
            
            # Evaluate analytical solution at nodes
            try:
                analytical_values = analytical_funcs[eq_idx](nodes, time)
                if np.isscalar(analytical_values):
                    analytical_values = np.full(n_nodes, analytical_values)
                
                # Compute weighted Euclidean error: sqrt(h) * ||u_num - u_exact||_2
                error_vector = eq_trace_values - analytical_values
                euclidean_norm = np.linalg.norm(error_vector)
                weighted_error = np.sqrt(h) * euclidean_norm
                domain_errors[eq_idx] = weighted_error
                
            except Exception as e:
                warnings.warn(f"Error evaluating analytical solution for equation {eq_idx}: {e}")
                domain_errors[eq_idx] = None
        
        return domain_errors
    
    def _compute_domain_bulk_error(self,
                                  bulk_solution: BulkData,
                                  problem: Problem,
                                  discretization: Discretization,
                                  time: float) -> Dict[int, Union[float, None]]:
        """Compute bulk L2 error for a single domain using 4-point quadrature."""
        neq = problem.neq
        n_elements = discretization.n_elements
        nodes = discretization.nodes
        
        # Check analytical solutions availability
        analytical_funcs = self._get_analytical_functions(problem)
        if analytical_funcs is None:
            warnings.warn(f"No analytical solutions found for bulk error computation")
            return {eq_idx: None for eq_idx in range(neq)}
        
        domain_errors = {}
        
        for eq_idx in range(neq):
            # Check if analytical function exists for this equation
            if eq_idx >= len(analytical_funcs) or analytical_funcs[eq_idx] is None:
                warnings.warn(f"No analytical solution for equation {eq_idx}")
                domain_errors[eq_idx] = None
                continue
            
            try:
                error_squared = 0.0
                
                # Integrate over each element using 4-point quadrature
                for elem_idx in range(n_elements):
                    x_left = nodes[elem_idx]
                    x_right = nodes[elem_idx + 1]
                    h_elem = x_right - x_left
                    
                    # Get bulk coefficients for this element and equation
                    element_coeffs = bulk_solution.get_element_data(elem_idx)
                    c0 = element_coeffs[2 * eq_idx]      # Left coefficient
                    c1 = element_coeffs[2 * eq_idx + 1]  # Right coefficient
                    
                    # Map quadrature nodes to physical element
                    xi_01 = (self.quad_nodes + 1) / 2  # Map [-1,1] to [0,1]
                    mapped_nodes = x_left + xi_01 * h_elem
                    
                    # Evaluate numerical solution at quadrature points
                    numerical_values = c0 * (1 - xi_01) + c1 * xi_01
                    
                    # Evaluate analytical solution at quadrature points
                    analytical_values = analytical_funcs[eq_idx](mapped_nodes, time)
                    if np.isscalar(analytical_values):
                        analytical_values = np.full_like(mapped_nodes, analytical_values)
                    
                    # Compute error function at quadrature points
                    error_function = numerical_values - analytical_values
                    
                    # Integrate using quadrature: weights are built into QUAD matrix
                    # For L2 norm: ∫(u_num - u_exact)^2 dx
                    error_squared_values = error_function ** 2
                    
                    # Use first row of QUAD matrix for integration (both rows should give same result)
                    element_contribution = h_elem * np.dot(self.quad_matrix[0, :], error_squared_values)
                    error_squared += element_contribution
                
                domain_errors[eq_idx] = np.sqrt(max(0, error_squared))
                
            except Exception as e:
                warnings.warn(f"Error computing bulk L2 error for equation {eq_idx}: {e}")
                domain_errors[eq_idx] = None
        
        return domain_errors
    
    def _get_analytical_functions(self, problem: Problem) -> Optional[List[Callable]]:
        """Extract analytical functions from Problem object."""
        if hasattr(problem, 'solution') and problem.solution is not None:
            if isinstance(problem.solution, (list, tuple)):
                return list(problem.solution)
        return None
    
    def _validate_inputs(self, solutions, problems, discretizations, solution_type: str):
        """Validate input arguments for consistency."""
        n_domains = len(problems)
        
        if len(solutions) != n_domains:
            raise ValueError(f"Number of {solution_type} solutions ({len(solutions)}) "
                           f"must match number of problems ({n_domains})")
        
        if len(discretizations) != n_domains:
            raise ValueError(f"Number of discretizations ({len(discretizations)}) "
                           f"must match number of problems ({n_domains})")
        
        # Validate individual domains
        for domain_idx in range(n_domains):
            problem = problems[domain_idx]
            discretization = discretizations[domain_idx]
            solution = solutions[domain_idx]
            
            if not hasattr(problem, 'neq'):
                raise ValueError(f"Problem {domain_idx} missing 'neq' attribute")
            
            if not hasattr(discretization, 'n_elements'):
                raise ValueError(f"Discretization {domain_idx} missing 'n_elements' attribute")
            
            if not hasattr(discretization, 'nodes'):
                raise ValueError(f"Discretization {domain_idx} missing 'nodes' attribute")
            
            # Validate solution dimensions
            if solution_type == "trace":
                expected_size = problem.neq * (discretization.n_elements + 1)
                if solution.size != expected_size:
                    raise ValueError(f"Trace solution {domain_idx} has size {solution.size}, "
                                   f"expected {expected_size}")
            elif solution_type == "bulk":
                expected_shape = (2 * problem.neq, discretization.n_elements)
                actual_shape = solution.get_data().shape
                if actual_shape != expected_shape:
                    raise ValueError(f"Bulk solution {domain_idx} has shape {actual_shape}, "
                                   f"expected {expected_shape}")
    
    def test(self) -> bool:
        """
        Test method to validate MinimalErrorEvaluator functionality.
        
        Returns:
            True if all tests pass, False otherwise
        """
        print("Testing MinimalErrorEvaluator")
        
        try:
            # Test 1: Initialization
            if self.quad_matrix is None or self.quad_nodes is None:
                print("FAIL: ElementaryMatrices initialization failed")
                return False
            
            if self.quad_matrix.shape != (2, 4):
                print(f"FAIL: Quadrature matrix shape {self.quad_matrix.shape} != (2, 4)")
                return False
            
            if self.quad_nodes.shape != (4,):
                print(f"FAIL: Quadrature nodes shape {self.quad_nodes.shape} != (4,)")
                return False
            
            print("PASS: ElementaryMatrices initialization and shapes")
            
            # Test 2: Input validation with invalid inputs
            try:
                self._validate_inputs([], [None], [], "trace")
                print("FAIL: No exception raised for mismatched input lengths")
                return False
            except ValueError:
                print("PASS: Input validation correctly raises ValueError")
            
            # Test 3: Analytical function extraction
            class MockProblem:
                def __init__(self, solution_list):
                    self.solution = solution_list
                    self.neq = len(solution_list) if solution_list else 0
            
            # Test with valid solution list
            mock_problem = MockProblem([lambda x, t: x + t, lambda x, t: x * t])
            funcs = self._get_analytical_functions(mock_problem)
            if funcs is None or len(funcs) != 2:
                print("FAIL: Analytical function extraction failed")
                return False
            
            # Test with no solutions
            mock_problem_no_sol = MockProblem(None)
            funcs_none = self._get_analytical_functions(mock_problem_no_sol)
            if funcs_none is not None:
                print("FAIL: Should return None for missing solutions")
                return False
            
            print("PASS: Analytical function extraction")
            
            # Test 4: Quadrature node mapping
            # Test that quadrature nodes are in expected range [-1, 1]
            if np.min(self.quad_nodes) < -1 or np.max(self.quad_nodes) > 1:
                print(f"FAIL: Quadrature nodes outside [-1,1]: {self.quad_nodes}")
                return False
            
            # Test mapping to [0,1]
            xi_01 = (self.quad_nodes + 1) / 2
            if np.min(xi_01) < 0 or np.max(xi_01) > 1:
                print(f"FAIL: Mapped nodes outside [0,1]: {xi_01}")
                return False
            
            print("PASS: Quadrature node mapping")
            
            print("✓ All MinimalErrorEvaluator tests passed!")
            return True
            
        except Exception as e:
            print(f"FAIL: Unexpected error during testing: {e}")
            return False
    
    def __str__(self) -> str:
        return (f"MinimalErrorEvaluator(quad_nodes={len(self.quad_nodes)}, "
                f"quad_matrix_shape={self.quad_matrix.shape})")
    
    def __repr__(self) -> str:
        return f"MinimalErrorEvaluator(elementary_matrices={self.elementary_matrices})"