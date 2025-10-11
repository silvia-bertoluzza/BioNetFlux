"""
Bulk data management for HDG method.
Alternative to BulkSolution with flexible set_data method.
"""

import numpy as np
from .problem import Problem 
from .discretization import Discretization
from ooc1d.utils.elementary_matrices import ElementaryMatrices
from typing import List, Optional, Union, Callable



class BulkData:
    """
    Manages bulk data for a single domain with flexible initialization.
    
    This class stores bulk coefficients in a 2*neq x n_elements array and provides
    multiple ways to set the data depending on the input format.
    """

    def __init__(self, problem: Problem, discretization: Discretization, dual: bool = False):
        """
        Initialize BulkData instance.
        
        Args:
            problem: The problem instance containing relevant information
            discretization: The discretization instance containing mesh information
            dual: Boolean flag for dual formulation (default False)

        """
        self.n_elements = discretization.n_elements
        self.neq = problem.neq
        self.dual = dual
        self.nodes = discretization.nodes

        h = problem.domain_length / discretization.n_elements

        elementary_matrices = ElementaryMatrices(orthonormal_basis=False)
        self.trace_matrix = elementary_matrices.get_matrix('T')
        self.mass_matrix = h * elementary_matrices.get_matrix('M')
        self.quad_matrix = h * elementary_matrices.get_matrix('QUAD')
        self.quad_nodes = elementary_matrices.get_matrix('qnodes')
        
        # Initialize data array: shape (2*neq, n_elements)
        self.data = np.zeros((2 * self.neq, self.n_elements))

    def set_data(self, 
                 input_data: Union[np.ndarray, List[Callable]], 
                 time: float = 0.0):
        """
        Set bulk data with multiple input formats.
        
        Args:
            input_data: Can be:
                - np.ndarray of shape (2*neq, n_elements): direct coefficient array
                - List of neq callable functions f(s,t): evaluate at nodes
                - np.ndarray of size neq*(n_elements+1): trace values at nodes
            time: Time for function evaluation
            trace_matrix: Required for reconstruction from trace values
        """
        if self.dual:
            self._set_data_dual(input_data, time)
        else:
            self._set_data_primal(input_data, time) 
    
    def _set_data_dual(self, input_data, time):
        """Handle data setting for dual formulation (dual=True)."""

        # Type 1: Direct 2*neq x n_elements array
        if isinstance(input_data, np.ndarray) and input_data.shape == (2 * self.neq, self.n_elements):
            self.data = input_data.copy()
            return
        
        # Type 2: List of callable functions
        if isinstance(input_data, list) and len(input_data) == self.neq:
            if self.quad_matrix is None or self.quad_nodes is None:
                raise ValueError("Quadrature data required for dual function-based initialization")
            self._integrate_from_functions(input_data, time, self.quad_matrix, self.quad_nodes)
            return    
        
        if isinstance(input_data, np.ndarray) and input_data.size == self.neq * (self.n_elements + 1):
            if self.mass_matrix is None or self.trace_matrix is None:
                raise ValueError("Quadrature data required for dual trace-based initialization")
            self._integrate_from_trace_vector(input_data, self.trace_matrix, self.mass_matrix)
            return
        
        
    def _integrate_from_functions(self, functions: List[Callable], time: float, quad_matrix: np.ndarray, quad_nodes: np.ndarray):
        """Integrate from a list of functions using quadrature."""
        print("Integrating from functions using quadrature...")
        
        for k in range(self.n_elements):  
            
            element_coeffs = []
            # Get element nodes
            left_node = self.nodes[k]
            right_node = self.nodes[k + 1]
                
            # Map quadrature nodes to element
            a, b = left_node, right_node
            mapped_nodes = 0.5 * (b - a) * quad_nodes + 0.5 * (a + b)
            
            for eq in range(self.neq):
                # Evaluate function at quadrature nodes
                try:
                    f_values = functions[eq](mapped_nodes, time)
                
                except Exception as e:
                    print(f"Warning: Could not evaluate function {eq} at element {k}: {e}")
                    f_values = np.zeros_like(mapped_nodes)
                
                # Compute integral using quadrature weights
                integral = quad_matrix @ f_values
                element_coeffs.extend(integral)    
                # Store result in data array
            self.data[:, k] = element_coeffs

    def _integrate_from_trace_vector(self, trace_vector: np.ndarray, trace_matrix: np.ndarray, mass_matrix: np.ndarray): 
        """Integrate from trace values using quadrature."""
        # Flatten the trace vector first to handle any shape (1D, row, or column vector)
        trace_flat = trace_vector.flatten()
        
        # Verify size
        expected_size = self.neq * (self.n_elements + 1)
        if trace_flat.size != expected_size:
            raise ValueError(f"Trace vector has size {trace_flat.size}, "
                           f"expected {expected_size} for {self.neq} equations "
                           f"and {self.n_elements + 1} nodes")

        for k in range(self.n_elements):
            
            # Construct element data
            element_coeffs = []
            
            for eq in range(self.neq):
                # Calculate indices for left and right nodes of element k for equation eq
                left_idx = eq * (self.n_elements + 1) + k
                right_idx = eq * (self.n_elements + 1) + (k + 1)
                
                # Extract trace values at element endpoints
                u_left = trace_flat[left_idx]
                u_right = trace_flat[right_idx]
                
                # Solve for bulk coefficients
                local_trace = np.array([u_left, u_right])
                try:
                    local_bulk = np.linalg.solve(trace_matrix, local_trace)
                    local_bulk = mass_matrix @ local_bulk  # Integrate using mass matrix
                    element_coeffs.extend(local_bulk)
                except np.linalg.LinAlgError:
                    print(f"Warning: Singular trace matrix at element {k}, using zeros")
                    element_coeffs.extend([0.0, 0.0])
            self.data[:, k] = element_coeffs


    def _set_data_primal(self, input_data, time):
        """Handle data setting for primal formulation (dual=False)."""
        
        trace_matrix = self.trace_matrix
        
        # Type 1: Direct 2*neq x n_elements array
        if isinstance(input_data, np.ndarray) and input_data.shape == (2 * self.neq, self.n_elements):
            self.data = input_data.copy()
            return
        
        # Type 2: List of callable functions
        if isinstance(input_data, list) and len(input_data) == self.neq:
            if trace_matrix is None:
                raise ValueError("trace_matrix required for function-based initialization")
            self._construct_from_functions(input_data, time)
            return
        
        # Type 3: Trace values vector of size neq*(n_elements+1)
        if isinstance(input_data, np.ndarray) and input_data.size == self.neq * (self.n_elements + 1):
            if trace_matrix is None:
                raise ValueError("trace_matrix required for trace-based initialization")
            self._construct_from_trace_vector(input_data, trace_matrix)
            return
        
        # Invalid input
        raise ValueError(
            f"Invalid input format. Expected:\n"
            f"- Array of shape ({2*self.neq}, {self.n_elements})\n"
            f"- List of {self.neq} callable functions\n"
            f"- Array of size {self.neq * (self.n_elements + 1)} (trace values)"
        )
    
    
    def _construct_from_functions(self, functions: List[Callable], time: float):
        """
        Construct bulk data from callable functions evaluated at nodes.
        
        Args:
            functions: List of neq functions f(s,t)
            time: Time for evaluation
            trace_matrix: Matrix to convert trace values to bulk coefficients
        """
        trace_matrix = self.trace_matrix
        self._validate_trace_matrix(trace_matrix)
        
        for k in range(self.n_elements):
            # Get element nodes
            left_node = self.nodes[k]
            right_node = self.nodes[k + 1]
            
            # Construct element data
            element_coeffs = []
            
            for eq in range(self.neq):
                try:
                    if callable(functions[eq]):
                        u_left = functions[eq](left_node, time)
                        u_right = functions[eq](right_node, time)
                    else:
                        u_left = u_right = 0.0
                except Exception as e:
                    print(f"Warning: Could not evaluate function {eq} at element {k}: {e}")
                    u_left = u_right = 0.0
                
                # Solve trace_matrix @ bulk_coeffs = [u_left, u_right]
                local_trace = np.array([u_left, u_right])
                try:
                    local_bulk = np.linalg.solve(trace_matrix, local_trace)
                    element_coeffs.extend(local_bulk)
                except np.linalg.LinAlgError:
                    print(f"Warning: Singular trace matrix at element {k}, using zeros")
                    element_coeffs.extend([0.0, 0.0])
            
            # Store in data array
            start_row = 0
            for eq in range(self.neq):
                self.data[start_row:start_row+2, k] = element_coeffs[eq*2:(eq+1)*2]
                start_row += 2
    
    def _construct_from_trace_vector(self, trace_vector: np.ndarray, trace_matrix: np.ndarray):
        """
        Construct bulk data from trace values at nodes.
        
        Args:
            trace_vector: Array of size neq*(n_elements+1) with trace values
            trace_matrix: Matrix to convert trace values to bulk coefficients
        """
        self._validate_trace_matrix(trace_matrix)
        
        # Flatten the trace vector first to handle any shape (1D, row, or column vector)
        trace_flat = trace_vector.flatten()
        
        # Verify size
        expected_size = self.neq * (self.n_elements + 1)
        if trace_flat.size != expected_size:
            raise ValueError(f"Trace vector has size {trace_flat.size}, "
                           f"expected {expected_size} for {self.neq} equations "
                           f"and {self.n_elements + 1} nodes")
        
        for k in range(self.n_elements):
            # Construct element data
            element_coeffs = []
            
            for eq in range(self.neq):
                # Calculate indices for left and right nodes of element k for equation eq
                left_idx = eq * (self.n_elements + 1) + k
                right_idx = eq * (self.n_elements + 1) + (k + 1)
                
                # Extract trace values at element endpoints
                u_left = trace_flat[left_idx]
                u_right = trace_flat[right_idx]
                
                # Solve for bulk coefficients
                local_trace = np.array([u_left, u_right])
                try:
                    local_bulk = np.linalg.solve(trace_matrix, local_trace)
                    element_coeffs.extend(local_bulk)
                except np.linalg.LinAlgError:
                    print(f"Warning: Singular trace matrix at element {k}, using zeros")
                    element_coeffs.extend([0.0, 0.0])
            
            # Store in data array
            start_row = 0
            for eq in range(self.neq):
                self.data[start_row:start_row+2, k] = element_coeffs[eq*2:(eq+1)*2]
                start_row += 2

    def _validate_trace_matrix(self, trace_matrix: np.ndarray):
        """Validate trace matrix dimensions and properties."""
        if trace_matrix is None:
            raise ValueError("trace_matrix cannot be None")
        
        expected_shape = (2, 2)  # Assuming 2x2 trace matrix for each equation
        if trace_matrix.shape != expected_shape:
            raise ValueError(f"Expected trace matrix shape {expected_shape}, got {trace_matrix.shape}")
        
        # Check if matrix is singular
        if np.abs(np.linalg.det(trace_matrix)) < 1e-14:
            raise ValueError("Trace matrix is singular")
    
    def get_data(self) -> np.ndarray:
        """Get copy of bulk data array."""
        return self.data.copy()
    
    def get_trace_values(self) -> np.ndarray:
        """
        Extract trace values at nodes (inverse of trace-based construction).
        
        Returns:
            Array of size neq*(n_elements+1) with trace values at nodes
        """
        # This is a placeholder - actual implementation would depend on 
        # the trace matrix and reconstruction method
        trace_values = np.zeros((self.neq, self.n_elements + 1))
        
        # Simple approximation: use bulk coefficients directly
        # In practice, this would involve proper reconstruction
        for k in range(self.n_elements):
            for eq in range(self.neq):
                start_row = eq * 2
                # Use first coefficient as approximation for both nodes
                trace_values[eq, k] = self.data[start_row, k]
                if k == self.n_elements - 1:  # Last element
                    trace_values[eq, k + 1] = self.data[start_row + 1, k]
        
        return trace_values.flatten()
    
    def get_element_data(self, element_idx: int) -> np.ndarray:
        """
        Get bulk coefficients for a specific element.
        
        Args:
            element_idx: Element index (0 to n_elements-1)
            
        Returns:
            Array of shape (2*neq,) with bulk coefficients
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise IndexError(f"Element index {element_idx} out of range")
        
        return self.data[:, element_idx].copy()
    
    def compute_mass(self, mass_matrix: np.ndarray) -> float:
        """
        Compute total mass using provided mass matrix.
        
        Args:
            mass_matrix: Mass matrix for integration
            
        Returns:
            Total mass
        """
        total_mass = 0.0
        
        for eq in range(self.neq):
            start_row = eq * 2
            end_row = start_row + 2
            eq_coeffs = self.data[start_row:end_row, :]
            
            # Mass contribution: integrate over all elements
            eq_mass = np.sum(mass_matrix @ eq_coeffs)
            total_mass += eq_mass
        
        return total_mass
    
    def __str__(self) -> str:
        return (f"BulkData(neq={self.neq}, elements={self.n_elements}, "
                f"dual={self.dual}, "
                f"data_range=[{np.min(self.data):.6e}, {np.max(self.data):.6e}])")
    
    def __repr__(self) -> str:
        return (f"BulkData(n_elements={self.n_elements}, neq={self.neq}, "
                f"dual={self.dual}, data_shape={self.data.shape})")
    
    def test(self) -> bool:
        """
        Test method to validate BulkData instance state and functionality.
        
        Returns:
            True if all tests pass, False otherwise
        """
        print(f"Testing BulkData instance: {self}")
        
        try:
            # Test 1: Check data array shape
            expected_shape = (2 * self.neq, self.n_elements)
            if self.data.shape != expected_shape:
                print(f"FAIL: Data shape {self.data.shape} != expected {expected_shape}")
                return False
            print(f"PASS: Data shape {self.data.shape}")
            
            # Test 2: Check for NaN or infinite values
            if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
                print("FAIL: Data contains NaN or infinite values")
                return False
            print("PASS: No NaN or infinite values in data")
            
            # Test 3: Test matrix properties
            if self.trace_matrix is not None:
                if self.trace_matrix.shape != (2, 2):
                    print(f"FAIL: Trace matrix shape {self.trace_matrix.shape} != (2, 2)")
                    return False
                det = np.linalg.det(self.trace_matrix)
                if abs(det) < 1e-14:
                    print(f"FAIL: Trace matrix is singular (det={det})")
                    return False
                print(f"PASS: Trace matrix is well-conditioned (det={det:.6e})")
            
            # Test 4: Test get_data method
            data_copy = self.get_data()
            if not np.array_equal(data_copy, self.data):
                print("FAIL: get_data() does not return correct copy")
                return False
            print("PASS: get_data() returns correct copy")
            
            # Test 5: Test get_element_data method
            if self.n_elements > 0:
                element_data = self.get_element_data(0)
                if element_data.shape != (2 * self.neq,):
                    print(f"FAIL: Element data shape {element_data.shape} != ({2 * self.neq},)")
                    return False
                print(f"PASS: get_element_data() returns correct shape")
            
            # Test 6: Test bounds checking
            try:
                self.get_element_data(-1)
                print("FAIL: No exception raised for negative element index")
                return False
            except IndexError:
                print("PASS: IndexError raised for negative element index")
            
            try:
                self.get_element_data(self.n_elements)
                print("FAIL: No exception raised for out-of-bounds element index")
                return False
            except IndexError:
                print("PASS: IndexError raised for out-of-bounds element index")
            
            print("All tests passed!")
            return True
            
        except Exception as e:
            print(f"FAIL: Unexpected error during testing: {e}")
            return False

