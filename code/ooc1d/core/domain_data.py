"""
Lightweight container for extracted domain-specific data.
"""

import numpy as np
from typing import List, Optional, Callable


class DomainData:
    """
    Lightweight container storing only essential extracted data for a domain.
    
    This avoids storing full problem/discretization objects and extracts
    only what's needed for bulk operations.
    """
    
    def __init__(self,
                 neq: int,
                 n_elements: int,
                 nodes: np.ndarray,
                 element_length: float,
                 mass_matrix: np.ndarray,
                 trace_matrix: np.ndarray,
                 initial_conditions: List[Optional[Callable]],
                 forcing_functions: List[Optional[Callable]]):
        """
        Initialize domain data container.
        
        Args:
            neq: Number of equations
            n_elements: Number of elements
            nodes: Node coordinates
            element_length: Length of elements
            mass_matrix: 2×2 mass matrix from static condensation
            trace_matrix: Trace matrix from static condensation
            initial_conditions: List of initial condition functions
            forcing_functions: List of forcing functions
        """
        self.neq = neq
        self.n_elements = n_elements
        self.nodes = nodes.copy()
        self.element_length = element_length
        self.mass_matrix = mass_matrix.copy()
        self.trace_matrix = trace_matrix.copy()
        self.initial_conditions = initial_conditions.copy()
        self.forcing_functions = forcing_functions.copy()
    
    def __str__(self) -> str:
        return (f"DomainData(neq={self.neq}, n_elements={self.n_elements}, "
                f"element_length={self.element_length})")
