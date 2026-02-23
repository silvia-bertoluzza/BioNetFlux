from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple
from .problem import Problem
from .discretization import Discretization, GlobalDiscretization

class StaticCondensationBase(ABC):
    """
    Abstract base class for static condensation implementations.
    Different problem types can inherit from this and implement their specific logic.

    Attributes:
        flux_orders: List of polynomial degrees for the flux variable of each
            equation.  Entry ``k`` is 0 if the flux of equation *k* is P0
            (1 DOF per element) or 1 if it is P1 (2 DOFs per element).
            Must be set by every concrete subclass before ``build_matrices()``
            is called.  Length must equal ``problem.neq``.
    """
    
    def __init__(self, problem: Problem, global_disc: GlobalDiscretization, elementary_matrices: Any, ipb: int=0):
        """
        Initialize static condensation for a specific problem type.
        
        Args:
            problem: Problem definition
            global_disc: Global discretization object
            elementary_matrices: Pre-computed elementary matrices
            ipb: Index of the problem/domain in case of multiple problems/domains
        """
        self.problem = problem
        self.discretization = global_disc.spatial_discretizations[ipb]
        self.elementary_matrices = elementary_matrices
        self.sc_matrices = {}
        self.dt = global_disc.dt
        self.tau = self.discretization.tau  # Stabilization parameters
        self.flux_orders: List[int] = []  # To be set by subclass
        
    @abstractmethod
    def build_matrices(self) -> Dict[str, np.ndarray]:
        """
        Build static condensation matrices (equivalent to scBlocks.m).
        Must be implemented by each problem type.
        
        Returns:
            Dictionary of pre-computed matrices for static condensation
        """
        pass
    
    @abstractmethod
    def static_condensation(self, 
                          local_trace: np.ndarray, 
                          local_source: np.ndarray, 
                          **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform static condensation step (equivalent to StaticC.m).
        Must be implemented by each problem type.
        
        Args:
            local_trace: Local trace unknowns
            local_source: Local source terms
            **kwargs: Additional problem-specific parameters
            
        Returns:
            Tuple of (local_solution, flux, flux_trace, jacobian)
        """
        pass
    
    @abstractmethod
    def assemble_forcing_term(self, *args, **kwargs) -> np.ndarray:
        """
        str
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Assembled right-hand side in correct format for static condensation
        """
        pass 
    
    @property
    def flux_dofs_per_element(self) -> List[int]:
        """Number of flux DOFs per element for each equation.

        Derived from ``flux_orders``: order 0 (P0) gives 1 DOF,
        order 1 (P1) gives 2 DOFs.
        """
        return [order + 1 for order in self.flux_orders]

    @property
    def total_flux_dofs_per_element(self) -> int:
        """Total number of flux DOFs per element across all equations."""
        return sum(self.flux_dofs_per_element)

    def _validate_flux_orders(self) -> None:
        """Validate that flux_orders has been properly set by the subclass.

        Raises:
            ValueError: If flux_orders length does not match neq or contains
                invalid values.
        """
        if len(self.flux_orders) != self.problem.neq:
            raise ValueError(
                f"flux_orders has length {len(self.flux_orders)} but problem "
                f"has neq={self.problem.neq}. Every concrete "
                f"StaticCondensation subclass must set self.flux_orders "
                f"to a list of length neq."
            )
        for i, order in enumerate(self.flux_orders):
            if order not in (0, 1):
                raise ValueError(
                    f"flux_orders[{i}] = {order}, expected 0 (P0) or 1 (P1)."
                )

    def get_matrices(self) -> Dict[str, np.ndarray]:
        """Get all pre-computed matrices."""
        return self.sc_matrices

 