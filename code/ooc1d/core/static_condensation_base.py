from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple
from .problem import Problem
from .discretization import Discretization, GlobalDiscretization

class StaticCondensationBase(ABC):
    """
    Abstract base class for static condensation implementations.
    Different problem types can inherit from this and implement their specific logic.
    """
    
    def __init__(self, problem: Problem, global_disc: GlobalDiscretization, elementary_matrices: Any, ipb: int=0):
        """
        Initialize static condensation for a specific problem type.
        
        Args:
            problem: Problem definition
            global_disc: Global discretization object
            elementary_matrices: Pre-computed elementary matrices
        """
        self.problem = problem
        self.discretization = global_disc.spatial_discretizations[ipb]
        self.elementary_matrices = elementary_matrices
        self.sc_matrices = {}
        self.dt = global_disc.dt
        self.tau = self.discretization.tau  # Stabilization parameters
        
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
    
    def get_matrices(self) -> Dict[str, np.ndarray]:
        """Get all pre-computed matrices."""
        return self.sc_matrices

    @abstractmethod
    def assemble_forcing_term(self, *args, **kwargs) -> np.ndarray:
        """
        Assemble the right-hand side for the static condensation system.
        Must be implemented by each problem type. Might depend on local sources, previous solutions, etc.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Assembled right-hand side in correct format for static condensation
        """
        pass 
 