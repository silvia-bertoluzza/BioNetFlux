import numpy as np
from typing import Callable, List, Optional, Union
from .discretization import Discretization, GlobalDiscretization

class Problem:
    """
    Problem definition class for 1D Keller-Segel type problems.
    
    Equivalent to MATLAB problem{ipb} structure.
    """
    
    def __init__(self, 
                 neq: int = 2,
                 domain_start: float = 0.0,
                 domain_length: float = 1.0,
                 parameters: np.ndarray = None,
                 problem_type: str = "keller_segel",
                 name: str = "unnamed_problem"):
        """
        Initialize problem parameters.
        
        Args:
            neq: Number of equations (typically 2)
            domain_start: Domain start coordinate (A)
            domain_length: Domain length (L)
            parameters: Vector of physical parameters (length n_parameters)
            problem_type: String identifying problem type for implementation selection
            name: Descriptive name for the problem instance
        """
        self.neq = neq
        self.domain_start = domain_start
        self.domain_length = domain_length
        self.domain_end = domain_start + domain_length
        self.name = name
        
        # Physical parameters as vector
        self.parameters = parameters if parameters is not None else np.array([1.0, 1.0, 0.0, 0.0])  # Default: [mu, nu, a, b]
        self.n_parameters = len(self.parameters)
        
        # Problem type for implementation selection
        self.type = problem_type
        
        # Variable names
        self.u_names = ['u', 'phi'] if neq == 2 else [f'u{i}' for i in range(neq)]
        
        # Functions (to be set by user)
        self.chi: Optional[Callable] = None
        self.dchi: Optional[Callable] = None
        self.force: List[Callable] = [lambda s, t: np.zeros_like(s)] * neq
        self.u0: List[Callable] = [lambda s: np.zeros_like(s)] * neq
        self.solution: List[Callable] = [lambda s, t: np.zeros_like(s)] * neq
        
        # Boundary conditions
        self.flux_u0: List[Callable] = [lambda t: 0.0] * neq  # Left boundary
        self.flux_u1: List[Callable] = [lambda t: 0.0] * neq  # Right boundary
        self.neumann_data = np.zeros(4)  # Boundary data array
        
        # Domain endpoint coordinates for visualization
        self.extrema = [(domain_start, 0.0), (domain_start + domain_length, 0.0)]
        
        # Names for unknown variables
        self.unknown_names = [f"Unknown n. {i+1}" for i in range(neq)]
    
    def set_chemotaxis(self, chi: Callable, dchi: Callable):
        """Set chemotactic sensitivity function and its derivative."""
        self.chi = chi
        self.dchi = dchi
        
    def set_force(self, equation_idx: int, force_func: Callable):
        """Set source term for specified equation."""
        self.force[equation_idx] = force_func
    
    def set_solution(self, equation_idx: int, solution_func: Callable):
        """Set solution term for specified equation."""
        self.solution[equation_idx] = solution_func

    def set_initial_condition(self, equation_idx: int, u0_func: Callable):
        """Set initial condition for specified equation."""
        self.u0[equation_idx] = u0_func
        
    def set_boundary_flux(self, equation_idx: int, 
                         left_flux: Optional[Callable] = None,
                         right_flux: Optional[Callable] = None):
        """Set boundary flux functions."""
        if left_flux is not None:
            self.flux_u0[equation_idx] = left_flux
        if right_flux is not None:
            self.flux_u1[equation_idx] = right_flux
    
    def get_parameter(self, index: int) -> float:
        """Get parameter by index."""
        return self.parameters[index]
    
    def set_parameter(self, index: int, value: float):
        """Set parameter by index."""
        self.parameters[index] = value
    
    def set_parameters(self, parameters: np.ndarray):
        """Set all parameters."""
        self.parameters = parameters
        self.n_parameters = len(parameters)
    
    def set_extrema(self, point1: tuple, point2: tuple):
        """
        Set the domain extrema coordinates.
        
        Args:
            point1: Tuple (x, y) for the left endpoint (corresponding to A)
            point2: Tuple (x, y) for the right endpoint (corresponding to A+L)
        """
        self.extrema = [point1, point2]

    def get_extrema(self):
        """
        Get the domain extrema coordinates.
        
        Returns:
            List of two tuples [(x1, y1), (x2, y2)] representing domain endpoints
        """
        return self.extrema
    
    def set_function(self, function_name: str, function: Callable):
        """
        Generic method to set any function as an attribute of the problem.
        
        This provides maximum flexibility for adding custom functions to problems
        while maintaining backward compatibility with existing specific methods.
        
        Args:
            function_name (str): Name of the attribute to create
            function (callable): Function to assign to the attribute
            
        Example:
            problem.set_function('lambda_function', lambda x: np.ones_like(x))
            problem.set_function('custom_source', lambda x, t: x**2 * t)
        """
        if not isinstance(function_name, str):
            raise TypeError(f"function_name must be a string, got {type(function_name)}")
        
        if not callable(function):
            raise TypeError(f"function must be callable, got {type(function)}")
        
        # Set the attribute dynamically
        setattr(self, function_name, function)
