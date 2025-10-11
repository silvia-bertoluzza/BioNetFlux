import numpy as np
from typing import List, Optional

class Discretization:
    """
    Spatial discretization for a single domain using finite elements.
    Handles only spatial mesh generation and element properties.
    """
    
    def __init__(self, n_elements: int, domain_start: float = 0.0, 
                 domain_length: float = 1.0, stab_constant: float = 1.0):
        self.n_elements = n_elements
        self.domain_start = domain_start
        self.domain_length = domain_length
        self.stab_constant = stab_constant
        
        # Spatial discretization
        self.n_nodes = n_elements + 1
        self.element_length = domain_length / n_elements
        
        # Generate spatial mesh
        self._generate_mesh()
    
    def _generate_mesh(self):
        """Generate the spatial mesh nodes and connectivity."""
        # Node coordinates
        self.nodes = np.linspace(self.domain_start, 
                                self.domain_start + self.domain_length, 
                                self.n_nodes)
        
        # Element connectivity (for linear elements)
        self.elements = np.array([[i, i+1] for i in range(self.n_elements)])
        
        # Element centers
        self.element_centers = self.nodes[:-1] + self.element_length / 2
    
    def get_mesh_info(self) -> dict:
        """Return mesh information dictionary."""
        return {
            'n_elements': self.n_elements,
            'n_nodes': self.n_nodes,
            'element_length': self.element_length,
            'nodes': self.nodes,
            'elements': self.elements,
            'element_centers': self.element_centers,
            'domain_start': self.domain_start,
            'domain_length': self.domain_length,
            'stab_constant': self.stab_constant
        }

    def set_tau(self, tau_values: List[float]):
        """Set stabilization parameters for each equation."""
        self.tau = np.array(tau_values)
        if len(self.tau) == 0:
            raise ValueError("Tau values list cannot be empty.")

class GlobalDiscretization:
    """
    Global discretization managing multiple spatial domains and time discretization.
    Coordinates temporal evolution across all domains.
    """
    
    def __init__(self, spatial_discretizations: List[Discretization]):
        self.spatial_discretizations = spatial_discretizations
        self.n_domains = len(spatial_discretizations)
        
        # Time discretization parameters (initially None)
        self.dt = None
        self.T = None
        self.n_time_steps = None
        self.time_points = None
        
        # Global mesh information
        self._compute_global_info()
    
    def _compute_global_info(self):
        """Compute global mesh information from all domains."""
        self.total_elements = sum(disc.n_elements for disc in self.spatial_discretizations)
        self.total_nodes = sum(disc.n_nodes for disc in self.spatial_discretizations)
        
    
    
    def set_time_parameters(self, dt: float, T: float):
        """Set global time discretization parameters."""
        self.dt = dt
        self.T = T
        self.n_time_steps = int(np.ceil(T / dt))
        self.time_points = np.linspace(0, T, self.n_time_steps + 1)
    
    def get_spatial_discretization(self, domain_index: int) -> Discretization:
        """Get spatial discretization for a specific domain."""
        if 0 <= domain_index < self.n_domains:
            return self.spatial_discretizations[domain_index]
        else:
            raise IndexError(f"Domain index {domain_index} out of range [0, {self.n_domains-1}]")
    
    def get_time_info(self) -> dict:
        """Return time discretization information."""
        return {
            'dt': self.dt,
            'T': self.T,
            'n_time_steps': self.n_time_steps,
            'time_points': self.time_points
        }
    
    def get_global_info(self) -> dict:
        """Return global discretization information."""
        return {
            'n_domains': self.n_domains,
            'total_elements': self.total_elements,
            'total_nodes': self.total_nodes,
            'global_start': self.global_start,
            'global_end': self.global_end,
            'global_length': self.global_length,
            'time_info': self.get_time_info(),
            'spatial_discretizations': [disc.get_mesh_info() for disc in self.spatial_discretizations]
        }
