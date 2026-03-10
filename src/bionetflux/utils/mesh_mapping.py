"""
Utility functions for mapping between parametric and physical mesh coordinates.

Provides simple coordinate transformation utilities for BioNetFlux domains.
"""

import numpy as np
from typing import Tuple, Union
from ..geometry.domain_geometry import DomainInfo
from ..core.discretization import Discretization


def parametric_to_physical_mesh(domain: DomainInfo, discretization: Discretization) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map parametric nodes from discretization to physical coordinates on domain.
    
    Creates a physical mesh containing the coordinates of the physical nodes
    corresponding to the parametric nodes in the discretization.
    
    Args:
        domain: Domain geometry information containing physical extrema
        discretization: Spatial discretization containing parametric node coordinates
        
    Returns:
        Tuple containing:
        - x_coords: Array of x-coordinates in physical space
        - y_coords: Array of y-coordinates in physical space
        
    Example:
        >>> # Create domain from (0,0) to (1,1)
        >>> domain = DomainInfo(
        ...     domain_id=0,
        ...     extrema_start=(0.0, 0.0),
        ...     extrema_end=(1.0, 1.0)
        ... )
        >>> # Create discretization with 5 elements
        >>> discretization = Discretization(n_elements=5, domain_length=1.0)
        >>> # Get physical coordinates
        >>> x_coords, y_coords = parametric_to_physical_mesh(domain, discretization)
    """
    # Get parametric node coordinates from discretization
    param_coords = discretization.nodes
    
    # Get physical extrema from domain
    extrema_start = domain.extrema_start  # (x1, y1)
    extrema_end = domain.extrema_end      # (x2, y2)
    
    # Normalize parameter coordinates to [0, 1]
    # Map from domain parameter space to unit interval
    param_min = domain.domain_start
    param_max = domain.domain_start + domain.domain_length
    t = (param_coords - param_min) / (param_max - param_min)
    
    # Linear interpolation between physical extrema
    x_coords = extrema_start[0] + t * (extrema_end[0] - extrema_start[0])
    y_coords = extrema_start[1] + t * (extrema_end[1] - extrema_start[1])
    
    return x_coords, y_coords


def create_physical_mesh_dict(domain: DomainInfo, discretization: Discretization) -> dict:
    """
    Create a comprehensive physical mesh dictionary with coordinate information.
    
    Args:
        domain: Domain geometry information
        discretization: Spatial discretization
        
    Returns:
        Dictionary containing:
        - 'nodes_physical': (N, 2) array of [x, y] coordinates
        - 'nodes_parametric': (N,) array of parametric coordinates
        - 'elements': Element connectivity from discretization
        - 'domain_info': Domain metadata
        - 'mesh_info': Discretization metadata
    """
    # Get physical coordinates
    x_coords, y_coords = parametric_to_physical_mesh(domain, discretization)
    
    # Combine into coordinate array
    nodes_physical = np.column_stack((x_coords, y_coords))
    
    # Build comprehensive mesh dictionary
    mesh_dict = {
        'nodes_physical': nodes_physical,
        'nodes_parametric': discretization.nodes,
        'elements': discretization.elements,
        'domain_info': {
            'domain_id': domain.domain_id,
            'extrema_start': domain.extrema_start,
            'extrema_end': domain.extrema_end,
            'domain_start': domain.domain_start,
            'domain_length': domain.domain_length,
            'name': domain.name,
            'euclidean_length': domain.euclidean_length()
        },
        'mesh_info': discretization.get_mesh_info()
    }
    
    return mesh_dict