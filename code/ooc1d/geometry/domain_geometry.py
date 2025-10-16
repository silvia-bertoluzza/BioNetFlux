import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DomainInfo:
    """
    Container for domain geometric information.
    """
    domain_id: int
    extrema_start: Tuple[float, float]  # (x1, y1)
    extrema_end: Tuple[float, float]    # (x2, y2)
    domain_start: float = 0.0           # Parameter space start
    domain_length: float = 1.0          # Parameter space length
    name: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate default domain_length from extrema if not set
        if self.domain_length == 1.0:  # Default value
            self.domain_length = self.euclidean_length()
    
    def euclidean_length(self) -> float:
        """Calculate Euclidean distance between extrema."""
        dx = self.extrema_end[0] - self.extrema_start[0]
        dy = self.extrema_end[1] - self.extrema_start[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def center_point(self) -> Tuple[float, float]:
        """Calculate center point between extrema."""
        x_center = (self.extrema_start[0] + self.extrema_end[0]) / 2
        y_center = (self.extrema_start[1] + self.extrema_end[1]) / 2
        return (x_center, y_center)
    
    def direction_vector(self) -> Tuple[float, float]:
        """Calculate unit direction vector from start to end."""
        dx = self.extrema_end[0] - self.extrema_start[0]
        dy = self.extrema_end[1] - self.extrema_start[1]
        length = self.euclidean_length()
        if length > 0:
            return (dx/length, dy/length)
        return (1.0, 0.0)


class DomainGeometry:
    """
    Lean geometry class for constructing and handling complex multi-domain geometries.
    
    Manages a collection of domains (segments) and provides interface methods 
    for problem and discretization setup.
    """
    
    def __init__(self, name: str = "unnamed_geometry"):
        """
        Initialize empty geometry.
        
        Args:
            name: Descriptive name for the geometry
        """
        self.name = name
        self.domains: List[DomainInfo] = []
        self._next_id = 0
        self._global_metadata: Dict[str, Any] = {}
    
    def add_domain(self, 
                   extrema_start: Tuple[float, float],
                   extrema_end: Tuple[float, float],
                   domain_start: Optional[float] = None,
                   domain_length: Optional[float] = None,
                   name: Optional[str] = None,
                   **metadata) -> int:
        """
        Add a domain (segment) to the geometry.
        
        Args:
            extrema_start: Start point (x1, y1) in physical space
            extrema_end: End point (x2, y2) in physical space
            domain_start: Parameter space start (default: 0.0)
            domain_length: Parameter space length (default: Euclidean distance)
            name: Optional domain name
            **metadata: Additional domain-specific metadata
            
        Returns:
            Domain ID (index) of the added domain
        """
        # Calculate default values
        if domain_start is None:
            domain_start = 0.0
        
        if domain_length is None:
            # Default to Euclidean distance between extrema
            dx = extrema_end[0] - extrema_start[0]
            dy = extrema_end[1] - extrema_start[1]
            domain_length = np.sqrt(dx*dx + dy*dy)
        
        # Generate default name if not provided
        if name is None:
            name = f"domain_{self._next_id}"
        
        # Create domain info
        domain_info = DomainInfo(
            domain_id=self._next_id,
            extrema_start=extrema_start,
            extrema_end=extrema_end,
            domain_start=domain_start,
            domain_length=domain_length,
            name=name,
            metadata=metadata
        )
        
        # Add to collection
        self.domains.append(domain_info)
        domain_id = self._next_id
        self._next_id += 1
        
        return domain_id
    
    def get_domain(self, domain_id: int) -> DomainInfo:
        """
        Retrieve domain information by ID.
        
        Args:
            domain_id: Domain index
            
        Returns:
            DomainInfo object containing all domain parameters
            
        Raises:
            IndexError: If domain_id is invalid
        """
        if domain_id < 0 or domain_id >= len(self.domains):
            raise IndexError(f"Domain ID {domain_id} out of range [0, {len(self.domains)-1}]")
        
        return self.domains[domain_id]
    
    def get_all_domains(self) -> List[DomainInfo]:
        """Get list of all domains."""
        return self.domains.copy()
    
    def num_domains(self) -> int:
        """Get number of domains in geometry."""
        return len(self.domains)
    
    def get_bounding_box(self) -> Dict[str, float]:
        """
        Calculate bounding box of entire geometry.
        
        Returns:
            Dictionary with keys: x_min, x_max, y_min, y_max
        """
        if not self.domains:
            return {'x_min': 0.0, 'x_max': 1.0, 'y_min': 0.0, 'y_max': 1.0}
        
        all_x = []
        all_y = []
        
        for domain in self.domains:
            all_x.extend([domain.extrema_start[0], domain.extrema_end[0]])
            all_y.extend([domain.extrema_start[1], domain.extrema_end[1]])
        
        return {
            'x_min': min(all_x),
            'x_max': max(all_x),
            'y_min': min(all_y),
            'y_max': max(all_y)
        }
    
    def set_global_metadata(self, **metadata):
        """Set global metadata for the entire geometry."""
        self.global_metadata.update(metadata)
    
    def get_global_metadata(self) -> Dict[str, Any]:
        """Get global metadata."""
        return self._global_metadata.copy()
    
    def get_domain_names(self) -> List[str]:
        """Get list of all domain names."""
        return [domain.name for domain in self.domains]
    
    def find_domain_by_name(self, name: str) -> Optional[int]:
        """
        Find domain ID by name.
        
        Args:
            name: Domain name to search for
            
        Returns:
            Domain ID if found, None otherwise
        """
        for domain in self.domains:
            if domain.name == name:
                return domain.domain_id
        return None
    
    def remove_domain(self, domain_id: int):
        """
        Remove domain by ID.
        
        Args:
            domain_id: Domain ID to remove
            
        Raises:
            IndexError: If domain_id is invalid
        """
        if domain_id < 0 or domain_id >= len(self.domains):
            raise IndexError(f"Domain ID {domain_id} out of range")
        
        # Remove domain and update IDs
        del self.domains[domain_id]
        
        # Renumber domain IDs to maintain consistency
        for i, domain in enumerate(self.domains):
            domain.domain_id = i
        
        self._next_id = len(self.domains)
    
    def total_length(self) -> float:
        """Calculate total Euclidean length of all domains."""
        return sum(domain.euclidean_length() for domain in self.domains)
    
    def summary(self) -> str:
        """Generate summary string of the geometry."""
        lines = [
            f"Geometry: {self.name}",
            f"Number of domains: {len(self.domains)}",
            f"Total length: {self.total_length():.3f}",
            "Domains:"
        ]
        
        for domain in self.domains:
            lines.append(f"  {domain.domain_id}: {domain.name}")
            lines.append(f"    Extrema: {domain.extrema_start} → {domain.extrema_end}")
            lines.append(f"    Parameter: [{domain.domain_start:.3f}, {domain.domain_start + domain.domain_length:.3f}]")
            lines.append(f"    Length: {domain.euclidean_length():.3f}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        """Support len() operation."""
        return len(self.domains)
    
    def __getitem__(self, domain_id: int) -> DomainInfo:
        """Support indexing: geometry[i]."""
        return self.get_domain(domain_id)
    
    def __iter__(self):
        """Support iteration over domains."""
        return iter(self.domains)
