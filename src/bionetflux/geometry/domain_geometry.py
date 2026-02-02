import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Constants for special boundary types
EXTERIOR_BOUNDARY = -1
PERIODIC_BOUNDARY = -2
SYMMETRY_BOUNDARY = -3


@dataclass
class ConnectionInfo:
    """
    Container for domain connection information.
    """
    domain1_id: int
    domain2_id: int
    parameter1: float  # Parameter value in domain1 where connection is made
    parameter2: float  # Parameter value in domain2 where connection is made
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_boundary_connection(self) -> bool:
        """Check if this is a boundary connection."""
        return self.domain2_id < 0
    
    def is_exterior_boundary(self) -> bool:
        """Check if this is an exterior boundary connection."""
        return self.domain2_id == EXTERIOR_BOUNDARY
    
    def is_periodic_boundary(self) -> bool:
        """Check if this is a periodic boundary connection."""
        return self.domain2_id == PERIODIC_BOUNDARY
    
    def is_symmetry_boundary(self) -> bool:
        """Check if this is a symmetry boundary connection."""
        return self.domain2_id == SYMMETRY_BOUNDARY
    
    def get_boundary_type(self) -> Optional[str]:
        """Get the boundary type as a string, or None if not a boundary connection."""
        if not self.is_boundary_connection():
            return None
        elif self.is_exterior_boundary():
            return "exterior"
        elif self.is_periodic_boundary():
            return "periodic"
        elif self.is_symmetry_boundary():
            return "symmetry"
        else:
            return f"unknown_boundary_{self.domain2_id}"


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
    display_color: str = "blue"         # Color for matplotlib plotting
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
        self.connections: List[ConnectionInfo] = []  # NEW: List of domain connections
        self._next_id = 0
        self._global_metadata: Dict[str, Any] = {}
    
    def add_domain(self, 
                   extrema_start: Tuple[float, float],
                   extrema_end: Tuple[float, float],
                   domain_start: Optional[float] = None,
                   domain_length: Optional[float] = None,
                   name: Optional[str] = None,
                   display_color: str = "blue",
                   **metadata) -> int:
        """
        Add a domain (segment) to the geometry.
        
        Args:
            extrema_start: Start point (x1, y1) in physical space
            extrema_end: End point (x2, y2) in physical space
            domain_start: Parameter space start (default: 0.0)
            domain_length: Parameter space length (default: Euclidean distance)
            name: Optional domain name
            display_color: Color for matplotlib plotting (default: "blue")
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
            display_color=display_color,
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
    
    def num_connections(self) -> int:
        """Get number of connections in geometry."""
        return len(self.connections)
    
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
        self._global_metadata.update(metadata)
    
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
    
    def add_connection(self,
                      domain1_id: int,
                      domain2_id: int,
                      parameter1: float,
                      parameter2: float = 0.0,
                      **metadata) -> int:
        """
        Add a connection between two domains or mark a boundary point.
        
        Args:
            domain1_id: ID of first domain
            domain2_id: ID of second domain, or boundary type constant (EXTERIOR_BOUNDARY, etc.)
            parameter1: Parameter value in domain1 where connection is made
            parameter2: Parameter value in domain2 (ignored for boundary connections, default: 0.0)
            **metadata: Additional connection-specific metadata
            
        Returns:
            Connection index in the connections list
            
        Raises:
            IndexError: If domain IDs are invalid
            ValueError: If parameter values are out of domain range
        """
        # Validate domain1_id
        if domain1_id < 0 or domain1_id >= len(self.domains):
            raise IndexError(f"Domain ID {domain1_id} out of range [0, {len(self.domains)-1}]")
        
        domain1 = self.domains[domain1_id]
        
        # Validate parameter1 range
        domain1_end = domain1.domain_start + domain1.domain_length
        if not (domain1.domain_start <= parameter1 <= domain1_end):
            raise ValueError(f"Parameter1 {parameter1} not in domain {domain1_id} range [{domain1.domain_start}, {domain1_end}]")
        
        # Handle boundary connections (domain2_id < 0)
        if domain2_id < 0:
            # Boundary connection - parameter2 is ignored
            connection = ConnectionInfo(
                domain1_id=domain1_id,
                domain2_id=domain2_id,
                parameter1=parameter1,
                parameter2=0.0,  # Not used for boundary connections
                metadata=metadata
            )
        else:
            # Regular domain-to-domain connection
            if domain2_id >= len(self.domains):
                raise IndexError(f"Domain ID {domain2_id} out of range [0, {len(self.domains)-1}]")
            
            domain2 = self.domains[domain2_id]
            
            # Validate parameter2 range
            domain2_end = domain2.domain_start + domain2.domain_length
            if not (domain2.domain_start <= parameter2 <= domain2_end):
                raise ValueError(f"Parameter2 {parameter2} not in domain {domain2_id} range [{domain2.domain_start}, {domain2_end}]")
            
            connection = ConnectionInfo(
                domain1_id=domain1_id,
                domain2_id=domain2_id,
                parameter1=parameter1,
                parameter2=parameter2,
                metadata=metadata
            )
        
        # Add to connections list
        self.connections.append(connection)
        return len(self.connections) - 1
    
    def add_exterior_boundary(self, domain_id: int, parameter: float, **metadata) -> int:
        """
        Convenience method to add an exterior boundary point.
        
        Args:
            domain_id: Domain ID
            parameter: Parameter value where boundary occurs
            **metadata: Additional boundary metadata (e.g., boundary_condition="neumann")
            
        Returns:
            Connection index
        """
        return self.add_connection(domain_id, EXTERIOR_BOUNDARY, parameter, **metadata)
    
    def add_periodic_boundary(self, domain_id: int, parameter: float, **metadata) -> int:
        """
        Convenience method to add a periodic boundary point.
        
        Args:
            domain_id: Domain ID
            parameter: Parameter value where boundary occurs
            **metadata: Additional boundary metadata
            
        Returns:
            Connection index
        """
        return self.add_connection(domain_id, PERIODIC_BOUNDARY, parameter, **metadata)
    
    def add_symmetry_boundary(self, domain_id: int, parameter: float, **metadata) -> int:
        """
        Convenience method to add a symmetry boundary point.
        
        Args:
            domain_id: Domain ID
            parameter: Parameter value where boundary occurs
            **metadata: Additional boundary metadata
            
        Returns:
            Connection index
        """
        return self.add_connection(domain_id, SYMMETRY_BOUNDARY, parameter, **metadata)
    
    def get_boundary_connections(self) -> List[ConnectionInfo]:
        """Get all boundary connections (exterior, periodic, symmetry)."""
        return [conn for conn in self.connections if conn.is_boundary_connection()]
    
    def get_interior_connections(self) -> List[ConnectionInfo]:
        """Get all interior (domain-to-domain) connections."""
        return [conn for conn in self.connections if not conn.is_boundary_connection()]
    
    def get_connections_by_type(self, boundary_type: str) -> List[ConnectionInfo]:
        """
        Get connections by type.
        
        Args:
            boundary_type: "exterior", "periodic", "symmetry", or "interior"
            
        Returns:
            List of connections of the specified type
        """
        if boundary_type == "interior":
            return self.get_interior_connections()
        elif boundary_type == "exterior":
            return [conn for conn in self.connections if conn.is_exterior_boundary()]
        elif boundary_type == "periodic":
            return [conn for conn in self.connections if conn.is_periodic_boundary()]
        elif boundary_type == "symmetry":
            return [conn for conn in self.connections if conn.is_symmetry_boundary()]
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    def total_length(self) -> float:
        """Calculate total Euclidean length of all domains."""
        return sum(domain.euclidean_length() for domain in self.domains)
    
    def summary(self) -> str:
        """Generate summary string of the geometry."""
        lines = [
            f"Geometry: {self.name}",
            f"Number of domains: {len(self.domains)}",
            f"Number of connections: {len(self.connections)}",
            f"Total length: {self.total_length():.3f}",
            "Domains:"
        ]
        
        for domain in self.domains:
            lines.append(f"  {domain.domain_id}: {domain.name}")
            lines.append(f"    Extrema: {domain.extrema_start} → {domain.extrema_end}")
            lines.append(f"    Parameter: [{domain.domain_start:.3f}, {domain.domain_start + domain.domain_length:.3f}]")
            lines.append(f"    Length: {domain.euclidean_length():.3f}")
        
        # Add connections summary with type information
        if self.connections:
            lines.append("Connections:")
            for i, connection in enumerate(self.connections):
                if connection.is_boundary_connection():
                    boundary_type = connection.get_boundary_type()
                    lines.append(f"  {i}: Domain {connection.domain1_id}@{connection.parameter1:.3f} → "
                               f"{boundary_type} boundary")
                else:
                    lines.append(f"  {i}: Domain {connection.domain1_id}@{connection.parameter1:.3f} ↔ "
                               f"Domain {connection.domain2_id}@{connection.parameter2:.3f}")
        
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
    
    def validate_geometry(self, verbose: bool = False) -> bool:
        """
        Validate the geometry for consistency and common issues.
        
        Args:
            verbose: Whether to print validation details
            
        Returns:
            True if geometry is valid, False otherwise
        """
        if verbose:
            print(f"Validating geometry: {self.name}")
        
        issues = []
        warnings = []
        
        # Check if geometry is empty
        if len(self.domains) == 0:
            issues.append("Geometry is empty (no domains)")
        
        # Validate individual domains
        for domain in self.domains:
            # Check for degenerate domains (zero length)
            if domain.euclidean_length() < 1e-12:
                issues.append(f"Domain {domain.domain_id} ({domain.name}) has zero length")
            
            # Check for negative parameter space length
            if domain.domain_length <= 0:
                issues.append(f"Domain {domain.domain_id} ({domain.name}) has non-positive parameter length")
            
            # Check for valid coordinates
            for coord_name, coord in [("start", domain.extrema_start), ("end", domain.extrema_end)]:
                if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
                    issues.append(f"Domain {domain.domain_id} ({domain.name}) has invalid {coord_name} coordinates")
                elif not all(isinstance(x, (int, float)) for x in coord):
                    issues.append(f"Domain {domain.domain_id} ({domain.name}) has non-numeric {coord_name} coordinates")
        
        # Check for duplicate domain names
        names = [domain.name for domain in self.domains if domain.name is not None]
        if len(names) != len(set(names)):
            duplicate_names = [name for name in set(names) if names.count(name) > 1]
            issues.append(f"Duplicate domain names found: {duplicate_names}")
        
        # Check for overlapping parameter spaces (WARNING, not failure)
        param_ranges = [(domain.domain_start, domain.domain_start + domain.domain_length) 
                       for domain in self.domains]
        for i, (start1, end1) in enumerate(param_ranges):
            for j, (start2, end2) in enumerate(param_ranges[i+1:], i+1):
                if not (end1 <= start2 or end2 <= start1):  # Overlapping ranges
                    domain1 = self.domains[i]
                    domain2 = self.domains[j]
                    warnings.append(f"Overlapping parameter spaces: {domain1.name} [{start1:.3f}, {end1:.3f}] "
                                  f"and {domain2.name} [{start2:.3f}, {end2:.3f}]")
        
        # Validate connections
        for i, connection in enumerate(self.connections):
            # Check domain1_id exists
            if connection.domain1_id >= len(self.domains):
                issues.append(f"Connection {i}: domain1_id {connection.domain1_id} does not exist")
                continue
            
            # Check parameter1 range
            domain1 = self.domains[connection.domain1_id]
            domain1_end = domain1.domain_start + domain1.domain_length
            if not (domain1.domain_start <= connection.parameter1 <= domain1_end):
                issues.append(f"Connection {i}: parameter1 {connection.parameter1} out of domain {connection.domain1_id} range")
            
            if connection.is_boundary_connection():
                # Validate boundary connection
                if connection.domain2_id not in [EXTERIOR_BOUNDARY, PERIODIC_BOUNDARY, SYMMETRY_BOUNDARY]:
                    warnings.append(f"Connection {i}: unknown boundary type {connection.domain2_id}")
            else:
                # Validate regular connection
                if connection.domain2_id >= len(self.domains):
                    issues.append(f"Connection {i}: domain2_id {connection.domain2_id} does not exist")
                    continue
                
                # Check parameter2 range
                domain2 = self.domains[connection.domain2_id]
                domain2_end = domain2.domain_start + domain2.domain_length
                if not (domain2.domain_start <= connection.parameter2 <= domain2_end):
                    issues.append(f"Connection {i}: parameter2 {connection.parameter2} out of domain {connection.domain2_id} range")
                
                # Check for self-connections
                if connection.domain1_id == connection.domain2_id:
                    warnings.append(f"Connection {i}: self-connection in domain {connection.domain1_id}")
        
        # Check for duplicate connections (updated for boundary connections)
        for i, conn1 in enumerate(self.connections):
            for j, conn2 in enumerate(self.connections[i+1:], i+1):
                if conn1.is_boundary_connection() and conn2.is_boundary_connection():
                    # Check for duplicate boundary connections
                    if (conn1.domain1_id == conn2.domain1_id and 
                        conn1.domain2_id == conn2.domain2_id and
                        abs(conn1.parameter1 - conn2.parameter1) < 1e-12):
                        warnings.append(f"Duplicate boundary connections: {i} and {j}")
                elif not conn1.is_boundary_connection() and not conn2.is_boundary_connection():
                    # Check for duplicate interior connections (bidirectional)
                    same_connection = (
                        (conn1.domain1_id == conn2.domain1_id and conn1.domain2_id == conn2.domain2_id and
                         abs(conn1.parameter1 - conn2.parameter1) < 1e-12 and abs(conn1.parameter2 - conn2.parameter2) < 1e-12) or
                        (conn1.domain1_id == conn2.domain2_id and conn1.domain2_id == conn2.domain1_id and
                         abs(conn1.parameter1 - conn2.parameter2) < 1e-12 and abs(conn1.parameter2 - conn2.parameter1) < 1e-12)
                    )
                    if same_connection:
                        warnings.append(f"Duplicate interior connections: {i} and {j}")
        
        # Report results
        if verbose:
            if issues:
                print(f"  Found {len(issues)} validation errors:")
                for issue in issues:
                    print(f"    ✗ {issue}")
            else:
                print("  ✓ No validation errors found")
            
            if warnings:
                print(f"  Found {len(warnings)} warnings:")
                for warning in warnings:
                    print(f"    ⚠ {warning}")
            else:
                print("  ✓ No warnings")
            
            # Connection validation summary
            if self.connections:
                print(f"  Connection validation: {len(self.connections)} connections checked")
        
        # Return True only if no critical issues (warnings don't affect validation)
        return len(issues) == 0
    
    def find_intersections(self, tolerance: float = 1e-6) -> List[Tuple[int, int, Tuple[float, float]]]:
        """
        Find intersection points between domain segments.
        
        This method finds both endpoint connections and actual line segment intersections.
        
        Args:
            tolerance: Distance tolerance for considering points as intersecting
            
        Returns:
            List of tuples (domain1_id, domain2_id, intersection_point)
        """
        intersections = []
        
        def segments_intersect(p1, q1, p2, q2, tol=tolerance):
            """
            Find intersection point between two line segments.
            
            Args:
                p1, q1: Start and end points of first segment
                p2, q2: Start and end points of second segment
                tol: Tolerance for numerical precision
                
            Returns:
                Intersection point (x, y) if segments intersect, None otherwise
            """
            x1, y1 = p1
            x2, y2 = q1
            x3, y3 = p2
            x4, y4 = q2
            
            # Calculate direction vectors
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x4 - x3, y4 - y3
            
            # Calculate determinant
            det = dx1 * dy2 - dy1 * dx2
            
            # Check if lines are parallel
            if abs(det) < tol:
                # Lines are parallel, check if they're collinear and overlapping
                # Vector from p1 to p2
                dx3, dy3 = x3 - x1, y3 - y1
                
                # Check if points are collinear using cross product
                cross = dx3 * dy1 - dy3 * dx1
                if abs(cross) < tol:
                    # Segments are collinear, check for overlap
                    # Project all points onto the line direction
                    if abs(dx1) > abs(dy1):  # Project onto x-axis
                        t1_start = 0.0
                        t1_end = 1.0
                        t2_start = (x3 - x1) / dx1 if abs(dx1) > tol else 0.0
                        t2_end = (x4 - x1) / dx1 if abs(dx1) > tol else 0.0
                    else:  # Project onto y-axis
                        t1_start = 0.0
                        t1_end = 1.0
                        t2_start = (y3 - y1) / dy1 if abs(dy1) > tol else 0.0
                        t2_end = (y4 - y1) / dy1 if abs(dy1) > tol else 0.0
                    
                    # Ensure t2_start <= t2_end
                    if t2_start > t2_end:
                        t2_start, t2_end = t2_end, t2_start
                    
                    # Check for overlap
                    overlap_start = max(t1_start, t2_start)
                    overlap_end = min(t1_end, t2_end)
                    
                    if overlap_start <= overlap_end + tol:
                        # Segments overlap, return midpoint of overlap
                        t_mid = (overlap_start + overlap_end) / 2
                        intersection_x = x1 + t_mid * dx1
                        intersection_y = y1 + t_mid * dy1
                        return (intersection_x, intersection_y)
                
                return None  # Parallel but not collinear, or no overlap
            
            # Calculate intersection parameters
            dx3, dy3 = x1 - x3, y1 - y3
            t = (dx2 * dy3 - dy2 * dx3) / det
            u = (dx1 * dy3 - dy1 * dx3) / det
            
            # Check if intersection point lies on both segments
            if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
                # Calculate intersection point
                intersection_x = x1 + t * dx1
                intersection_y = y1 + t * dy1
                return (intersection_x, intersection_y)
            
            return None
        
        def point_distance(p1, p2):
            """Calculate distance between two points."""
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Check all pairs of domains
        for i, domain1 in enumerate(self.domains):
            for j, domain2 in enumerate(self.domains[i+1:], i+1):
                # Get segment endpoints
                seg1_start = domain1.extrema_start
                seg1_end = domain1.extrema_end
                seg2_start = domain2.extrema_start  
                seg2_end = domain2.extrema_end
                
                # Find segment intersection
                intersection_point = segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end)
                
                if intersection_point is not None:
                    intersections.append((i, j, intersection_point))
                else:
                    # Also check for close endpoints (endpoint connections)
                    endpoints1 = [seg1_start, seg1_end]
                    endpoints2 = [seg2_start, seg2_end]
                    
                    for ep1 in endpoints1:
                        for ep2 in endpoints2:
                            if point_distance(ep1, ep2) <= tolerance:
                                # Use the midpoint as intersection
                                midpoint = ((ep1[0] + ep2[0]) / 2, (ep1[1] + ep2[1]) / 2)
                                intersections.append((i, j, midpoint))
                                break
                        else:
                            continue
                        break
        
        return intersections
    
    def get_connectivity_info(self, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Analyze geometry connectivity.
        
        Args:
            tolerance: Distance tolerance for considering points as connected
            
        Returns:
            Dictionary with connectivity information
        """
        intersections = self.find_intersections(tolerance)
        
        # Build adjacency information
        adjacency = {i: set() for i in range(len(self.domains))}
        for domain1_id, domain2_id, _ in intersections:
            adjacency[domain1_id].add(domain2_id)
            adjacency[domain2_id].add(domain1_id)
        
        # Find connected components
        visited = set()
        components = []
        
        for domain_id in range(len(self.domains)):
            if domain_id not in visited:
                component = []
                stack = [domain_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(adjacency[current] - visited)
                
                components.append(component)
        
        return {
            'intersections': intersections,
            'adjacency': adjacency,
            'connected_components': components,
            'num_components': len(components),
            'is_connected': len(components) == 1,
            'isolated_domains': [comp[0] for comp in components if len(comp) == 1]
        }
    
    def suggest_parameter_spacing(self, gap: float = 0.1) -> List[Tuple[float, float]]:
        """
        Suggest non-overlapping parameter space ranges for all domains.
        
        Args:
            gap: Minimum gap between parameter ranges
            
        Returns:
            List of (start, length) tuples for each domain
        """
        suggestions = []
        current_start = 0.0
        
        # Sort domains by current parameter start for consistent ordering
        sorted_domains = sorted(self.domains, key=lambda d: d.domain_start)
        
        for domain in sorted_domains:
            # Use the domain's preferred length, or Euclidean length as fallback
            length = domain.domain_length if domain.domain_length > 0 else domain.euclidean_length()
            
            suggestions.append((current_start, length))
            current_start += length + gap
        
        return suggestions
    
    @classmethod
    def create_test_geometries(cls) -> Dict[str, 'DomainGeometry']:
        """
        Create a collection of test geometries for validation and testing.
        
        Returns:
            Dictionary of test geometry instances
        """
        geometries = {}
        
        # 1. Simple linear chain
        linear = cls("linear_chain")
        linear.add_domain((0.0, 0.0), (1.0, 0.0), name="segment1", display_color="red")
        linear.add_domain((1.0, 0.0), (2.0, 0.0), name="segment2", display_color="green")
        linear.add_domain((2.0, 0.0), (3.0, 0.0), name="segment3", display_color="blue")
        geometries["linear"] = linear
        
        # 2. T-junction
        t_junction = cls("t_junction")
        t_junction.add_domain((0.0, -1.0), (0.0, 1.0), name="main_channel", display_color="darkblue")
        t_junction.add_domain((0.0, 0.0), (1.0, 0.0), name="side_branch", display_color="orange")
        geometries["t_junction"] = t_junction
        
        # 3. Grid network
        grid = cls("grid_network")
        # Vertical segments
        grid.add_domain((-0.5, 0.0), (-0.5, 1.0), name="left_vertical", display_color="purple")
        grid.add_domain((0.5, 0.0), (0.5, 1.0), name="right_vertical", display_color="purple")
        # Horizontal connectors
        colors = ["red", "orange", "yellow", "green"]
        for i, y in enumerate([0.2, 0.4, 0.6, 0.8]):
            color = colors[i % len(colors)]
            grid.add_domain((-0.5, y), (0.5, y), name=f"horizontal_{i+1}", display_color=color)
        geometries["grid"] = grid
        
        # 4. Star network
        star = cls("star_network")
        center = (0.0, 0.0)
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        star_colors = ["red", "orange", "yellow", "green", "blue", "purple"]
        for i, angle in enumerate(angles):
            end_point = (np.cos(angle), np.sin(angle))
            star.add_domain(center, end_point, name=f"branch_{i+1}", display_color=star_colors[i])
        geometries["star"] = star
        
        # 5. Complex branching
        branching = cls("complex_branching")
        # Main trunk
        branching.add_domain((0.0, 0.0), (0.0, 2.0), name="trunk", display_color="brown")
        # Primary branches
        branching.add_domain((0.0, 1.0), (1.0, 1.5), name="branch_1a", display_color="darkgreen")
        branching.add_domain((0.0, 1.0), (-1.0, 1.5), name="branch_1b", display_color="darkgreen")
        # Secondary branches
        branching.add_domain((1.0, 1.5), (1.5, 2.0), name="branch_2a", display_color="lightgreen")
        branching.add_domain((1.0, 1.5), (1.5, 1.0), name="branch_2b", display_color="lightgreen")
        geometries["branching"] = branching
        
        # 6. Degenerate case (for testing validation)
        degenerate = cls("degenerate_test")
        degenerate.add_domain((0.0, 0.0), (0.0, 0.0), name="zero_length", display_color="red")  # Zero length
        degenerate.add_domain((1.0, 1.0), (2.0, 1.0), domain_start=0.5, domain_length=1.0, name="overlap1", display_color="orange")
        degenerate.add_domain((3.0, 1.0), (4.0, 1.0), domain_start=1.0, domain_length=1.0, name="overlap2", display_color="yellow")
        geometries["degenerate"] = degenerate
        
        return geometries
    
    def run_self_test(self, verbose: bool = True) -> bool:
        """
        Run comprehensive self-test on the geometry.
        
        Args:
            verbose: Whether to print detailed test results
            
        Returns:
            True if all tests pass, False otherwise
        """
        if verbose:
            print(f"Running self-test for geometry: {self.name}")
            print("=" * 50)
        
        all_passed = True
        
        # Test 1: Basic functionality
        if verbose:
            print("Test 1: Basic functionality")
        
        try:
            # Test domain addition and retrieval
            original_count = len(self.domains)
            test_id = self.add_domain((10.0, 10.0), (11.0, 11.0), name="test_domain")
            
            if len(self.domains) != original_count + 1:
                if verbose:
                    print("  ✗ Domain addition failed")
                all_passed = False
            elif self.get_domain(test_id).name != "test_domain":
                if verbose:
                    print("  ✗ Domain retrieval failed")
                all_passed = False
            else:
                if verbose:
                    print("  ✓ Domain addition and retrieval")
            
            # Clean up test domain
            self.remove_domain(test_id)
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Basic functionality test failed: {e}")
            all_passed = False
        
        # Test 2: Geometry validation
        if verbose:
            print("Test 2: Geometry validation")
        
        try:
            is_valid = self.validate_geometry(verbose=False)
            if verbose:
                print(f"  {'✓' if is_valid else '✗'} Geometry validation: {'PASS' if is_valid else 'FAIL'}")
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Validation test failed: {e}")
            all_passed = False
        
        # Test 3: Connectivity analysis
        if verbose:
            print("Test 3: Connectivity analysis")
        
        try:
            connectivity = self.get_connectivity_info()
            if verbose:
                print(f"  ✓ Found {len(connectivity['intersections'])} intersections")
                print(f"  ✓ {connectivity['num_components']} connected component(s)")
                if connectivity['isolated_domains']:
                    print(f"  ! {len(connectivity['isolated_domains'])} isolated domain(s)")
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Connectivity analysis failed: {e}")
            all_passed = False
        
        # Test 4: Bounding box calculation
        if verbose:
            print("Test 4: Bounding box calculation")
        
        try:
            bbox = self.get_bounding_box()
            required_keys = ['x_min', 'x_max', 'y_min', 'y_max']
            if all(key in bbox for key in required_keys):
                if verbose:
                    print(f"  ✓ Bounding box: x[{bbox['x_min']:.2f}, {bbox['x_max']:.2f}], y[{bbox['y_min']:.2f}, {bbox['y_max']:.2f}]")
            else:
                if verbose:
                    print("  ✗ Bounding box missing required keys")
                all_passed = False
                
        except Exception as e:
            if verbose:
                print(f"  ✗ Bounding box calculation failed: {e}")
            all_passed = False
        
        # Test 5: Parameter space suggestions
        if verbose:
            print("Test 5: Parameter space analysis")
        
        try:
            suggestions = self.suggest_parameter_spacing()
            if len(suggestions) == len(self.domains):
                if verbose:
                    print("  ✓ Parameter space suggestions generated")
            else:
                if verbose:
                    print("  ✗ Parameter space suggestion count mismatch")
                all_passed = False
                
        except Exception as e:
            if verbose:
                print(f"  ✗ Parameter space analysis failed: {e}")
            all_passed = False
        
        # Test 6: Enhanced connection functionality with boundary support
        if verbose:
            print("Test 6: Enhanced connection functionality")
        
        try:
            original_connections = len(self.connections)
            
            # Add test domains if none exist
            if len(self.domains) < 2:
                test_dom1 = self.add_domain((0.0, 0.0), (1.0, 0.0), name="test_domain_1")
                test_dom2 = self.add_domain((1.0, 0.0), (2.0, 0.0), name="test_domain_2")
            else:
                test_dom1 = 0
                test_dom2 = 1
            
            # Test interior connection
            interior_conn_id = self.add_connection(
                domain1_id=test_dom1,
                domain2_id=test_dom2,
                parameter1=self.domains[test_dom1].domain_start + self.domains[test_dom1].domain_length,
                parameter2=self.domains[test_dom2].domain_start
            )
            
            # Test boundary connections
            exterior_conn_id = self.add_exterior_boundary(test_dom1, self.domains[test_dom1].domain_start)
            periodic_conn_id = self.add_periodic_boundary(test_dom2, self.domains[test_dom2].domain_start + self.domains[test_dom2].domain_length)
            
            # Test connection type queries
            boundary_connections = self.get_boundary_connections()
            interior_connections = self.get_interior_connections()
            exterior_connections = self.get_connections_by_type("exterior")
            
            if len(self.connections) != original_connections + 3:
                if verbose:
                    print("  ✗ Connection addition failed")
                all_passed = False
            elif len(boundary_connections) != 2 or len(interior_connections) != 1:
                if verbose:
                    print("  ✗ Connection type classification failed")
                all_passed = False
            elif len(exterior_connections) != 1:
                if verbose:
                    print("  ✗ Exterior connection query failed")
                all_passed = False
            else:
                if verbose:
                    print("  ✓ Enhanced connection functionality")
            
            # Test helper methods
            conn = self.get_connection(exterior_conn_id)
            if not conn.is_boundary_connection() or not conn.is_exterior_boundary():
                if verbose:
                    print("  ✗ Boundary connection helper methods failed")
                all_passed = False
            else:
                if verbose:
                    print("  ✓ Connection helper methods")
            
            # Clean up test connections
            self.remove_connection(periodic_conn_id)
            self.remove_connection(exterior_conn_id)
            self.remove_connection(interior_conn_id)
            
            # Clean up test domains if we added them
            if len(self.domains) >= 2 and self.domains[-1].name == "test_domain_2":
                self.remove_domain(len(self.domains) - 1)
                self.remove_domain(len(self.domains) - 1)
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Enhanced connection functionality test failed: {e}")
            all_passed = False
        
        if verbose:
            print("=" * 50)
            print(f"Self-test result: {'PASS' if all_passed else 'FAIL'}")
        
        return all_passed
    
def build_grid_geometry(N: int = 4):
    """
    Build a default OoC grid geometry with vertical segments and horizontal connectors.
    Includes explicit connections for constraint generation.
    
    Args:
        N: Number of horizontal segments in each section (default: 4)
    
    Returns:
        DomainGeometry: Default grid geometry instance
    """
    print(f"Creating default custom grid geometry with N={N}...")
    
    geometry = DomainGeometry("default_ooc_grid_geometry")
    
    # Vertical segments
    # S1: Left vertical segment
    geometry.add_domain(
        extrema_start=(-1.0, -1.0),
        extrema_end=(-1.0, 1.0),
        name="S1_left_vertical",
        display_color="blue"
    )
    
    # S2: Lower middle vertical segment  
    geometry.add_domain(
        extrema_start=(0.0, -1.0),
        extrema_end=(0.0, -0.1),
        name="S2_lower_middle_vertical",
        display_color="green"
    )
    
    # S3: Upper middle vertical segment
    geometry.add_domain(
        extrema_start=(0.0, 0.1),
        extrema_end=(0.0, 1.0),
        name="S3_upper_middle_vertical",
        display_color="green"
    )
    
    # S4: Right vertical segment
    geometry.add_domain(
        extrema_start=(1.0, -1.0),
        extrema_end=(1.0, 1.0),
        name="S4_right_vertical",
        display_color="blue"
    )
    
    # Horizontal connectors - Lower section (-0.9 < y < -0.2)
    y_lower_values = np.linspace(-0.9, -0.2, N)
    
    print(f"  Adding {N} lower horizontal connectors at y = {y_lower_values}")
    
    # Lower connectors: S1 to S2
    for i, y_pos in enumerate(y_lower_values):
        geometry.add_domain(
            extrema_start=(-1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"lower_S1_S2_{i+1}",
            display_color="red"
        )
    
    # Lower connectors: S4 to S2  
    for i, y_pos in enumerate(y_lower_values):
        geometry.add_domain(
            extrema_start=(1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"lower_S4_S2_{i+1}",
            display_color="red"
        )
    
    # Horizontal connectors - Upper section (0.2 < y < 0.9)
    y_upper_values = np.linspace(0.2, 0.9, N)
    
    print(f"  Adding {N} upper horizontal connectors at y = {y_upper_values}")
    
    # Upper connectors: S1 to S3
    for i, y_pos in enumerate(y_upper_values):
        geometry.add_domain(
            extrema_start=(-1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"upper_S1_S3_{i+1}",
            display_color="red"
        )
    
    # Upper connectors: S4 to S3
    for i, y_pos in enumerate(y_upper_values):
        geometry.add_domain(
            extrema_start=(1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"upper_S4_S3_{i+1}",
            display_color="red"
        )
    
    # =============================================================================
    # ADD EXPLICIT CONNECTIONS TO GEOMETRY
    # =============================================================================
    print("  Adding explicit connections to geometry...")
    
    # External boundary conditions for vertical segments (domains 0, 1, 2, 3)
    for domain_idx in [0, 1, 2, 3]:  # S1, S2, S3, S4
        # Add exterior boundary at start of domain
        geometry.add_exterior_boundary(domain_idx, 0.0)  # domain_start
        # Add exterior boundary at end of domain
        domain_length = geometry.get_domain(domain_idx).domain_length
        geometry.add_exterior_boundary(domain_idx, domain_length)  # domain_end
    
    # Interior connections for horizontal-vertical intersections
    # S1 connections: start of lower S1->S2 and upper S1->S3 connectors with S1 (domain 0)
    s1_left_lower = list(range(4, 4+N))        # Lower S1->S2 connectors  
    s1_left_upper = list(range(4+2*N, 4+3*N))  # Upper S1->S3 connectors
    s1_connections = s1_left_lower + s1_left_upper
    
    for domain_idx in s1_connections:
        # Get horizontal segment info
        horizontal_domain_info = geometry.get_domain(domain_idx)
        intersection_y = horizontal_domain_info.extrema_start[1]  # y-coordinate at S1 end
        
        # Map to S1 parameter space: S1 spans y ∈ [-1, 1], param ∈ [0, domain_length]
        s1_param = (intersection_y + 1.0) / 2.0 * geometry.get_domain(0).domain_length
        
        # Add connection between horizontal connector start and S1
        geometry.add_connection(
            domain1_id=domain_idx,  # Horizontal connector
            domain2_id=0,           # S1 (left vertical)
            parameter1=0.0,         # Start of horizontal (at S1)
            parameter2=s1_param     # Corresponding point on S1
        )
    
    # S4 connections: start of lower S4->S2 and upper S4->S3 connectors with S4 (domain 3)
    s4_right_lower = list(range(4+N, 4+2*N))    # Lower S4->S2 connectors
    s4_right_upper = list(range(4+3*N, 4+4*N))  # Upper S4->S3 connectors  
    s4_connections = s4_right_lower + s4_right_upper
    
    for domain_idx in s4_connections:
        # Get horizontal segment info
        horizontal_domain_info = geometry.get_domain(domain_idx)
        intersection_y = horizontal_domain_info.extrema_start[1]  # y-coordinate at S4 end
        
        # Map to S4 parameter space: S4 spans y ∈ [-1, 1], param ∈ [0, domain_length]
        s4_param = (intersection_y + 1.0) / 2.0 * geometry.get_domain(3).domain_length
        
        # Add connection between horizontal connector start and S4
        geometry.add_connection(
            domain1_id=domain_idx,  # Horizontal connector  
            domain2_id=3,           # S4 (right vertical)
            parameter1=0.0,         # Start of horizontal (at S4)
            parameter2=s4_param     # Corresponding point on S4
        )

    # S2 connections: end of S1->S2 connectors + end of S4->S2 connectors
    s2_from_s1 = list(range(4, 4+N))        # End of S1->S2 connectors connects to S2
    s2_from_s4 = list(range(4+N, 4+2*N))    # End of S4->S2 connectors connects to S2  
    
    # End of S1->S2 connectors with S2
    for domain_idx in s2_from_s1:
        horizontal_domain_info = geometry.get_domain(domain_idx)
        intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate at S2 end
        
        # Map to S2 parameter space: S2 spans y ∈ [-1, -0.1], param ∈ [0, domain_length]
        s2_y_start, s2_y_end = -1.0, -0.1
        s2_param = (intersection_y - s2_y_start) / (s2_y_end - s2_y_start) * geometry.get_domain(1).domain_length
        
        # Add connection between horizontal connector end and S2
        geometry.add_connection(
            domain1_id=domain_idx,  # S1->S2 connector
            domain2_id=1,           # S2 (lower middle vertical)
            parameter1=geometry.get_domain(domain_idx).domain_length,  # End of horizontal (at S2)
            parameter2=s2_param     # Corresponding point on S2
        )
    
    # End of S4->S2 connectors with S2  
    for domain_idx in s2_from_s4:
        horizontal_domain_info = geometry.get_domain(domain_idx)
        intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate at S2 end
        
        # Map to S2 parameter space
        s2_y_start, s2_y_end = -1.0, -0.1
        s2_param = (intersection_y - s2_y_start) / (s2_y_end - s2_y_start) * geometry.get_domain(1).domain_length
        
        # Add connection between horizontal connector end and S2
        geometry.add_connection(
            domain1_id=domain_idx,  # S4->S2 connector
            domain2_id=1,           # S2 (lower middle vertical)
            parameter1=geometry.get_domain(domain_idx).domain_length,  # End of horizontal (at S2)
            parameter2=s2_param     # Corresponding point on S2
        )
    
    # S3 connections: end of S1->S3 connectors + end of S4->S3 connectors
    s3_from_s1 = list(range(4+2*N, 4+3*N))  # End of S1->S3 connectors connects to S3
    s3_from_s4 = list(range(4+3*N, 4+4*N))  # End of S4->S3 connectors connects to S3
    
    # End of S1->S3 connectors with S3
    for domain_idx in s3_from_s1:
        horizontal_domain_info = geometry.get_domain(domain_idx)
        intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate at S3 end
        
        # Map to S3 parameter space: S3 spans y ∈ [0.1, 1.0], param ∈ [0, domain_length]
        s3_y_start, s3_y_end = 0.1, 1.0
        s3_param = (intersection_y - s3_y_start) / (s3_y_end - s3_y_start) * geometry.get_domain(2).domain_length
        
        # Add connection between horizontal connector end and S3
        geometry.add_connection(
            domain1_id=domain_idx,  # S1->S3 connector
            domain2_id=2,           # S3 (upper middle vertical)
            parameter1=geometry.get_domain(domain_idx).domain_length,  # End of horizontal (at S3)
            parameter2=s3_param     # Corresponding point on S3
        )
    
    # End of S4->S3 connectors with S3
    for domain_idx in s3_from_s4:
        horizontal_domain_info = geometry.get_domain(domain_idx)
        intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate at S3 end
        
        # Map to S3 parameter space
        s3_y_start, s3_y_end = 0.1, 1.0
        s3_param = (intersection_y - s3_y_start) / (s3_y_end - s3_y_start) * geometry.get_domain(2).domain_length
        
        # Add connection between horizontal connector end and S3
        geometry.add_connection(
            domain1_id=domain_idx,  # S4->S3 connector
            domain2_id=2,           # S3 (upper middle vertical)  
            parameter1=geometry.get_domain(domain_idx).domain_length,  # End of horizontal (at S3)
            parameter2=s3_param     # Corresponding point on S3
        )
    
    print(f"✓ Default grid geometry created:")
    print(f"  - 4 vertical segments (S1, S2, S3, S4)")
    print(f"  - {2*N} lower horizontal connectors (-0.9 < y < -0.2)")
    print(f"  - {2*N} upper horizontal connectors (0.2 < y < 0.9)")
    print(f"  - Total domains: {geometry.num_domains()}")
    print(f"  - Total connections: {geometry.num_connections()}")
    print(f"    - Boundary connections: {len(geometry.get_boundary_connections())}")
    print(f"    - Interior connections: {len(geometry.get_interior_connections())}")
    
    return geometry

def build_arc_sequence_geometry(N: int = 2, start: float = 0.0, length: float = 1.0):
    """
    Build an arc sequence geometry with N aligned segments.
    
    Creates N consecutive segments of equal length:
    - Segment 1: from (start, 0.5) to (start + length, 0.5) 
    - Segment 2: from (start + length, 0.5) to (start + 2*length, 0.5)
    - ...
    - Segment N: from (start + (N-1)*length, 0.5) to (start + N*length, 0.5)
    
    Each segment has parameter space [0, 1] scaled to its physical length.
    
    Args:
        N: Number of segments (default: 2)
        start: Starting x-coordinate (default: 0.0)
        length: Length of each segment (default: 1.0)
    
    With connections:
    - Exterior boundary at start of first segment (parameter 0)
    - N-1 interior connections between consecutive segments
    - Exterior boundary at end of last segment (parameter 1)
    
    Returns:
        DomainGeometry: Arc sequence geometry instance
    """
    if N < 1:
        raise ValueError(f"Number of segments N must be at least 1, got {N}")
    if length <= 0:
        raise ValueError(f"Segment length must be positive, got {length}")
    
    print(f"Creating arc sequence geometry with N={N} segments...")
    print(f"  Start: x={start}, segment length: {length}")
    print(f"  Total span: x ∈ [{start}, {start + N*length}]")
    
    geometry = DomainGeometry("arc_sequence_geometry")
    
    # Add N aligned segments
    colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]
    
    for i in range(N):
        x_start = start + i * length
        x_end = start + (i + 1) * length
        
        geometry.add_domain(
            extrema_start=(x_start, 0.5),
            extrema_end=(x_end, 0.5),
            domain_start=x_start,
            domain_length=length,  # Normalized parameter space [0, 1] for each segment
            name=f"segment_{i+1}",
            display_color=colors[i % len(colors)]
        )
    
    # Add boundary connections
    # Exterior boundary at start of first segment (domain 0, parameter 0)
    geometry.add_exterior_boundary(
        domain_id=0,
        parameter=start,
        boundary_condition="neumann"  # Example metadata
    )
    
    # Exterior boundary at end of last segment (domain N-1, parameter 1)
    geometry.add_exterior_boundary(
        domain_id=N-1,
        parameter=start + length * N,
        boundary_condition="neumann"  # Example metadata
    )
    
    # Add N-1 interior connections between consecutive segments
    for i in range(N-1):
        geometry.add_connection(
            domain1_id=i,        # Current segment
            domain2_id=i+1,      # Next segment
            parameter1 = start + length * (i+1),      # End of current segment
            parameter2 = start + length * (i+1),      # Start of next segment
            connection_type="trace_continuity"
        )
    
    print(f"✓ Arc sequence geometry created:")
    print(f"  - {N} aligned segments along y=0.5")
    print(f"  - Each segment: length={length}, parameter space [0, 1]")
    print(f"  - Total physical length: {N*length}")
    print(f"  - Total domains: {geometry.num_domains()}")
    print(f"  - Total connections: {geometry.num_connections()}")
    print(f"    - Boundary connections: {len(geometry.get_boundary_connections())}")
    print(f"    - Interior connections: {len(geometry.get_interior_connections())}")
    
    return geometry

