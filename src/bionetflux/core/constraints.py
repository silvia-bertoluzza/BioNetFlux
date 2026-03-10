"""
Constraint module for boundary and junction conditions.
Handles Lagrange multipliers for various constraint types.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from enum import Enum


class ConstraintType(Enum):
    """Types of constraints."""
    # Boundary conditions (single domain)
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    
    # Junction conditions (two domains)
    TRACE_CONTINUITY = "trace_continuity"
    KEDEM_KATCHALSKY = "kedem_katchalsky"


class Constraint:
    """
    Base class for all constraint types.
    """
    
    def __init__(self, 
                 constraint_type: ConstraintType,
                 equation_index: int,
                 domains: List[int],
                 positions: List[float],
                 parameters: Optional[np.ndarray] = None,
                 data_function: Optional[Callable] = None):
        """
        Initialize constraint.
        
        Args:
            constraint_type: Type of constraint
            equation_index: Which equation (0, 1, ...) this constraint applies to
            domains: List of domain indices (length 1 for boundary, 2 for junction)
            positions: Position coordinates in each domain
            parameters: Parameters for the constraint (e.g., Robin coefficients, KK parameters)
            data_function: Function f(t) providing constraint data over time
        """
        self.type = constraint_type
        self.equation_index = equation_index
        self.domains = domains
        self.positions = positions
        self.parameters = parameters if parameters is not None else np.array([])
        self.data_function = data_function if data_function is not None else lambda t: 0.0
        
        # Validate input
        if len(domains) != len(positions):
            raise ValueError("Number of domains must match number of positions")
        
        if constraint_type in [ConstraintType.DIRICHLET, ConstraintType.NEUMANN, ConstraintType.ROBIN]:
            if len(domains) != 1:
                raise ValueError("Boundary conditions require exactly one domain")
        else:
            if len(domains) != 2:
                raise ValueError("Junction conditions require exactly two domains")
    
    @property
    def is_boundary_condition(self) -> bool:
        """Check if this is a boundary condition."""
        return self.type in [ConstraintType.DIRICHLET, ConstraintType.NEUMANN, ConstraintType.ROBIN]
    
    @property
    def is_junction_condition(self) -> bool:
        """Check if this is a junction condition."""
        return not self.is_boundary_condition
    
    @property
    def n_multipliers(self) -> int:
        """Number of Lagrange multipliers for this constraint."""
        return 1 if self.is_boundary_condition else 2
    
    def get_data(self, time: float) -> float:
        """Get constraint data at given time."""
        return self.data_function(time)


class ConstraintManager:
    """
    Manages all constraints and their associated Lagrange multipliers.
    """
    
    def __init__(self):
        """Initialize empty constraint manager."""
        self.constraints: List[Constraint] = []
        self._node_mappings: List[List[int]] = []  # Node indices for each constraint
    
    def add_constraint(self, constraint: Constraint) -> int:
        """
        Add a constraint to the system.
        
        Args:
            constraint: Constraint to add
            
        Returns:
            Index of the added constraint
        """
        constraint_index = len(self.constraints)
        self.constraints.append(constraint)
        self._node_mappings.append([])  # Will be filled by map_to_discretizations
        return constraint_index
    
    def add_dirichlet(self, 
                     equation_index: int, 
                     domain_index: int, 
                     position: float,
                     data_function: Optional[Callable] = None) -> int:
        """Add Dirichlet boundary condition."""
        constraint = Constraint(
            ConstraintType.DIRICHLET,
            equation_index,
            [domain_index],
            [position],
            data_function=data_function
        )
        return self.add_constraint(constraint)
    
    def add_neumann(self, 
                   equation_index: int, 
                   domain_index: int, 
                   position: float,
                   data_function: Optional[Callable] = None) -> int:
        """Add Neumann boundary condition."""
        constraint = Constraint(
            ConstraintType.NEUMANN,
            equation_index,
            [domain_index],
            [position],
            data_function=data_function
        )
        return self.add_constraint(constraint)
    
    def add_robin(self, 
                 equation_index: int, 
                 domain_index: int, 
                 position: float,
                 alpha: float, 
                 beta: float,
                 data_function: Optional[Callable] = None) -> int:
        """Add Robin boundary condition: alpha * u + beta * du/dn = data."""
        constraint = Constraint(
            ConstraintType.ROBIN,
            equation_index,
            [domain_index],
            [position],
            parameters=np.array([alpha, beta]),
            data_function=data_function
        )
        return self.add_constraint(constraint)
    
    def add_trace_continuity(self, 
                           equation_index: int,
                           domain1_index: int, 
                           domain2_index: int,
                           position1: float, 
                           position2: float) -> int:
        """Add trace continuity condition: u1 = u2."""
        constraint = Constraint(
            ConstraintType.TRACE_CONTINUITY,
            equation_index,
            [domain1_index, domain2_index],
            [position1, position2]
        )
        return self.add_constraint(constraint)
    
    
    def add_kedem_katchalsky(self, 
                           equation_index: int,
                           domain1_index: int, 
                           domain2_index: int,
                           position1: float, 
                           position2: float,
                           permeability: float) -> int:
        """Add Kedem-Katchalsky condition: flux = -P * (u1 - u2)."""
        constraint = Constraint(
            ConstraintType.KEDEM_KATCHALSKY,
            equation_index,
            [domain1_index, domain2_index],
            [position1, position2],
            parameters=np.array([permeability])
        )
        return self.add_constraint(constraint)
    
    # =========================================================================
    # Query and override methods
    # =========================================================================
    
    def find_constraints(self,
                         domain_index: Optional[int] = None,
                         equation_index: Optional[int] = None,
                         constraint_type: Optional[ConstraintType] = None,
                         position: Optional[float] = None,
                         tol: float = 1e-10) -> List[int]:
        """Find constraints matching the given filters.

        All filter parameters are optional. When multiple filters are given
        they are combined with AND logic: a constraint must satisfy every
        specified filter to be included in the result.

        Args:
            domain_index: If given, keep only constraints whose domains list
                contains this domain index.
            equation_index: If given, keep only constraints for this equation.
            constraint_type: If given, keep only constraints of this type.
            position: If given, keep only constraints that have at least one
                position entry within *tol* of this value.
            tol: Absolute tolerance for the position comparison.

        Returns:
            List of matching constraint indices (may be empty).
        """
        matches = []
        for i, c in enumerate(self.constraints):
            if domain_index is not None and domain_index not in c.domains:
                continue
            if equation_index is not None and c.equation_index != equation_index:
                continue
            if constraint_type is not None and c.type != constraint_type:
                continue
            if position is not None:
                if not any(abs(p - position) <= tol for p in c.positions):
                    continue
            matches.append(i)
        return matches
    
    def replace_constraint(self, index: int, new_constraint: Constraint) -> None:
        """Replace an existing constraint in-place.

        The new constraint occupies the same position in the list so that
        all other constraint indices (and therefore multiplier offsets)
        remain unchanged.

        Important:
            The node mapping for this constraint is cleared.  The caller
            must call ``map_to_discretizations`` after all replacements
            are done.

        Args:
            index: Index of the constraint to replace.
            new_constraint: The replacement Constraint object.

        Raises:
            IndexError: If *index* is out of range.
            ValueError: If the replacement would change the number of
                Lagrange multipliers (boundary ↔ junction swap), which
                would silently corrupt the global system.
        """
        if index < 0 or index >= len(self.constraints):
            raise IndexError(
                f"Constraint index {index} out of range "
                f"(0–{len(self.constraints) - 1})"
            )
        old = self.constraints[index]
        if new_constraint.n_multipliers != old.n_multipliers:
            raise ValueError(
                f"Cannot replace a {old.type.value} constraint "
                f"({old.n_multipliers} multiplier(s)) with a "
                f"{new_constraint.type.value} constraint "
                f"({new_constraint.n_multipliers} multiplier(s)): "
                f"this would change the global multiplier count"
            )
        self.constraints[index] = new_constraint
        self._node_mappings[index] = []  # must re-map
    
    # =========================================================================
    # Factory methods (create Constraint objects without appending)
    # =========================================================================
    
    def make_dirichlet(self,
                       equation_index: int,
                       domain_index: int,
                       position: float,
                       data_function: Optional[Callable] = None) -> Constraint:
        """Create a Dirichlet constraint without adding it to the manager.

        Args:
            equation_index: Which equation this BC applies to.
            domain_index: Domain index.
            position: Position coordinate in the domain.
            data_function: Function f(t) providing the Dirichlet data.

        Returns:
            A Constraint object (not yet added to the manager).
        """
        return Constraint(
            ConstraintType.DIRICHLET,
            equation_index,
            [domain_index],
            [position],
            data_function=data_function,
        )
    
    def make_neumann(self,
                     equation_index: int,
                     domain_index: int,
                     position: float,
                     data_function: Optional[Callable] = None) -> Constraint:
        """Create a Neumann constraint without adding it to the manager.

        Args:
            equation_index: Which equation this BC applies to.
            domain_index: Domain index.
            position: Position coordinate in the domain.
            data_function: Function f(t) providing the Neumann data.

        Returns:
            A Constraint object (not yet added to the manager).
        """
        return Constraint(
            ConstraintType.NEUMANN,
            equation_index,
            [domain_index],
            [position],
            data_function=data_function,
        )
    
    def make_robin(self,
                   equation_index: int,
                   domain_index: int,
                   position: float,
                   alpha: float,
                   beta: float,
                   data_function: Optional[Callable] = None) -> Constraint:
        """Create a Robin constraint without adding it to the manager.

        Robin condition: alpha * u + beta * du/dn = data.

        Args:
            equation_index: Which equation this BC applies to.
            domain_index: Domain index.
            position: Position coordinate in the domain.
            alpha: Coefficient for the trace value.
            beta: Coefficient for the flux.
            data_function: Function f(t) providing the Robin data.

        Returns:
            A Constraint object (not yet added to the manager).
        """
        return Constraint(
            ConstraintType.ROBIN,
            equation_index,
            [domain_index],
            [position],
            parameters=np.array([alpha, beta]),
            data_function=data_function,
        )
    
    def make_trace_continuity(self,
                              equation_index: int,
                              domain1_index: int,
                              domain2_index: int,
                              position1: float,
                              position2: float) -> Constraint:
        """Create a trace-continuity constraint without adding it to the manager.

        Trace continuity: u1 = u2.

        Args:
            equation_index: Which equation this condition applies to.
            domain1_index: First domain index.
            domain2_index: Second domain index.
            position1: Position in the first domain.
            position2: Position in the second domain.

        Returns:
            A Constraint object (not yet added to the manager).
        """
        return Constraint(
            ConstraintType.TRACE_CONTINUITY,
            equation_index,
            [domain1_index, domain2_index],
            [position1, position2],
        )
    
    def make_kedem_katchalsky(self,
                              equation_index: int,
                              domain1_index: int,
                              domain2_index: int,
                              position1: float,
                              position2: float,
                              permeability: float) -> Constraint:
        """Create a Kedem-Katchalsky constraint without adding it to the manager.

        KK condition: flux = -P * (u1 - u2).

        Args:
            equation_index: Which equation this condition applies to.
            domain1_index: First domain index.
            domain2_index: Second domain index.
            position1: Position in the first domain.
            position2: Position in the second domain.
            permeability: Membrane permeability coefficient.

        Returns:
            A Constraint object (not yet added to the manager).
        """
        return Constraint(
            ConstraintType.KEDEM_KATCHALSKY,
            equation_index,
            [domain1_index, domain2_index],
            [position1, position2],
            parameters=np.array([permeability]),
        )
    
    def map_to_discretizations(self, discretizations: List) -> None:
        """
        Map constraint positions to discretization nodes.
        
        Args:
            discretizations: List of spatial discretizations for each domain
        """
        for i, constraint in enumerate(self.constraints):
            node_indices = []
            
            for domain_idx, position in zip(constraint.domains, constraint.positions):
                # Find closest node in discretization
                disc = discretizations[domain_idx]
                nodes = np.linspace(disc.domain_start, 
                                  disc.domain_start + disc.domain_length, 
                                  disc.n_elements + 1)
                closest_node = np.argmin(np.abs(nodes - position))
                node_indices.append(closest_node)
            
            self._node_mappings[i] = node_indices
    
    def get_node_indices(self, constraint_index: int) -> List[int]:
        """Get discretization node indices for a constraint."""
        return self._node_mappings[constraint_index]
    
    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return len(self.constraints)
    
    @property
    def n_multipliers(self) -> int:
        """Total number of Lagrange multipliers."""
        return sum(c.n_multipliers for c in self.constraints)
    
    def get_constraints_by_domain(self, domain_index: int) -> List[int]:
        """Get indices of all constraints involving a specific domain."""
        return [i for i, c in enumerate(self.constraints) 
                if domain_index in c.domains]
    
    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[int]:
        """Get indices of all constraints of a specific type."""
        return [i for i, c in enumerate(self.constraints) 
                if c.type == constraint_type]
    
    def get_multiplier_data(self, time: float) -> np.ndarray:
        """Get constraint data for all multipliers at given time."""
        data = []
        for constraint in self.constraints:
            constraint_data = constraint.get_data(time)
            # Add data for each multiplier (1 for boundary, 2 for junction)
            for _ in range(constraint.n_multipliers):
                data.append(constraint_data)
        return np.array(data)
    
    def compute_constraint_residuals(self, 
                                   trace_solutions: List[np.ndarray], 
                                   multiplier_values: np.ndarray, 
                                   time: float,
                                   domain_data_list: List = None) -> np.ndarray:
        """
        Compute constraint residuals for all constraints.
        
        Args:
            trace_solutions: List of trace solution vectors for each domain
            multiplier_values: Vector of all Lagrange multiplier values (containing flux values)
            time: Current time for time-dependent constraint data
            domain_data_list: List of domain data objects containing nodes information
            
        Returns:
            Vector of constraint residuals matching multiplier structure
        """
        
        if domain_data_list is None:
            raise ValueError("domain_data_list parameter is required for proper constraint evaluation")
        
        residuals = []
        multiplier_idx = 0
        
        for constraint_idx, constraint in enumerate(self.constraints):
            node_indices = self.get_node_indices(constraint_idx)
            
            if constraint.type == ConstraintType.DIRICHLET:
                # Dirichlet: u - g = 0
                domain_idx = constraint.domains[0]
                node_idx = node_indices[0]
                eq_idx = constraint.equation_index
                
                # Extract trace value at the constraint node using domain data
                n_nodes = len(domain_data_list[domain_idx].nodes)
                trace_idx = eq_idx * n_nodes + node_idx
                u_value = trace_solutions[domain_idx][trace_idx]
                
                constraint_data = constraint.get_data(time)
                    
                residual = u_value - constraint_data
                residuals.append(residual)
                multiplier_idx += 1
                
            elif constraint.type == ConstraintType.NEUMANN:
                # Neumann: flux - g = 0
                # Flux value is stored in multiplier_values
                flux_value = multiplier_values[multiplier_idx]
                constraint_data = constraint.get_data(time)
                residual = flux_value - constraint_data
                residuals.append(residual)
                multiplier_idx += 1

            elif constraint.type == ConstraintType.ROBIN:
                # Robin: alpha * u + beta * flux - g = 0
                domain_idx = constraint.domains[0]
                node_idx = node_indices[0]
                eq_idx = constraint.equation_index
                
                # Extract trace value using domain data
                n_nodes = len(domain_data_list[domain_idx].nodes)
                trace_idx = eq_idx * n_nodes + node_idx
                u_value = trace_solutions[domain_idx][trace_idx]
                
                alpha, beta = constraint.parameters[0], constraint.parameters[1]
                constraint_data = constraint.get_data(time)
                
                # Flux value is stored in multiplier_values
                flux_value = multiplier_values[multiplier_idx]
                residual = alpha * u_value + beta * flux_value - constraint_data
                residuals.append(residual)
                multiplier_idx += 1
                
            elif constraint.type == ConstraintType.TRACE_CONTINUITY:
                # Trace continuity: u1 - u2 = 0
                domain1_idx, domain2_idx = constraint.domains
                node1_idx, node2_idx = node_indices
                eq_idx = constraint.equation_index
                
                # Extract trace values from both domains using domain data
                n_nodes1 = len(domain_data_list[domain1_idx].nodes)
                n_nodes2 = len(domain_data_list[domain2_idx].nodes)
                
                trace_idx1 = eq_idx * n_nodes1 + node1_idx
                trace_idx2 = eq_idx * n_nodes2 + node2_idx
                
                u1_value = trace_solutions[domain1_idx][trace_idx1]
                u2_value = trace_solutions[domain2_idx][trace_idx2]
                
                flux1_value = multiplier_values[multiplier_idx]
                flux2_value = multiplier_values[multiplier_idx + 1]

                residual = u1_value - u2_value
                residuals.extend([residual, flux1_value + flux2_value])  # Two multipliers for junction condition
                multiplier_idx += 2
                
            elif constraint.type == ConstraintType.KEDEM_KATCHALSKY:
                # Kedem-Katchalsky: flux = -P * (u1 - u2)
                domain1_idx, domain2_idx = constraint.domains
                node1_idx, node2_idx = node_indices
                eq_idx = constraint.equation_index
                
                # Extract trace values using domain data
                n_nodes1 = len(domain_data_list[domain1_idx].nodes)
                n_nodes2 = len(domain_data_list[domain2_idx].nodes)
                
                trace_idx1 = eq_idx * n_nodes1 + node1_idx
                trace_idx2 = eq_idx * n_nodes2 + node2_idx
                
                u1_value = trace_solutions[domain1_idx][trace_idx1]
                u2_value = trace_solutions[domain2_idx][trace_idx2]
                
                permeability = constraint.parameters[0]
                
                # Flux values are stored in multiplier_values
                flux1_value = multiplier_values[multiplier_idx]
                flux2_value = multiplier_values[multiplier_idx + 1]
                
                # KK condition: flux1 + flux2 = -P * (u1 - u2)
                u_jump = u1_value - u2_value
                expected_flux_sum = permeability * u_jump
                
                # Placeholder residuals (would need actual flux values)
                residuals.extend([flux1_value - expected_flux_sum, flux2_value + expected_flux_sum])
                multiplier_idx += 2
        
        return np.array(residuals)
    
    def _get_equations_per_domain(self, domain_idx: int) -> int:
        """
        Helper to get number of equations per domain.
        
        WARNING: This method assumes neq=2 and should ONLY be used for Keller-Segel problems.
        For general problems, use the domain_data_list parameter in compute_constraint_residuals
        and access len(domain_data_list[domain_idx].nodes) directly.
        
        This method is deprecated and will be removed in future versions.
        """
        import warnings
        warnings.warn(
            "The _get_equations_per_domain method assumes neq=2 and should only be used for "
            "Keller-Segel problems. Use domain_data_list parameter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return 2  # Assuming neq = 2 for Keller-Segel ONLY
