"""
Lean Global assembly module for combining domain flux jumps and constraint residuals.
Uses the lean BulkDataManager approach where framework objects are passed as parameters.
"""

import numpy as np
from typing import List, Tuple, Optional

from .discretization import GlobalDiscretization
from .flux_jump import domain_flux_jump
from .constraints import ConstraintManager
from .lean_bulk_data_manager import BulkDataManager
from .bulk_data import BulkData


class GlobalAssembler:
    """
    Lean assembler that uses BulkDataManager with external framework objects.
    
    This implementation separates concerns by using a lean BulkDataManager that
    doesn't store framework objects internally, requiring them to be passed
    as parameters to methods that need them.
    """
    
    def __init__(self, 
                 domain_data_list: List,
                 constraint_manager: Optional[ConstraintManager] = None):
        """
        Initialize lean global assembler with pre-extracted domain data.
        
        Args:
            domain_data_list: List of DomainData objects with essential information
            constraint_manager: Optional constraint manager for boundary/junction conditions
        """
        self.bulk_manager = BulkDataManager(domain_data_list)
        self.constraint_manager = constraint_manager
        
        self.n_domains = len(domain_data_list)
        
        # Compute DOF structure from domain data
        self._compute_dof_structure()
    
    @classmethod
    def from_framework_objects(cls,
                              problems: List,
                              global_discretization,
                              static_condensations: List,
                              constraint_manager: Optional[ConstraintManager] = None):
        """
        Factory method to create LeanGlobalAssembler from framework objects.
        
        Args:
            problems: List of Problem instances for each domain
            global_discretization: GlobalDiscretization instance
            static_condensations: List of static condensation implementations
            constraint_manager: Optional constraint manager
            
        Returns:
            LeanGlobalAssembler instance
        """
        # Extract domain data using BulkDataManager static method
        domain_data_list = BulkDataManager.extract_domain_data_list(
            problems, global_discretization.spatial_discretizations, static_condensations
        )
        
        return cls(domain_data_list, constraint_manager)
    
    def _compute_dof_structure(self):
        """Compute the global DOF structure from domain data."""
        self.domain_trace_sizes = []
        self.domain_trace_offsets = []
        
        total_trace_dofs = 0
        for domain_data in self.bulk_manager.domain_data_list:
            n_nodes = domain_data.n_elements + 1
            trace_size = domain_data.neq * n_nodes
            
            self.domain_trace_sizes.append(trace_size)
            self.domain_trace_offsets.append(total_trace_dofs)
            total_trace_dofs += trace_size
        
        self.total_trace_dofs = total_trace_dofs
        self.n_multipliers = self.constraint_manager.n_multipliers if self.constraint_manager else 0
        self.total_dofs = self.total_trace_dofs + self.n_multipliers
                              
    def assemble_residual_and_jacobian(self, 
                                     global_solution: np.ndarray,
                                     forcing_terms: List[np.ndarray],
                                     static_condensations: List,
                                     time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble global residual and Jacobian from domain flux jumps and constraints.
        If we put the nonlinear static condensation equation in the form F(U;F_ext) = 0,
        where U is the trace solution and F_ext are the forcing terms 
        In our framework, F_ext is pre-computed as dt * f + M * u_old, 
        so we pass it in as forcing_terms. 
        
        Args:
            global_solution: Global solution vector [trace_solutions; multipliers]
            forcing_terms: List of forcing term arrays for each domain (already computed)
            static_condensations: List of static condensation objects for flux jump computation
            time: Current time (for constraint evaluation)
            
        Returns:
            tuple: (residual, jacobian) - Global residual vector and Jacobian matrix
        """
        # Validate inputs
        if len(forcing_terms) != self.n_domains:
            raise ValueError(f"Number of forcing terms ({len(forcing_terms)}) != number of domains ({self.n_domains})")
        
        if len(static_condensations) != self.n_domains:
            raise ValueError(f"Number of static condensations ({len(static_condensations)}) != number of domains ({self.n_domains})")
        
        # Extract trace solutions and multipliers from global solution
        trace_solutions = self._extract_trace_solutions(global_solution)
        multipliers = global_solution[self.total_trace_dofs:] if self.n_multipliers > 0 else np.array([])
        
        # Initialize global residual and Jacobian
        residual = np.zeros(self.total_dofs)
        jacobian = np.zeros((self.total_dofs, self.total_dofs))
        
        # Assemble domain contributions
        for i in range(self.n_domains):
            # Validate forcing term shape
            expected_rows = 2 * self.bulk_manager.domain_data_list[i].neq
            expected_cols = self.bulk_manager.domain_data_list[i].n_elements
            if forcing_terms[i].shape != (expected_rows, expected_cols):
                raise ValueError(f"Domain {i} forcing term shape {forcing_terms[i].shape} != expected ({expected_rows}, {expected_cols})")
            
            # Compute domain flux jump using static condensation
            U, F, JF = domain_flux_jump(
                trace_solutions[i].reshape(-1, 1),
                forcing_terms[i],
                None, None,
                static_condensations[i]
            )
            
            # Add domain residual to global residual
            start_idx = self.domain_trace_offsets[i]
            end_idx = start_idx + self.domain_trace_sizes[i]
            residual[start_idx:end_idx] += F.flatten()
            
            # Add domain Jacobian to global Jacobian
            jacobian[start_idx:end_idx, start_idx:end_idx] += JF
        
        # Add constraint contributions if present
        if self.constraint_manager is not None and self.n_multipliers > 0:
            # Add multiplier contributions to trace residuals
            multiplier_idx = 0
            
            for constraint_idx, constraint in enumerate(self.constraint_manager.constraints):
                node_indices = self.constraint_manager.get_node_indices(constraint_idx)
                
                if constraint.is_boundary_condition:
                    # Single domain constraint
                    domain_idx = constraint.domains[0]
                    node_idx = node_indices[0]
                    eq_idx = constraint.equation_index
                    
                    # Calculate global trace index
                    domain_offset = self.domain_trace_offsets[domain_idx]
                    domain_data = self.bulk_manager.domain_data_list[domain_idx]
                    n_nodes = domain_data.n_elements + 1
                    trace_idx = domain_offset + eq_idx * n_nodes + node_idx

                    print(f"DEBUG: Index of trace dof {trace_idx}, index of multiplier dof {multiplier_idx}") 
                    # Add multiplier value to trace residual
                    residual[trace_idx] += multipliers[multiplier_idx]
                    multiplier_idx += 1
                    
                else:
                    # Junction constraint - affects two domains
                    domain1_idx, domain2_idx = constraint.domains
                    node1_idx, node2_idx = node_indices
                    eq_idx = constraint.equation_index
                    
                    # Calculate global trace indices for both domains
                    domain1_offset = self.domain_trace_offsets[domain1_idx]
                    domain2_offset = self.domain_trace_offsets[domain2_idx]
                    domain1_data = self.bulk_manager.domain_data_list[domain1_idx]
                    domain2_data = self.bulk_manager.domain_data_list[domain2_idx]
                    n_nodes1 = domain1_data.n_elements + 1
                    n_nodes2 = domain2_data.n_elements + 1
                    
                    trace1_idx = domain1_offset + eq_idx * n_nodes1 + node1_idx
                    trace2_idx = domain2_offset + eq_idx * n_nodes2 + node2_idx
                    
                    # Add multiplier values to both trace residuals
                    residual[trace1_idx] += multipliers[multiplier_idx]
                    residual[trace2_idx] += multipliers[multiplier_idx + 1]
                    multiplier_idx += 2
            
            # Compute constraint residuals
            constraint_residuals = self.constraint_manager.compute_constraint_residuals(
                trace_solutions, multipliers, time
            )            
            
            # Add constraint residuals to global residual
            residual[self.total_trace_dofs:] = constraint_residuals
            
            # Add constraint Jacobian contributions
            self._add_constraint_jacobian_contributions(
                jacobian, trace_solutions, multipliers, time
            )
        
        return residual, jacobian

    def bulk_by_static_condensation(self, 
                                     global_solution: np.ndarray,
                                     forcing_terms: List[np.ndarray],
                                     static_condensations: List,
                                     time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble global residual and Jacobian from domain flux jumps and constraints.
        If we put the nonlinear static condensation equation in the form F(U;F_ext) = 0,
        where U is the trace solution and F_ext are the forcing terms 
        In our framework, F_ext is pre-computed as dt * f + M * u_old, 
        so we pass it in as forcing_terms. 
        
        Args:
            global_solution: Global solution vector [trace_solutions; multipliers]
            forcing_terms: List of forcing term arrays for each domain (already computed)
            static_condensations: List of static condensation objects for flux jump computation
            time: Current time (for constraint evaluation)
            
        Returns:
            tuple: (residual, jacobian) - Global residual vector and Jacobian matrix
        """
        # Validate inputs
        if len(forcing_terms) != self.n_domains:
            raise ValueError(f"Number of forcing terms ({len(forcing_terms)}) != number of domains ({self.n_domains})")
        
        if len(static_condensations) != self.n_domains:
            raise ValueError(f"Number of static condensations ({len(static_condensations)}) != number of domains ({self.n_domains})")
        
        # Extract trace solutions and multipliers from global solution
        trace_solutions = self._extract_trace_solutions(global_solution)
        multipliers = global_solution[self.total_trace_dofs:] if self.n_multipliers > 0 else np.array([])
        
        # Initialize global residual and Jacobian
        bulk_solution = []
        
        # Assemble domain contributions
        for i in range(self.n_domains):
            # Validate forcing term shape
            expected_rows = 2 * self.bulk_manager.domain_data_list[i].neq
            expected_cols = self.bulk_manager.domain_data_list[i].n_elements
            if forcing_terms[i].shape != (expected_rows, expected_cols):
                raise ValueError(f"Domain {i} forcing term shape {forcing_terms[i].shape} != expected ({expected_rows}, {expected_cols})")
            
            # Compute domain flux jump using static condensation
            U, F, JF = domain_flux_jump(
                trace_solutions[i].reshape(-1, 1),
                forcing_terms[i],
                None, None,
                static_condensations[i]
            )
            
            bulk_solution.append(U)

        return bulk_solution


    def compute_forcing_terms(self,
                            bulk_data_list: List[BulkData],
                            problems: List,
                            discretizations: List,
                            time: float,
                            dt: float) -> List[np.ndarray]:
        """
        Compute forcing terms for all domains using the lean BulkDataManager.
        
        Args:
            bulk_data_list: List of BulkData objects from previous time step
            problems: List of Problem objects
            discretizations: List of discretization objects
            time: Current time
            dt: Time step size
            
        Returns:
            List of forcing term arrays, one per domain
        """
        # Validate framework objects
        self.bulk_manager._validate_framework_objects(
            problems=problems,
            discretizations=discretizations,
            operation_name="compute_forcing_terms"
        )
        
        # Use BulkDataManager to compute forcing terms
        return self.bulk_manager.compute_forcing_terms(
            bulk_data_list, problems, discretizations, time, dt
        )
    
    def _extract_trace_solutions(self, global_solution: np.ndarray) -> List[np.ndarray]:
        """Extract individual domain trace solutions from global solution vector."""
        trace_solutions = []
        
        for i in range(self.n_domains):
            start_idx = self.domain_trace_offsets[i]
            end_idx = start_idx + self.domain_trace_sizes[i]
            trace_solutions.append(global_solution[start_idx:end_idx])
        
        return trace_solutions
    
    def _add_constraint_jacobian_contributions(self, 
                                             jacobian: np.ndarray,
                                             trace_solutions: List[np.ndarray],
                                             multipliers: np.ndarray,
                                             time: float):
        """
        Add constraint Jacobian contributions to global Jacobian.
        
        Args:
            jacobian: Global Jacobian matrix to modify
            trace_solutions: List of trace solution arrays
            multipliers: Array of constraint multipliers
            time: Current time
        """
        multiplier_idx = 0
        
        for constraint_idx, constraint in enumerate(self.constraint_manager.constraints):
            node_indices = self.constraint_manager.get_node_indices(constraint_idx)
            
            if constraint.is_boundary_condition:
                # Boundary constraint: affects one domain
                domain_idx = constraint.domains[0]
                node_idx = node_indices[0]
                eq_idx = constraint.equation_index
                
                # Global indices
                domain_offset = self.domain_trace_offsets[domain_idx]
                domain_data = self.bulk_manager.domain_data_list[domain_idx]
                n_nodes = domain_data.n_elements + 1
                trace_idx = domain_offset + eq_idx * n_nodes + node_idx
                multiplier_global_idx = self.total_trace_dofs + multiplier_idx
                
                # Add constraint Jacobian terms
                if constraint.type.value == "dirichlet":
                    # Constraint: u - g = 0
                    jacobian[multiplier_global_idx, trace_idx] = 1.0
                elif constraint.type.value == "neumann":
                    # Constraint: flux - g = 0
                    jacobian[multiplier_global_idx, multiplier_global_idx] = 1.0
                
                # Add coupling terms
                jacobian[trace_idx, multiplier_global_idx] = 1.0
                
                multiplier_idx += 1
                
            else:
                # Junction constraint: affects two domains
                domain1_idx, domain2_idx = constraint.domains
                node1_idx, node2_idx = node_indices
                eq_idx = constraint.equation_index
                
                # Global indices for both domains
                domain1_offset = self.domain_trace_offsets[domain1_idx]
                domain2_offset = self.domain_trace_offsets[domain2_idx]
                domain1_data = self.bulk_manager.domain_data_list[domain1_idx]
                domain2_data = self.bulk_manager.domain_data_list[domain2_idx]
                n_nodes1 = domain1_data.n_elements + 1
                n_nodes2 = domain2_data.n_elements + 1
                
                trace1_idx = domain1_offset + eq_idx * n_nodes1 + node1_idx
                trace2_idx = domain2_offset + eq_idx * n_nodes2 + node2_idx
                
                multiplier1_global_idx = self.total_trace_dofs + multiplier_idx
                multiplier2_global_idx = self.total_trace_dofs + multiplier_idx + 1
                
                if constraint.type.value == "trace_continuity":
                    # Constraint equations
                    jacobian[multiplier1_global_idx, trace1_idx] = 1.0
                    jacobian[multiplier1_global_idx, trace2_idx] = -1.0
                    jacobian[multiplier2_global_idx, multiplier1_global_idx] = 1.0
                    jacobian[multiplier2_global_idx, multiplier2_global_idx] = 1.0
                    
                    # Coupling terms
                    jacobian[trace1_idx, multiplier1_global_idx] = 1.0
                    jacobian[trace2_idx, multiplier2_global_idx] = 1.0
                
                elif constraint.type.value == "kedem_katchalsky":
                    # KK constraint with permeability
                    P = constraint.parameters[0]
                    
                    jacobian[multiplier1_global_idx, multiplier1_global_idx] = 1.0
                    jacobian[multiplier1_global_idx, trace1_idx] = -P
                    jacobian[multiplier1_global_idx, trace2_idx] = P
                    jacobian[multiplier2_global_idx, multiplier1_global_idx] = 1.0
                    jacobian[multiplier2_global_idx, multiplier2_global_idx] = 1.0
                    
                    jacobian[trace1_idx, multiplier1_global_idx] = 1.0
                    jacobian[trace2_idx, multiplier2_global_idx] = 1.0
                
                multiplier_idx += 2
    
    def create_initial_guess_from_bulk_data(self, 
                                           bulk_data_list: List[BulkData]) -> np.ndarray:
        """
        Create initial guess for global solution vector from BulkData objects.
        
        Args:
            bulk_data_list: List of BulkData objects
            
        Returns:
            Initial guess for global solution vector
        """
        initial_guess = np.zeros(self.total_dofs)
        
        # Extract trace values from BulkData objects
        for i in range(self.n_domains):
            start_idx = self.domain_trace_offsets[i]
            end_idx = start_idx + self.domain_trace_sizes[i]
            
            # Get trace values from BulkData
            trace_values = bulk_data_list[i].get_trace_values()
            initial_guess[start_idx:end_idx] = trace_values
        
        # Multipliers start at zero
        initial_guess[self.total_trace_dofs:] = 0.0
        
        return initial_guess
    
    def create_initial_guess_from_problems(self, 
                                         problems: List,
                                         discretizations: List,
                                         time: float = 0.0) -> np.ndarray:
        """
        Create initial guess directly from problem initial conditions.
        
        Args:
            problems: List of Problem objects
            discretizations: List of discretization objects
            time: Time for initial condition evaluation
            
        Returns:
            Initial guess for global solution vector
        """
        # Validate framework objects
        self.bulk_manager._validate_framework_objects(
            problems=problems,
            discretizations=discretizations,
            operation_name="create_initial_guess_from_problems"
        )
        
        initial_guess = np.zeros(self.total_dofs)
        
        # Initialize trace solutions with initial conditions
        for i in range(self.n_domains):
            start_idx = self.domain_trace_offsets[i]
            end_idx = start_idx + self.domain_trace_sizes[i]
            
            # Set initial conditions from problem definition
            problem = problems[i]
            discretization = discretizations[i]
            domain_data = self.bulk_manager.domain_data_list[i]
            n_nodes = discretization.n_elements + 1
            
            nodes = discretization.nodes
            
            for eq in range(problem.neq):
                eq_start = start_idx + eq * n_nodes
                eq_end = eq_start + n_nodes
                
                # Use initial condition function if available
                if (domain_data.initial_conditions[eq] is not None and 
                    callable(domain_data.initial_conditions[eq])):
                    try:
                        initial_values = domain_data.initial_conditions[eq](nodes, time)
                        initial_guess[eq_start:eq_end] = initial_values
                    except Exception as e:
                        print(f"Warning: Could not evaluate initial condition for domain {i}, equation {eq}: {e}")
                        initial_guess[eq_start:eq_end] = 0.0
                else:
                    initial_guess[eq_start:eq_end] = 0.0
        
        # Multipliers start at zero
        initial_guess[self.total_trace_dofs:] = 0.0
        
        return initial_guess
    
    def get_domain_solutions(self, global_solution: np.ndarray) -> List[np.ndarray]:
        """Extract domain trace solutions from global solution."""
        return self._extract_trace_solutions(global_solution)
    
    def get_multipliers(self, global_solution: np.ndarray) -> np.ndarray:
        """Extract constraint multipliers from global solution."""
        return global_solution[self.total_trace_dofs:] if self.n_multipliers > 0 else np.array([])
    
    def initialize_bulk_data(self, 
                           problems: List,
                           discretizations: List,
                           time: float = 0.0) -> List[BulkData]:
        """
        Initialize BulkData objects using lean BulkDataManager.
        
        Args:
            problems: List of Problem objects
            discretizations: List of discretization objects
            time: Initial time
            
        Returns:
            List of initialized BulkData objects
        """
        return self.bulk_manager.initialize_all_bulk_data(problems, discretizations, time)
        
    
    def compute_mass_conservation(self, bulk_data_list: List[BulkData]) -> float:
        """
        Compute total mass for conservation check using lean BulkDataManager.
        
        Args:
            bulk_data_list: List of BulkData instances
            
        Returns:
            Total mass across all domains
        """
        return self.bulk_manager.compute_total_mass(bulk_data_list)
    
    def get_num_domains(self) -> int:
        """Get number of domains."""
        return self.n_domains
    
    def get_domain_info(self, domain_idx: int):
        """Get domain data for inspection."""
        return self.bulk_manager.get_domain_info(domain_idx)
    
    def test(self, 
             problems: List = None,
             discretizations: List = None,
             static_condensations: List = None) -> bool:
        """
        Test the LeanGlobalAssembler functionality.
        
        Args:
            problems: List of Problem objects for testing (optional)
            discretizations: List of discretization objects for testing (optional)
            static_condensations: List of static condensation objects for testing (optional)
            
        Returns:
            True if all tests pass, False otherwise
        """
        print(f"Testing LeanGlobalAssembler with {self.n_domains} domains")

        if problems is None or discretizations is None or static_condensations is None:
            print("FAIL: Missing required arguments")
            return False

        # Test 0: Check parameter list lengths match
        if len(problems) != len(discretizations):
            print(f"FAIL: Length mismatch - problems ({len(problems)}) != discretizations ({len(discretizations)})")
            return False
        
        if len(problems) != len(static_condensations):
            print(f"FAIL: Length mismatch - problems ({len(problems)}) != static_condensations ({len(static_condensations)})")
            return False
        
        if len(discretizations) != len(static_condensations):
            print(f"FAIL: Length mismatch - discretizations ({len(discretizations)}) != static_condensations ({len(static_condensations)})")
            return False
        
        if len(problems) != self.n_domains:
            print(f"FAIL: Length mismatch - problems ({len(problems)}) != n_domains ({self.n_domains})")
            return False
        
        print(f"PASS: Parameter list lengths validated ({len(problems)} domains)")

        try:
            # Test 1: Test BulkDataManager
            if not self.bulk_manager.test(problems, discretizations, static_condensations):
                print("FAIL: BulkDataManager test failed")
                return False
            print("PASS: BulkDataManager test passed")
            
            # Test 2: DOF structure validation
            if self.total_dofs <= 0:
                print("FAIL: Invalid total DOF count")
                return False
            
            expected_trace_dofs = sum(
                dd.neq * (dd.n_elements + 1) for dd in self.bulk_manager.domain_data_list
            )
            if self.total_trace_dofs != expected_trace_dofs:
                print(f"FAIL: Trace DOF count mismatch: {self.total_trace_dofs} != {expected_trace_dofs}")
                return False
            
            print(f"PASS: DOF structure validated (trace={self.total_trace_dofs}, multipliers={self.n_multipliers})")
            
            # Test 3: Domain offset validation
            for i in range(self.n_domains):
                expected_size = self.bulk_manager.domain_data_list[i].neq * (self.bulk_manager.domain_data_list[i].n_elements + 1)
                if self.domain_trace_sizes[i] != expected_size:
                    print(f"FAIL: Domain {i} trace size mismatch: {self.domain_trace_sizes[i]} != {expected_size}")
                    return False
            
            # Check offset consistency
            for i in range(1, self.n_domains):
                expected_offset = self.domain_trace_offsets[i-1] + self.domain_trace_sizes[i-1]
                if self.domain_trace_offsets[i] != expected_offset:
                    print(f"FAIL: Domain {i} offset mismatch: {self.domain_trace_offsets[i]} != {expected_offset}")
                    return False
            
            print("PASS: Domain offsets and sizes validated")
            
            # Test 4: Initial guess creation (if framework objects provided)
            if problems is not None and discretizations is not None:
                try:
                    # Test BulkData-based initial guess
                    bulk_data_list = self.initialize_bulk_data(problems, discretizations, time=0.0)
                    initial_guess_bd = self.create_initial_guess_from_bulk_data(bulk_data_list)
                    
                    if initial_guess_bd.shape != (self.total_dofs,):
                        print(f"FAIL: Initial guess shape {initial_guess_bd.shape} != ({self.total_dofs},)")
                        return False
                    
                    # Test problem-based initial guess
                    initial_guess_prob = self.create_initial_guess_from_problems(problems, discretizations, time=0.0)
                    
                    if initial_guess_prob.shape != (self.total_dofs,):
                        print(f"FAIL: Problem initial guess shape {initial_guess_prob.shape} != ({self.total_dofs},)")
                        return False
                    
                    # Validate ranges
                    if np.any(np.isnan(initial_guess_bd)) or np.any(np.isinf(initial_guess_bd)):
                        print("FAIL: BulkData initial guess contains NaN or infinite values")
                        return False
                    
                    if np.any(np.isnan(initial_guess_prob)) or np.any(np.isinf(initial_guess_prob)):
                        print("FAIL: Problem initial guess contains NaN or infinite values")
                        return False
                    
                    print("PASS: Initial guess creation tests passed")
                    
                    # Test 5: Forcing term computation
                    if problems is not None and discretizations is not None:
                        try:
                            bulk_data_list = self.initialize_bulk_data(problems, discretizations, time=0.0)
                            
                            # Test forcing term computation separately
                            forcing_terms = self.compute_forcing_terms(
                                bulk_data_list, problems, discretizations, time=0.1, dt=0.01
                            )
                            
                            if len(forcing_terms) != self.n_domains:
                                print(f"FAIL: Forcing terms count {len(forcing_terms)} != {self.n_domains}")
                                return False
                            
                            # Validate forcing term shapes
                            for i, forcing_term in enumerate(forcing_terms):
                                expected_rows = 2 * self.bulk_manager.domain_data_list[i].neq
                                expected_cols = self.bulk_manager.domain_data_list[i].n_elements
                                if forcing_term.shape != (expected_rows, expected_cols):
                                    print(f"FAIL: Domain {i} forcing term shape {forcing_term.shape} != expected ({expected_rows}, {expected_cols})")
                                    return False
                            
                            print("PASS: Forcing term computation test passed")
                            
                            # Test 6: Residual and Jacobian assembly with pre-computed forcing terms
                            test_solution = np.random.rand(self.total_dofs) * 0.1  # Small random values
                            
                            try:
                                # Extract forcing terms as List[np.ndarray] from BulkData.data objects
                                forcing_terms_data = [bulk_sol.get_data() for bulk_sol in bulk_data_list]
                                
                                residual, jacobian = self.assemble_residual_and_jacobian(
                                    global_solution=test_solution,
                                    forcing_terms=forcing_terms_data,
                                    static_condensations=static_condensations,
                                    time=0.1
                                )
                                
                                if residual.shape != (self.total_dofs,):
                                    print(f"FAIL: Residual shape {residual.shape} != ({self.total_dofs},)")
                                    return False
                                
                                if jacobian.shape != (self.total_dofs, self.total_dofs):
                                    print(f"FAIL: Jacobian shape {jacobian.shape} != ({self.total_dofs}, {self.total_dofs})")
                                    return False
                                
                                # Check for invalid values
                                if np.any(np.isnan(residual)) or np.any(np.isinf(residual)):
                                    print("FAIL: Residual contains NaN or infinite values")
                                    return False
                                
                                if np.any(np.isnan(jacobian)) or np.any(np.isinf(jacobian)):
                                    print("FAIL: Jacobian contains NaN or infinite values")
                                    return False
                                
                                print(f"PASS: Residual and Jacobian assembly test passed")
                                print(f"  Residual range: [{np.min(residual):.6e}, {np.max(residual):.6e}]")
                                print(f"  Jacobian range: [{np.min(jacobian):.6e}, {np.max(jacobian):.6e}]")
                                
                                # Test with zero forcing terms
                                zero_forcing_terms = [np.zeros_like(ft) for ft in forcing_terms_data]
                                zero_residual, zero_jacobian = self.assemble_residual_and_jacobian(
                                    global_solution=test_solution,
                                    forcing_terms=zero_forcing_terms,
                                    static_condensations=static_condensations,
                                    time=0.1
                                )
                                
                                print(f"PASS: Zero forcing terms test passed")
                                print(f"  Zero residual range: [{np.min(zero_residual):.6e}, {np.max(zero_residual):.6e}]")
                                
                            except Exception as e:
                                print(f"FAIL: Residual/Jacobian assembly failed: {e}")
                                import traceback
                                traceback.print_exc()
                                return False
                                
                        except Exception as e:
                            print(f"FAIL: Framework integration test failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return False
                    else:
                        print("SKIP: No framework objects provided - skipping forcing term tests")
                    
                except Exception as e:
                    print(f"FAIL: Framework integration test failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print("SKIP: No framework objects provided - skipping integration tests")
            
            # Test 7: Solution extraction
            test_solution = np.random.rand(self.total_dofs)
            domain_solutions = self.get_domain_solutions(test_solution)
            multipliers = self.get_multipliers(test_solution)
            
            if len(domain_solutions) != self.n_domains:
                print(f"FAIL: Domain solutions count {len(domain_solutions)} != {self.n_domains}")
                return False
            
            if len(multipliers) != self.n_multipliers:
                print(f"FAIL: Multipliers count {len(multipliers)} != {self.n_multipliers}")
                return False
            
            # Validate domain solution sizes
            for i, domain_sol in enumerate(domain_solutions):
                expected_size = self.domain_trace_sizes[i]
                if len(domain_sol) != expected_size:
                    print(f"FAIL: Domain {i} solution size {len(domain_sol)} != {expected_size}")
                    return False
            
            print("PASS: Solution extraction tests passed")
            
            # Test 8: Mass conservation (if framework objects provided)
            if problems is not None and discretizations is not None:
                try:
                    bulk_data_list = self.initialize_bulk_data(problems, discretizations, time=0.0)
                    total_mass = self.compute_mass_conservation(bulk_data_list)
                    
                    if np.isnan(total_mass) or np.isinf(total_mass):
                        print(f"FAIL: Total mass is NaN or infinite: {total_mass}")
                        return False
                    
                    print(f"PASS: Mass conservation test passed (total_mass={total_mass:.6e})")
                    
                except Exception as e:
                    print(f"FAIL: Mass conservation test failed: {e}")
                    return False
            else:
                print("SKIP: Mass conservation test requires framework objects")
            
            # Test 9: Constraint handling (if constraints exist)
            if self.constraint_manager is not None and self.n_multipliers > 0:
                print(f"PASS: Constraint manager present with {self.n_multipliers} multipliers")
                
                # Test constraint residual structure
                test_trace_solutions = self.get_domain_solutions(test_solution)
                test_multipliers = self.get_multipliers(test_solution)
                
                try:
                    constraint_residuals = self.constraint_manager.compute_constraint_residuals(
                        test_trace_solutions, test_multipliers, time=0.1
                    )
                    
                    if len(constraint_residuals) != self.n_multipliers:
                        print(f"FAIL: Constraint residual count {len(constraint_residuals)} != {self.n_multipliers}")
                        return False
                    
                    print("PASS: Constraint handling test passed")
                    
                except Exception as e:
                    print(f"FAIL: Constraint handling test failed: {e}")
                    return False
            else:
                print("SKIP: No constraints defined")
            
            # Test 10: Factory method (if framework objects provided)
            if problems is not None and discretizations is not None and static_condensations is not None:
                try:
                    # Create mock global discretization
                    class MockGlobalDiscretization:
                        def __init__(self, discretizations):
                            self.spatial_discretizations = discretizations
                    
                    global_disc = MockGlobalDiscretization(discretizations)
                    
                    factory_assembler = GlobalAssembler.from_framework_objects(
                        problems, global_disc, static_condensations, self.constraint_manager
                    )
                    
                    if factory_assembler.n_domains != self.n_domains:
                        print("FAIL: Factory method created assembler with wrong domain count")
                        return False
                    
                    if factory_assembler.total_dofs != self.total_dofs:
                        print("FAIL: Factory method created assembler with wrong DOF count")
                        return False
                    
                    print("PASS: Factory method test passed")
                    
                except Exception as e:
                    print(f"FAIL: Factory method test failed: {e}")
                    return False
            else:
                print("SKIP: Factory method test requires all framework objects")
            
            print("✓ All LeanGlobalAssembler tests passed!")
            return True
            
        except Exception as e:
            print(f"FAIL: Unexpected error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def __str__(self) -> str:
        return (f"LeanGlobalAssembler(domains={self.n_domains}, "
                f"trace_dofs={self.total_trace_dofs}, "
                f"multipliers={self.n_multipliers}, "
                f"total_dofs={self.total_dofs})")
    
    
    def __repr__(self) -> str:
        return (f"LeanGlobalAssembler(n_domains={self.n_domains}, "
                f"domain_trace_sizes={self.domain_trace_sizes}, "
                f"n_multipliers={self.n_multipliers})")
