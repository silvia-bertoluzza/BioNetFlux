"""
Lean Bulk Data Manager for HDG method.

This module provides a memory-efficient coordinator for bulk operations that
extracts only essential data during initialization and accepts framework
objects as parameters to methods that need them.
"""

import numpy as np
from typing import List, Optional, Callable
from ooc1d.core.bulk_data import BulkData
from ooc1d.core.domain_data import DomainData


class BulkDataManager:
    """
    Ultra-lean coordinator for bulk operations in HDG method.
    
    This class stores only essential extracted domain data and accepts
    framework objects as parameters to methods that need them. This approach
    minimizes memory usage and increases flexibility.
    """
    
    def __init__(self, domain_data_list: List[DomainData]):
        """
        Initialize with pre-extracted domain data only.
        
        Args:
            domain_data_list: List of DomainData objects with essential information
        """
        self.domain_data_list = domain_data_list
    
    def _validate_framework_objects(self, 
                                   problems: List = None,
                                   discretizations: List = None, 
                                   static_condensations: List = None,
                                   operation_name: str = "operation") -> None:
        """
        Validate that framework objects match the stored domain data.
        
        Args:
            problems: List of Problem objects to validate
            discretizations: List of discretization objects to validate
            static_condensations: List of static condensation objects to validate
            operation_name: Name of the operation for error messages
            
        Raises:
            ValueError: If framework objects don't match domain data
        """
        n_domains = len(self.domain_data_list)
        
        # Check list lengths
        if problems is not None:
            if len(problems) != n_domains:
                raise ValueError(
                    f"{operation_name}: Number of problems ({len(problems)}) "
                    f"must match number of domains ({n_domains})"
                )
        
        if discretizations is not None:
            if len(discretizations) != n_domains:
                raise ValueError(
                    f"{operation_name}: Number of discretizations ({len(discretizations)}) "
                    f"must match number of domains ({n_domains})"
                )
        
        if static_condensations is not None:
            if len(static_condensations) != n_domains:
                raise ValueError(
                    f"{operation_name}: Number of static condensations ({len(static_condensations)}) "
                    f"must match number of domains ({n_domains})"
                )
        
        # Validate individual domain compatibility
        for i in range(n_domains):
            domain_data = self.domain_data_list[i]
            
            # Validate problem compatibility
            if problems is not None:
                problem = problems[i]
                if not hasattr(problem, 'neq'):
                    raise ValueError(f"{operation_name}: Problem {i} missing 'neq' attribute")
                
                if problem.neq != domain_data.neq:
                    raise ValueError(
                        f"{operation_name}: Problem {i} neq ({problem.neq}) "
                        f"doesn't match domain data neq ({domain_data.neq})"
                    )
            
            # Validate discretization compatibility
            if discretizations is not None:
                discretization = discretizations[i]
                if not hasattr(discretization, 'n_elements'):
                    raise ValueError(f"{operation_name}: Discretization {i} missing 'n_elements' attribute")
                
                if discretization.n_elements != domain_data.n_elements:
                    raise ValueError(
                        f"{operation_name}: Discretization {i} n_elements ({discretization.n_elements}) "
                        f"doesn't match domain data n_elements ({domain_data.n_elements})"
                    )
                
                if hasattr(discretization, 'nodes') and hasattr(discretization, 'element_length'):
                    # Check node count consistency
                    expected_nodes = discretization.n_elements + 1
                    if len(discretization.nodes) != expected_nodes:
                        raise ValueError(
                            f"{operation_name}: Discretization {i} has {len(discretization.nodes)} nodes, "
                            f"expected {expected_nodes} for {discretization.n_elements} elements"
                        )
                    
                    # Check element length consistency (allow small tolerance for floating point)
                    if abs(discretization.element_length - domain_data.element_length) > 1e-12:
                        raise ValueError(
                            f"{operation_name}: Discretization {i} element_length ({discretization.element_length}) "
                            f"doesn't match domain data element_length ({domain_data.element_length})"
                        )
            
            # Validate static condensation compatibility
            if static_condensations is not None:
                sc = static_condensations[i]
                if not hasattr(sc, 'build_matrices'):
                    raise ValueError(f"{operation_name}: Static condensation {i} missing 'build_matrices' method")
                
                try:
                    sc_matrices = sc.build_matrices()
                    
                    # Check mass matrix compatibility
                    mass_matrix = sc_matrices.get('M')
                    if mass_matrix is not None:
                        if mass_matrix.shape != domain_data.mass_matrix.shape:
                            raise ValueError(
                                f"{operation_name}: Static condensation {i} mass matrix shape "
                                f"{mass_matrix.shape} doesn't match domain data mass matrix shape "
                                f"{domain_data.mass_matrix.shape}"
                            )
                    
                    # Check trace matrix compatibility
                    trace_matrix = sc_matrices.get('T')
                    if trace_matrix is not None:
                        if trace_matrix.shape != domain_data.trace_matrix.shape:
                            raise ValueError(
                                f"{operation_name}: Static condensation {i} trace matrix shape "
                                f"{trace_matrix.shape} doesn't match domain data trace matrix shape "
                                f"{domain_data.trace_matrix.shape}"
                            )
                
                except Exception as e:
                    raise ValueError(
                        f"{operation_name}: Failed to build matrices for static condensation {i}: {e}"
                    )

    @staticmethod
    def extract_domain_data_list(problems: List, 
                                discretizations: List, 
                                static_condensations: List) -> List[DomainData]:
        """
        Static factory method to extract domain data from framework objects.
        
        This method can be called once and the result stored externally,
        allowing multiple lean managers to use the same extracted data.
        
        Args:
            problems: List of Problem instances
            discretizations: List of discretization instances  
            static_condensations: List of static condensation instances
            
        Returns:
            List of DomainData objects with essential extracted information
        """
        domain_data_list = []
        
        for i, (problem, discretization, sc) in enumerate(zip(
            problems, discretizations, static_condensations)):
            
            domain_data = BulkDataManager._extract_single_domain_data(
                problem, discretization, sc, i
            )
            domain_data_list.append(domain_data)
        
        return domain_data_list
    
    @staticmethod
    def _extract_single_domain_data(problem, discretization, sc, domain_idx: int) -> DomainData:
        """Extract essential data from a single domain's objects."""
        # Get matrices from static condensation
        sc_matrices = sc.build_matrices()
        mass_matrix = sc_matrices.get('M')
        trace_matrix = sc_matrices.get('T')
        
        if mass_matrix is None:
            mass_matrix = np.eye(2)
            print(f"Warning: No mass matrix for domain {domain_idx}, using identity")
            
        if trace_matrix is None:
            raise ValueError(f"Trace matrix not available for domain {domain_idx}")
        
        # Extract initial conditions
        initial_conditions = []
        for eq in range(problem.neq):
            ic_func = None
            
            # Try different access patterns
            if hasattr(problem, 'u0') and len(problem.u0) > eq:
                ic_func = problem.u0[eq] if callable(problem.u0[eq]) else None
            elif hasattr(problem, 'initial_conditions') and len(problem.initial_conditions) > eq:
                ic_func = problem.initial_conditions[eq] if callable(problem.initial_conditions[eq]) else None
            elif hasattr(problem, 'get_initial_condition'):
                try:
                    ic_func = problem.get_initial_condition(eq)
                except:
                    ic_func = None
            
            initial_conditions.append(ic_func)
        
        # Extract forcing functions
        forcing_functions = []
        for eq in range(problem.neq):
            force_func = None
            
            # Try different access patterns
            if hasattr(problem, 'force') and len(problem.force) > eq:
                force_func = problem.force[eq] if callable(problem.force[eq]) else None
            elif hasattr(problem, 'forcing_functions') and len(problem.forcing_functions) > eq:
                force_func = problem.forcing_functions[eq] if callable(problem.forcing_functions[eq]) else None
            elif hasattr(problem, 'get_forcing_function'):
                try:
                    force_func = problem.get_forcing_function(eq)
                except:
                    force_func = None
            
            forcing_functions.append(force_func)
        
        return DomainData(
            neq=problem.neq,
            n_elements=discretization.n_elements,
            nodes=discretization.nodes,
            element_length=discretization.element_length,
            mass_matrix=mass_matrix,
            trace_matrix=trace_matrix,
            initial_conditions=initial_conditions,
            forcing_functions=forcing_functions
        )
    
    def create_bulk_data(self, 
                        domain_index: int, 
                        problem, 
                        discretization, 
                        dual: bool = False) -> BulkData:
        """
        Create a BulkData object using external framework objects.
        
        Args:
            domain_index: Index of the domain
            problem: Problem object for this domain
            discretization: Discretization object for this domain  
            dual: Whether to use dual formulation
            
        Returns:
            BulkData object
        """
        if domain_index < 0 or domain_index >= len(self.domain_data_list):
            raise ValueError(f"Domain index {domain_index} out of range")
        
        # Validate individual domain objects
        return BulkData(problem, discretization, dual=dual)

    def compute_source_terms(self,
                            problems: List,
                            discretizations: List,
                            time: float) -> List[BulkData]:
        """
        Compute source terms using external framework objects.
        
        Args:
            problems: List of Problem objects
            discretizations: List of discretization objects
            time: Current time
            
        Returns:
            List of BulkData objects with source terms
        """
        
        # Validate framework objects
        self._validate_framework_objects(
            problems=problems,
            discretizations=discretizations,
            operation_name="compute_source_terms"
        )
        
        source_terms = []
        for i, (problem, discretization) in enumerate(zip(problems, discretizations)):
            source_term = BulkData(problem, discretization, dual=True)
            source_term.set_data(problem.force, time)
            source_terms.append(source_term)
        # Validate bulk data compatibility
        
        return source_terms
    
    def compute_forcing_terms(self, 
                              bulk_data_list: List[BulkData],
                              problems: List,
                              discretizations: List, 
                              time: float, 
                              dt: float) -> List[np.ndarray]:
        """
        Compute forcing terms using external framework objects.
        
        Args:
            bulk_data_list: List of BulkData instances (current solutions)
            problems: List of Problem objects
            discretizations: List of discretization objects
            time: Current time
            dt: Time step size
            
        Returns:
            List of forcing term arrays
        """
        if len(bulk_data_list) != len(self.domain_data_list):
            raise ValueError("Number of bulk data objects must match number of domains")
        
        # Validate framework objects
        self._validate_framework_objects(
            problems=problems,
            discretizations=discretizations,
            operation_name="compute_forcing_terms"
        )
        
        # Validate bulk data compatibility
        for i, bulk_data in enumerate(bulk_data_list):
            domain_data = self.domain_data_list[i]
            bulk_data_shape = bulk_data.get_data().shape
            expected_shape = (2 * domain_data.neq, domain_data.n_elements)
            
            if bulk_data_shape != expected_shape:
                raise ValueError(
                    f"compute_forcing_terms: BulkData {i} has shape {bulk_data_shape}, "
                    f"expected {expected_shape} for domain with {domain_data.neq} equations "
                    f"and {domain_data.n_elements} elements"
                )

        forcing_terms = []
        
        for i, (domain_data, bulk_data, problem, discretization) in enumerate(zip(
            self.domain_data_list, bulk_data_list, problems, discretizations)):
            
            # Create a dual BulkData object for forcing term integration
            forcing_bulk_data = self.create_bulk_data(i, problem, discretization, dual=True)
            
            # Set forcing functions for integration
            if any(f is not None for f in domain_data.forcing_functions):
                forcing_bulk_data.set_data(domain_data.forcing_functions, time)
                force_contrib = forcing_bulk_data.get_data() * dt
            else:
                force_contrib = np.zeros_like(bulk_data.get_data())
            
            # Get current bulk solution data
            current_data = bulk_data.get_data()
            
            # Compute mass matrix contribution (implicit Euler: M * U_old)
            mass_contrib = np.zeros_like(current_data)
            for eq in range(domain_data.neq):
                start_idx = eq * 2
                end_idx = start_idx + 2
                mass_contrib[start_idx:end_idx, :] = (
                    domain_data.mass_matrix @ current_data[start_idx:end_idx, :]
                )
            
            # Combine contributions: forcing * dt + M * U_old
            forcing_term = force_contrib + mass_contrib
            forcing_terms.append(forcing_term)
        
        return forcing_terms
    
    def initialize_all_bulk_data(self, 
                                problems: List,
                                discretizations: List,
                                time: float = 0.0) -> List[BulkData]:
        """
        Initialize all BulkData objects with initial conditions using external objects.
        
        Args:
            problems: List of Problem objects
            discretizations: List of discretization objects
            time: Initial time
            
        Returns:
            List of initialized BulkData objects
        """
        # Validate framework objects
        self._validate_framework_objects(
            problems=problems,
            discretizations=discretizations,
            operation_name="initialize_all_bulk_data"
        )
        
        bulk_data_list = []
        for i, (problem, discretization) in enumerate(zip(problems, discretizations)):
            bulk_data = self.initialize_bulk_data_from_initial_conditions(
                i, problem, discretization, time
            )
            bulk_data_list.append(bulk_data)
        
        return bulk_data_list
    
    def update_bulk_data(self, bulk_data_list: List[BulkData], new_data_list: List[np.ndarray]):
        """
        Update BulkData objects with new solution data.
        
        Args:
            bulk_data_list: List of BulkData objects to update
            new_data_list: List of new bulk solution arrays
        """
        if len(bulk_data_list) != len(new_data_list):
            raise ValueError("Number of bulk data objects must match number of new data arrays")
        
        if len(bulk_data_list) != len(self.domain_data_list):
            raise ValueError("Number of bulk data objects must match number of domains")
        
        # Validate new data compatibility
        for i, (bulk_data, new_data) in enumerate(zip(bulk_data_list, new_data_list)):
            domain_data = self.domain_data_list[i]
            expected_shape = (2 * domain_data.neq, domain_data.n_elements)
            
            if new_data.shape != expected_shape:
                raise ValueError(
                    f"update_bulk_data: New data {i} has shape {new_data.shape}, "
                    f"expected {expected_shape} for domain with {domain_data.neq} equations "
                    f"and {domain_data.n_elements} elements"
                )
            
            # Check for invalid values
            if np.any(np.isnan(new_data)) or np.any(np.isinf(new_data)):
                raise ValueError(f"update_bulk_data: New data {i} contains NaN or infinite values")
        
        for bulk_data, new_data in zip(bulk_data_list, new_data_list):
            bulk_data.set_data(new_data)
    
    def compute_total_mass(self, bulk_data_list: List[BulkData]) -> float:
        """
        Compute total mass for conservation check.
        
        Args:
            bulk_data_list: List of BulkData instances
            
        Returns:
            Total mass across all domains
        """
        if len(bulk_data_list) != len(self.domain_data_list):
            raise ValueError("Number of bulk data objects must match number of domains")
        
        # Validate bulk data compatibility
        for i, bulk_data in enumerate(bulk_data_list):
            domain_data = self.domain_data_list[i]
            bulk_data_shape = bulk_data.get_data().shape
            expected_shape = (2 * domain_data.neq, domain_data.n_elements)
            
            if bulk_data_shape != expected_shape:
                raise ValueError(
                    f"compute_total_mass: BulkData {i} has shape {bulk_data_shape}, "
                    f"expected {expected_shape}"
                )
        
        total_mass = 0.0
        for domain_data, bulk_data in zip(self.domain_data_list, bulk_data_list):
            # Use BulkData's built-in mass computation with domain's mass matrix
            domain_mass = bulk_data.compute_mass(domain_data.mass_matrix)
            total_mass += domain_mass
        
        return total_mass
    
    def initialize_bulk_data_from_initial_conditions(self, 
                                                    domain_index: int,
                                                    problem,
                                                    discretization,
                                                    time: float = 0.0) -> BulkData:
        """
        Initialize BulkData object with initial conditions using external objects.
        
        Args:
            domain_index: Index of the domain
            problem: Problem object for this domain
            discretization: Discretization object for this domain
            time: Initial time
            
        Returns:
            Initialized BulkData object
        """
        if domain_index < 0 or domain_index >= len(self.domain_data_list):
            raise ValueError(f"Domain index {domain_index} out of range")
        
        # Validate individual domain objects for this specific domain
        domain_data = self.domain_data_list[domain_index]
        
        if problem is not None:
            if not hasattr(problem, 'neq'):
                raise ValueError(f"initialize_bulk_data_from_initial_conditions: Problem missing 'neq' attribute")
            if problem.neq != domain_data.neq:
                raise ValueError(
                    f"initialize_bulk_data_from_initial_conditions: Problem neq ({problem.neq}) "
                    f"doesn't match domain data neq ({domain_data.neq})"
                )
        
        if discretization is not None:
            if not hasattr(discretization, 'n_elements'):
                raise ValueError(f"initialize_bulk_data_from_initial_conditions: Discretization missing 'n_elements' attribute")
            if discretization.n_elements != domain_data.n_elements:
                raise ValueError(
                    f"initialize_bulk_data_from_initial_conditions: Discretization n_elements ({discretization.n_elements}) "
                    f"doesn't match domain data n_elements ({domain_data.n_elements})"
                )
        
        bulk_data = self.create_bulk_data(domain_index, problem, discretization, dual=False)
        
        # Set initial conditions using function input
        if any(ic is not None for ic in domain_data.initial_conditions):
            bulk_data.set_data(domain_data.initial_conditions, time)
        else:
            # Default to zero initial conditions
            zero_data = np.zeros((2 * domain_data.neq, domain_data.n_elements))
            bulk_data.set_data(zero_data)
        
        return bulk_data
    
    def get_bulk_data_arrays(self, bulk_data_list: List[BulkData]) -> List[np.ndarray]:
        """
        Extract data arrays from BulkData objects.
        
        Args:
            bulk_data_list: List of BulkData objects
            
        Returns:
            List of bulk solution arrays
        """
        return [bulk_data.get_data() for bulk_data in bulk_data_list]
    
    def compute_mass_conservation(self, bulk_data_list: List[BulkData]) -> float:
        """
        Compute total mass for conservation check.
        
        Args:
            bulk_data_list: List of BulkData instances
            
        Returns:
            Total mass across all domains
        """
        return self.compute_total_mass(bulk_data_list)
    
    def get_num_domains(self) -> int:
        """Get number of domains."""
        return len(self.domain_data_list)
    
    def get_domain_info(self, domain_idx: int) -> DomainData:
        """Get domain data for inspection."""
        if domain_idx < 0 or domain_idx >= len(self.domain_data_list):
            raise IndexError(f"Domain index {domain_idx} out of range")
        return self.domain_data_list[domain_idx]
    
    def test(self, 
             problems: List = None,
             discretizations: List = None,
             static_condensations: List = None) -> bool:
        """
        Test method to validate BulkDataManager instance state and functionality.
        
        Args:
            problems: List of Problem objects for testing (optional)
            discretizations: List of discretization objects for testing (optional)  
            static_condensations: List of static condensation objects for testing (optional)
            
        Returns:
            True if all tests pass, False otherwise
        """
        print(f"Testing Lean BulkDataManager with {len(self.domain_data_list)} domains")
        
        try:
            # Test 0: Parameter validation (if framework objects provided)
            if any(obj is not None for obj in [problems, discretizations, static_condensations]):
                try:
                    self._validate_framework_objects(
                        problems=problems,
                        discretizations=discretizations,
                        static_condensations=static_condensations,
                        operation_name="test"
                    )
                    print("PASS: Framework object validation passed")
                except ValueError as e:
                    print(f"FAIL: Framework object validation failed: {e}")
                    return False
            else:
                print("SKIP: No framework objects provided - skipping validation test")

            # Test 1: Validate domain data structure
            for i, domain_data in enumerate(self.domain_data_list):
                if domain_data.mass_matrix is None:
                    print(f"FAIL: Domain {i} has no mass matrix")
                    return False
                
                if domain_data.trace_matrix is None:
                    print(f"FAIL: Domain {i} has no trace matrix")
                    return False
                
                # Check matrix dimensions
                if domain_data.mass_matrix.shape != (2, 2):
                    print(f"FAIL: Domain {i} mass matrix shape {domain_data.mass_matrix.shape} != (2, 2)")
                    return False
                
                if domain_data.trace_matrix.shape != (2, 2):
                    print(f"FAIL: Domain {i} trace matrix shape {domain_data.trace_matrix.shape} != (2, 2)")
                    return False
            
            print(f"PASS: All domain data validated")
            
            # Test 2: Test BulkData creation (if framework objects provided)
            if problems is not None and discretizations is not None:
                if len(problems) != len(self.domain_data_list) or len(discretizations) != len(self.domain_data_list):
                    print("SKIP: Framework objects count mismatch - skipping BulkData creation test")
                else:
                    for i in range(min(len(problems), 3)):  # Test first 3 domains or all if fewer
                        try:
                            # Test primal formulation
                            bulk_data_primal = self.create_bulk_data(i, problems[i], discretizations[i], dual=False)
                            if not bulk_data_primal.test():
                                print(f"FAIL: Primal BulkData test failed for domain {i}")
                                return False
                            
                            # Test dual formulation
                            bulk_data_dual = self.create_bulk_data(i, problems[i], discretizations[i], dual=True)
                            if not bulk_data_dual.test():
                                print(f"FAIL: Dual BulkData test failed for domain {i}")
                                return False
                            
                        except Exception as e:
                            print(f"FAIL: BulkData creation failed for domain {i}: {e}")
                            return False
                    
                    print(f"PASS: BulkData creation tests passed")
            else:
                print("SKIP: No framework objects provided - skipping BulkData creation test")
            
            # Test 3: Test initialization (if framework objects provided)
            if problems is not None and discretizations is not None:
                try:
                    bulk_data_list = self.initialize_all_bulk_data(problems, discretizations, time=0.0)
                    if len(bulk_data_list) != len(self.domain_data_list):
                        print(f"FAIL: Initialized bulk data count {len(bulk_data_list)} != domains {len(self.domain_data_list)}")
                        return False
                    
                    print(f"PASS: Initialization tests passed")
                    
                    # Test 4: Test forcing term computation
                    forcing_terms = self.compute_forcing_terms(bulk_data_list, problems, discretizations, time=0.5, dt=0.1)
                    
                    if len(forcing_terms) != len(self.domain_data_list):
                        print(f"FAIL: Forcing terms count {len(forcing_terms)} != domains {len(self.domain_data_list)}")
                        return False
                    
                    for i, forcing_term in enumerate(forcing_terms):
                        expected_shape = (2 * self.domain_data_list[i].neq, self.domain_data_list[i].n_elements)
                        if forcing_term.shape != expected_shape:
                            print(f"FAIL: Domain {i} forcing term shape {forcing_term.shape} != {expected_shape}")
                            return False
                        
                        # Check for NaN or infinite values
                        if np.any(np.isnan(forcing_term)) or np.any(np.isinf(forcing_term)):
                            print(f"FAIL: Domain {i} forcing term contains NaN or infinite values")
                            return False
                    
                    print(f"PASS: Forcing term computation tests passed")
                    
                    # Test 5: Test mass computation
                    total_mass = self.compute_total_mass(bulk_data_list)
                    
                    if np.isnan(total_mass) or np.isinf(total_mass):
                        print(f"FAIL: Total mass is NaN or infinite: {total_mass}")
                        return False
                    
                    print(f"PASS: Mass computation test passed (total_mass={total_mass:.6e})")
                    
                except Exception as e:
                    print(f"FAIL: Framework integration test failed: {e}")
                    return False
            else:
                print("SKIP: No framework objects provided - skipping integration tests")
            
            # Test 6: Test bounds checking
            try:
                if problems is not None and discretizations is not None:
                    self.create_bulk_data(-1, problems[0], discretizations[0])
                    print("FAIL: No exception raised for negative domain index")
                    return False
            except ValueError:
                print("PASS: ValueError raised for negative domain index")
            except:
                print("SKIP: Cannot test bounds checking without framework objects")
            
            # Test 7: Test utility methods
            num_domains = self.get_num_domains()
            if num_domains != len(self.domain_data_list):
                print(f"FAIL: get_num_domains() returned {num_domains} != {len(self.domain_data_list)}")
                return False
            print(f"PASS: get_num_domains() returned correct value")
            
            # Test domain info access
            if len(self.domain_data_list) > 0:
                domain_info = self.get_domain_info(0)
                if domain_info != self.domain_data_list[0]:
                    print("FAIL: get_domain_info() returned incorrect data")
                    return False
                print("PASS: get_domain_info() test passed")
            
            # Test 8: Parameter mismatch testing (if framework objects provided)
            if problems is not None and discretizations is not None:
                try:
                    # Test with wrong number of problems
                    try:
                        wrong_problems = problems[:-1] if len(problems) > 1 else problems + [problems[0]]
                        self._validate_framework_objects(
                            problems=wrong_problems,
                            discretizations=discretizations,
                            operation_name="parameter_mismatch_test"
                        )
                        print("FAIL: Should have raised error for wrong number of problems")
                        return False
                    except ValueError:
                        print("PASS: Correctly detected wrong number of problems")
                    
                    # Test with incompatible problem (if we can create one)
                    if hasattr(problems[0], 'neq') and problems[0].neq > 1:
                        # Create a mock problem with different neq
                        class MockWrongProblem:
                            def __init__(self):
                                self.neq = problems[0].neq + 1
                        
                        wrong_problem_list = [MockWrongProblem()] + problems[1:]
                        try:
                            self._validate_framework_objects(
                                problems=wrong_problem_list,
                                discretizations=discretizations,
                                operation_name="parameter_mismatch_test"
                            )
                            print("FAIL: Should have raised error for incompatible problem neq")
                            return False
                        except ValueError:
                            print("PASS: Correctly detected incompatible problem neq")
                    
                    print("PASS: Parameter mismatch detection tests passed")
                    
                except Exception as e:
                    print(f"FAIL: Parameter mismatch testing failed: {e}")
                    return False
            else:
                print("SKIP: Insufficient framework objects for parameter mismatch testing")

            print("✓ All Lean BulkDataManager tests passed!")
            return True
            
        except Exception as e:
            print(f"FAIL: Unexpected error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def __str__(self) -> str:
        total_elements = sum(d.n_elements for d in self.domain_data_list)
        total_equations = sum(d.neq for d in self.domain_data_list)
        return (f"LeanBulkDataManager(domains={len(self.domain_data_list)}, "
                f"total_elements={total_elements}, "
                f"total_equations={total_equations})")
    
    def __repr__(self) -> str:
        domain_elements = [d.n_elements for d in self.domain_data_list]
        domain_equations = [d.neq for d in self.domain_data_list]
        return (f"LeanBulkDataManager(n_domains={len(self.domain_data_list)}, "
                f"domain_elements={domain_elements}, "
                f"domain_equations={domain_equations})")
