import numpy as np
from typing import Callable, List, Optional, Union, Dict
from .discretization import Discretization, GlobalDiscretization

class Problem:
    """
    Problem definition base class for 1D Keller-Segel type problems.
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
        """
        Set initial condition for specified equation.
        
        Args:
            equation_idx: Index of the equation
            u0_func: Function that takes either:
                    - f(s): spatial-only function 
                    - f(s, t): space-time function (will be evaluated at t=0)
        """
        import inspect
        
        # Get function signature to determine number of parameters
        try:
            sig = inspect.signature(u0_func)
            n_params = len(sig.parameters)
            
            if n_params == 1:
                # Function takes only spatial argument - create wrapper that can handle both call patterns
                def robust_wrapper(*args, **kwargs):
                    # Always call the original function with just the first argument (s)
                    return u0_func(args[0])
                self.u0[equation_idx] = robust_wrapper
            elif n_params >= 2:
                # Function takes spatial and temporal arguments - create wrapper that evaluates at t=0
                def robust_wrapper(*args, **kwargs):
                    if len(args) == 1:
                        # Called with just s
                        return u0_func(args[0], 0.0)
                    elif len(args) >= 2:
                        # Called with s and t, but we force t=0 for initial condition
                        return u0_func(args[0], 0.0)
                    else:
                        # Use keyword arguments
                        s = kwargs.get('s', args[0] if args else 0)
                        return u0_func(s, 0.0)
                self.u0[equation_idx] = robust_wrapper
            else:
                # Function takes no arguments - create constant function
                constant_value = u0_func()
                def robust_wrapper(*args, **kwargs):
                    s = args[0] if args else kwargs.get('s', np.array([0.0]))
                    return constant_value * np.ones_like(s)
                self.u0[equation_idx] = robust_wrapper
                
        except Exception:
            # Fallback: create a universal wrapper that tries different call patterns
            def universal_wrapper(*args, **kwargs):
                test_s = args[0] if args else kwargs.get('s', np.array([0.0]))
                
                # Try single argument first
                try:
                    return u0_func(test_s)
                except TypeError:
                    pass
                
                # Try dual argument with t=0
                try:
                    return u0_func(test_s, 0.0)
                except TypeError:
                    pass
                
                # Try no arguments (constant)
                try:
                    constant_value = u0_func()
                    return constant_value * np.ones_like(test_s)
                except TypeError:
                    pass
                
                # If all else fails, return zeros
                return np.zeros_like(test_s)
            
            self.u0[equation_idx] = universal_wrapper

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
    
    def validate_problem(self, verbose: bool = False) -> bool:
        """
        Validate the problem configuration for consistency and common issues.
        
        Args:
            verbose: Whether to print validation details
            
        Returns:
            True if problem is valid, False otherwise
        """
        if verbose:
            print(f"Validating problem: {self.name}")
        
        issues = []
        warnings = []
        
        # Check basic parameters
        if self.neq <= 0:
            issues.append("Number of equations must be positive")
        
        if self.domain_length <= 0:
            issues.append("Domain length must be positive")
        
        if self.parameters is None:
            issues.append("Parameters array is None")
        elif len(self.parameters) == 0:
            issues.append("Parameters array is empty")
        elif not all(isinstance(p, (int, float)) for p in self.parameters):
            issues.append("All parameters must be numeric")
        
        # Check parameter consistency
        if self.n_parameters != len(self.parameters):
            issues.append(f"Parameter count mismatch: n_parameters={self.n_parameters}, actual={len(self.parameters)}")
        
        # Check domain consistency
        expected_end = self.domain_start + self.domain_length
        if abs(self.domain_end - expected_end) > 1e-12:
            issues.append(f"Domain end inconsistent: expected={expected_end}, actual={self.domain_end}")
        
        # Check function lists
        expected_function_count = self.neq
        if len(self.force) != expected_function_count:
            issues.append(f"Force function count mismatch: expected={expected_function_count}, actual={len(self.force)}")
        
        if len(self.u0) != expected_function_count:
            issues.append(f"Initial condition count mismatch: expected={expected_function_count}, actual={len(self.u0)}")
        
        if len(self.solution) != expected_function_count:
            issues.append(f"Solution function count mismatch: expected={expected_function_count}, actual={len(self.solution)}")
        
        if len(self.flux_u0) != expected_function_count:
            issues.append(f"Left boundary flux count mismatch: expected={expected_function_count}, actual={len(self.flux_u0)}")
        
        if len(self.flux_u1) != expected_function_count:
            issues.append(f"Right boundary flux count mismatch: expected={expected_function_count}, actual={len(self.flux_u1)}")
        
        # Check extrema
        if not isinstance(self.extrema, (list, tuple)) or len(self.extrema) != 2:
            issues.append("Extrema must be a list/tuple of 2 points")
        else:
            for i, point in enumerate(self.extrema):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    issues.append(f"Extrema point {i} must be a 2D coordinate tuple")
                elif not all(isinstance(coord, (int, float)) for coord in point):
                    issues.append(f"Extrema point {i} must have numeric coordinates")
        
        # Check problem type
        valid_types = ["keller_segel", "organ_on_chip", "generic"]
        if self.type not in valid_types:
            warnings.append(f"Unknown problem type '{self.type}', expected one of {valid_types}")
        
        # Check chemotaxis functions for Keller-Segel problems
        if self.type == "keller_segel":
            if self.chi is None:
                warnings.append("Keller-Segel problem missing chemotaxis function chi")
            if self.dchi is None:
                warnings.append("Keller-Segel problem missing chemotaxis derivative dchi")
        
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
        
        return len(issues) == 0
    
    def test_functions(self, verbose: bool = False) -> bool:
        """
        Test that all functions can be called without errors.
        
        Args:
            verbose: Whether to print test details
            
        Returns:
            True if all function tests pass, False otherwise
        """
        if verbose:
            print(f"Testing functions for problem: {self.name}")
        
        test_passed = True
        
        # Create test inputs
        test_s = np.linspace(self.domain_start, self.domain_end, 5)
        test_t = 0.5
        test_scalar = 1.0
        
        # Test force functions
        for i, force_func in enumerate(self.force):
            try:
                result = force_func(test_s, test_t)
                if not isinstance(result, np.ndarray) or result.shape != test_s.shape:
                    if verbose:
                        print(f"  ✗ Force function {i} returned invalid shape")
                    test_passed = False
                elif verbose:
                    print(f"  ✓ Force function {i}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Force function {i} failed: {e}")
                test_passed = False
        
        # Test initial condition functions
        for i, u0_func in enumerate(self.u0):
            try:
                result = u0_func(test_s)
                if not isinstance(result, np.ndarray) or result.shape != test_s.shape:
                    if verbose:
                        print(f"  ✗ Initial condition {i} returned invalid shape")
                    test_passed = False
                elif verbose:
                    print(f"  ✓ Initial condition {i}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Initial condition {i} failed: {e}")
                test_passed = False
        
        # Test solution functions
        for i, sol_func in enumerate(self.solution):
            try:
                result = sol_func(test_s, test_t)
                if not isinstance(result, np.ndarray) or result.shape != test_s.shape:
                    if verbose:
                        print(f"  ✗ Solution function {i} returned invalid shape")
                    test_passed = False
                elif verbose:
                    print(f"  ✓ Solution function {i}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Solution function {i} failed: {e}")
                test_passed = False
        
        # Test boundary flux functions
        for i, flux_func in enumerate(self.flux_u0):
            try:
                result = flux_func(test_t)
                if not isinstance(result, (int, float, np.number)):
                    if verbose:
                        print(f"  ✗ Left boundary flux {i} returned non-scalar")
                    test_passed = False
                elif verbose:
                    print(f"  ✓ Left boundary flux {i}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Left boundary flux {i} failed: {e}")
                test_passed = False
        
        for i, flux_func in enumerate(self.flux_u1):
            try:
                result = flux_func(test_t)
                if not isinstance(result, (int, float, np.number)):
                    if verbose:
                        print(f"  ✗ Right boundary flux {i} returned non-scalar")
                    test_passed = False
                elif verbose:
                    print(f"  ✓ Right boundary flux {i}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Right boundary flux {i} failed: {e}")
                test_passed = False
        
        # Test chemotaxis functions if they exist
        if self.chi is not None:
            try:
                result = self.chi(test_s)
                if not isinstance(result, np.ndarray) or result.shape != test_s.shape:
                    if verbose:
                        print("  ✗ Chemotaxis function chi returned invalid shape")
                    test_passed = False
                elif verbose:
                    print("  ✓ Chemotaxis function chi")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Chemotaxis function chi failed: {e}")
                test_passed = False
        
        if self.dchi is not None:
            try:
                result = self.dchi(test_s)
                if not isinstance(result, np.ndarray) or result.shape != test_s.shape:
                    if verbose:
                        print("  ✗ Chemotaxis derivative dchi returned invalid shape")
                    test_passed = False
                elif verbose:
                    print("  ✓ Chemotaxis derivative dchi")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Chemotaxis derivative dchi failed: {e}")
                test_passed = False
        
        return test_passed
    
    @classmethod
    def create_test_problems(cls) -> Dict[str, 'Problem']:
        """
        Create a collection of test problems for validation and testing.
        
        Returns:
            Dictionary of test problem instances
        """
        problems = {}
        
        # 1. Basic Keller-Segel problem
        ks_basic = cls(neq=2, domain_start=0.0, domain_length=1.0,
                      parameters=np.array([1.0, 2.0, 0.1, 0.0]),
                      problem_type="keller_segel", name="KS_basic")
        
        # Set basic functions
        ks_basic.set_chemotaxis(lambda x: np.ones_like(x), lambda x: np.zeros_like(x))
        ks_basic.set_initial_condition(0, lambda s: np.exp(-s**2))
        ks_basic.set_initial_condition(1, lambda s: np.sin(np.pi * s))
        ks_basic.set_force(0, lambda s, t: 0.1 * np.ones_like(s))
        ks_basic.set_force(1, lambda s, t: np.zeros_like(s))
        
        problems["ks_basic"] = ks_basic
        
        # 2. Organ-on-Chip problem
        ooc_basic = cls(neq=4, domain_start=0.0, domain_length=2.0,
                       parameters=np.array([1e-9, 0.001, 1e-4, 1e-5, 1e-3]),
                       problem_type="organ_on_chip", name="OoC_basic")
        
        # Set initial conditions for all 4 equations
        ooc_basic.set_initial_condition(0, lambda s: 0.5 * np.ones_like(s))  # nutrients
        ooc_basic.set_initial_condition(1, lambda s: np.zeros_like(s))       # waste
        ooc_basic.set_initial_condition(2, lambda s: 0.1 * np.ones_like(s))  # cells
        ooc_basic.set_initial_condition(3, lambda s: 0.05 * np.ones_like(s)) # growth factors
        
        # Set source terms
        for i in range(4):
            ooc_basic.set_force(i, lambda s, t: np.zeros_like(s))
        
        problems["ooc_basic"] = ooc_basic
        
        # 3. Custom extrema problem
        custom_geom = cls(neq=2, domain_start=1.0, domain_length=3.0,
                         parameters=np.array([0.5, 1.5, 0.2, 0.1]),
                         problem_type="generic", name="custom_geometry")
        
        custom_geom.set_extrema((2.0, 1.0), (5.0, 4.0))  # Non-standard geometry
        custom_geom.set_initial_condition(0, lambda s: s * (4.0 - s))  # Parabolic
        custom_geom.set_initial_condition(1, lambda s: np.exp(-0.5 * s))  # Exponential
        
        problems["custom_geometry"] = custom_geom
        
        # 4. Analytical solution problem
        analytical = cls(neq=2, domain_start=0.0, domain_length=np.pi,
                        parameters=np.array([1.0, 1.0, 0.0, 0.0]),
                        problem_type="keller_segel", name="analytical_test")
        
        # Set analytical solutions
        analytical.set_solution(0, lambda s, t: np.sin(s) * np.exp(-t))
        analytical.set_solution(1, lambda s, t: np.cos(s) * np.exp(-0.5*t))
        analytical.set_initial_condition(0, lambda s: np.sin(s))
        analytical.set_initial_condition(1, lambda s: np.cos(s))
        
        problems["analytical"] = analytical
        
        # 5. Invalid problem (for testing validation)
        invalid = cls(neq=0, domain_start=1.0, domain_length=-1.0,  # Invalid parameters
                     parameters=np.array([]), problem_type="invalid", name="invalid_test")
        
        problems["invalid"] = invalid
        
        return problems
    
    def run_self_test(self, verbose: bool = True) -> bool:
        """
        Run comprehensive self-test on the problem.
        
        Args:
            verbose: Whether to print detailed test results
            
        Returns:
            True if all tests pass, False otherwise
        """
        if verbose:
            print(f"Running self-test for problem: {self.name}")
            print("=" * 50)
        
        all_passed = True
        
        # Test 1: Problem validation
        if verbose:
            print("Test 1: Problem validation")
        
        try:
            is_valid = self.validate_problem(verbose=False)
            if verbose:
                print(f"  {'✓' if is_valid else '✗'} Problem validation: {'PASS' if is_valid else 'FAIL'}")
            if not is_valid:
                all_passed = False
        except Exception as e:
            if verbose:
                print(f"  ✗ Validation test failed: {e}")
            all_passed = False
        
        # Test 2: Function testing
        if verbose:
            print("Test 2: Function testing")
        
        try:
            functions_ok = self.test_functions(verbose=False)
            if verbose:
                print(f"  {'✓' if functions_ok else '✗'} Function tests: {'PASS' if functions_ok else 'FAIL'}")
            if not functions_ok:
                all_passed = False
        except Exception as e:
            if verbose:
                print(f"  ✗ Function testing failed: {e}")
            all_passed = False
        
        # Test 3: Parameter operations
        if verbose:
            print("Test 3: Parameter operations")
        
        try:
            # Test parameter access
            if len(self.parameters) > 0:
                original_param = self.get_parameter(0)
                self.set_parameter(0, original_param + 1.0)
                new_param = self.get_parameter(0)
                self.set_parameter(0, original_param)  # Restore
                
                if abs(new_param - (original_param + 1.0)) < 1e-12:
                    if verbose:
                        print("  ✓ Parameter get/set operations")
                else:
                    if verbose:
                        print("  ✗ Parameter operations failed")
                    all_passed = False
            else:
                if verbose:
                    print("  ⚠ No parameters to test")
        except Exception as e:
            if verbose:
                print(f"  ✗ Parameter operations failed: {e}")
            all_passed = False
        
        # Test 4: Extrema operations
        if verbose:
            print("Test 4: Extrema operations")
        
        try:
            original_extrema = self.get_extrema()
            test_extrema = [(10.0, 20.0), (30.0, 40.0)]
            self.set_extrema(test_extrema[0], test_extrema[1])
            new_extrema = self.get_extrema()
            self.set_extrema(original_extrema[0], original_extrema[1])  # Restore
            
            if new_extrema == test_extrema:
                if verbose:
                    print("  ✓ Extrema get/set operations")
            else:
                if verbose:
                    print("  ✗ Extrema operations failed")
                all_passed = False
        except Exception as e:
            if verbose:
                print(f"  ✗ Extrema operations failed: {e}")
            all_passed = False
        
        # Test 5: Dynamic function setting
        if verbose:
            print("Test 5: Dynamic function setting")
        
        try:
            test_func = lambda x: x**2
            self.set_function("test_function", test_func)
            
            if hasattr(self, "test_function") and callable(self.test_function):
                test_result = self.test_function(2.0)
                if abs(test_result - 4.0) < 1e-12:
                    if verbose:
                        print("  ✓ Dynamic function setting")
                else:
                    if verbose:
                        print("  ✗ Dynamic function execution failed")
                    all_passed = False
            else:
                if verbose:
                    print("  ✗ Dynamic function setting failed")
                all_passed = False
        except Exception as e:
            if verbose:
                print(f"  ✗ Dynamic function setting failed: {e}")
            all_passed = False
        
        if verbose:
            print("=" * 50)
            print(f"Self-test result: {'PASS' if all_passed else 'FAIL'}")
        
        return all_passed
