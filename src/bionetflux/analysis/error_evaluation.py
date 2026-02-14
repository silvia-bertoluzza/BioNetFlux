"""
L2 Error Evaluation Module for BioNetFlux

This module provides functionality to compute L2 errors between numerical and analytical solutions,
perform convergence analysis, and evaluate solution quality metrics.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
import warnings
import matplotlib.pyplot as plt


def retrieve_analytical_solution(problems: List) -> Dict:
    """
    Extract analytical solutions from a list of problem objects.
    
    Args:
        problems: List of problem objects that may contain analytical solutions
    
    Returns:
        Dictionary with analytical functions organized by domain index
        Format: {'domain_0': [func1, func2, ...], 'domain_1': [...], ...}
    """
    analytical_solutions = {}
    
    for i, problem in enumerate(problems):
        domain_key = f'domain_{i}'
        
        if hasattr(problem, 'solution') and problem.solution is not None:
            # Use the actual analytical solution from the problem
            if callable(problem.solution):
                # Single function - replicate for all equations in this problem
                warnings.warn(f"Problem {i}: Single callable solution found and replicated for all {problem.neq} equations. "
                            f"Consider providing a list of callables, one per equation, for more accurate analytical solutions.")
                analytical_solutions[domain_key] = [problem.solution] * problem.neq           
            elif isinstance(problem.solution, (list, tuple)):
                # Multiple functions for multiple equations 
                # This is the relevant case for us. The attribute solution of a problem object 
                # is a list of functions, one for each equation.
                if len(problem.solution) == problem.neq:
                    analytical_solutions[domain_key] = list(problem.solution)
                else:
                    warnings.warn(f"Problem {i}: Number of solution functions ({len(problem.solution)}) "
                                f"doesn't match number of equations ({problem.neq}). Using zero analytical solution.")
                    analytical_solutions[domain_key] = None
            else:
                warnings.warn(f"Problem {i}: solution attribute exists but is not callable or list. "
                            f"Type: {type(problem.solution)}. Using zero analytical solution.")
                analytical_solutions[domain_key] = None
        else:
            warnings.warn(f"Problem {i}: No solution attribute found. Using zero analytical solution.")
            analytical_solutions[domain_key] = None
    
    return analytical_solutions


class ErrorEvaluator:
    """
    Evaluates L2 errors between numerical and analytical solutions.
    Supports both pointwise and integrated error measures with HDG-specific trace error formulation.
    """
    
    def __init__(self, problems: List, discretizations: List):
        """
        Initialize the L2 error evaluator.
        
        Args:
            problems: List of problem objects with analytical solutions
            discretizations: List of spatial discretization objects
        """
        self.problems = problems
        self.discretizations = discretizations
        self.n_domains = len(problems)
        
        # Validate inputs
        if len(discretizations) != self.n_domains:
            raise ValueError("Number of discretizations must match number of problems")
        
        # Automatically extract analytical solutions from problems
        self.analytical_solutions = retrieve_analytical_solution(problems)
        
        # ====================================================================
        # DEBUG SECTION: Print analytical solutions information
        # ====================================================================
        print("\n" + "="*60)
        print("DEBUG: ErrorEvaluator Initialization")
        print("="*60)
        print(f"Number of domains: {self.n_domains}")
        print(f"Number of problems: {len(self.problems)}")
        print(f"Number of discretizations: {len(self.discretizations)}")
        print("\nAnalytical solutions extracted:")
        
        


        if not self.analytical_solutions:
            print("  ❌ No analytical solutions found in any domain")
        else:
            for domain_key, functions in self.analytical_solutions.items():
                print(f"  {domain_key}:")
                if functions is None:
                    print(f"    ❌ None (no analytical solution available)")
                elif isinstance(functions, list):
                    print(f"    ✅ List with {len(functions)} functions:")
                    for i, func in enumerate(functions):
                        if callable(func):
                            print(f"      Equation {i}: {func.__name__} (callable)")
                            try: 
                                print(f"        Test call: {func(0.5, 1.0)}")  # Test call with sample inputs
                            except Exception as e:
                                print(f"        ❌ Test call failed: {e}")
                        else:
                            print(f"      Equation {i}: {type(func).__name__} (not callable)")
        
        # Test evaluation of all callable functions from problem.solution at (0.5, 1.5)
        print("\nTest evaluation at point (0.5, 1.5):")
        for i, problem in enumerate(self.problems):
            if hasattr(problem, 'solution') and problem.solution is not None:
                print(f"  Problem {i}:")
                if callable(problem.solution):
                    try:
                        result = problem.solution(0.5, 1.5)
                        print(f"    Single function: {result}")
                    except Exception as e:
                        print(f"    ❌ Single function evaluation failed: {e}")
                elif isinstance(problem.solution, (list, tuple)):
                    for eq_idx, func in enumerate(problem.solution):
                        if callable(func):
                            try:
                                result = func(0.5, 1.5)
                                print(f"    Equation {eq_idx}: {result}")
                            except Exception as e:
                                print(f"    ❌ Equation {eq_idx} evaluation failed: {e}")
                        else:
                            print(f"    ❌ Equation {eq_idx}: not callable")
                else:
                    print(f"    ❌ Solution is not callable or list")
            else:
                print(f"  Problem {i}: No solution attribute")
        
        # Check consistency with problem equations
        print("\nConsistency check with problem.neq:")
        for i, problem in enumerate(self.problems):
            domain_key = f'domain_{i}'
            functions = self.analytical_solutions.get(domain_key, None)
            print(f"  Domain {i}: problem.neq={problem.neq}, functions={len(functions) if functions else 0}")
            
            

            if functions and len(functions) != problem.neq:
                print(f"    ⚠️  WARNING: Mismatch between problem.neq and number of analytical functions")
        
        
        print("="*60)
        print("END DEBUG: ErrorEvaluator Initialization")
        print("="*60 + "\n")
        # ====================================================================
        # END DEBUG SECTION
        # ====================================================================
        
    
    def compute_trace_error(self, 
                        numerical_solutions: List[np.ndarray], 
                        time: float,
                        analytical_functions: Optional[List[List[Callable]]] = None) -> Dict:
        """
        Compute Euclidean and max errors between numerical and analytical solutions.
                
        Args:
            numerical_solutions: List of numerical trace solutions for each domain
            time: Current time for analytical solution evaluation
            analytical_functions: Optional list of analytical functions per domain/equation
                                 If None, uses automatically extracted solutions from problems
        
        Returns:
            Dictionary with standard error metrics per domain and equation
        """
        # Determine maximum number of equations to set default alpha
        max_equations = max(problem.neq for problem in self.problems)
                 
        # Use provided analytical functions or fall back to extracted ones
        if analytical_functions is None:
            analytical_functions = [self.analytical_solutions.get(f'domain_{i}', None) 
                                   for i in range(self.n_domains)]
        
        results = {
            'domain_errors': {},  # organized by (domain_idx, eq_idx)
            'global_errors': [],  # global error for each equation
            'time': time,
            'error_type': 'trace'
            }
        
        # Track errors per equation across all domains
        max_equations = max(problem.neq for problem in self.problems)
        global_error_squared = [0.0] * max_equations
        global_solution_norm_squared = [0.0] * max_equations
        max_pointwise_error = [0.0] * max_equations
        solution_max_norm = [0.0] * max_equations
        
        for domain_idx in range(self.n_domains):
            problem = self.problems[domain_idx]
            discretization = self.discretizations[domain_idx]
            numerical_sol = numerical_solutions[domain_idx]
            
            domain_result = self._compute_domain_trace_error(
                problem, discretization, numerical_sol, time, 
                analytical_functions[domain_idx] if analytical_functions and domain_idx < len(analytical_functions) else None
            )
            
            # Store individual equation errors with domain/equation indexing
            for eq_error, eq_norm in zip(domain_result['equation_errors'], domain_result['solution_norms']):
                eq_idx = eq_error['equation_idx']
                results['domain_errors'][(domain_idx, eq_idx)] = eq_error
                
                # Accumulate global error per equation
                if eq_idx < len(global_error_squared):
                    global_error_squared[eq_idx] += eq_error['l2_error']**2
                    global_solution_norm_squared[eq_idx] += eq_norm['solution_norm']**2
                    max_pointwise_error[eq_idx] = max(max_pointwise_error[eq_idx], eq_error['max_pointwise_error'])
                    solution_max_norm[eq_idx] = max(solution_max_norm[eq_idx], eq_norm['solution_max_norm'])
            
        # Compute global errors per equation
        for eq_idx in range(max_equations):
            global_error = np.sqrt(global_error_squared[eq_idx])
            if global_solution_norm_squared[eq_idx] > 1e-14:
                relative_error = np.sqrt(global_error_squared[eq_idx] / global_solution_norm_squared[eq_idx])
            else:
                relative_error = np.inf
                
            results['global_errors'].append({
                'equation_idx': eq_idx,
                'euclidean_error': global_error,
                'euclidean_relative_error': relative_error,
                'euclidean_solution_norm': np.sqrt(global_solution_norm_squared[eq_idx]),
                'max_error': max_pointwise_error[eq_idx],
                'max_relative_error': max_pointwise_error[eq_idx] / solution_max_norm[eq_idx] if solution_max_norm[eq_idx] > 1e-14 else np.inf,
                'max_solution_norm': solution_max_norm
            })
                   
        return results

    def _compute_domain_trace_error(self, 
                                       problem, 
                                       discretization, 
                                       numerical_sol: np.ndarray, 
                                       time: float,
                                       analytical_functions: Optional[List[Callable]] = None) -> Dict:
        """
        Compute euclidean and maximum trace error for a single domain.
        
        Euclidean trace error formulation:
        - For each node: pointwise_error = numerical_value - analytical_value
        - Local mesh size: h = domain_length / n_elements
        - Euclidean error for equation eq: ||pointwise_errors||_2 (Euclidean norm)
        - Maximum pointwise error: max(|pointwise_errors|)
        
        Args:
            problem: Problem object for the domain
            discretization: Spatial discretization for the domain
            numerical_sol: Numerical solution array
            time: Current time
            analytical_functions: List of analytical functions per equation
        
        Returns:
            Dictionary with domain-specific Euclidean trace error metrics
        """
        nodes = discretization.nodes
        n_nodes = len(nodes)
        n_elements = discretization.n_elements
        neq = problem.neq
        
        # Get analytical functions
        if analytical_functions is None:
            analytical_functions = self._get_analytical_functions(problem)
        
        max_pointwise_error = 0.0
        equation_errors = []
        solution_norms = []
        
        for eq_idx in range(neq):           
            # Extract numerical solution for this equation
            eq_start = eq_idx * n_nodes
            eq_end = eq_start + n_nodes
            numerical_values = numerical_sol[eq_start:eq_end]
            
            # Compute analytical solution at nodes
            if analytical_functions and eq_idx < len(analytical_functions):
                analytical_values = np.array([analytical_functions[eq_idx](x, time) for x in nodes])
            else:
                # Fallback: assume zero analytical solution with warning
                analytical_values = np.zeros_like(numerical_values)
                warnings.warn(f"No analytical solution available for equation {eq_idx}, using zero")
            
            # Compute pointwise errors
            pointwise_errors = numerical_values - analytical_values
            
            # Compute euclidean and maximum norms of pointwise errors
            eq_euclidean_norm = np.sqrt(np.sum(pointwise_errors**2))
            eq_max_pointwise_error = np.max(np.abs(pointwise_errors))
            
            # Update maximum pointwise error across all equations
            max_pointwise_error = max(max_pointwise_error, eq_max_pointwise_error)
            
            eq_solution_norm = np.sqrt(np.sum(analytical_values**2))
            eq_solution_max_norm = np.max(np.abs(analytical_values))
            
            # Store equation-specific results
            eq_result = {
                'equation_idx': eq_idx,
                'l2_error': eq_euclidean_norm,
                'max_pointwise_error': eq_max_pointwise_error,
                'relative_l2_error': eq_euclidean_norm / eq_solution_norm if eq_solution_norm > 1e-14 else np.inf,
                'relative_max_error': eq_max_pointwise_error / eq_solution_max_norm if eq_solution_max_norm > 1e-14 else np.inf,
                'n_nodes': n_nodes,
            }
            
            eq_solution_norms = {
                'equation_idx': eq_idx,
                'solution_norm': eq_solution_norm,  
                'solution_max_norm': eq_solution_max_norm        
            }
            
            solution_norms.append(eq_solution_norms)
            equation_errors.append(eq_result)
        
        return {
            'domain_idx': getattr(problem, 'domain_idx', 0),
            'equation_errors': equation_errors,
            'solution_norms': solution_norms,
            'error_type': 'trace'
        }
 
    def compute_bulk_error(self, 
                          bulk_solutions: List, 
                          time: float,
                          analytical_functions: Optional[List[List[Callable]]] = None) -> Dict:
        """
        Compute L2 errors between numerical bulk solutions and analytical solutions.
        Handles discontinuous Galerkin bulk solutions.
        
        Args:
            bulk_solutions: List of BulkData objects for each domain
            time: Current time for analytical solution evaluation
            analytical_functions: Optional list of analytical functions per domain/equation
                                 If None, uses automatically extracted solutions from problems
        
        Returns:
            Dictionary with bulk error metrics per domain and equation
        """
        # Use provided analytical functions or fall back to extracted ones
        if analytical_functions is None:
            analytical_functions = [self.analytical_solutions.get(f'domain_{i}', None) 
                                   for i in range(self.n_domains)]
        
        
        results = {
            'domain_errors': {},  # organized by (domain_idx, eq_idx)
            'global_errors': [],  # global error for each equation
            'time': time,
            'error_type': 'bulk'
            }
        
        # results = {
        #     'domain_errors': [],
        #     'equation_errors': {},  # organized by (domain_idx, eq_idx)
        #     'global_error_per_equation': [],  # global error for each equation
        #     'relative_global_error_per_equation': [],  # relative global error for each equation
        #     'time': time,
        #     'error_type': 'bulk'
        # }
        
        # Track errors per equation across all domains
        max_equations = max(problem.neq for problem in self.problems)
        global_error_squared = [0.0] * max_equations
        global_solution_norm_squared = [0.0] * max_equations
        # max_pointwise_error = 0.0
        
        for domain_idx in range(self.n_domains):
            problem = self.problems[domain_idx]
            discretization = self.discretizations[domain_idx]
            bulk_data = bulk_solutions[domain_idx]
            
            # Extract the numpy array from BulkData object
            # bulk_solutions is a list of BulkData objects, each containing a .data attribute which is the actual numpy array of shape (2*neq, n_elements)
            # it can also be a list of raw numpy array, so we check for that as well
            bulk_sol = bulk_data.data if hasattr(bulk_data, 'data') else bulk_data
            
            domain_result = self._compute_domain_bulk_error(
                problem, discretization, bulk_sol, time, 
                analytical_functions[domain_idx] if analytical_functions else None
            )
            
            
            # Store individual equation errors with domain/equation indexing
            for eq_error, eq_norm in zip(domain_result['equation_errors'], domain_result['solution_norms']):
                eq_idx = eq_error['equation_idx']
                results['domain_errors'][(domain_idx, eq_idx)] = eq_error
                
                # Accumulate global error per equation
                if eq_idx < len(global_error_squared):
                    global_error_squared[eq_idx] += eq_error['l2_error']**2
                    global_solution_norm_squared[eq_idx] += eq_norm['solution_norm']**2
            
                
        # Compute global errors per equation
        for eq_idx in range(max_equations):
            global_error = np.sqrt(global_error_squared[eq_idx])
            if global_solution_norm_squared[eq_idx] > 1e-14:
                relative_error = np.sqrt(global_error_squared[eq_idx] / global_solution_norm_squared[eq_idx])
            else:
                relative_error = np.inf
   
            results['global_errors'].append({
                'equation_idx': eq_idx,
                'l2_error': global_error,
                'l2_relative_error': relative_error,
                'l2_solution_norm': np.sqrt(global_solution_norm_squared[eq_idx])
            })
        
        return results
    
    def _compute_domain_bulk_error(self, 
                                  problem, 
                                  discretization, 
                                  bulk_sol: np.ndarray, 
                                  time: float,
                                  analytical_functions: Optional[List[Callable]] = None) -> Dict:
        """
        Compute L2 error for bulk solution in a single domain.
        Handles discontinuous Galerkin basis functions.
        
        Args:
            problem: Problem object for the domain
            discretization: Spatial discretization for the domain
            bulk_sol: Bulk solution array (2*neq, n_elements) - each column has coefficients for one element
            time: Current time
            analytical_functions: List of analytical functions per equation
        
        Returns:
            Dictionary with domain-specific bulk error metrics
        """
        n_elements = discretization.n_elements
        neq = problem.neq
        
        # Get analytical functions
        if analytical_functions is None:
            analytical_functions = self._get_analytical_functions(problem)
        
        # max_pointwise_error = 0.0
        equation_errors = []
        solution_norms = []
                
        # Validate bulk solution structure
        expected_shape = (2 * neq, n_elements)
        if bulk_sol.shape != expected_shape:
            warnings.warn(f"Expected bulk solution shape {expected_shape}, got {bulk_sol.shape}")
            # Try to reshape if possible
            if bulk_sol.size == 2 * neq * n_elements:
                bulk_sol = bulk_sol.reshape(expected_shape)
            else:
                raise ValueError(f"Cannot reshape bulk solution from {bulk_sol.shape} to {expected_shape}")
        
        for eq_idx in range(neq):
            # Extract coefficients for this equation across all elements
            # For equation eq_idx, coefficients are at rows [2*eq_idx, 2*eq_idx+1]
            eq_coeff_row_start = 2 * eq_idx
            eq_coeff_row_end = eq_coeff_row_start + 2
            eq_coeffs = bulk_sol[eq_coeff_row_start:eq_coeff_row_end, :]  # Shape: (2, n_elements)
            
            # ========================================================================
            # DEBUG: Plot discontinuous piecewise linear bulk solution for equation eq_idx
            # ========================================================================
            import matplotlib.pyplot as plt
            
            x_plot = []
            y_plot = []
            
            for elem_idx in range(n_elements):
                x_left = discretization.nodes[elem_idx]
                x_right = discretization.nodes[elem_idx + 1]
                elem_coeffs = eq_coeffs[:, elem_idx]  # [coeff0, coeff1]
                
                # Evaluate at element endpoints using DG basis functions
                # At xi = -1: phi_0 = 1, phi_1 = 0 => u_left = coeff0
                # At xi = +1: phi_0 = 0, phi_1 = 1 => u_right = coeff1
                u_left = elem_coeffs[0]
                u_right = elem_coeffs[1]
                
                x_plot.extend([x_left, x_right])
                y_plot.extend([u_left, u_right])
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'Numerical (Eq {eq_idx})')
            
            # Plot analytical solution if available
            if analytical_functions and eq_idx < len(analytical_functions):
                x_analytical = np.linspace(discretization.nodes[0], discretization.nodes[-1], 200)
                y_analytical = [analytical_functions[eq_idx](x, time) for x in x_analytical]
                plt.plot(x_analytical, y_analytical, 'r--', linewidth=2, label=f'Analytical (Eq {eq_idx})')
            else:
                # Zero analytical solution
                x_analytical = np.linspace(discretization.nodes[0], discretization.nodes[-1], 200)
                y_analytical = np.zeros_like(x_analytical)
                plt.plot(x_analytical, y_analytical, 'r--', linewidth=2, label=f'Analytical (Zero, Eq {eq_idx})')
            
            # Mark element boundaries
            for boundary in discretization.nodes:
                plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
            
            plt.grid(True, alpha=0.3)
            plt.xlabel('x')
            plt.ylabel(f'Solution (Equation {eq_idx})')
            plt.title(f'DEBUG: Bulk Solution - Domain {getattr(problem, "domain_idx", 0)}, '
                     f'Eq {eq_idx}, t={time:.4f}, {n_elements} elements')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            print(f"DEBUG: Eq {eq_idx} coeffs shape: {eq_coeffs.shape}")
            # ========================================================================
            
            # Compute error using Gaussian quadrature on each element
            eq_error_squared = 0.0
            eq_solution_norm_squared = 0.0
            # eq_max_error = 0.0
            
            # Gaussian quadrature points and weights for [-1, 1]
            gauss_points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            gauss_weights = np.array([1.0, 1.0])
            
            for elem_idx in range(n_elements):
                # Element domain
                x_left = discretization.nodes[elem_idx]
                x_right = discretization.nodes[elem_idx + 1]
                h_elem = x_right - x_left
                
                # Element coefficients for this equation: eq_coeffs[:, elem_idx] -> [coeff0, coeff1]
                elem_coeffs = eq_coeffs[:, elem_idx]  # Shape: (2,)
                
                # Integrate over element using Gaussian quadrature
                for gp, gw in zip(gauss_points, gauss_weights):
                    # Map Gauss point from [-1, 1] to physical element
                    x_phys = 0.5 * ((1 - gp) * x_left + (1 + gp) * x_right)
                    
                    # Evaluate DG basis functions at Gauss point (linear basis)
                    # phi_0(xi) = (1 - xi)/2, phi_1(xi) = (1 + xi)/2
                    phi_0 = (1 - gp) / 2
                    phi_1 = (1 + gp) / 2
                    basis_values = np.array([phi_0, phi_1])
                    
                    # Numerical solution at Gauss point
                    numerical_value = np.dot(elem_coeffs, basis_values)
                    
                    # Analytical solution at Gauss point
                    if analytical_functions and eq_idx < len(analytical_functions):
                        analytical_value = analytical_functions[eq_idx](x_phys, time)
                    else:
                        analytical_value = 0.0
                        if elem_idx == 0:  # Warn only once per equation
                            warnings.warn(f"No analytical solution available for bulk equation {eq_idx}, using zero")
                    
                    # Pointwise error
                    pointwise_error = numerical_value - analytical_value
                    
                    # Accumulate L2 error (with Jacobian = h_elem/2)
                    jacobian = h_elem / 2
                    eq_error_squared += (pointwise_error**2) * gw * jacobian
                    eq_solution_norm_squared += (analytical_value**2) * gw * jacobian
                    
                    # Track maximum pointwise error
                    # eq_max_error = max(eq_max_error, abs(pointwise_error))
            
            # max_pointwise_error = max(max_pointwise_error, eq_max_error)
            
            # Store equation-specific results
            equation_errors.append({
                'equation_idx': eq_idx,
                'l2_error': np.sqrt(eq_error_squared),
                'solution_norm': np.sqrt(eq_solution_norm_squared),
            })
        
            solution_norms.append({
                'equation_idx': eq_idx,
                'solution_norm': np.sqrt(eq_solution_norm_squared),
            })
            
        return {
            'domain_idx': getattr(problem, 'domain_idx', 0),
            'equation_errors': equation_errors,
            'solution_norms': solution_norms,
            'error_type': 'bulk'
        }
                
    def _integrate_trapezoidal(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Integrate using trapezoidal rule.
        
        Args:
            x: Coordinate points
            y: Function values at points
        
        Returns:
            Integrated value
        """
        if len(x) != len(y):
            raise ValueError("x and y arrays must have same length")
        if len(x) < 2:
            return 0.0
        
        return np.trapz(y, x)
    
    def _get_analytical_functions(self, problem) -> Optional[List[Callable]]:
        """
        WARNING: This method is wrong, it should be removed.  Use retrieve_analytical_solution instead.
        Extract analytical functions from problem object.
        
        Args:
            problem: Problem object
        
        Returns:
            List of analytical functions or None if not available
        """
        warnings.warn(
            "The _get_analytical_functions method is deprecated and incorrect. "
            "Use retrieve_analytical_solution() instead. "
            "This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Try common attribute names for analytical solutions
        for attr_name in ['analytical_solution', 'exact_solution', 'analytical_functions']:
            if hasattr(problem, attr_name):
                analytical = getattr(problem, attr_name)
                if callable(analytical):
                    # Single function - wrap in list
                    return [analytical]
                elif isinstance(analytical, (list, tuple)):
                    # Multiple functions
                    return list(analytical)
        
        return None
    
    def compute_convergence_rate(self, 
                                errors: List[float], 
                                mesh_sizes: List[float]) -> Tuple[float, float]:
        """
        Compute convergence rate from L2 errors and mesh sizes.
        
        Args:
            errors: List of L2 errors for different mesh sizes
            mesh_sizes: List of corresponding mesh sizes (h values)
        
        Returns:
            Tuple of (convergence_rate, correlation_coefficient)
        """
        if len(errors) != len(mesh_sizes) or len(errors) < 2:
            raise ValueError("Need at least 2 error/mesh_size pairs")
        
        # Remove zero or negative errors (log will fail)
        valid_indices = [i for i, err in enumerate(errors) if err > 1e-16]
        if len(valid_indices) < 2:
            warnings.warn("Insufficient valid errors for convergence analysis")
            return 0.0, 0.0
        
        valid_errors = [errors[i] for i in valid_indices]
        valid_mesh_sizes = [mesh_sizes[i] for i in valid_indices]
        
        # Linear regression in log-log space: log(error) = p * log(h) + c
        log_h = np.log(valid_mesh_sizes)
        log_error = np.log(valid_errors)
        
        # Fit: log_error = p * log_h + c
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        p, c = np.linalg.lstsq(A, log_error, rcond=None)[0]
        
        # Correlation coefficient
        correlation = np.corrcoef(log_h, log_error)[0, 1]
        
        return p, correlation
    
    def generate_error_report(self, trace_errors: Optional[Dict] = None, bulk_errors: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive formatted error analysis report for trace and/or bulk errors.
        
        Args:
            trace_errors: Optional results from compute_trace_error
            bulk_errors: Optional results from compute_bulk_error
        
        Returns:
            Formatted string report
        """
        if trace_errors is None and bulk_errors is None:
            return "No error data provided for report generation."
        
        report = []
        report.append("="*70)
        
        # Determine report title
        if trace_errors is not None and bulk_errors is not None:
            report.append("COMPREHENSIVE ERROR ANALYSIS REPORT")
            report.append("TRACE AND BULK ERRORS")
        elif trace_errors is not None:
            report.append("TRACE ERROR ANALYSIS REPORT")
        else:
            report.append("BULK ERROR ANALYSIS REPORT")
            
        report.append("="*70)
        
        # Time information
        if trace_errors is not None:
            report.append(f"Time: {trace_errors['time']:.6f}")
        elif bulk_errors is not None:
            report.append(f"Time: {bulk_errors['time']:.6f}")
        
        report.append("")
        
        # ====================================================================
        # SECTION 1: TRACE ERRORS
        # ====================================================================
        if trace_errors is not None:
            report.append("TRACE ERROR ANALYSIS")
            report.append("-" * 40)
            
            # Global trace errors
            if 'global_errors' in trace_errors and trace_errors['global_errors']:
                report.append("Global Trace Errors per Equation:")
                for eq_result in trace_errors['global_errors']:
                    eq_idx = eq_result['equation_idx']
                    report.append(f"  Equation {eq_idx + 1}:")
                    report.append(f"    Euclidean Error: {eq_result['euclidean_error']:.6e}")
                    report.append(f"    Relative Error:  {eq_result['euclidean_relative_error']:.6e}")
                    report.append(f"    Solution Norm:   {eq_result['euclidean_solution_norm']:.6e}")
                    if 'max_error' in eq_result:
                        report.append(f"    Max Error:       {eq_result['max_error']:.6e}")
                        report.append(f"    Max Rel. Error:  {eq_result['max_relative_error']:.6e}")
            
            # Domain-wise trace errors
            if 'domain_errors' in trace_errors and trace_errors['domain_errors']:
                report.append("\nDomain-wise Trace Errors:")
                
                # Group by domains
                domains = {}
                for (domain_idx, eq_idx), eq_error in trace_errors['domain_errors'].items():
                    if domain_idx not in domains:
                        domains[domain_idx] = []
                    domains[domain_idx].append((eq_idx, eq_error))
                
                # Sort and report by domain
                for domain_idx in sorted(domains.keys()):
                    equations = sorted(domains[domain_idx], key=lambda x: x[0])
                    report.append(f"  Domain {domain_idx + 1}:")
                    
                    for eq_idx, eq_error in equations:
                        report.append(f"    Equation {eq_idx + 1}:")
                        report.append(f"      L2 Error:            {eq_error['l2_error']:.6e}")
                        if 'relative_l2_error' in eq_error:
                            report.append(f"      Relative L2 Error:   {eq_error['relative_l2_error']:.6e}")
                        if 'max_pointwise_error' in eq_error:
                            report.append(f"      Max Pointwise Error: {eq_error['max_pointwise_error']:.6e}")
                        if 'relative_max_error' in eq_error:
                            report.append(f"      Rel. Max Error:      {eq_error['relative_max_error']:.6e}")
                        if 'n_nodes' in eq_error:
                            report.append(f"      Nodes:               {eq_error['n_nodes']}")
            
            report.append("")
        
        # ====================================================================
        # SECTION 2: BULK ERRORS
        # ====================================================================
        if bulk_errors is not None:
            report.append("BULK ERROR ANALYSIS")
            report.append("-" * 40)
            
            # Global bulk errors
            if 'global_errors' in bulk_errors and bulk_errors['global_errors']:
                report.append("Global Bulk Errors per Equation:")
                for eq_result in bulk_errors['global_errors']:
                    eq_idx = eq_result['equation_idx']
                    report.append(f"  Equation {eq_idx + 1}:")
                    report.append(f"    L2 Error:        {eq_result['l2_error']:.6e}")
                    report.append(f"    Relative Error:  {eq_result['l2_relative_error']:.6e}")
                    report.append(f"    Solution Norm:   {eq_result['l2_solution_norm']:.6e}")
            
            # Domain-wise bulk errors
            if 'domain_errors' in bulk_errors and bulk_errors['domain_errors']:
                report.append("\nDomain-wise Bulk Errors:")
                
                # Group by domains
                domains = {}
                for (domain_idx, eq_idx), eq_error in bulk_errors['domain_errors'].items():
                    if domain_idx not in domains:
                        domains[domain_idx] = []
                    domains[domain_idx].append((eq_idx, eq_error))
                
                # Sort and report by domain
                for domain_idx in sorted(domains.keys()):
                    equations = sorted(domains[domain_idx], key=lambda x: x[0])
                    report.append(f"  Domain {domain_idx + 1}:")
                    
                    for eq_idx, eq_error in equations:
                        report.append(f"    Equation {eq_idx + 1}:")
                        report.append(f"      L2 Error:      {eq_error['l2_error']:.6e}")
                        if 'solution_norm' in eq_error:
                            report.append(f"      Solution Norm: {eq_error['solution_norm']:.6e}")
            
            report.append("")
        
        # ====================================================================
        # SECTION 3: SUMMARY STATISTICS
        # ====================================================================
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        
        # Overall statistics
        if trace_errors is not None and 'global_errors' in trace_errors:
            total_trace_equations = len(trace_errors['global_errors'])
            total_trace_error_sq = sum(eq['euclidean_error']**2 for eq in trace_errors['global_errors'])
            report.append(f"Trace Analysis:")
            report.append(f"  Equations analyzed:     {total_trace_equations}")
            report.append(f"  Overall global error:   {np.sqrt(total_trace_error_sq):.6e}")
        
        if bulk_errors is not None and 'global_errors' in bulk_errors:
            total_bulk_equations = len(bulk_errors['global_errors'])
            total_bulk_error_sq = sum(eq['l2_error']**2 for eq in bulk_errors['global_errors'])
            report.append(f"Bulk Analysis:")
            report.append(f"  Equations analyzed:     {total_bulk_equations}")
            report.append(f"  Overall global L2 error: {np.sqrt(total_bulk_error_sq):.6e}")
        
        # Combined statistics if both are available
        if trace_errors is not None and bulk_errors is not None:
            report.append(f"Combined Analysis:")
            if 'domain_errors' in trace_errors and 'domain_errors' in bulk_errors:
                total_domains_trace = len(set(k[0] for k in trace_errors['domain_errors'].keys()))
                total_domains_bulk = len(set(k[0] for k in bulk_errors['domain_errors'].keys()))
                report.append(f"  Domains with trace data: {total_domains_trace}")
                report.append(f"  Domains with bulk data:  {total_domains_bulk}")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
    
    def get_equation_error(self, error_results: Dict, domain_idx: int, equation_idx: int) -> Optional[Dict]:
        """
        Get error results for a specific equation in a specific domain.
        
        Args:
            error_results: Results from compute_trace_error or compute_bulk_error
            domain_idx: Domain index
            equation_idx: Equation index
        
        Returns:
            Error dictionary for the specified equation or None if not found
        """
        return error_results['equation_errors'].get((domain_idx, equation_idx), None)
    
    def get_global_equation_error(self, error_results: Dict, equation_idx: int) -> Optional[Dict]:
        """
        Get global error results for a specific equation across all domains.
        
        Args:
            error_results: Results from compute_trace_error
            equation_idx: Equation index
        
        Returns:
            Global error dictionary for the specified equation or None if not found
        """
        if equation_idx < len(error_results['global_error_per_equation']):
            return error_results['global_error_per_equation'][equation_idx]
        return None

    def get_analytical_solutions(self) -> Dict:
        """
        Get the automatically extracted analytical solutions.
        
        Returns:
            Dictionary with analytical functions per domain
        """
        return self.analytical_solutions.copy()
    
    def has_analytical_solution(self, domain_idx: int) -> bool:
        """
        Check if a domain has analytical solutions available.
        
        Args:
            domain_idx: Index of the domain to check
        
        Returns:
            True if analytical solutions are available, False otherwise
        """
        domain_key = f'domain_{domain_idx}'
        return (domain_key in self.analytical_solutions and 
                self.analytical_solutions[domain_key] is not None)

        
    # def compute_hdg_trace_error(self, 
    #                     numerical_solutions: List[np.ndarray], 
    #                     time: float,
    #                     analytical_functions: Optional[List[List[Callable]]] = None,
    #                     alpha: Optional[np.ndarray] = None,
    #                     use_hdg_formulation: bool = True) -> Dict:
    #     """
    #     Compute L2 errors between numerical and analytical solutions using HDG trace error formulation.
        
    #     According to HDG theory, the trace error is defined as:
    #     error = h^alpha[eq] * ||pointwise_errors||_2
    #     where h is the mesh size and alpha[eq] is the equation-specific exponent.
        
    #     Args:
    #         numerical_solutions: List of numerical trace solutions for each domain
    #         time: Current time for analytical solution evaluation
    #         analytical_functions: Optional list of analytical functions per domain/equation
    #                              If None, uses automatically extracted solutions from problems
    #         alpha: Optional array of mesh-size scaling exponents per equation (default: zeros vector)
    #         use_hdg_formulation: If True, uses HDG h^alpha scaling; if False, uses standard L2 integration
        
    #     Returns:
    #         Dictionary with HDG-specific error metrics per domain and equation
    #     """
    #     # Determine maximum number of equations to set default alpha
    #     max_equations = max(problem.neq for problem in self.problems)
        
    #     # Set default alpha values (zeros vector) if not provided
    #     if alpha is None:
    #         alpha = np.zeros(max_equations)
    #     else:
    #         alpha = np.asarray(alpha)
    #         if alpha.size == 1:
    #             # Broadcast scalar to vector
    #             alpha = np.full(max_equations, alpha.item())
    #         elif len(alpha) < max_equations:
    #             # Extend with zeros if too short
    #             alpha_extended = np.zeros(max_equations)
    #             alpha_extended[:len(alpha)] = alpha
    #             alpha = alpha_extended
        
    #     # Use provided analytical functions or fall back to extracted ones
    #     if analytical_functions is None:
    #         analytical_functions = [self.analytical_solutions.get(f'domain_{i}', None) 
    #                                for i in range(self.n_domains)]
        
    #     results = {
    #         'domain_errors': [],
    #         'equation_errors': {},  # organized by (domain_idx, eq_idx)
    #         'global_error_per_equation': [],  # global error for each equation
    #         'global_error': 0.0,
    #         'max_error': 0.0,
    #         'time': time,
    #         'alpha': alpha.copy(),
    #         'hdg_formulation': use_hdg_formulation,
    #         'error_formulation': 'HDG trace error (h^alpha[eq] scaling)' if use_hdg_formulation else 'Standard L2 integration'
    #     }
        
    #     # Track errors per equation across all domains
    #     max_equations = max(problem.neq for problem in self.problems)
    #     global_error_squared_per_eq = [0.0] * max_equations
    #     global_solution_norm_squared_per_eq = [0.0] * max_equations
    #     max_pointwise_error = 0.0
        
    #     for domain_idx in range(self.n_domains):
    #         problem = self.problems[domain_idx]
    #         discretization = self.discretizations[domain_idx]
    #         numerical_sol = numerical_solutions[domain_idx]
            
    #         domain_result = self._compute_domain_hdg_trace_error(
    #             problem, discretization, numerical_sol, time, 
    #             analytical_functions[domain_idx] if analytical_functions and domain_idx < len(analytical_functions) else None,
    #             alpha, use_hdg_formulation
    #         )
            
    #         results['domain_errors'].append(domain_result)
            
    #         # Store individual equation errors with domain/equation indexing
    #         for eq_error, eq_norm in zip(domain_result['equation_errors'], domain_result['solution_norms']):
    #             eq_idx = eq_error['equation_idx']
    #             results['equation_errors'][(domain_idx, eq_idx)] = eq_error
                
    #             # Accumulate global error per equation
    #             if eq_idx < len(global_error_squared_per_eq):
    #                 global_error_squared_per_eq[eq_idx] += eq_error['hdg_error_squared'] if use_hdg_formulation else eq_error['l2_error_squared']
    #                 global_solution_norm_squared_per_eq[eq_idx] += eq_norm['solution_norm']**2
    #                 max_pointwise_error_per_eq[eq_idx] = max(max_pointwise_error_per_eq[eq_idx], eq_error['max_pointwise_error'])
    #                 solution_max_norm_per_eq[eq_idx] = max(solution_max_norm_per_eq[eq_idx], eq_norm['solution_max_norm'])
            
    #         max_pointwise_error = max(max_pointwise_error, domain_result['max_pointwise_error'])
        
    #     # Compute global errors per equation
    #     for eq_idx in range(max_equations):
    #         global_error = np.sqrt(global_error_squared_per_eq[eq_idx])
    #         if global_solution_norm_squared_per_eq[eq_idx] > 1e-14:
    #             relative_error = np.sqrt(global_error_squared_per_eq[eq_idx] / global_solution_norm_squared_per_eq[eq_idx])
    #         else:
    #             relative_error = np.inf
                
    #         results['global_error_per_equation'].append({
    #             'equation_idx': eq_idx,
    #             'global_hdg_error' if use_hdg_formulation else 'global_l2_error': global_error,
    #             'global_relative_error': relative_error,
    #             'global_solution_norm': np.sqrt(global_solution_norm_squared_per_eq[eq_idx])
    #         })
        
    #     # Overall global error (sum of all equations)
    #     total_error_squared = sum(global_error_squared_per_eq)
    #     total_solution_norm_squared = sum(global_solution_norm_squared_per_eq)
        
    #     results['global_error'] = np.sqrt(total_error_squared)
    #     results['max_error'] = max_pointwise_error
        
    #     if total_solution_norm_squared > 1e-14:
    #         results['relative_global_error'] = np.sqrt(total_error_squared / total_solution_norm_squared)
    #     else:
    #         results['relative_global_error'] = np.inf
            
    #     return results
    
    # def _compute_domain_hdg_trace_error(self, 
    #                                    problem, 
    #                                    discretization, 
    #                                    numerical_sol: np.ndarray, 
    #                                    time: float,
    #                                    analytical_functions: Optional[List[Callable]] = None,
    #                                    alpha: np.ndarray = None,
    #                                    use_hdg_formulation: bool = True) -> Dict:
    #     """
    #     Compute HDG trace error for a single domain using h^alpha[eq] scaling.
        
    #     HDG trace error formulation:
    #     - For each node: pointwise_error = numerical_value - analytical_value
    #     - Local mesh size: h = domain_length / n_elements
    #     - HDG error for equation eq: h^alpha[eq] * ||pointwise_errors||_2 (Euclidean norm)
        
    #     Args:
    #         problem: Problem object for the domain
    #         discretization: Spatial discretization for the domain
    #         numerical_sol: Numerical solution array
    #         time: Current time
    #         analytical_functions: List of analytical functions per equation
    #         alpha: Array of mesh-size scaling exponents per equation
    #         use_hdg_formulation: If True, uses HDG h^alpha scaling; if False, uses standard L2
        
    #     Returns:
    #         Dictionary with domain-specific HDG trace error metrics
    #     """
    #     nodes = discretization.nodes
    #     n_nodes = len(nodes)
    #     n_elements = discretization.n_elements
    #     neq = problem.neq
        
    #     # Set default alpha values (zeros vector) if not provided
    #     if alpha is None:
    #         alpha = np.zeros(neq)
    #     else:
    #         # Ensure alpha has enough entries for all equations
    #         if len(alpha) < neq:
    #             alpha_extended = np.zeros(neq)
    #             alpha_extended[:len(alpha)] = alpha
    #             alpha = alpha_extended
        
    #     # Compute characteristic mesh size for this domain
    #     h = problem.domain_length / n_elements if n_elements > 0 else 1.0
        
    #     # Get analytical functions
    #     if analytical_functions is None:
    #         analytical_functions = self._get_analytical_functions(problem)
        
    #     max_pointwise_error = 0.0
    #     equation_errors = []
        
    #     for eq_idx in range(neq):
    #         # Get equation-specific alpha value
    #         alpha_eq = alpha[eq_idx] if eq_idx < len(alpha) else 0.0
    #         h_alpha_eq = h**alpha_eq
            
    #         # Extract numerical solution for this equation
    #         eq_start = eq_idx * n_nodes
    #         eq_end = eq_start + n_nodes
    #         numerical_values = numerical_sol[eq_start:eq_end, 0] if numerical_sol.ndim == 2 else numerical_sol[eq_start:eq_end]
            
    #         # Compute analytical solution at nodes
    #         if analytical_functions and eq_idx < len(analytical_functions):
    #             analytical_values = np.array([analytical_functions[eq_idx](x, time) for x in nodes])
    #         else:
    #             # Fallback: assume zero analytical solution with warning
    #             analytical_values = np.zeros_like(numerical_values)
    #             warnings.warn(f"No analytical solution available for equation {eq_idx}, using zero")
            
    #         # Compute pointwise errors
    #         pointwise_errors = numerical_values - analytical_values
            
    #         if use_hdg_formulation:
    #             # HDG formulation: h^alpha[eq] * ||pointwise_errors||_2 (Euclidean norm)
    #             euclidean_norm_squared = np.sum(pointwise_errors**2)
    #             hdg_error_squared = (h_alpha_eq**2) * euclidean_norm_squared
    #             hdg_error = h_alpha_eq * np.sqrt(euclidean_norm_squared)
                
    #             # For consistency, also compute standard L2 error
    #             l2_error_squared = self._integrate_trapezoidal(nodes, pointwise_errors**2)
    #             l2_error = np.sqrt(l2_error_squared)
    #         else:
    #             # Standard L2 formulation (for comparison)
    #             l2_error_squared = self._integrate_trapezoidal(nodes, pointwise_errors**2)
    #             l2_error = np.sqrt(l2_error_squared)
    #             hdg_error_squared = l2_error_squared
    #             hdg_error = l2_error
            
    #         # Compute solution norm (always using L2 integration)
    #         eq_solution_norm_squared = self._integrate_trapezoidal(nodes, analytical_values**2)
    #         eq_solution_norm = np.sqrt(eq_solution_norm_squared)
            
    #         # Track maximum pointwise error
    #         eq_max_error = np.max(np.abs(pointwise_errors))
    #         max_pointwise_error = max(max_pointwise_error, eq_max_error)
            
    #         # Store equation-specific results
    #         eq_result = {
    #             'equation_idx': eq_idx,
    #             'hdg_error': hdg_error,
    #             'hdg_error_squared': hdg_error_squared,
    #             'l2_error': l2_error if use_hdg_formulation else hdg_error,
    #             'l2_error_squared': l2_error_squared if use_hdg_formulation else hdg_error_squared,
    #             'solution_norm': eq_solution_norm,
    #             'solution_norm_squared': eq_solution_norm_squared,
    #             'max_pointwise_error': eq_max_error,
    #             'relative_error': hdg_error / eq_solution_norm if eq_solution_norm > 1e-14 else np.inf,
    #             'mesh_size': h,
    #             'alpha_eq': alpha_eq,
    #             'h_alpha_eq': h_alpha_eq,
    #             'euclidean_norm': np.sqrt(np.sum(pointwise_errors**2)) if use_hdg_formulation else None,
    #             'n_nodes': n_nodes,
    #             'numerical_values': numerical_values.copy(),
    #             'analytical_values': analytical_values.copy(),
    #             'pointwise_errors': pointwise_errors.copy(),
    #             'formulation': 'HDG' if use_hdg_formulation else 'L2'
    #         }
            
    #         equation_errors.append(eq_result)
        
    #     return {
    #         'domain_idx': getattr(problem, 'domain_idx', 0),
    #         'max_pointwise_error': max_pointwise_error,
    #         'equation_errors': equation_errors,
    #         'nodes': nodes.copy(),
    #         'n_equations': neq,
    #         'mesh_size': h,
    #         'alpha': alpha[:neq].copy(),  # Store only relevant alpha values for this domain
    #         'n_elements': n_elements,
    #         'formulation': 'HDG' if use_hdg_formulation else 'L2'
    #     }

def create_analytical_solutions_example() -> Dict:
    """
    THIS FUNCTION IS FOR TESTING PURPOSES ONLY. DO NOT USE IN PRODUCTION.
    Example analytical solutions for testing purposes.
    
    Returns:
        Dictionary with example analytical functions
    """
    warnings.warn(
        "The create_analytical_solutions_example function is for testing purposes only. "
        "Do not use this function in production code. "
        "Use retrieve_analytical_solution() to get real analytical solutions from problem objects.",
        UserWarning,
        stacklevel=2
    )
    
    def pressure_analytical(x: float, t: float) -> float:
        """Example: exponentially decaying pressure wave"""
        return np.exp(-t) * np.sin(np.pi * x)
    
    def flow_analytical(x: float, t: float) -> float:
        """Example: flow derived from pressure gradient"""
        return -np.pi * np.exp(-t) * np.cos(np.pi * x)
    
    return {
        'domain_0': [pressure_analytical, flow_analytical],
        # Add more domains as needed
    }
