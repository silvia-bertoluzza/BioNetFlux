"""
BioNetFlux-specific Newton solver for nonlinear systems in time stepping.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import time


@dataclass
class NewtonResult:
    """Result container for Newton iteration."""
    
    converged: bool
    iterations: int
    final_solution: np.ndarray
    final_residual_norm: float
    residual_history: List[float]
    step_norms: List[float]
    jacobian_condition: Optional[float] = None
    computation_time: float = 0.0
    
    def __str__(self) -> str:
        status = "CONVERGED" if self.converged else "FAILED"
        return (f"NewtonResult({status}, {self.iterations} iterations, "
                f"||R||={self.final_residual_norm:.6e})")


class NewtonSolver:
    """
    BioNetFlux-specific Newton solver for nonlinear systems.
    
    This solver is designed specifically for BioNetFlux time stepping,
    using the global assembler structure and static condensation framework.
    """
    
    def __init__(self, tolerance: float = 1e-10, max_iterations: int = 20,
                 verbose: bool = False):
        """
        Initialize BioNetFlux Newton solver.
        
        Parameters:
            tolerance: Convergence tolerance for residual norm
            max_iterations: Maximum number of Newton iterations
            verbose: Whether to print iteration progress
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    def solve(self, 
              initial_guess: np.ndarray,
              global_assembler,
              forcing_terms: List[np.ndarray],
              static_condensations: List,
              current_time: float,  # Renamed from 'time' to 'current_time'
              tolerance: Optional[float] = None,
              max_iterations: Optional[int] = None) -> NewtonResult:
        """
        Solve nonlinear system using BioNetFlux global assembler.
        
        Parameters:
            initial_guess: Starting point for Newton iteration
            global_assembler: BioNetFlux GlobalAssembler instance
            forcing_terms: List of forcing term arrays for each domain
            static_condensations: List of StaticCondensation instances
            current_time: Current simulation time for evaluation  # Updated docstring
            tolerance: Override default convergence tolerance
            max_iterations: Override default max iterations
            
        Returns:
            NewtonResult with solution and convergence info
        """
        start_time = time.time()  # Now this works correctly
        
        # Use instance defaults or override
        tol = tolerance if tolerance is not None else self.tolerance
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        
        # Initialize iteration variables
        newton_solution = initial_guess.copy()
        residual_history = []
        step_norms = []
        jacobian_condition = None
        
        if self.verbose:
            print(f"  Newton solver: tolerance={tol:.1e}, max_iterations={max_iter}")
            print(f"    Time: {current_time:.6f}")  # Updated variable name
            print(f"    Forcing terms: {len(forcing_terms)} domains")
        
        for iteration in range(max_iter):
            # Evaluate residual and Jacobian using BioNetFlux global assembler
            try:
                current_residual, current_jacobian = global_assembler.assemble_residual_and_jacobian(
                    global_solution=newton_solution,
                    forcing_terms=forcing_terms,
                    static_condensations=static_condensations,
                    time=current_time  # Updated variable name
                )
            except Exception as e:
                if self.verbose:
                    print(f"    Iteration {iteration}: Residual/Jacobian assembly failed ({e})")
                return NewtonResult(
                    converged=False,
                    iterations=iteration,
                    final_solution=newton_solution,
                    final_residual_norm=np.inf,
                    residual_history=residual_history,
                    step_norms=step_norms,
                    computation_time=time.time() - start_time
                )
            
            # Compute residual norm
            residual_norm = np.linalg.norm(current_residual)
            residual_history.append(residual_norm)
            
            # Check for convergence
            if residual_norm <= tol:
                if self.verbose:
                    print(f"    ✓ Newton converged in {iteration + 1} iterations")
                    print(f"      Final residual norm: {residual_norm:.6e}")
                return NewtonResult(
                    converged=True,
                    iterations=iteration + 1,
                    final_solution=newton_solution,
                    final_residual_norm=residual_norm,
                    residual_history=residual_history,
                    step_norms=step_norms,
                    jacobian_condition=jacobian_condition,
                    computation_time=time.time() - start_time
                )
            
            # Check Jacobian condition number
            try:
                jacobian_condition = np.linalg.cond(current_jacobian)
                if jacobian_condition > 1e12:
                    if self.verbose:
                        print(f"    ⚠ Warning: Jacobian poorly conditioned (cond = {jacobian_condition:.2e})")
            except Exception:
                jacobian_condition = np.inf
                if self.verbose:
                    print(f"    ⚠ Warning: Could not compute Jacobian condition number")
            
            if self.verbose:
                print(f"    Newton iteration {iteration}: ||R|| = {residual_norm:.6e}")
                if jacobian_condition is not None and jacobian_condition < np.inf:
                    print(f"                                 cond(J) = {jacobian_condition:.2e}")
            
            # Solve linear system: J * delta_x = -R
            try:
                delta_x = np.linalg.solve(current_jacobian, -current_residual)
            except np.linalg.LinAlgError as e:
                if self.verbose:
                    print(f"    ✗ Newton failed: Linear system singular ({e})")
                return NewtonResult(
                    converged=False,
                    iterations=iteration + 1,
                    final_solution=newton_solution,
                    final_residual_norm=residual_norm,
                    residual_history=residual_history,
                    step_norms=step_norms,
                    jacobian_condition=jacobian_condition,
                    computation_time=time.time() - start_time
                )
            
            # Store step norm and update solution
            step_norm = np.linalg.norm(delta_x)
            step_norms.append(step_norm)
            newton_solution += delta_x
            
            if self.verbose and step_norm > 0:
                print(f"                                 ||δx|| = {step_norm:.6e}")
        
        # Maximum iterations reached
        final_residual_norm = residual_history[-1] if residual_history else np.inf
        if self.verbose:
            print(f"    ✗ Newton failed to converge after {max_iter} iterations")
            print(f"      Final residual norm: {final_residual_norm:.6e}")
        
        return NewtonResult(
            converged=False,
            iterations=max_iter,
            final_solution=newton_solution,
            final_residual_norm=final_residual_norm,
            residual_history=residual_history,
            step_norms=step_norms,
            jacobian_condition=jacobian_condition,
            computation_time=time.time() - start_time
        )
    
    def solve_with_line_search(self, 
                              initial_guess: np.ndarray,
                              global_assembler,
                              forcing_terms: List[np.ndarray],
                              static_condensations: List,
                              current_time: float,  # Renamed parameter
                              alpha_init: float = 1.0,
                              alpha_min: float = 1e-4) -> NewtonResult:
        """
        BioNetFlux Newton solver with simple backtracking line search.
        
        This is a more robust version that can handle cases where full Newton
        steps lead to divergence.
        
        Parameters:
            initial_guess: Starting point
            global_assembler: BioNetFlux GlobalAssembler instance
            forcing_terms: List of forcing term arrays for each domain
            static_condensations: List of StaticCondensation instances
            current_time: Current simulation time for evaluation  # Updated docstring
            alpha_init: Initial step size (1.0 = full Newton step)
            alpha_min: Minimum step size before giving up
            
        Returns:
            NewtonResult with solution and convergence info
        """
        start_time = time.time()
        
        newton_solution = initial_guess.copy()
        residual_history = []
        step_norms = []
        jacobian_condition = None
        
        if self.verbose:
            print(f"  Newton solver with line search:")
            print(f"    tolerance={self.tolerance:.1e}, max_iterations={self.max_iterations}")
            print(f"    alpha_init={alpha_init}, alpha_min={alpha_min}")
            print(f"    Time: {current_time:.6f}")  # Updated variable name
        
        for iteration in range(self.max_iterations):
            # Evaluate residual and Jacobian at current point
            try:
                current_residual, current_jacobian = global_assembler.assemble_residual_and_jacobian(
                    global_solution=newton_solution,
                    forcing_terms=forcing_terms,
                    static_condensations=static_condensations,
                    time=current_time  # Updated variable name
                )
            except Exception as e:
                if self.verbose:
                    print(f"    Iteration {iteration}: Assembly failed ({e})")
                break
            
            residual_norm = np.linalg.norm(current_residual)
            residual_history.append(residual_norm)
            
            # Check convergence
            if residual_norm <= self.tolerance:
                if self.verbose:
                    print(f"    ✓ Newton with line search converged in {iteration + 1} iterations")
                    print(f"      Final residual norm: {residual_norm:.6e}")
                return NewtonResult(
                    converged=True,
                    iterations=iteration + 1,
                    final_solution=newton_solution,
                    final_residual_norm=residual_norm,
                    residual_history=residual_history,
                    step_norms=step_norms,
                    jacobian_condition=jacobian_condition,
                    computation_time=time.time() - start_time
                )
            
            # Compute Newton direction
            try:
                jacobian_condition = np.linalg.cond(current_jacobian)
                newton_direction = np.linalg.solve(current_jacobian, -current_residual)
            except np.linalg.LinAlgError:
                if self.verbose:
                    print(f"    ✗ Singular Jacobian at iteration {iteration}")
                break
            
            if self.verbose:
                print(f"    Iteration {iteration}: ||R|| = {residual_norm:.6e}, "
                      f"cond(J) = {jacobian_condition:.2e}")
            
            # Line search
            alpha = alpha_init
            newton_solution_new = None
            residual_new_norm = np.inf
            
            line_search_attempts = 0
            max_line_search_attempts = 10
            
            while alpha >= alpha_min and line_search_attempts < max_line_search_attempts:
                line_search_attempts += 1
                solution_candidate = newton_solution + alpha * newton_direction
                
                try:
                    # Evaluate residual at candidate point
                    residual_candidate, _ = global_assembler.assemble_residual_and_jacobian(
                        global_solution=solution_candidate,
                        forcing_terms=forcing_terms,
                        static_condensations=static_condensations,
                        time=current_time  # Updated variable name
                    )
                    residual_candidate_norm = np.linalg.norm(residual_candidate)
                    
                    # Accept if residual decreased (simple Armijo condition)
                    if residual_candidate_norm < residual_norm:
                        newton_solution_new = solution_candidate
                        residual_new_norm = residual_candidate_norm
                        break
                        
                except Exception:
                    pass  # Try smaller step
                
                alpha *= 0.5  # Backtrack
            
            if newton_solution_new is None:
                if self.verbose:
                    print(f"    ✗ Line search failed at iteration {iteration} after {line_search_attempts} attempts")
                break
            
            step_norms.append(alpha * np.linalg.norm(newton_direction))
            newton_solution = newton_solution_new
            
            if self.verbose:
                print(f"      Line search: ||R|| {residual_norm:.6e} → {residual_new_norm:.6e}, α = {alpha:.3f}")
        
        # Failed to converge
        final_residual_norm = residual_history[-1] if residual_history else np.inf
        if self.verbose:
            print(f"    ✗ Newton with line search failed to converge")
            print(f"      Final residual norm: {final_residual_norm:.6e}")
        
        return NewtonResult(
            converged=False,
            iterations=len(residual_history),
            final_solution=newton_solution,
            final_residual_norm=final_residual_norm,
            residual_history=residual_history,
            step_norms=step_norms,
            jacobian_condition=jacobian_condition,
            computation_time=time.time() - start_time
        )
    
    def solve_with_damping(self,
                          initial_guess: np.ndarray,
                          global_assembler,
                          forcing_terms: List[np.ndarray],
                          static_condensations: List,
                          current_time: float,  # Renamed parameter
                          damping_factor: float = 0.8) -> NewtonResult:
        """
        BioNetFlux Newton solver with fixed damping.
        
        Uses a fixed damping factor for all Newton steps, which can help
        with convergence for ill-conditioned problems.
        
        Parameters:
            initial_guess: Starting point
            global_assembler: BioNetFlux GlobalAssembler instance
            forcing_terms: List of forcing term arrays for each domain
            static_condensations: List of StaticCondensation instances
            current_time: Current simulation time for evaluation  # Updated docstring
            damping_factor: Fixed damping factor (0 < damping_factor <= 1)
            
        Returns:
            NewtonResult with solution and convergence info
        """
        if not (0 < damping_factor <= 1):
            raise ValueError(f"Damping factor must be in (0, 1], got {damping_factor}")
        
        start_time = time.time()
        
        newton_solution = initial_guess.copy()
        residual_history = []
        step_norms = []
        jacobian_condition = None
        
        if self.verbose:
            print(f"  Newton solver with damping:")
            print(f"    tolerance={self.tolerance:.1e}, max_iterations={self.max_iterations}")
            print(f"    damping_factor={damping_factor}")
            print(f"    Time: {current_time:.6f}")  # Updated variable name
        
        for iteration in range(self.max_iterations):
            # Evaluate residual and Jacobian
            try:
                current_residual, current_jacobian = global_assembler.assemble_residual_and_jacobian(
                    global_solution=newton_solution,
                    forcing_terms=forcing_terms,
                    static_condensations=static_condensations,
                    time=current_time  # Updated variable name
                )
            except Exception as e:
                if self.verbose:
                    print(f"    Iteration {iteration}: Assembly failed ({e})")
                break
            
            residual_norm = np.linalg.norm(current_residual)
            residual_history.append(residual_norm)
            
            # Check convergence
            if residual_norm <= self.tolerance:
                if self.verbose:
                    print(f"    ✓ Damped Newton converged in {iteration + 1} iterations")
                    print(f"      Final residual norm: {residual_norm:.6e}")
                return NewtonResult(
                    converged=True,
                    iterations=iteration + 1,
                    final_solution=newton_solution,
                    final_residual_norm=residual_norm,
                    residual_history=residual_history,
                    step_norms=step_norms,
                    jacobian_condition=jacobian_condition,
                    computation_time=time.time() - start_time
                )
            
            # Solve linear system with damping
            try:
                jacobian_condition = np.linalg.cond(current_jacobian)
                delta_x = np.linalg.solve(current_jacobian, -current_residual)
                damped_delta_x = damping_factor * delta_x
            except np.linalg.LinAlgError as e:
                if self.verbose:
                    print(f"    ✗ Singular Jacobian at iteration {iteration}: {e}")
                break
            
            step_norms.append(np.linalg.norm(damped_delta_x))
            newton_solution += damped_delta_x
            
            if self.verbose:
                print(f"    Iteration {iteration}: ||R|| = {residual_norm:.6e}, "
                      f"cond(J) = {jacobian_condition:.2e}, ||δx|| = {step_norms[-1]:.6e}")
        
        # Failed to converge
        final_residual_norm = residual_history[-1] if residual_history else np.inf
        if self.verbose:
            print(f"    ✗ Damped Newton failed to converge")
            print(f"      Final residual norm: {final_residual_norm:.6e}")
        
        return NewtonResult(
            converged=False,
            iterations=len(residual_history),
            final_solution=newton_solution,
            final_residual_norm=final_residual_norm,
            residual_history=residual_history,
            step_norms=step_norms,
            jacobian_condition=jacobian_condition,
            computation_time=time.time() - start_time
        )
