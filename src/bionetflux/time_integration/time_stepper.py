"""
Main time stepper for advancing BioNetFlux solutions by one time step.
"""

from dataclasses import dataclass
import numpy as np
import time
from typing import List, Optional, Tuple
from .newton_solver import NewtonSolver, NewtonResult
# from .time_step_result import TimeStepResult

@dataclass
class TimeStepResult:
    """Container for time step results and convergence information."""
    
    # Core results
    converged: bool
    iterations: int
    final_residual_norm: float
    updated_solution: np.ndarray
    updated_bulk_data: List
    computation_time: float
    
    # Optional detailed info
    residual_history: Optional[List[float]] = None
    jacobian_condition: Optional[float] = None
    newton_step_norms: Optional[List[float]] = None
    
    def __str__(self) -> str:
        status = "CONVERGED" if self.converged else "FAILED"
        return (f"TimeStepResult({status}, {self.iterations} iterations, "
                f"residual={self.final_residual_norm:.6e}, "
                f"time={self.computation_time:.4f}s)")
    
    def summary(self) -> str:
        """Generate a detailed summary string."""
        lines = [
            f"Time Step Result Summary:",
            f"  Status: {'CONVERGED' if self.converged else 'FAILED TO CONVERGE'}",
            f"  Newton iterations: {self.iterations}",
            f"  Final residual norm: {self.final_residual_norm:.6e}",
            f"  Computation time: {self.computation_time:.4f} seconds",
            f"  Solution vector size: {len(self.updated_solution)}",
            f"  Bulk data domains: {len(self.updated_bulk_data)}"
        ]
        
        if self.jacobian_condition is not None:
            lines.append(f"  Final Jacobian condition: {self.jacobian_condition:.2e}")
        
        if self.residual_history is not None:
            lines.append(f"  Residual reduction: {self.residual_history[0]:.2e} → {self.residual_history[-1]:.2e}")
        
        return "\n".join(lines)


class TimeStepper:
    """
    Coordinates single time step advancement using implicit Euler + Newton.
    
    This class encapsulates all the logic needed to advance a BioNetFlux
    solution by one time step, replacing the complex while loop in examples.
    """
    
    def __init__(self, setup, newton_solver: Optional[NewtonSolver] = None,
                 verbose: bool = True):
        """
        Initialize time stepper with solver setup and Newton solver.
        
        Parameters:
            setup: SolverSetup instance with all framework components
            newton_solver: Newton solver instance (creates default if None)
            verbose: Whether to print progress information
        """
        self.setup = setup
        self.newton_solver = newton_solver or NewtonSolver(verbose=verbose)
        self.verbose = verbose
        
        # Cache frequently used components
        self.bulk_manager = setup.bulk_data_manager
        self.global_assembler = setup.global_assembler
        self.static_condensations = setup.static_condensations
        self.problems = setup.problems
        self.discretizations = setup.global_discretization.spatial_discretizations
    
    def initialize_solution(self) -> Tuple[np.ndarray, List]:
        """
        Initialize the solution at time t=0.
        
        This method sets up the initial global solution vector and bulk data,
        corresponding to Step 3, Step 4, and lines 226-233 of the example file.
        
        Returns:
            Tuple[np.ndarray, List]: (current_solution, current_bulk_data) at t=0
        """
        if self.verbose:
            print("Initializing solution at t=0...")
        
        # Step 3: Create initial conditions (trace solutions and multipliers)
        trace_solutions, multipliers = self.setup.create_initial_conditions()
        
        if self.verbose:
            print(f"    ✓ Initial trace solutions created for {len(trace_solutions)} domains")
            print(f"    ✓ Initial multipliers created: {len(multipliers)} values")
        
        # Step 4: Assemble global solution vector
        current_solution = self.setup.create_global_solution_vector(trace_solutions, multipliers)
        
        if self.verbose:
            print(f"    ✓ Global solution vector assembled: shape {current_solution.shape}")
        
        # Initialize bulk data for all domains (lines 226-233 equivalent)
        current_bulk_data = self.bulk_manager.initialize_all_bulk_data(
            problems=self.problems,
            discretizations=self.discretizations,
            time=0.0
        )
        
        if self.verbose:
            print(f"    ✓ Bulk data initialized for {len(current_bulk_data)} domains")
            print(f"    ✓ Initialization at t=0 completed")
        
        return current_solution, current_bulk_data
    
    def advance_time_step(self, 
                         current_solution: np.ndarray,
                         current_bulk_data: List, 
                         current_time: float,
                         dt: float) -> TimeStepResult:
        """
        Advance the simulation by one time step using implicit Euler + Newton.
        
        This method replaces the entire while loop body from the original example.
        
        Parameters:
            current_solution: Global solution vector at current time
            current_bulk_data: List of BulkData objects at current time  
            current_time: Current simulation time
            dt: Time step size
        
        Returns:
            TimeStepResult containing updated solution and convergence info
        """
        start_time = time.time()
        new_time = current_time + dt
        
        if self.verbose:
            print(f"  TimeStepper: advancing from t={current_time:.6f} to t={new_time:.6f}")
        
        # Step 1: Compute source terms at new time
        try:
            source_terms = self.bulk_manager.compute_source_terms(
                problems=self.problems,
                discretizations=self.discretizations,
                time=new_time
            )
            if self.verbose:
                print(f"    ✓ Source terms computed")
        except Exception as e:
            if self.verbose:
                print(f"    ✗ Source term computation failed: {e}")
            return self._create_failed_result(current_solution, current_bulk_data, 
                                            f"Source computation failed: {e}",
                                            time.time() - start_time)
        
        # Step 2: Assemble forcing terms (right-hand side) for static condensation
        try:
            forcing_terms = []
            for i, (bulk_sol, source, static_cond) in enumerate(
                zip(current_bulk_data, source_terms, self.static_condensations)):
                
                forcing_term = static_cond.assemble_forcing_term(
                    previous_bulk_solution=bulk_sol.data,
                    external_force=source.data
                )
                forcing_terms.append(forcing_term)
            
            if self.verbose:
                print(f"    ✓ Forcing terms assembled for {len(forcing_terms)} domains")
        except Exception as e:
            if self.verbose:
                print(f"    ✗ Forcing term assembly failed: {e}")
            return self._create_failed_result(current_solution, current_bulk_data,
                                            f"Forcing assembly failed: {e}",
                                            time.time() - start_time)
        
        # Step 3: Solve nonlinear system using BioNetFlux-specific Newton solver
        if self.verbose:
            print(f"    Starting Newton iterations...")
        
        newton_result = self.newton_solver.solve(
            initial_guess=current_solution,  # Use current_solution directly as Newton initial guess
            global_assembler=self.global_assembler,
            forcing_terms=forcing_terms,
            static_condensations=self.static_condensations,
            current_time=new_time  # Updated parameter name
        )
        
        if not newton_result.converged:
            if self.verbose:
                print(f"    ✗ Newton solver failed to converge")
            return TimeStepResult(
                converged=False,
                iterations=newton_result.iterations,
                final_residual_norm=newton_result.final_residual_norm,
                updated_solution=newton_result.final_solution,
                updated_bulk_data=current_bulk_data,  # Keep old bulk data
                computation_time=time.time() - start_time,
                residual_history=newton_result.residual_history,
                jacobian_condition=newton_result.jacobian_condition,
                newton_step_norms=newton_result.step_norms
            )
        
        # Step 5: Update bulk data via static condensation
        try:
            updated_bulk_solutions = self.global_assembler.bulk_by_static_condensation(
                global_solution=newton_result.final_solution,
                forcing_terms=forcing_terms,
                static_condensations=self.static_condensations,
                time=new_time
            )
            
            # Update bulk_data objects with new solutions
            # Following the pattern from evolution+plotting_example.py lines 342-350
            updated_bulk_data = []
            
            for i, new_bulk_array in enumerate(updated_bulk_solutions):
                
                # Create new BulkData object or update existing one
                # Copy the structure from the original example
                new_bulk_data = self.bulk_manager.create_bulk_data(
                    domain_index=i,
                    problem=self.problems[i],
                    discretization=self.discretizations[i]
                )
                # Directly set the data array (following example pattern)
                new_bulk_data.data = new_bulk_array.copy()
                updated_bulk_data.append(new_bulk_data)
            
            if self.verbose:
                print(f"    ✓ Bulk data updated for {len(updated_bulk_data)} domains")
                
        except Exception as e:
            if self.verbose:
                print(f"    ✗ Bulk data update failed: {e}")
            # Still return the Newton solution, but with old bulk data
            updated_bulk_data = current_bulk_data
        
        # Step 6: Create and return successful result
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"    ✓ Time step completed successfully in {total_time:.4f}s")
            print(f"    ✓ Newton converged in {newton_result.iterations} iterations")
            print(f"    ✓ Final residual norm: {newton_result.final_residual_norm:.6e}")
        
        return TimeStepResult(
            converged=True,
            iterations=newton_result.iterations,
            final_residual_norm=newton_result.final_residual_norm,
            updated_solution=newton_result.final_solution,
            updated_bulk_data=updated_bulk_data,
            computation_time=total_time,
            residual_history=newton_result.residual_history,
            jacobian_condition=newton_result.jacobian_condition,
            newton_step_norms=newton_result.step_norms
        )
    
    def _create_failed_result(self, solution, bulk_data, error_msg, comp_time):
        """Helper to create a failed TimeStepResult."""
        if self.verbose:
            print(f"    ✗ Time step failed: {error_msg}")
        
        return TimeStepResult(
            converged=False,
            iterations=0,
            final_residual_norm=np.inf,
            updated_solution=solution,
            updated_bulk_data=bulk_data,
            computation_time=comp_time
        )
    
    def advance_multiple_steps(self, 
                              initial_solution: np.ndarray,
                              initial_bulk_data: List,
                              start_time: float,
                              dt: float,
                              n_steps: int,
                              stop_on_failure: bool = True) -> List[TimeStepResult]:
        """
        Advance multiple time steps in sequence.
        
        Parameters:
            initial_solution: Starting global solution
            initial_bulk_data: Starting bulk data
            start_time: Starting time
            dt: Time step size
            n_steps: Number of steps to take
            stop_on_failure: Whether to stop if a step fails
            
        Returns:
            List of TimeStepResult objects, one per step
        """
        results = []
        current_solution = initial_solution.copy()
        current_bulk_data = initial_bulk_data
        current_time = start_time
        
        if self.verbose:
            print(f"TimeStepper: advancing {n_steps} steps from t={start_time:.6f}")
        
        for step in range(n_steps):
            if self.verbose:
                print(f"\n--- Time Step {step + 1}/{n_steps} ---")
            
            result = self.advance_time_step(
                current_solution=current_solution,
                current_bulk_data=current_bulk_data,
                current_time=current_time,
                dt=dt
            )
            
            results.append(result)
            
            if not result.converged:
                if self.verbose:
                    print(f"  ✗ Step {step + 1} failed")
                if stop_on_failure:
                    if self.verbose:
                        print(f"  Stopping due to convergence failure")
                    break
            else:
                # Update state for next iteration
                current_solution = result.updated_solution
                current_bulk_data = result.updated_bulk_data
                current_time += dt
        
        if self.verbose:
            successful_steps = sum(1 for r in results if r.converged)
            print(f"\nMulti-step advancement: {successful_steps}/{len(results)} steps successful")
        
        return results
    
    def get_adaptive_stepper(self, dt_min: float = 1e-6, dt_max: float = 1.0,
                           safety_factor: float = 0.8) -> 'AdaptiveTimeStepper':
        """
        Get an adaptive time stepper based on this time stepper.
        
        Parameters:
            dt_min: Minimum allowed time step
            dt_max: Maximum allowed time step
            safety_factor: Safety factor for time step adjustment
            
        Returns:
            AdaptiveTimeStepper instance
        """
        return AdaptiveTimeStepper(
            self.setup, self.newton_solver, self.verbose,
            dt_min, dt_max, safety_factor
        )


class AdaptiveTimeStepper(TimeStepper):
    """
    Time stepper with adaptive time step control.
    
    Extends TimeStepper to automatically adjust dt based on Newton convergence.
    """
    
    def __init__(self, setup, newton_solver=None, verbose=True,
                 dt_min=1e-6, dt_max=1.0, safety_factor=0.8):
        """
        Initialize adaptive time stepper.
        
        Parameters:
            setup: SolverSetup instance
            newton_solver: Newton solver (created if None)
            verbose: Print progress information
            dt_min: Minimum time step
            dt_max: Maximum time step  
            safety_factor: Factor for time step adjustment
        """
        super().__init__(setup, newton_solver, verbose)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety_factor = safety_factor
    
    def advance_time_step_adaptive(self, 
                                  current_solution: np.ndarray,
                                  current_bulk_data: List,
                                  current_time: float,
                                  dt_suggested: float) -> Tuple[TimeStepResult, float]:
        """
        Advance with adaptive time step control.
        
        Parameters:
            current_solution: Current global solution
            current_bulk_data: Current bulk data
            current_time: Current time
            dt_suggested: Suggested time step size
            
        Returns:
            Tuple of (TimeStepResult, dt_next) where dt_next is suggested next step
        """
        dt_current = max(self.dt_min, min(dt_suggested, self.dt_max))
        max_retries = 5
        
        if self.verbose:
            print(f"  AdaptiveTimeStepper: trying dt={dt_current:.6f}")
        
        for retry in range(max_retries):
            # Try time step with current dt
            result = self.advance_time_step(
                current_solution, current_bulk_data, current_time, dt_current
            )
            
            if result.converged:
                # Success! Suggest next time step based on Newton performance
                if result.iterations <= 3:
                    # Converged quickly - can increase dt
                    dt_next = min(dt_current * 1.2, self.dt_max)
                    if self.verbose and dt_next > dt_current:
                        print(f"    → Increasing dt: {dt_current:.6f} → {dt_next:.6f}")
                elif result.iterations <= 8:
                    # Reasonable convergence - keep dt
                    dt_next = dt_current
                else:
                    # Slow convergence - reduce dt for next step
                    dt_next = max(dt_current * 0.8, self.dt_min)
                    if self.verbose and dt_next < dt_current:
                        print(f"    → Decreasing dt for next step: {dt_current:.6f} → {dt_next:.6f}")
                
                return result, dt_next
            
            else:
                # Failed to converge - reduce dt and retry
                dt_current = max(dt_current * 0.5, self.dt_min)
                
                if dt_current <= self.dt_min:
                    if self.verbose:
                        print(f"    ✗ Reached minimum dt={self.dt_min:.2e}, giving up")
                    return result, self.dt_min
                
                if self.verbose:
                    print(f"    Retry {retry + 1}: reducing dt to {dt_current:.6f}")
        
        # All retries failed
        if self.verbose:
            print(f"    ✗ All {max_retries} retries failed")
        
        return result, dt_current
