"""
BioNetFlux Picard solver for nonlinear systems in time stepping.

The Picard linearisation freezes the nonlinear coefficients inside each static
condensation (e.g. chi(phi_avg)) at their value from the *previous* iteration.
This is achieved by threading the per-element bulk solution (U) and flux data
(J) computed at iteration k into the assembler call for iteration k+1, where
they are forwarded to static_condensation via the ``prev_local_solution`` and
``prev_flux`` keyword arguments.

The outer convergence loop is identical to NewtonSolver; only the assembler
call differs (capture_flux=True, prev lists forwarded).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import time


@dataclass
class PicardResult:
    """Result container for Picard iteration."""

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
        return (
            f"PicardResult({status}, {self.iterations} iterations, "
            f"||R||={self.final_residual_norm:.6e})"
        )


class PicardSolver:
    """
    BioNetFlux Picard solver for nonlinear systems in time stepping.

    At each iteration the per-element bulk solution (U) and flux data (J) from
    the previous iteration are passed to the assembler so that each static
    condensation implementation can freeze its nonlinear coefficients.

    The first iteration always runs without previous data (identical to one
    Newton step), so the solver degrades gracefully when the initial guess is
    used as the first frozen state.
    """

    def __init__(
        self,
        tolerance: float = 1e-10,
        max_iterations: int = 50,
        verbose: bool = False,
    ):
        """
        Initialise BioNetFlux Picard solver.

        Parameters:
            tolerance: Convergence tolerance for the residual norm.
            max_iterations: Maximum number of Picard iterations.
            verbose: Whether to print iteration progress.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose

    def solve(
        self,
        initial_guess: np.ndarray,
        global_assembler,
        forcing_terms: List[np.ndarray],
        static_condensations: List,
        current_time: float,
        tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> PicardResult:
        """
        Solve nonlinear system using the Picard linearisation.

        Parameters:
            initial_guess: Starting point for Picard iteration.
            global_assembler: BioNetFlux GlobalAssembler instance.
            forcing_terms: List of forcing term arrays for each domain.
            static_condensations: List of StaticCondensation instances.
            current_time: Current simulation time.
            tolerance: Override default convergence tolerance.
            max_iterations: Override default max iterations.

        Returns:
            PicardResult with solution and convergence information.
        """
        start_time = time.time()

        tol = tolerance if tolerance is not None else self.tolerance
        max_iter = max_iterations if max_iterations is not None else self.max_iterations

        picard_solution = initial_guess.copy()
        residual_history: List[float] = []
        step_norms: List[float] = []
        jacobian_condition: Optional[float] = None

        # Previous-iteration per-element data; None on the first call so that
        # the first Picard step is equivalent to one Newton step.
        prev_U_list: Optional[List[np.ndarray]] = None
        prev_J_list: Optional[List[np.ndarray]] = None

        if self.verbose:
            print(f"  Picard solver: tolerance={tol:.1e}, max_iterations={max_iter}")
            print(f"    Time: {current_time:.6f}")

        for iteration in range(max_iter):
            # Assemble with capture_flux=True to obtain per-element U and J
            # for this iteration, which become the frozen data for the next.
            try:
                current_residual, current_jacobian, captured_U_list, captured_J_list = (
                    global_assembler.assemble_residual_and_jacobian(
                        global_solution=picard_solution,
                        forcing_terms=forcing_terms,
                        static_condensations=static_condensations,
                        time=current_time,
                        prev_U_list=prev_U_list,
                        prev_J_list=prev_J_list,
                        capture_flux=True,
                    )
                )
            except Exception as e:
                if self.verbose:
                    print(f"    Iteration {iteration}: assembly failed ({e})")
                return PicardResult(
                    converged=False,
                    iterations=iteration,
                    final_solution=picard_solution,
                    final_residual_norm=np.inf,
                    residual_history=residual_history,
                    step_norms=step_norms,
                    computation_time=time.time() - start_time,
                )

            residual_norm = np.linalg.norm(current_residual)
            residual_history.append(residual_norm)

            if self.verbose:
                print(f"    Picard iteration {iteration}: ||R|| = {residual_norm:.6e}")

            if residual_norm <= tol:
                if self.verbose:
                    print(f"    ✓ Picard converged in {iteration + 1} iterations")
                return PicardResult(
                    converged=True,
                    iterations=iteration + 1,
                    final_solution=picard_solution,
                    final_residual_norm=residual_norm,
                    residual_history=residual_history,
                    step_norms=step_norms,
                    jacobian_condition=jacobian_condition,
                    computation_time=time.time() - start_time,
                )

            # Check Jacobian condition number
            try:
                jacobian_condition = np.linalg.cond(current_jacobian)
                if self.verbose and jacobian_condition > 1e12:
                    print(
                        f"    ⚠ Warning: Jacobian poorly conditioned "
                        f"(cond = {jacobian_condition:.2e})"
                    )
            except Exception:
                jacobian_condition = np.inf

            # Solve linearised system: J * delta_x = -R
            try:
                delta_x = np.linalg.solve(current_jacobian, -current_residual)
            except np.linalg.LinAlgError as e:
                if self.verbose:
                    print(f"    ✗ Picard failed: Linear system singular ({e})")
                return PicardResult(
                    converged=False,
                    iterations=iteration + 1,
                    final_solution=picard_solution,
                    final_residual_norm=residual_norm,
                    residual_history=residual_history,
                    step_norms=step_norms,
                    jacobian_condition=jacobian_condition,
                    computation_time=time.time() - start_time,
                )

            step_norm = np.linalg.norm(delta_x)
            step_norms.append(step_norm)
            picard_solution = picard_solution + delta_x

            # Freeze the current iteration's data for the next iteration.
            prev_U_list = captured_U_list
            prev_J_list = captured_J_list

            if self.verbose:
                print(f"                                 ||δx|| = {step_norm:.6e}")

        final_residual_norm = residual_history[-1] if residual_history else np.inf
        if self.verbose:
            print(
                f"    ✗ Picard failed to converge after {max_iter} iterations; "
                f"||R|| = {final_residual_norm:.6e}"
            )
        return PicardResult(
            converged=False,
            iterations=max_iter,
            final_solution=picard_solution,
            final_residual_norm=final_residual_norm,
            residual_history=residual_history,
            step_norms=step_norms,
            jacobian_condition=jacobian_condition,
            computation_time=time.time() - start_time,
        )
