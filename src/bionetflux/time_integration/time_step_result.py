"""
Data structures for time step results.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import time


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
