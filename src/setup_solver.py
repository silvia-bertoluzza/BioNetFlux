"""
Backward-compatible shim for setup_solver.

The actual implementation has moved to bionetflux.setup_solver.
This file re-exports all public names so that existing scripts using
    from setup_solver import quick_setup, SolverSetup
continue to work.

For new code, prefer:
    from bionetflux.setup_solver import quick_setup, SolverSetup
"""

# Re-export everything from the canonical location
from bionetflux.setup_solver import (  # noqa: F401
    SolverSetup,
    create_solver_setup,
    quick_setup,
)
