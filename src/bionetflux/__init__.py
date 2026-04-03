
"""BioNetFlux: Multi-Domain Biological Network Flow Simulation Framework"""

__version__ = "1.0.0"

# Main exports for convenience
from .core.problem import Problem
from .geometry.domain_geometry import DomainGeometry, DomainInfo
from .visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
from .setup_solver import SolverSetup, quick_setup, create_solver_setup

__all__ = [
    "Problem",
    "DomainGeometry",
    "DomainInfo",
    "LeanMatplotlibPlotter",
    "SolverSetup",
    "quick_setup",
    "create_solver_setup",
]

