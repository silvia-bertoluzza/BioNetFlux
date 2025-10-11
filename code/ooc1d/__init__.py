"""
OOC1D: HDG Solver for 1D Keller-Segel Problems
Python port of the MATLAB implementation
"""

from .core.problem import Problem
from .core.discretization import Discretization

__version__ = "0.1.0"
__all__ = ["Problem", "Discretization"]
