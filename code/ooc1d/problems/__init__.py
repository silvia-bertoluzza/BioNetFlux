"""
Problem definitions for OOC1D library.
Contains implementations of various test problems and examples.
"""
from .test_problem import create_global_framework as create_test_problem
from .triple_arc import create_global_framework as create_triple_arc
from .T_junction import create_global_framework as create_T_junction

__all__ = [
    "create_global_framework",
]
