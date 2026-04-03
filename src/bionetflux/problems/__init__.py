"""
Problem definitions for the BioNetFlux library.
Contains implementations of various test problems and examples.
"""
from .custom_problem_template import create_global_framework as create_custom_problem

__all__ = [
    "create_custom_problem",
]
