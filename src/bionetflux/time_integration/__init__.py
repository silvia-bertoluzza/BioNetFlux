"""
Time integration module for BioNetFlux.

This module provides time-stepping functionality for advancing solutions
in time using implicit Euler with Newton iteration.
"""

from .time_stepper import TimeStepper, TimeStepResult
from .newton_solver import NewtonSolver, NewtonResult

__all__ = ['TimeStepper', 'NewtonSolver', 'NewtonResult', 'TimeStepResult']
