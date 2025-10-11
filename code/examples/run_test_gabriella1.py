#!/usr/bin/env python3
"""
Example script to run TestGabriella1 problem.
Equivalent to running main.m with TestGabriella1 selected.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from python_port.ooc1d.problems.single_arc1 import create_test_gabriella1
from ooc1d.core.solver import HDGSolver
from ooc1d.utils.visualization import plot_solution

def main():
    """Run the TestGabriella1 problem."""
    print("Running TestGabriella1 problem...")
    
    # Create problem definition
    problems, discretizations, kappa, kk_params = create_test_gabriella1()
    
    # Create solver
    solver = HDGSolver(problems, discretizations, kappa, kk_params)
    
    # Solve
    results = solver.solve(verbose=True)
    
    # Plot results
    plot_solution(results, problems, discretizations)
    
    plt.show()
    print("Simulation complete.")

if __name__ == "__main__":
    main()
