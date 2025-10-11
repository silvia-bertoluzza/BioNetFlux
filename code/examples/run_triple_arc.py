#!/usr/bin/env python3
"""
Example script to run TripleArc problem.
Equivalent to running main.m with TripleArc selected.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from ooc1d.problems.triple_arc import create_triple_arc

def main():
    """Run the TripleArc problem."""
    print("Running TripleArc problem...")
    print("Three connected domains with different source terms")
    
    # Create problem definition
    problems, discretizations, kappa, kk_params = create_triple_arc()
    
    print(f"Number of domains: {len(problems)}")
    for i, (problem, discretization) in enumerate(zip(problems, discretizations)):
        print(f"Domain {i+1}: [{problem.domain_start:.0f}, {problem.domain_end:.0f}] "
              f"with {discretization.n_elements} elements")
        print(f"  Parameters: {problem.parameters}")
        print(f"  Problem type: {problem.type}")
        print(f"  Time steps: {discretization.n_time_steps}, dt = {discretization.dt}")
    
    print(f"\nInterface parameters:")
    print(f"kappa =\n{kappa}")
    print(f"kk_params =\n{kk_params}")
    
    print("\nProblem and discretization construction completed successfully!")
    
    # Stop here - comment out solver creation and execution
    # from ooc1d.core.solver import HDGSolver
    # from ooc1d.utils.visualization import plot_solution
    
    # # Create solver
    # solver = HDGSolver(problems, discretizations, kappa, kk_params)
    
    # # Solve
    # results = solver.solve(verbose=True)
    
    # # Plot results
    # plot_solution(results, problems, discretizations)
    
    # # Additional plots for multi-domain analysis
    # plot_multi_domain_analysis(results, problems, discretizations)
    
    # plt.show()
    # print("Simulation complete.")

if __name__ == "__main__":
    main()
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Total system mass conservation
    ax = axes[0, 1]
    total_masses_u = [sum(masses[i][0] for i in range(len(problems))) for masses in results['masses']]
    total_masses_phi = [sum(masses[i][1] for i in range(len(problems))) for masses in results['masses']]
    ax.plot(times, total_masses_u, label='Total u')
    ax.plot(times, total_masses_phi, label='Total φ')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Mass')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Newton iterations per time step
    ax = axes[1, 0]
    ax.plot(times[1:], results['newton_iterations'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Newton Iterations')
    ax.grid(True)
    
    # Plot 4: Interface flux visualization (placeholder)
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Interface Flux\n(To be implemented)', 
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Interface Analysis')
    
    plt.tight_layout()

if __name__ == "__main__":
    main()
