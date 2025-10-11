"""
Solution visualization module for HDG system.

This module provides plotting capabilities for HDG solutions, including
trace solutions, mass evolution, and other diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import os


class SolutionPlotter:
    """
    Handles visualization of HDG solutions and system diagnostics.
    
    This class provides methods for plotting:
    - Trace solutions across domains
    - Mass evolution over time
    - Convergence diagnostics
    - Error analysis plots
    """
    
    def __init__(self, output_dir: str = ".", figsize: tuple = (12, 8)):
        """
        Initialize the solution plotter.
        
        Args:
            output_dir: Directory where plots will be saved
            figsize: Default figure size (width, height) in inches
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.colors = ['m', 'b', 'g', 'c', 'r', 'k', 'y', 'orange', 'purple', 'brown']
        self.markers = ['*', 'o', '+', 'x', 's', 'd', '^', 'v', '<', '>']
        self.equation_names = {
            0: ('Cell density u', 'u'),
            1: ('Chemoattractant φ', 'φ'),
            2: ('Equation 2', 'Eq2'),
            3: ('Equation 3', 'Eq3')
        }
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_trace_solution(self, 
                           problems: List,
                           discretizations: List,
                           trace_solutions: List[np.ndarray],
                           problem_name: str,
                           current_time: float,
                           save_plot: bool = True,
                           show_plot: bool = False) -> Optional[str]:
        """
        Plot the trace solution across all domains.
        
        Args:
            problems: List of problem objects
            discretizations: List of discretization objects
            trace_solutions: List of trace solution arrays for each domain
            problem_name: Name of the problem for plot title
            current_time: Current simulation time
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
            
        Returns:
            Filename of saved plot if save_plot=True, None otherwise
        """
        try:
            n_equations = problems[0].neq
            fig, axes = plt.subplots(n_equations, 1, figsize=self.figsize)
            if n_equations == 1:
                axes = [axes]
            
            # Plot each domain
            for i, (problem, discretization, trace_solution) in enumerate(
                zip(problems, discretizations, trace_solutions)):
                
                n_nodes = discretization.n_elements + 1
                nodes = discretization.nodes
                
                # Get proper x-coordinates for this domain
                x_coords = self._get_domain_coordinates(problem, nodes)
                
                # Plot each equation
                for eq in range(n_equations):
                    ax = axes[eq]
                    
                    # Extract trace values for this equation
                    trace_values = np.zeros(n_nodes)
                    for j in range(n_nodes):
                        node_idx = eq * n_nodes + j
                        trace_values[j] = trace_solution[node_idx]
                    
                    # Plot with different colors/markers for different domains
                    color = self.colors[i % len(self.colors)]
                    marker = self.markers[i % len(self.markers)]
                    
                    ax.plot(x_coords, trace_values, color=color, marker=marker, 
                           linewidth=2, markersize=6, 
                           label=f'Domain {i+1}')
                    
                    # Set labels and title
                    eq_name, eq_symbol = self.equation_names.get(eq, (f'Equation {eq+1}', f'Eq{eq+1}'))
                    ax.set_ylabel(eq_name, fontsize=12)
                    ax.set_title(f'{eq_name} at t = {current_time:.3f}', fontsize=14)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
            
            # Set x-label for the bottom subplot
            axes[-1].set_xlabel('Position', fontsize=12)
            
            # Adjust layout first
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            
            # Add main title with proper spacing
            plt.suptitle(f'Trace Solution - {problem_name}', fontsize=16)
            
            # Save and/or show the plot
            filename = None
            if save_plot:
                filename = f"trace_solution_{problem_name.replace(' ', '_')}_t{current_time:.3f}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename
            
        except Exception as e:
            print(f"Warning: Could not create trace solution plot: {e}")
            return None
    
    def plot_mass_evolution(self,
                           mass_evolution: List[float],
                           dt: float,
                           problem_name: str,
                           save_plot: bool = True,
                           show_plot: bool = False) -> Optional[str]:
        """
        Plot the evolution of total mass over time.
        
        Args:
            mass_evolution: List of mass values at each time step
            dt: Time step size
            problem_name: Name of the problem for plot title
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
            
        Returns:
            Filename of saved plot if save_plot=True, None otherwise
        """
        try:
            times = np.arange(len(mass_evolution)) * dt
            
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(times, mass_evolution, 'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Total Mass', fontsize=12)
            ax.set_title(f'Mass Evolution - {problem_name}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add mass conservation info
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            if initial_mass != 0:
                conservation_ratio = final_mass / initial_mass
                ax.text(0.02, 0.98, f'Conservation ratio: {conservation_ratio:.6f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save and/or show the plot
            filename = None
            if save_plot:
                filename = f"mass_evolution_{problem_name.replace(' ', '_')}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename
            
        except Exception as e:
            print(f"Warning: Could not create mass evolution plot: {e}")
            return None
    
    def plot_convergence_history(self,
                                convergence_data: List[List[float]],
                                problem_name: str,
                                save_plot: bool = True,
                                show_plot: bool = False) -> Optional[str]:
        """
        Plot Newton iteration convergence history.
        
        Args:
            convergence_data: List of residual norms for each time step
            problem_name: Name of the problem for plot title
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
            
        Returns:
            Filename of saved plot if save_plot=True, None otherwise
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            for i, residuals in enumerate(convergence_data[:10]):  # Show first 10 time steps
                if residuals:
                    iterations = range(1, len(residuals) + 1)
                    ax.semilogy(iterations, residuals, 'o-', 
                               label=f'Time step {i+1}', alpha=0.7)
            
            ax.set_xlabel('Newton Iteration', fontsize=12)
            ax.set_ylabel('Residual Norm', fontsize=12)
            ax.set_title(f'Newton Convergence History - {problem_name}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Save and/or show the plot
            filename = None
            if save_plot:
                filename = f"convergence_history_{problem_name.replace(' ', '_')}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename
            
        except Exception as e:
            print(f"Warning: Could not create convergence history plot: {e}")
            return None
    
    def create_summary_plot(self,
                           problems: List,
                           discretizations: List,
                           trace_solutions: List[np.ndarray],
                           mass_evolution: List[float],
                           problem_name: str,
                           final_time: float,
                           dt: float,
                           save_plot: bool = True,
                           show_plot: bool = False) -> Optional[str]:
        """
        Create a comprehensive summary plot with multiple subplots.
        
        Args:
            problems: List of problem objects
            discretizations: List of discretization objects
            trace_solutions: List of trace solution arrays for each domain
            mass_evolution: List of mass values at each time step
            problem_name: Name of the problem for plot title
            final_time: Final simulation time
            dt: Time step size
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
            
        Returns:
            Filename of saved plot if save_plot=True, None otherwise
        """
        try:
            n_equations = problems[0].neq
            n_rows = n_equations + 1  # +1 for mass evolution
            
            fig = plt.figure(figsize=(16, 4 * n_rows))
            
            # Plot trace solutions
            for eq in range(n_equations):
                ax = plt.subplot(n_rows, 1, eq + 1)
                
                for i, (problem, discretization, trace_solution) in enumerate(
                    zip(problems, discretizations, trace_solutions)):
                    
                    n_nodes = discretization.n_elements + 1
                    nodes = discretization.nodes
                    x_coords = self._get_domain_coordinates(problem, nodes)
                    
                    # Extract trace values for this equation
                    trace_values = np.zeros(n_nodes)
                    for j in range(n_nodes):
                        node_idx = eq * n_nodes + j
                        trace_values[j] = trace_solution[node_idx]
                    
                    color = self.colors[i % len(self.colors)]
                    marker = self.markers[i % len(self.markers)]
                    
                    ax.plot(x_coords, trace_values, color=color, marker=marker, 
                           linewidth=2, markersize=6, label=f'Domain {i+1}')
                
                eq_name, _ = self.equation_names.get(eq, (f'Equation {eq+1}', f'Eq{eq+1}'))
                ax.set_ylabel(eq_name, fontsize=12)
                ax.set_title(f'{eq_name} at t = {final_time:.3f}', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Plot mass evolution
            ax = plt.subplot(n_rows, 1, n_rows)
            times = np.arange(len(mass_evolution)) * dt
            ax.plot(times, mass_evolution, 'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Total Mass', fontsize=12)
            ax.set_title('Mass Evolution', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add mass conservation info
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            if initial_mass != 0:
                conservation_ratio = final_mass / initial_mass
                ax.text(0.02, 0.98, f'Conservation ratio: {conservation_ratio:.6f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Adjust layout first, leaving space for main title
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Add main title with proper spacing
            plt.suptitle(f'Solution Summary - {problem_name}', fontsize=18)
            
            # Save and/or show the plot
            filename = None
            if save_plot:
                filename = f"solution_summary_{problem_name.replace(' ', '_')}_T{final_time:.1f}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename
            
        except Exception as e:
            print(f"Warning: Could not create summary plot: {e}")
            return None
    
    def _get_domain_coordinates(self, problem, nodes: np.ndarray) -> np.ndarray:
        """
        Get proper x-coordinates for a domain based on extrema points.
        
        Args:
            problem: Problem object with extrema information
            nodes: Node coordinates from discretization
            
        Returns:
            Array of x-coordinates for plotting
        """
        if hasattr(problem, 'extrema') and len(problem.extrema) >= 2:
            x_start = problem.extrema[0][0]  # x-coordinate of first extrema point
            x_end = problem.extrema[1][0]    # x-coordinate of second extrema point
            
            # Map nodes from problem domain to extrema coordinates
            if hasattr(problem, 'A') and hasattr(problem, 'L'):
                # Map from [A, A+L] to [x_start, x_end]
                domain_length = problem.L
                domain_start = problem.A
                x_coords = x_start + (nodes - domain_start) * (x_end - x_start) / domain_length
            else:
                # Linear mapping assuming nodes span the domain
                node_span = nodes[-1] - nodes[0]
                if node_span > 0:
                    x_coords = x_start + (nodes - nodes[0]) * (x_end - x_start) / node_span
                else:
                    x_coords = np.full_like(nodes, x_start)
        else:
            # Fallback to using nodes directly
            x_coords = nodes
        
        return x_coords
