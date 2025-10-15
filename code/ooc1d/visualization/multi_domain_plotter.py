import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Optional, Tuple, Callable
import os


class MultiDomainPlotter:
    """
    Comprehensive visualization class for multi-domain, multi-equation problems.
    
    Handles visualization of neq solutions across ndom connected domain segments,
    with support for continuous plots, domain boundary highlighting, and time evolution.
    """
    
    def __init__(self, problems: List, discretizations: List, 
                 equation_names: Optional[List[str]] = None,
                 max_figure_width: float = 16.0,
                 max_figure_height: float = 12.0,
                 subplot_aspect_ratio: float = 0.7):
        """
        Initialize MultiDomainPlotter.
        
        Args:
            problems: List of Problem instances for each domain
            discretizations: List of Discretization instances for each domain
            equation_names: Optional list of names for equations (e.g., ['u', 'ω', 'v', 'φ'])
            max_figure_width: Maximum figure width in inches (default: 16.0)
            max_figure_height: Maximum figure height in inches (default: 12.0)
            subplot_aspect_ratio: Height/width ratio for individual subplots (default: 0.7)
        """
        self.problems = problems
        self.discretizations = discretizations
        self.ndom = len(problems)
        self.neq = problems[0].neq if problems else 2
        
        # Figure size control parameters
        self.max_figure_width = max_figure_width
        self.max_figure_height = max_figure_height
        self.subplot_aspect_ratio = subplot_aspect_ratio
        
        # Set default equation names if not provided
        if equation_names is None:
            if self.neq == 4:  # OrganOnChip
                self.equation_names = ['u', 'ω', 'v', 'φ']
            elif self.neq == 2:  # Keller-Segel
                self.equation_names = ['u', 'φ']
            else:
                self.equation_names = [f'Eq {i+1}' for i in range(self.neq)]
        else:
            self.equation_names = equation_names
        
        # Color scheme for different equations
        self.equation_colors = {
            0: {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
            1: {'color': 'red', 'linestyle': '--', 'marker': 's'},
            2: {'color': 'green', 'linestyle': '-.', 'marker': '^'},
            3: {'color': 'purple', 'linestyle': ':', 'marker': 'd'},
        }
        
        # Compute global coordinate mapping
        self._compute_global_coordinates()
    
    def _compute_global_coordinates(self):
        """Compute continuous global coordinates across all domains."""
        self.global_x = []
        self.domain_boundaries = []
        self.domain_labels = []
        
        for i, discretization in enumerate(self.discretizations):
            problem = self.problems[i]
            domain_start = problem.domain_start
            domain_length = problem.domain_length
            
            # Local coordinates for this domain
            local_x = domain_start + discretization.nodes 
            
            self.global_x.extend(local_x.tolist())
            
            # Track domain boundaries (except for first domain start)
            if i > 0:
                self.domain_boundaries.append(domain_start)
            
            # Domain labels
            domain_center = domain_start + domain_length / 2
            self.domain_labels.append((domain_center, f'Domain {i+1}'))
        
        self.global_x = np.array(self.global_x)
    
    def _extract_global_solution(self, trace_solutions: List[np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Extract and concatenate solutions across all domains for each equation.
        
        Args:
            trace_solutions: List of trace solution arrays, one per domain
            
        Returns:
            Dictionary mapping equation index to global solution array
        """
        global_solutions = {eq_idx: [] for eq_idx in range(self.neq)}
        
        for domain_idx, trace in enumerate(trace_solutions):
            discretization = self.discretizations[domain_idx]
            n_nodes = len(discretization.nodes)
            
            for eq_idx in range(self.neq):
                eq_start = eq_idx * n_nodes
                eq_end = eq_start + n_nodes
                eq_solution = trace[eq_start:eq_end]
                global_solutions[eq_idx].extend(eq_solution.tolist())
        
        # Convert to numpy arrays
        for eq_idx in range(self.neq):
            global_solutions[eq_idx] = np.array(global_solutions[eq_idx])
        
        return global_solutions
    
    def _compute_optimal_figure_size(self, n_rows: int, n_cols: int, 
                                   layout_type: str = "standard") -> Tuple[float, float]:
        """
        Compute optimal figure size based on number of subplots and screen constraints.
        
        Args:
            n_rows: Number of subplot rows
            n_cols: Number of subplot columns
            layout_type: Type of layout ("standard", "wide", "tall")
            
        Returns:
            Tuple of (width, height) in inches
        """
        if layout_type == "wide":
            # For continuous plots - prioritize width
            base_width_per_col = min(8.0, self.max_figure_width / max(n_cols, 1))
            base_height_per_row = base_width_per_col * self.subplot_aspect_ratio
            
        elif layout_type == "tall":
            # For domain comparison - prioritize height
            base_height_per_row = min(4.0, self.max_figure_height / max(n_rows, 1))
            base_width_per_col = base_height_per_row / self.subplot_aspect_ratio
            
        else:  # standard
            # Balanced approach
            base_width_per_col = min(6.0, self.max_figure_width / max(n_cols, 1))
            base_height_per_row = min(4.0, self.max_figure_height / max(n_rows, 1))
        
        # Calculate total dimensions
        total_width = base_width_per_col * n_cols
        total_height = base_height_per_row * n_rows
        
        # Apply maximum constraints
        if total_width > self.max_figure_width:
            scale_factor = self.max_figure_width / total_width
            total_width = self.max_figure_width
            total_height *= scale_factor
            
        if total_height > self.max_figure_height:
            scale_factor = self.max_figure_height / total_height
            total_height = self.max_figure_height
            total_width *= scale_factor
        
        # Ensure minimum readable size
        total_width = max(total_width, 8.0)
        total_height = max(total_height, 6.0)
        
        return total_width, total_height
    
    def _setup_figure_with_optimal_size(self, n_rows: int, n_cols: int, 
                                      layout_type: str = "standard") -> Tuple[plt.Figure, np.ndarray]:
        """
        Create figure with optimal size and return figure and axes.
        
        Args:
            n_rows: Number of subplot rows
            n_cols: Number of subplot columns  
            layout_type: Type of layout for size optimization
            
        Returns:
            Tuple of (figure, axes_array)
        """
        fig_width, fig_height = self._compute_optimal_figure_size(n_rows, n_cols, layout_type)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        return fig, axes

    def plot_continuous_solution(self, trace_solutions: List[np.ndarray], 
                                time: float = 0.0,
                                analytical_solutions: Optional[Dict[int, Callable]] = None,
                                title_prefix: str = "Multi-Domain Solution",
                                save_filename: Optional[str] = None,
                                show_domain_boundaries: bool = True,
                                show_domain_labels: bool = True,
                                figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Create continuous multi-panel plot across all domains.
        
        Args:
            trace_solutions: List of trace solution arrays
            time: Current time for title and analytical comparison
            analytical_solutions: Dict mapping equation index to analytical function
            title_prefix: Prefix for plot title
            save_filename: Optional filename to save plot
            show_domain_boundaries: Whether to show vertical lines at domain boundaries
            show_domain_labels: Whether to show domain labels
            figsize: Optional manual figure size override (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        # Extract global solutions
        global_solutions = self._extract_global_solution(trace_solutions)
        
        # Create figure with optimal sizing
        if figsize is not None:
            fig, axes = plt.subplots(self.neq, 1, figsize=figsize)
        else:
            fig, axes = self._setup_figure_with_optimal_size(self.neq, 1, "wide")
            axes = axes.flatten()  # Convert to 1D for easier indexing
        
        if self.neq == 1:
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        
        # Plot each equation
        for eq_idx in range(self.neq):
            ax = axes[eq_idx]
            style = self.equation_colors.get(eq_idx, {'color': 'black', 'linestyle': '-', 'marker': 'o'})
            
            # Plot discrete solution
            y_values = global_solutions[eq_idx]
            ax.plot(self.global_x, y_values, 
                   color=style['color'], linestyle=style['linestyle'], 
                   marker=style['marker'], markersize=4, linewidth=2,
                   label=f'{self.equation_names[eq_idx]} (discrete)')
            
            # Plot analytical solution if available
            if analytical_solutions and eq_idx in analytical_solutions:
                try:
                    analytical_y = analytical_solutions[eq_idx](self.global_x, time)
                    ax.plot(self.global_x, analytical_y, 
                           color=style['color'], linestyle=':', linewidth=3, alpha=0.7,
                           label=f'{self.equation_names[eq_idx]} (analytical)')
                except Exception as e:
                    print(f"Warning: Could not plot analytical solution for equation {eq_idx}: {e}")
            
            # Customize subplot
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f'{self.equation_names[eq_idx]}')
            ax.legend(fontsize=10)
            
            # Show domain boundaries
            if show_domain_boundaries:
                for boundary in self.domain_boundaries:
                    ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Show domain labels
            if show_domain_labels and eq_idx == 0:  # Only on top subplot
                for center, label in self.domain_labels:
                    ax.text(center, ax.get_ylim()[1], label, 
                           ha='center', va='bottom', fontsize=9, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            # Add statistics
            y_min, y_max = np.min(y_values), np.max(y_values)
            ax.text(0.02, 0.98, f'Range: [{y_min:.3e}, {y_max:.3e}]',
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set common x-label only on bottom subplot
        axes[-1].set_xlabel('Position')
        
        # Main title
        plt.suptitle(f'{title_prefix} at t = {time:.4f}\n{self.ndom} Domains, {self.neq} Equations',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if requested
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved as: {save_filename}")
        
        return fig
    
    def plot_domain_comparison(self, trace_solutions: List[np.ndarray],
                              time: float = 0.0,
                              title_prefix: str = "Domain-wise Comparison",
                              figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Create side-by-side domain view with all equations overlaid per domain.
        
        Args:
            trace_solutions: List of trace solution arrays
            time: Current time for title
            title_prefix: Prefix for plot title
            figsize: Optional manual figure size override (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure with optimal sizing
        if figsize is not None:
            fig, axes = plt.subplots(1, self.ndom, figsize=figsize)
        else:
            fig, axes = self._setup_figure_with_optimal_size(1, self.ndom, "standard")
            axes = axes.flatten()  # Convert to 1D for easier indexing
        
        if self.ndom == 1:
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        
        for domain_idx in range(self.ndom):
            ax = axes[domain_idx]
            discretization = self.discretizations[domain_idx]
            problem = self.problems[domain_idx]
            
            # Local coordinates
            domain_start = problem.domain_start
            domain_length = problem.domain_length
            local_x = domain_start + discretization.nodes 
            
            # Plot all equations for this domain
            trace = trace_solutions[domain_idx]
            n_nodes = len(discretization.nodes)
            
            for eq_idx in range(self.neq):
                eq_start = eq_idx * n_nodes
                eq_end = eq_start + n_nodes
                eq_values = trace[eq_start:eq_end]
                
                style = self.equation_colors.get(eq_idx, {'color': 'black', 'linestyle': '-', 'marker': 'o'})
                ax.plot(local_x, eq_values,
                       color=style['color'], linestyle=style['linestyle'],
                       marker=style['marker'], markersize=4, linewidth=2,
                       label=f'{self.equation_names[eq_idx]}')
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Position')
            ax.set_ylabel('Solution Value')
            ax.set_title(f'Domain {domain_idx + 1}\n[{problem.domain_start:.2f}, {problem.domain_start + problem.domain_length:.2f}]')
            ax.legend()
        
        plt.suptitle(f'{title_prefix} at t = {time:.4f}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_solution_evolution(self, initial_traces: List[np.ndarray],
                               final_traces: List[np.ndarray],
                               initial_time: float = 0.0,
                               final_time: float = 1.0,
                               title_prefix: str = "Solution Evolution",
                               figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Compare initial vs final solutions across all domains.
        
        Args:
            initial_traces: Initial trace solutions
            final_traces: Final trace solutions
            initial_time: Initial time
            final_time: Final time
            title_prefix: Prefix for plot title
            figsize: Optional manual figure size override (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        # Extract global solutions
        initial_global = self._extract_global_solution(initial_traces)
        final_global = self._extract_global_solution(final_traces)
        
        # Create figure with optimal sizing
        if figsize is not None:
            fig, axes = plt.subplots(self.neq, 1, figsize=figsize)
        else:
            fig, axes = self._setup_figure_with_optimal_size(self.neq, 1, "wide")
            axes = axes.flatten()  # Convert to 1D for easier indexing
        
        if self.neq == 1:
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        
        for eq_idx in range(self.neq):
            ax = axes[eq_idx]
            style = self.equation_colors.get(eq_idx, {'color': 'black'})
            
            # Plot initial and final
            ax.plot(self.global_x, initial_global[eq_idx],
                   color=style['color'], linestyle='-', linewidth=2, alpha=0.7,
                   marker='o', markersize=3, label=f'{self.equation_names[eq_idx]} (t={initial_time:.2f})')
            
            ax.plot(self.global_x, final_global[eq_idx],
                   color=style['color'], linestyle='--', linewidth=2,
                   marker='s', markersize=3, label=f'{self.equation_names[eq_idx]} (t={final_time:.2f})')
            
            # Show domain boundaries
            for boundary in self.domain_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
            
            # Calculate and show change statistics
            max_change = np.max(np.abs(final_global[eq_idx] - initial_global[eq_idx]))
            initial_norm = np.linalg.norm(initial_global[eq_idx])
            relative_change = max_change / (initial_norm + 1e-12)
            
            ax.text(0.02, 0.98, f'Max Δ: {max_change:.3e}\nRel Δ: {relative_change:.3e}',
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f'{self.equation_names[eq_idx]}')
            ax.legend()
        
        axes[-1].set_xlabel('Position')
        
        plt.suptitle(f'{title_prefix}: t = {initial_time:.2f} → {final_time:.2f}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_time_animation(self, solution_history: List[List[np.ndarray]],
                             time_history: List[float],
                             save_filename: Optional[str] = None,
                             fps: int = 10,
                             figsize: Optional[Tuple[float, float]] = None) -> FuncAnimation:
        """
        Create animated plot showing time evolution.
        
        Args:
            solution_history: List of trace solution lists for each time step
            time_history: List of time values
            save_filename: Optional filename to save animation (as .gif or .mp4)
            fps: Frames per second for animation
            figsize: Optional manual figure size override (width, height)
            
        Returns:
            FuncAnimation object
        """
        # Setup figure with optimal sizing
        if figsize is not None:
            fig, axes = plt.subplots(self.neq, 1, figsize=figsize)
        else:
            fig, axes = self._setup_figure_with_optimal_size(self.neq, 1, "wide")
            axes = axes.flatten()  # Convert to 1D for easier indexing
        
        if self.neq == 1:
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        
        # Initialize empty line objects
        lines = []
        for eq_idx in range(self.neq):
            ax = axes[eq_idx]
            style = self.equation_colors.get(eq_idx, {'color': 'black'})
            
            line, = ax.plot([], [], color=style['color'], linewidth=2, 
                           marker='o', markersize=3, label=self.equation_names[eq_idx])
            lines.append(line)
            
            ax.set_xlim(np.min(self.global_x), np.max(self.global_x))
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f'{self.equation_names[eq_idx]}')
            ax.legend()
            
            # Show domain boundaries
            for boundary in self.domain_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
        
        axes[-1].set_xlabel('Position')
        
        # Compute global y-limits
        all_solutions = []
        for traces in solution_history:
            global_sols = self._extract_global_solution(traces)
            for eq_idx in range(self.neq):
                all_solutions.extend(global_sols[eq_idx])
        
        y_min, y_max = np.min(all_solutions), np.max(all_solutions)
        y_range = y_max - y_min
        for ax in axes:
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        # Animation function
        def animate(frame):
            traces = solution_history[frame]
            time = time_history[frame]
            global_sols = self._extract_global_solution(traces)
            
            for eq_idx in range(self.neq):
                lines[eq_idx].set_data(self.global_x, global_sols[eq_idx])
            
            plt.suptitle(f'Time Evolution: t = {time:.4f}', fontsize=14, fontweight='bold')
            return lines
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(solution_history), 
                           interval=1000//fps, blit=False, repeat=True)
        
        # Save if requested
        if save_filename:
            if save_filename.endswith('.gif'):
                anim.save(save_filename, writer='pillow', fps=fps)
            elif save_filename.endswith('.mp4'):
                anim.save(save_filename, writer='ffmpeg', fps=fps)
            print(f"✓ Animation saved as: {save_filename}")
        
        return anim
    
    def set_figure_size_limits(self, max_width: float = 16.0, max_height: float = 12.0, 
                              aspect_ratio: float = 0.7):
        """
        Update figure size constraints.
        
        Args:
            max_width: Maximum figure width in inches
            max_height: Maximum figure height in inches  
            aspect_ratio: Height/width ratio for individual subplots
        """
        self.max_figure_width = max_width
        self.max_figure_height = max_height
        self.subplot_aspect_ratio = aspect_ratio
        print(f"✓ Figure size limits updated: max {max_width}×{max_height} inches, aspect ratio {aspect_ratio}")
