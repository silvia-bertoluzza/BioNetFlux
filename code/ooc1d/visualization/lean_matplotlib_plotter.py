import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple
import matplotlib.colors as mcolors


class LeanMatplotlibPlotter:
    """
    Lean matplotlib-based visualization class for multi-domain solutions.
    
    Supports two visualization modes:
    1. 2D curve mode: z = f(x) curves where z is solution, x is position
    2. Flat 3D mode: thick segments in (x,y) plane with z shown via colormap
    """
    
    def __init__(self, 
                 problems: List, 
                 discretizations: List,
                 equation_names: Optional[List[str]] = None,
                 figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize the lean plotter.
        
        Args:
            problems: List of Problem instances
            discretizations: List of Discretization instances  
            equation_names: Optional equation names for labeling
            figsize: Default figure size
        """
        self.problems = problems
        self.discretizations = discretizations
        self.ndom = len(problems)
        self.neq = problems[0].neq if problems else 2
        self.figsize = figsize
        
        # Set equation names with Greek letters
        if equation_names is None:
            if self.neq == 4:  # OrganOnChip
                self.equation_names = ['u', 'ω', 'v', 'φ']
            elif self.neq == 2:  # Keller-Segel
                self.equation_names = ['u', 'φ']
            else:
                self.equation_names = [f'Eq_{i+1}' for i in range(self.neq)]
        else:
            self.equation_names = equation_names
        
        # Color schemes
        self.equation_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        self.equation_colormaps = ['viridis', 'plasma', 'inferno', 'cividis', 'magma', 'turbo']
        
        # Compute global coordinate mapping
        self._compute_coordinates()
    
    def _compute_coordinates(self):
        """Compute global coordinates across all domains."""
        self.all_coords = []
        self.domain_boundaries = []  # Keep for backward compatibility, but won't use
        self.domain_info = []
        
        # Track bounding box
        all_domain_starts = []
        all_domain_ends = []
        all_x_coords = []
        all_y_coords = []
        
        for i, (discretization, problem) in enumerate(zip(self.discretizations, self.problems)):
            # Use discretization.nodes directly (they already contain the full coordinates)
            param_coords = discretization.nodes
            self.all_coords.extend(param_coords.tolist())
            
            # Get domain extrema for 2D positioning
            extrema_start = problem.extrema[0]  # (x1, y1)
            extrema_end = problem.extrema[1]    # (x2, y2)
            
            # Track domain extents for bounding box
            domain_end = problem.domain_start + problem.domain_length
            all_domain_starts.append(problem.domain_start)
            all_domain_ends.append(domain_end)
            all_x_coords.extend([extrema_start[0], extrema_end[0]])
            all_y_coords.extend([extrema_start[1], extrema_end[1]])
            
            # Store domain info for 2D/3D positioning using extrema
            self.domain_info.append({
                'start': problem.domain_start,
                'end': domain_end,
                'length': problem.domain_length,
                'n_nodes': len(discretization.nodes),
                'extrema_start': extrema_start,
                'extrema_end': extrema_end,
                'center_x': (extrema_start[0] + extrema_end[0]) / 2,
                'center_y': (extrema_start[1] + extrema_end[1]) / 2,
            })
        
        self.all_coords = np.array(self.all_coords)
        
        # Compute bounding box from extrema coordinates
        self.bounding_box = {
            'x_min': min(all_x_coords),
            'x_max': max(all_x_coords),
            'y_min': min(all_y_coords),
            'y_max': max(all_y_coords),
        }
        
        # Use parameter coordinate range for backward compatibility
        self.x_min, self.x_max = min(all_domain_starts), max(all_domain_ends)

    def _map_param_to_extrema(self, domain_idx: int, param_coords: np.ndarray) -> tuple:
        """
        Map parameter coordinates to 2D extrema coordinates for a domain.
        
        Args:
            domain_idx: Domain index
            param_coords: 1D parameter coordinates
            
        Returns:
            Tuple (x_coords, y_coords) in 2D space
        """
        domain_info = self.domain_info[domain_idx]
        extrema_start = domain_info['extrema_start']
        extrema_end = domain_info['extrema_end']
        
        # Normalize parameter coordinates to [0, 1]
        param_min, param_max = domain_info['start'], domain_info['end']
        t = (param_coords - param_min) / (param_max - param_min)
        
        # Linear interpolation between extrema
        x_coords = extrema_start[0] + t * (extrema_end[0] - extrema_start[0])
        y_coords = extrema_start[1] + t * (extrema_end[1] - extrema_start[1])
        
        return x_coords, y_coords

    def _extract_solutions(self, trace_solutions: List[np.ndarray]) -> Dict[int, np.ndarray]:
        """Extract and concatenate solutions for each equation across domains."""
        global_solutions = {eq_idx: [] for eq_idx in range(self.neq)}
        
        for domain_idx, trace in enumerate(trace_solutions):
            n_nodes = self.domain_info[domain_idx]['n_nodes']
            
            for eq_idx in range(self.neq):
                eq_start = eq_idx * n_nodes
                eq_end = eq_start + n_nodes
                eq_solution = trace[eq_start:eq_end]
                global_solutions[eq_idx].extend(eq_solution.tolist())
        
        # Convert to numpy arrays
        for eq_idx in range(self.neq):
            global_solutions[eq_idx] = np.array(global_solutions[eq_idx])
        
        return global_solutions
    
    def plot_2d_curves(self, 
                      trace_solutions: List[np.ndarray],
                      title: str = "2D Solution Curves",
                      show_bounding_box: bool = True,
                      show_mesh_points: bool = True,
                      save_filename: Optional[str] = None) -> plt.Figure:
        """
        Plot solutions as 2D curves: z = f(x) with separate subplots for each domain.
        
        Args:
            trace_solutions: List of trace solutions for each domain
            title: Plot title
            show_bounding_box: Whether to show domain boundaries
            show_mesh_points: Whether to show dots at mesh points
            save_filename: Optional filename for saving
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure with subplots for each domain
        fig, axes = plt.subplots(self.ndom, 1, figsize=(self.figsize[0], self.figsize[1] * self.ndom * 0.6))
        if self.ndom == 1:
            axes = [axes]
        
        # Process each domain separately
        coord_start = 0
        for domain_idx in range(self.ndom):
            ax = axes[domain_idx]
            domain_info = self.domain_info[domain_idx]
            n_nodes = domain_info['n_nodes']
            
            # Get coordinates for this domain
            coord_end = coord_start + n_nodes
            domain_coords = self.all_coords[coord_start:coord_end]
            
            # Extract solutions for this domain
            trace = trace_solutions[domain_idx]
            domain_solutions = {}
            for eq_idx in range(self.neq):
                eq_start = eq_idx * n_nodes
                eq_end = eq_start + n_nodes
                eq_solution = trace[eq_start:eq_end]
                domain_solutions[eq_idx] = eq_solution
            
            # Find solution range for this domain
            all_domain_values = []
            for eq_idx in range(self.neq):
                all_domain_values.extend(domain_solutions[eq_idx])
            z_min, z_max = np.min(all_domain_values), np.max(all_domain_values)
            z_range = z_max - z_min
            z_padding = z_range * 0.1 if z_range > 0 else 1.0
            
            # Plot each equation for this domain
            for eq_idx in range(self.neq):
                color = self.equation_colors[eq_idx % len(self.equation_colors)]
                eq_name = self.equation_names[eq_idx]
                
                # Plot curve
                ax.plot(domain_coords, domain_solutions[eq_idx], 
                       color=color, linewidth=2, label=eq_name, alpha=0.8)
                
                # Plot mesh points if requested
                if show_mesh_points:
                    ax.scatter(domain_coords, domain_solutions[eq_idx], 
                              color=color, s=30, alpha=0.7, zorder=5)
            
            # Add domain boundaries
            if show_bounding_box:
                ax.axvline(x=domain_info['start'], color='gray', linestyle='--', 
                          alpha=0.6, linewidth=1, label='Domain boundary' if domain_idx == 0 else "")
                ax.axvline(x=domain_info['end'], color='gray', linestyle='--', 
                          alpha=0.6, linewidth=1)
            
            # Formatting for this subplot
            ax.set_ylabel('Solution Value', fontsize=12)
            ax.set_title(f'Domain {domain_idx + 1}: {self.problems[domain_idx].name}', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set nice limits for this domain
            coord_range = domain_info['end'] - domain_info['start']
            coord_padding = coord_range * 0.02 if coord_range > 0 else 0.1
            ax.set_xlim(domain_info['start'] - coord_padding, domain_info['end'] + coord_padding)
            ax.set_ylim(z_min - z_padding, z_max + z_padding)
            
            coord_start = coord_end
        
        # Only add x-label to bottom subplot
        axes[-1].set_xlabel('Position', fontsize=12)
        
        # Add main title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save if requested
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"✓ 2D curves saved as: {save_filename}")
        
        return fig
    
    def plot_flat_3d(self, 
                    trace_solutions: List[np.ndarray],
                    equation_idx: int = 0,
                    title: Optional[str] = None,
                    segment_width: float = 0.1,
                    save_filename: Optional[str] = None,
                    view_angle: Tuple[float, float] = (30, 45)) -> plt.Figure:
        """
        Plot solution as thick segments in (x,y) plane with z colormap.
        
        Args:
            trace_solutions: List of trace solutions for each domain
            equation_idx: Which equation to visualize
            title: Plot title (auto-generated if None)
            segment_width: Width of domain segments
            save_filename: Optional filename for saving
            view_angle: 3D view angles (elevation, azimuth)
            
        Returns:
            Matplotlib Figure object
        """
        # Extract solution for specified equation
        solutions = self._extract_solutions(trace_solutions)
        eq_solution = solutions[equation_idx]
        eq_name = self.equation_names[equation_idx]
        colormap = self.equation_colormaps[equation_idx % len(self.equation_colormaps)]
        
        # Create 3D plot
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each domain as a thick segment
        coord_start = 0
        vmin, vmax = np.min(eq_solution), np.max(eq_solution)
        
        for domain_idx in range(self.ndom):
            domain_info = self.domain_info[domain_idx]
            n_nodes = domain_info['n_nodes']
            
            # Get coordinates and solution for this domain
            coord_end = coord_start + n_nodes
            param_coords = self.all_coords[coord_start:coord_end]
            domain_solution = eq_solution[coord_start:coord_end]
            coord_start = coord_end
            
            # Map parameter coordinates to 2D extrema coordinates
            x_coords, y_coords = self._map_param_to_extrema(domain_idx, param_coords)
            
            # Plot the segment as colored line segments at z=0 with rounded ends
            for i in range(len(param_coords) - 1):
                x_seg = [x_coords[i], x_coords[i+1]]
                y_seg = [y_coords[i], y_coords[i+1]]
                z_seg = [0, 0]  # Flat segments at z=0
                
                # Color based on average solution value
                color_val = (domain_solution[i] + domain_solution[i+1]) / 2
                color = plt.cm.get_cmap(colormap)((color_val - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
                
                # Use solid_capstyle='round' for rounded ends
                ax.plot(x_seg, y_seg, z_seg, color=color, linewidth=8, alpha=0.8, solid_capstyle='round')
            
            # Add solution points above the segment
            ax.scatter(x_coords, y_coords, domain_solution, 
                      c=domain_solution, cmap=colormap, s=50, 
                      vmin=vmin, vmax=vmax, alpha=0.9)
            
            # Connect flat segment to solution points with thin lines
            for i in range(len(param_coords)):
                ax.plot([x_coords[i], x_coords[i]], 
                       [y_coords[i], y_coords[i]], 
                       [0, domain_solution[i]], 
                       'k-', alpha=0.3, linewidth=0.5)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label(f'{eq_name} Solution', fontsize=12)
        
        # Formatting
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel(f'{eq_name}', fontsize=12)
        
        if title is None:
            title = f'{eq_name} Solution - Flat 3D View'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set nice limits based on bounding box
        x_range = self.bounding_box['x_max'] - self.bounding_box['x_min']
        y_range = self.bounding_box['y_max'] - self.bounding_box['y_min']
        x_padding = x_range * 0.05 if x_range > 0 else 0.1
        y_padding = y_range * 0.05 if y_range > 0 else 0.1
        
        ax.set_xlim(self.bounding_box['x_min'] - x_padding, self.bounding_box['x_max'] + x_padding)
        ax.set_ylim(self.bounding_box['y_min'] - y_padding, self.bounding_box['y_max'] + y_padding)
        
        z_range = vmax - vmin
        z_padding = z_range * 0.1 if z_range > 0 else 1.0
        ax.set_zlim(min(0, vmin - z_padding), vmax + z_padding)
        
        plt.tight_layout()
        
        # Save if requested
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Flat 3D plot saved as: {save_filename}")
        
        return fig

    def plot_birdview(self, 
                     trace_solutions: List[np.ndarray],
                     equation_idx: int = 0,
                     title: Optional[str] = None,
                     segment_width: float = 0.1,
                     save_filename: Optional[str] = None,
                     show_colorbar: bool = True,
                     show_bounding_box: bool = True) -> plt.Figure:
        """
        Plot solution as thick color-coded segments in 2D xy plane (bird's eye view).
        
        Args:
            trace_solutions: List of trace solutions for each domain
            equation_idx: Which equation to visualize
            title: Plot title (auto-generated if None)
            segment_width: Width of domain segments
            save_filename: Optional filename for saving
            show_colorbar: Whether to show colorbar
            show_bounding_box: Whether to show bounding box
            
        Returns:
            Matplotlib Figure object
        """
        # Extract solution for specified equation
        solutions = self._extract_solutions(trace_solutions)
        eq_solution = solutions[equation_idx]
        eq_name = self.equation_names[equation_idx]
        colormap = self.equation_colormaps[equation_idx % len(self.equation_colormaps)]
        
        # Create 2D plot
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Plot each domain as a thick colored segment
        coord_start = 0
        vmin, vmax = np.min(eq_solution), np.max(eq_solution)
        
        # Create colormap normalization
        if vmax > vmin:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = plt.Normalize(vmin=vmin-1, vmax=vmin+1)
        
        for domain_idx in range(self.ndom):
            domain_info = self.domain_info[domain_idx]
            n_nodes = domain_info['n_nodes']
            
            # Get coordinates and solution for this domain
            coord_end = coord_start + n_nodes
            param_coords = self.all_coords[coord_start:coord_end]
            domain_solution = eq_solution[coord_start:coord_end]
            coord_start = coord_end
            
            # Map parameter coordinates to 2D extrema coordinates
            x_coords, y_coords = self._map_param_to_extrema(domain_idx, param_coords)
            
            # Plot the segment as colored line segments
            for i in range(len(param_coords) - 1):
                x_start, x_end = x_coords[i], x_coords[i+1]
                y_start, y_end = y_coords[i], y_coords[i+1]
                
                # Color based on average solution value
                color_val = (domain_solution[i] + domain_solution[i+1]) / 2
                color = plt.cm.get_cmap(colormap)(norm(color_val))
                
                # Draw thick line segment
                ax.plot([x_start, x_end], [y_start, y_end], 
                       color=color, linewidth=segment_width*100, alpha=0.8, solid_capstyle='round')
            
            # Add solution points as circles
            ax.scatter(x_coords, y_coords, 
                      c=domain_solution, cmap=colormap, s=80, 
                      vmin=vmin, vmax=vmax, alpha=0.9, 
                      edgecolors='black', linewidth=0.5, zorder=5)
            
            # Add domain label
            domain_center_x = domain_info['center_x']
            domain_center_y = domain_info['center_y']
            ax.text(domain_center_x, domain_center_y + segment_width*2, f'Domain {domain_idx+1}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add colorbar if requested
        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
            cbar.set_label(f'{eq_name} Solution', fontsize=12)
        
        # Add bounding box if requested
        if show_bounding_box:
            from matplotlib.patches import Rectangle
            bbox_width = self.bounding_box['x_max'] - self.bounding_box['x_min']
            bbox_height = self.bounding_box['y_max'] - self.bounding_box['y_min']
            
            rect = Rectangle(
                (self.bounding_box['x_min'], self.bounding_box['y_min']),
                bbox_width, bbox_height,
                linewidth=2, edgecolor='gray', facecolor='none', 
                linestyle='--', alpha=0.7, label='Bounding box'
            )
            ax.add_patch(rect)
        
        # Formatting
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        
        if title is None:
            title = f'{eq_name} Solution - Bird\'s Eye View'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set nice limits based on bounding box
        x_range = self.bounding_box['x_max'] - self.bounding_box['x_min']
        y_range = self.bounding_box['y_max'] - self.bounding_box['y_min']
        x_padding = x_range * 0.05 if x_range > 0 else 0.1
        y_padding = y_range * 0.05 if y_range > 0 else 0.1
        
        ax.set_xlim(self.bounding_box['x_min'] - x_padding, self.bounding_box['x_max'] + x_padding)
        ax.set_ylim(self.bounding_box['y_min'] - y_padding, self.bounding_box['y_max'] + y_padding)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if requested
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Bird's eye view plot saved as: {save_filename}")
        
        return fig

    def plot_comparison(self, 
                       initial_traces: List[np.ndarray],
                       final_traces: List[np.ndarray],
                       initial_time: float = 0.0,
                       final_time: float = 1.0,
                       save_filename: Optional[str] = None,
                       show_bounding_box: bool = True) -> plt.Figure:
        """
        Compare initial vs final solutions using 2D curves.
        
        Args:
            initial_traces: Initial trace solutions
            final_traces: Final trace solutions
            initial_time: Initial time
            final_time: Final time
            save_filename: Optional filename for saving
            
        Returns:
            Matplotlib Figure object
        """
        # Extract solutions
        initial_solutions = self._extract_solutions(initial_traces)
        final_solutions = self._extract_solutions(final_traces)
        
        # Create figure with subplots
        fig, axes = plt.subplots(self.neq, 1, figsize=(self.figsize[0], self.figsize[1] * self.neq * 0.7))
        if self.neq == 1:
            axes = [axes]
        
        for eq_idx in range(self.neq):
            ax = axes[eq_idx]
            color = self.equation_colors[eq_idx % len(self.equation_colors)]
            eq_name = self.equation_names[eq_idx]
            
            # Plot initial and final
            ax.plot(self.all_coords, initial_solutions[eq_idx], 
                   color=color, linewidth=2, linestyle='-', alpha=0.7,
                   label=f'{eq_name} (t={initial_time:.2f})')
            
            ax.plot(self.all_coords, final_solutions[eq_idx], 
                   color=color, linewidth=2, linestyle='--', alpha=0.9,
                   label=f'{eq_name} (t={final_time:.2f})')
            
            # Add bounding box boundaries
            if show_bounding_box:
                ax.axvline(x=self.bounding_box['x_min'], color='gray', linestyle=':', alpha=0.5)
                ax.axvline(x=self.bounding_box['x_max'], color='gray', linestyle=':', alpha=0.5)
            
            # Calculate change statistics
            max_change = np.max(np.abs(final_solutions[eq_idx] - initial_solutions[eq_idx]))
            initial_norm = np.linalg.norm(initial_solutions[eq_idx])
            relative_change = max_change / (initial_norm + 1e-12)
            
            # Add statistics text
            ax.text(0.02, 0.98, f'Max Δ: {max_change:.3e}\nRel Δ: {relative_change:.3e}',
                   transform=ax.transAxes, verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            ax.set_ylabel(f'{eq_name}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        axes[-1].set_xlabel('Position', fontsize=12)
        
        fig.suptitle(f'Solution Evolution: t = {initial_time:.2f} → {final_time:.2f}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        # Save if requested
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved as: {save_filename}")
        plt.show()
        return fig
    
    def show_all(self):
        """Show all created plots."""
        plt.show()
    
    def close_all(self):
        """Close all plots."""
        plt.close('all')