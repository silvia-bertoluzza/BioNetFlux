# BioNetFlux Documentation

![BioNetFlux Logo](../assets/bionetflux_logo.png)

---

**BioNetFlux: Multi-Domain Biological Network Flow Simulation**

*A Python framework for simulating biological transport phenomena on complex network geometries*

---

![Barra Bar](../assets/barra_bar.png)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Module Documentation](#module-documentation)
4. [Getting Started](#getting-started)
5. [Creating New Problems](#creating-new-problems)
6. [Geometry Module Guide](#geometry-module-guide)
7. [Visualization System](#visualization-system)
8. [Example Applications](#example-applications)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

BioNetFlux is a computational framework designed for simulating biological transport phenomena on complex network geometries. The framework specializes in solving coupled partial differential equations (PDEs) on multi-domain networks, with particular focus on:

- **Keller-Segel chemotaxis models**: Cell migration driven by chemical gradients
- **Organ-on-Chip systems**: Microfluidic device simulations with multiple compartments
- **Multi-domain networks**: Complex geometries with junction conditions and interface constraints

### Key Features

- **Multi-Domain Support**: Handle complex network topologies with arbitrary domain connections
- **Geometry Management**: Intuitive geometry definition using the `DomainGeometry` class
- **Flexible Constraints**: Support for Neumann, Dirichlet, and Kedem-Katchalsky junction conditions
- **Advanced Visualization**: 2D curve plots, 3D flat views, and bird's eye network visualization
- **Time Evolution**: Implicit time stepping with Newton-Raphson nonlinear solver
- **Static Condensation**: Efficient element-level solution elimination

---

## Architecture Overview

```
BioNetFlux/
├── code/
│   ├── ooc1d/
│   │   ├── core/           # Core mathematical components
│   │   ├── geometry/       # Geometry management
│   │   ├── problems/       # Problem definitions
│   │   ├── solver/         # Numerical solvers
│   │   └── visualization/  # Plotting and visualization
│   ├── setup_solver.py    # Main setup interface
│   └── test_*.py          # Example test files
└── docs/                  # Documentation
```

### Core Components

1. **Problem Definition**: Physical parameters, equations, and boundary conditions
2. **Geometry Management**: Domain layout and network topology
3. **Discretization**: Finite element spatial discretization
4. **Constraint System**: Interface conditions and boundary constraints
5. **Time Evolution**: Implicit time stepping with Newton solver
6. **Visualization**: Multi-mode plotting system

---

## Module Documentation

### Core Module (`ooc1d.core`)

#### Problem Class (`problem.py`)

The `Problem` class encapsulates the physics of a single domain:

```python
class Problem:
    def __init__(self, neq, domain_start, domain_length, parameters, 
                 problem_type, name):
        # Physical domain definition
        # Equation parameters
        # Problem identification
```

**Key Methods:**
- `set_chemotaxis(chi, dchi)`: Define chemotaxis functions
- `set_force(eq_idx, force_func)`: Set source terms
- `set_solution(eq_idx, sol_func)`: Set analytical solutions
- `set_initial_condition(eq_idx, ic_func)`: Define initial conditions
- `set_extrema(start_point, end_point)`: Set 2D spatial coordinates

#### Discretization Classes (`discretization.py`)

```python
class Discretization:
    # Single domain spatial discretization
    # Finite element nodes and connectivity
    
class GlobalDiscretization:
    # Multi-domain discretization management
    # Time stepping parameters
```

#### Constraint Management (`constraints.py`)

```python
class ConstraintManager:
    # Interface and boundary condition management
    def add_neumann(eq_idx, domain_idx, coordinate, flux_func)
    def add_trace_continuity(eq_idx, dom1_idx, dom2_idx, coord1, coord2)
    def add_kedem_katchalsky(eq_idx, dom1_idx, dom2_idx, coord1, coord2, perm)
```

### Geometry Module (`ooc1d.geometry`)

#### DomainGeometry Class (`domain_geometry.py`)

The geometry module provides intuitive tools for defining complex network topologies:

```python
class DomainGeometry:
    def __init__(self, name="unnamed_geometry"):
        # Initialize empty geometry
    
    def add_domain(self, extrema_start, extrema_end, domain_start=None, 
                   domain_length=None, name=None, **metadata):
        # Add a domain segment to the network
        
    def get_domain(self, domain_id):
        # Retrieve domain information
        
    def get_bounding_box(self):
        # Calculate network bounding box
```

**Domain Information Structure:**
```python
@dataclass
class DomainInfo:
    domain_id: int
    extrema_start: Tuple[float, float]  # Physical coordinates
    extrema_end: Tuple[float, float]
    domain_start: float                 # Parameter space
    domain_length: float
    name: str
    metadata: Dict[str, Any]
```

### Solver Module (`ooc1d.solver`)

#### Setup Interface (`setup_solver.py`)

```python
def quick_setup(problem_module, validate=True):
    # Automatic problem setup from module
    # Returns configured solver setup
    
class SolverSetup:
    # Complete solver configuration
    def create_initial_conditions()
    def create_global_solution_vector()
    def extract_domain_solutions()
```

### Visualization Module (`ooc1d.visualization`)

#### LeanMatplotlibPlotter (`lean_matplotlib_plotter.py`)

Three complementary visualization modes:

1. **2D Curve Plots**: Traditional solution vs. position plots (separate subplot per domain)
2. **Flat 3D View**: Network segments with solution-colored scatter points above
3. **Bird's Eye View**: Top-down network view with color-coded segments

```python
class LeanMatplotlibPlotter:
    def plot_2d_curves(trace_solutions, title, show_mesh_points, save_filename)
    def plot_flat_3d(trace_solutions, equation_idx, view_angle, save_filename)
    def plot_birdview(trace_solutions, equation_idx, time, save_filename)
```

---

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BioNetFlux
```

2. Set up Python path:
```python
import sys
sys.path.insert(0, '/path/to/BioNetFlux/code')
```

### Basic Usage

```python
from setup_solver import quick_setup
from ooc1d.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

# Load a problem
setup = quick_setup("ooc1d.problems.my_problem", validate=True)

# Create initial conditions
trace_solutions, multipliers = setup.create_initial_conditions()

# Initialize visualization
plotter = LeanMatplotlibPlotter(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations
)

# Plot initial conditions
plotter.plot_2d_curves(trace_solutions, title="Initial Conditions")
plotter.plot_birdview(trace_solutions, equation_idx=0, time=0.0)
```

---

## Creating New Problems

### Problem Structure Template

Create a new file in `ooc1d/problems/` following this structure:

```python
# File: ooc1d/problems/my_new_problem.py
import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager
from ..geometry import DomainGeometry

def create_global_framework():
    """
    Create a new multi-domain problem.
    Returns: problems, global_discretization, constraint_manager, problem_name
    """
    # 1. Global parameters
    neq = 2  # Number of equations
    T = 1.0  # Final time
    dt = 0.1  # Time step
    problem_name = "My New Problem"
    
    # 2. Physical parameters
    parameters = np.array([param1, param2, param3, param4])
    
    # 3. Define functions (chemotaxis, sources, solutions, etc.)
    def chi(x): return np.ones_like(x)
    def dchi(x): return np.zeros_like(x)
    def source_u(s, t): return 0.0 * s
    def source_phi(s, t): return 0.0 * s
    def initial_u(s, t=0.0): return np.ones_like(s)
    def initial_phi(s, t=0.0): return np.zeros_like(s)
    
    # 4. Create geometry
    geometry = DomainGeometry("my_geometry")
    # Add domains using geometry.add_domain(...)
    
    # 5. Create problems from geometry
    problems = []
    discretizations = []
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        # Create Problem and Discretization objects
    
    # 6. Set up constraints
    constraint_manager = ConstraintManager()
    # Add boundary and interface constraints
    
    # 7. Return framework components
    return problems, global_discretization, constraint_manager, problem_name
```

### Keller-Segel Problems

For chemotaxis problems, include:

```python
# Chemotaxis sensitivity function
def chi(x):
    k1, k2 = 3.9e-9, 5.e-6
    return k1 / (k2 + x)**2

def dchi(x):
    k1, k2 = 3.9e-9, 5.e-6
    return -k1 * 2 / (k2 + x)**3

# Set chemotaxis for all problems
for problem in problems:
    problem.set_chemotaxis(chi, dchi)
    problem.set_force(0, source_u)      # Cell equation source
    problem.set_force(1, source_phi)    # Chemical equation source
```

### Organ-on-Chip Problems

For microfluidic systems, focus on:

```python
# Multi-compartment setup
compartments = ["inlet", "cell_chamber", "outlet", "waste"]

# Different parameters per compartment
parameters_list = [
    np.array([D1, v1, k1, 0.0]),  # Inlet: high flow
    np.array([D2, v2, k2, k_cell]),  # Cell chamber: cell interaction
    np.array([D3, v3, k3, 0.0]),  # Outlet: medium flow
    np.array([D4, v4, k4, 0.0])   # Waste: low flow
]

# Junction conditions with permeabilities
permeabilities = [0.8, 1.0, 0.9]  # Between compartments
```

---

## Geometry Module Guide

### Simple Linear Network

```python
geometry = DomainGeometry("linear_chain")

# Add sequential domains
geometry.add_domain(
    extrema_start=(0.0, 0.0),
    extrema_end=(1.0, 0.0),
    name="segment1"
)

geometry.add_domain(
    extrema_start=(1.0, 0.0),
    extrema_end=(2.0, 0.0),
    name="segment2"
)
```

### T-Junction Network

```python
geometry = DomainGeometry("t_junction")

# Main channel
geometry.add_domain(
    extrema_start=(0.0, -1.0),
    extrema_end=(0.0, 1.0),
    name="main_channel"
)

# Side branch
geometry.add_domain(
    extrema_start=(0.0, 0.0),
    extrema_end=(1.0, 0.0),
    name="side_branch"
)
```

### Grid Network

```python
geometry = DomainGeometry("grid_network")

# Vertical segments
for i, x_pos in enumerate([-0.5, 0.5]):
    geometry.add_domain(
        extrema_start=(x_pos, 0.0),
        extrema_end=(x_pos, 1.0),
        name=f"vertical_{i}"
    )

# Horizontal connectors
for i, y_pos in enumerate([0.2, 0.4, 0.6, 0.8]):
    geometry.add_domain(
        extrema_start=(-0.5, y_pos),
        extrema_end=(0.5, y_pos),
        name=f"horizontal_{i}"
    )
```

### Complex Branching Network

```python
geometry = DomainGeometry("branching_network")

# Main trunk
geometry.add_domain(
    extrema_start=(0.0, 0.0),
    extrema_end=(0.0, 2.0),
    name="trunk"
)

# Branches at different levels
branch_angles = [30, 60, 120, 150]  # degrees
for i, angle in enumerate(branch_angles):
    angle_rad = np.radians(angle)
    length = 1.0
    end_x = length * np.cos(angle_rad)
    end_y = 1.0 + length * np.sin(angle_rad)
    
    geometry.add_domain(
        extrema_start=(0.0, 1.0),
        extrema_end=(end_x, end_y),
        name=f"branch_{i}"
    )
```

---

## Visualization System

### 2D Curve Plots

Best for analyzing solution profiles along individual domains:

```python
plotter.plot_2d_curves(
    trace_solutions=solutions,
    title="Solution Profiles",
    show_mesh_points=True,
    save_filename="solution_curves.png"
)
```

Features:
- Separate subplot per domain
- All equations shown in each domain
- Mesh point markers
- Domain boundary indicators

### Flat 3D View

Ideal for understanding network topology with solution values:

```python
plotter.plot_flat_3d(
    trace_solutions=solutions,
    equation_idx=0,
    view_angle=(30, 45),
    save_filename="network_3d.png"
)
```

Features:
- Network segments in xy-plane
- Solution values as colored scatter points above
- Connecting lines from segments to solution points
- Rotatable 3D view

### Bird's Eye View

Perfect for network-level solution analysis:

```python
plotter.plot_birdview(
    trace_solutions=solutions,
    equation_idx=0,
    time=current_time,
    save_filename="network_overview.png"
)
```

Features:
- Top-down network view
- Color-coded segment thickness
- Solution point markers
- Clean network overview

---

## Example Applications

### Example 1: Simple Keller-Segel Chain

```python
# File: examples/simple_keller_segel.py
import sys
sys.path.insert(0, '../code')

from setup_solver import quick_setup
from ooc1d.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

def main():
    # Setup problem
    setup = quick_setup("ooc1d.problems.KS_with_geometry", validate=True)
    
    # Get initial conditions
    trace_solutions, multipliers = setup.create_initial_conditions()
    
    # Initialize plotter
    plotter = LeanMatplotlibPlotter(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations
    )
    
    # Plot initial state
    plotter.plot_2d_curves(trace_solutions, title="Initial State")
    plotter.plot_birdview(trace_solutions, equation_idx=0, time=0.0)
    
    # Time evolution
    dt = setup.global_discretization.dt
    T = 0.5
    current_time = 0.0
    global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
    
    while current_time < T:
        # Newton iteration (simplified)
        current_time += dt
        # ... solver steps ...
        
        # Extract solutions
        final_traces, _ = setup.extract_domain_solutions(global_solution)
        
        # Visualize
        plotter.plot_birdview(final_traces, equation_idx=0, time=current_time)
    
    plotter.show_all()

if __name__ == "__main__":
    main()
```

### Example 2: Complex Grid Network

```python
# File: examples/grid_network_example.py
import sys
sys.path.insert(0, '../code')

from setup_solver import quick_setup
from ooc1d.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

def main():
    # Load complex grid problem
    setup = quick_setup("ooc1d.problems.KS_grid_geometry", validate=True)
    
    print(f"Problem: {setup.get_problem_info()['problem_name']}")
    print(f"Domains: {setup.get_problem_info()['num_domains']}")
    
    # Initial conditions
    trace_solutions, multipliers = setup.create_initial_conditions()
    
    # Visualization
    plotter = LeanMatplotlibPlotter(
        problems=setup.problems,
        discretizations=setup.global_discretization.spatial_discretizations,
        figsize=(15, 10)
    )
    
    # Multiple views of initial state
    plotter.plot_2d_curves(
        trace_solutions, 
        title="Grid Network - Domain Profiles",
        save_filename="grid_profiles.png"
    )
    
    for eq_idx in range(2):  # Both equations
        plotter.plot_flat_3d(
            trace_solutions,
            equation_idx=eq_idx,
            title=f"Grid Network - {plotter.equation_names[eq_idx]} (3D)",
            save_filename=f"grid_3d_eq{eq_idx}.png"
        )
        
        plotter.plot_birdview(
            trace_solutions,
            equation_idx=eq_idx,
            time=0.0,
            save_filename=f"grid_birdview_eq{eq_idx}.png"
        )
    
    plotter.show_all()

if __name__ == "__main__":
    main()
```

---

## API Reference

### Quick Setup Function

```python
setup_solver.quick_setup(problem_module: str, validate: bool = True) -> SolverSetup
```

**Parameters:**
- `problem_module`: Import path to problem definition (e.g., "ooc1d.problems.my_problem")
- `validate`: Whether to validate setup after creation

**Returns:** Configured `SolverSetup` object

### SolverSetup Class

```python
class SolverSetup:
    def get_problem_info() -> Dict[str, Any]
    def create_initial_conditions() -> Tuple[List[np.ndarray], np.ndarray]
    def create_global_solution_vector(traces, multipliers) -> np.ndarray
    def extract_domain_solutions(global_solution) -> Tuple[List[np.ndarray], np.ndarray]
```

### DomainGeometry Class

```python
class DomainGeometry:
    def add_domain(extrema_start: Tuple[float, float],
                   extrema_end: Tuple[float, float],
                   domain_start: float = None,
                   domain_length: float = None,
                   name: str = None,
                   **metadata) -> int
    
    def get_domain(domain_id: int) -> DomainInfo
    def get_bounding_box() -> Dict[str, float]
    def num_domains() -> int
    def summary() -> str
```

### LeanMatplotlibPlotter Class

```python
class LeanMatplotlibPlotter:
    def __init__(problems, discretizations, equation_names=None, figsize=(12,8))
    
    def plot_2d_curves(trace_solutions, title, show_mesh_points=True,
                       save_filename=None) -> plt.Figure
    
    def plot_flat_3d(trace_solutions, equation_idx=0, view_angle=(30,45),
                     save_filename=None) -> plt.Figure
    
    def plot_birdview(trace_solutions, equation_idx=0, time=0.0,
                      save_filename=None) -> plt.Figure
    
    def plot_comparison(initial_traces, final_traces, initial_time=0.0,
                        final_time=1.0, save_filename=None) -> plt.Figure
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Ensure correct path setup
import sys
sys.path.insert(0, '/path/to/BioNetFlux/code')
```

**2. Geometry Validation**
```python
# Check geometry before problem creation
geometry = DomainGeometry("test")
# ... add domains ...
print(geometry.summary())  # Verify domain layout
print(geometry.get_bounding_box())  # Check coordinates
```

**3. Constraint Setup**
```python
# Verify constraint mapping
constraint_manager.map_to_discretizations(discretizations)
print(f"Total constraints: {constraint_manager.n_multipliers}")
```

**4. Solution Convergence**
```python
# Monitor Newton iteration
newton_tolerance = 1e-10
max_newton_iterations = 20

# Check residual norms during iteration
if residual_norm > newton_tolerance:
    print(f"Convergence issue: residual = {residual_norm:.2e}")
```

### Performance Optimization

1. **Mesh Resolution**: Balance accuracy vs. computational cost
2. **Time Step Size**: Use adaptive time stepping for stability
3. **Newton Tolerance**: Adjust based on problem requirements
4. **Domain Decomposition**: Optimize domain sizes for load balancing

### Debugging Tips

1. **Visualization**: Use all three plot types to understand solution behavior
2. **Parameter Validation**: Check physical parameter ranges
3. **Constraint Verification**: Ensure proper interface connectivity
4. **Solution Monitoring**: Track solution norms and residuals

---

## Contact and Support

For questions, issues, or contributions:

- **Repository**: [BioNetFlux GitHub]
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: Submit via GitHub Issues

---

**BioNetFlux Development Team**
*Multi-Domain Biological Network Flow Simulation Framework*

---
