# BioNetFlux Documentation

![BioNetFlux Logo](../Logos/BioNetFlux.png)

---

**BioNetFlux: Multi-Domain Biological Network Flow Simulation**

*A Python framework for simulating biological transport phenomena on complex network geometries*

---

**Acknowledgements** The development of *BioNetFlux* was carried out with the
support of the Italian Ministry of Research, under the complementary action NRRP "D34Health
- Digital Driven Diagnostics, prognostics and therapeutics for sustainable Health care" (Grant
#PNC0000001). An AI language model (Claude)
was used to assist in translating and extending an existing MATLAB implementation—originally
written entirely by the author—into Python. The resulting Python code was reviewed, corrected,
and fully validated by the author to ensure mathematical and numerical consistency with the
MATLAB version.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Module Documentation](#module-documentation)
4. [Getting Started](#getting-started)
5. [Creating New Problems](#creating-new-problems)
6. [Geometry Module Guide](#geometry-module-guide)
7. [Time Integration](#time-integration)
8. [Visualization System](#visualization-system)
9. [Example Applications](#example-applications)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

BioNetFlux is a computational framework designed for simulating biological transport phenomena on complex network geometries. The framework specializes in solving coupled partial differential equations (PDEs) on multi-domain networks, with particular focus on:

- **Keller-Segel chemotaxis models**: Cell migration driven by chemical gradients
- **Organ-on-Chip systems**: Microfluidic device simulations with multiple compartments
- **Multi-domain networks**: Complex geometries with junction conditions and interface constraints

### Key Features

- **Multi-Domain Support**: Handle complex network topologies with arbitrary domain connections
- **Geometry Management**: Intuitive geometry definition using the `DomainGeometry` class
- **Flexible Constraints**: Support for Neumann, Dirichlet, Robin, trace-continuity, and Kedem-Katchalsky junction conditions
- **Advanced Visualization**: 2D curve plots, 3D flat views, and bird’s eye network visualization
- **Time Evolution**: Implicit Euler time stepping with Newton-Raphson nonlinear solver
- **Adaptive Time Stepping**: Automatic time step control based on Newton convergence
- **Static Condensation**: Efficient element-level solution elimination via factory pattern
- **TOML Configuration**: Problem parameters managed through TOML config files

---

## Architecture Overview

```
BioNetFlux/
├── src/
│   ├── setup_solver.py              # Backward-compatible shim
│   └── bionetflux/                  # Core framework package (v1.0.0)
│       ├── __init__.py              # Public exports
│       ├── setup_solver.py          # SolverSetup, quick_setup(), create_solver_setup()
│       ├── core/                    # Core mathematical components (14 files)
│       ├── geometry/                # Network geometry + maze data
│       ├── problems/                # Problem definitions (7 files)
│       ├── time_integration/        # TimeStepper, NewtonSolver
│       ├── utils/                   # Config, elementary matrices, mesh mapping
│       ├── visualization/           # LeanMatplotlibPlotter
│       └── analysis/                # Error evaluation
├── config/                              # TOML parameter files
├── tests/                               # Pytest test suite (21 files)
├── examples/                            # Example scripts (5 files)
└── docs/                                # Documentation
```

### Core Components

1. **Problem Definition** (`core/problem.py`): Physical parameters, equations, and boundary conditions
2. **Geometry Management** (`geometry/domain_geometry.py`): Domain layout and network topology
3. **Discretization** (`core/discretization.py`): Finite element spatial discretization
4. **Constraint System** (`core/constraints.py`): Interface conditions and boundary constraints
5. **Global Assembly** (`core/lean_global_assembly.py`): Global system assembly via `GlobalAssembler`
6. **Static Condensation** (`core/static_condensation_*.py`): Element-level solution elimination
7. **Time Integration** (`time_integration/`): `TimeStepper` + `NewtonSolver`
8. **Visualization** (`visualization/lean_matplotlib_plotter.py`): Multi-mode plotting system

---

## Module Documentation

### Core Module (`bionetflux.core`)

#### Problem Class (`problem.py`)

The `Problem` class encapsulates the physics of a single domain:

```python
class Problem:
    def __init__(self, neq=2, domain_start=0.0, domain_length=1.0,
                 parameters=None, problem_type="keller_segel",
                 name="unnamed_problem"):
        # Physical domain definition, equation parameters, problem identification
```

**Key Methods:**
- `set_chemotaxis(chi, dchi)`: Define chemotaxis sensitivity function and its derivative
- `set_force(eq_idx, force_func)`: Set source terms
- `set_solution(eq_idx, sol_func)`: Set analytical solutions (for error computation)
- `set_flux_solution(eq_idx, flux_func)`: Set analytical flux solution
- `set_initial_condition(eq_idx, u0_func)`: Define initial conditions
- `set_boundary_flux(eq_idx, left_flux, right_flux)`: Set boundary flux functions
- `set_extrema(point1, point2)`: Set 2D spatial coordinates for visualization
- `validate_problem(verbose)`: Validate problem configuration for consistency

#### Discretization Classes (`discretization.py`)

```python
class Discretization:
    def __init__(self, n_elements, domain_start=0.0, domain_length=1.0,
                 stab_constant=1.0):
        # Single domain spatial discretization with finite element nodes

class GlobalDiscretization:
    def __init__(self, spatial_discretizations):
        # Multi-domain discretization management + time parameters
    def set_time_parameters(self, dt, T):
        # Set global time discretization parameters
```

**Utility Function:**
- `compute_n_elements_from_h(domain_length, h)`: Compute number of elements from target mesh size (guaranteed even, min 4)

#### Constraint Management (`constraints.py`)

```python
class ConstraintType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    TRACE_CONTINUITY = "trace_continuity"
    KEDEM_KATCHALSKY = "kedem_katchalsky"

class ConstraintManager:
    def add_dirichlet(eq_idx, domain_idx, position, data_function=None)
    def add_neumann(eq_idx, domain_idx, position, data_function=None)
    def add_robin(eq_idx, domain_idx, position, alpha, beta, data_function=None)
    def add_trace_continuity(eq_idx, dom1_idx, dom2_idx, pos1, pos2)
    def add_kedem_katchalsky(eq_idx, dom1_idx, dom2_idx, pos1, pos2, permeability)
    # Also: find_constraints(), replace_constraint(), make_*() factory methods
```

#### Global Assembly (`lean_global_assembly.py`)

```python
class GlobalAssembler:
    @classmethod
    def from_framework_objects(cls, problems, global_discretization,
                              static_condensations, constraint_manager=None)
    def assemble_residual_and_jacobian(global_solution, forcing_terms,
                                     static_condensations, time)
    def compute_forcing_terms(bulk_data_list, problems, discretizations, time, dt)
```

#### Static Condensation (`static_condensation_factory.py`)

```python
class StaticCondensationFactory:
    @classmethod
    def create(cls, problem, global_disc, elementary_matrices, i=0)
        # Creates KellerSegelStaticCondensation or StaticCondensationOOC
        # based on problem.problem_type
    @classmethod
    def register_implementation(cls, problem_type, implementation_class)
```

### Geometry Module (`bionetflux.geometry`)

#### DomainGeometry Class (`domain_geometry.py`)

The geometry module provides tools for defining complex network topologies:

```python
class DomainGeometry:
    def __init__(self, name="unnamed_geometry"):
        # Initialize empty geometry

    def add_domain(self, extrema_start, extrema_end, domain_start=None,
                   domain_length=None, name=None, display_color="blue", **metadata):
        # Add a domain segment; returns domain_id

    def add_connection(self, domain1_id, domain2_id, parameter1, parameter2=0.0, **metadata):
        # Add connection between domains

    def add_exterior_boundary(self, domain_id, parameter, **metadata):
        # Convenience method for exterior boundary

    def get_domain(self, domain_id) -> DomainInfo
    def num_domains() -> int
    def num_connections() -> int
    def get_bounding_box() -> dict
    def validate_geometry(verbose=False) -> bool
    def summary() -> str
```

**Data Structures:**
```python
@dataclass
class DomainInfo:
    domain_id: int
    extrema_start: Tuple[float, float]  # Physical coordinates
    extrema_end: Tuple[float, float]
    domain_start: float                 # Parameter space
    domain_length: float
    name: str
    display_color: str
    metadata: Dict[str, Any]

@dataclass
class ConnectionInfo:
    domain1_id: int
    domain2_id: int
    parameter1: float
    parameter2: float
    metadata: Dict[str, Any]
```

**Factory Functions:**
- `build_grid_geometry(Nx, Ny, ...)`: Create rectangular grid network
- `build_arc_sequence_geometry(N, start, length)`: Create sequential arc geometry
- `create_maze_geometry(maze_name)`: Load predefined maze topology from CSV data

### Time Integration Module (`bionetflux.time_integration`)

#### TimeStepper (`time_stepper.py`)

```python
class TimeStepper:
    def __init__(self, setup, newton_solver=None, verbose=True)
    def initialize_solution() -> Tuple[np.ndarray, List]
    def advance_time_step(current_solution, current_bulk_data,
                         current_time, dt) -> TimeStepResult
    def advance_multiple_steps(initial_solution, initial_bulk_data,
                              start_time, dt, n_steps) -> List[TimeStepResult]
    def get_adaptive_stepper(dt_min, dt_max, safety_factor) -> AdaptiveTimeStepper

class AdaptiveTimeStepper(TimeStepper):
    def advance_time_step_adaptive(current_solution, current_bulk_data,
                                  current_time, dt_suggested) -> Tuple[TimeStepResult, float]

@dataclass
class TimeStepResult:
    converged: bool
    iterations: int
    final_residual_norm: float
    updated_solution: np.ndarray
    updated_bulk_data: List
    computation_time: float
    residual_history: Optional[List[float]]
```

#### NewtonSolver (`newton_solver.py`)

```python
class NewtonSolver:
    def __init__(self, tolerance=1e-10, max_iterations=20, verbose=False)
    def solve(initial_guess, global_assembler, forcing_terms,
             static_condensations, current_time) -> NewtonResult
    def solve_with_line_search(initial_guess, global_assembler, forcing_terms,
                              static_condensations, current_time) -> NewtonResult

@dataclass
class NewtonResult:
    converged: bool
    iterations: int
    final_solution: np.ndarray
    final_residual_norm: float
    residual_history: List[float]
    step_norms: List[float]
```

### Visualization Module (`bionetflux.visualization`)

#### LeanMatplotlibPlotter (`lean_matplotlib_plotter.py`)

Three complementary visualization modes:

1. **2D Curve Plots**: Traditional solution vs. position plots (separate subplot per domain)
2. **Flat 3D View**: Network segments with solution-colored scatter points above
3. **Bird’s Eye View**: Top-down network view with color-coded segments

```python
class LeanMatplotlibPlotter:
    def __init__(self, problems, discretizations, equation_names=None,
                 figsize=(12, 8), output_dir=None)
    def plot_2d_curves(trace_solutions, title="2D Solution Curves",
                      show_bounding_box=True, show_mesh_points=True,
                      save_filename=None) -> plt.Figure
    def plot_flat_3d(trace_solutions, equation_idx=0, title=None,
                    segment_width=0.1, save_filename=None,
                    view_angle=(30, 45)) -> plt.Figure
    def plot_birdview(trace_solutions, equation_idx=0, time=0.0,
                     save_filename=None) -> plt.Figure
    def plot_comparison(initial_traces, final_traces, initial_time=0.0,
                       final_time=1.0, save_filename=None) -> plt.Figure
```

### Analysis Module (`bionetflux.analysis`)

```python
class ErrorEvaluator:
    # L2 error computation between numerical and analytical solutions
```

### Utilities Module (`bionetflux.utils`)

| Component | Purpose |
|-----------|---------|
| `config_manager.py` | `BaseConfigManager` — TOML file loading and parameter management |
| `elementary_matrices.py` | `ElementaryMatrices` — Reference element matrices (SymPy) |
| `mesh_mapping.py` | Coordinate transformations between reference and physical elements |

---

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BioNetFlux
```

2. Install in development mode:
```bash
pip install -e .
```

### Basic Usage

```python
from bionetflux import quick_setup, LeanMatplotlibPlotter
from bionetflux.time_integration import TimeStepper

# Set up a problem with geometry and config
setup = quick_setup(
    problem_module="bionetflux.problems.ks_problem",
    validate=True,
    config_file="config/ks_parameters.toml",
    geometry=geometry  # optional DomainGeometry instance
)

# Initialize time stepper and solution
time_stepper = TimeStepper(setup, verbose=True)
current_solution, current_bulk_data = time_stepper.initialize_solution()

# Time evolution
dt = setup.global_discretization.dt
T = setup.global_discretization.T
current_time = 0.0

while current_time < T - dt / 2:
    result = time_stepper.advance_time_step(
        current_solution=current_solution,
        current_bulk_data=current_bulk_data,
        current_time=current_time,
        dt=dt
    )
    if result.converged:
        current_solution = result.updated_solution
        current_bulk_data = result.updated_bulk_data
        current_time += dt
    else:
        print(f"Newton failed at t={current_time}")
        break

# Visualize results
final_traces, final_multipliers = setup.extract_domain_solutions(current_solution)
plotter = LeanMatplotlibPlotter(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations
)
plotter.plot_2d_curves(final_traces, title="Final Solution")
plotter.plot_birdview(final_traces, equation_idx=0, time=current_time)
```

---

## Creating New Problems

### Problem Structure Template

Create a new file in `bionetflux/problems/` following this structure
(see `custom_problem_template.py` for a complete starting point):

```python
# File: bionetflux/problems/my_new_problem.py
import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization, compute_n_elements_from_h
from ..core.constraints import ConstraintManager
from ..geometry.domain_geometry import DomainGeometry

def create_global_framework(geometry=None, config_file=None):
    """
    Create a new multi-domain problem.
    Returns: (problems, global_discretization, constraint_manager, problem_name)
    """
    # 1. Global parameters
    neq = 2  # Number of equations
    T = 1.0  # Final time
    dt = 0.1  # Time step
    problem_name = "My New Problem"

    # 2. Physical parameters
    parameters = np.array([param1, param2, param3, param4])

    # 3. Define functions (chemotaxis, sources, initial conditions)
    def chi(x): return np.ones_like(x)
    def dchi(x): return np.zeros_like(x)
    def source_u(s, t): return 0.0 * s
    def initial_u(s, t=0.0): return np.ones_like(s)

    # 4. Create or use provided geometry
    if geometry is None:
        geometry = DomainGeometry("my_geometry")
        geometry.add_domain(extrema_start=(0, 0), extrema_end=(1, 0), name="seg_0")
        geometry.add_domain(extrema_start=(1, 0), extrema_end=(2, 0), name="seg_1")

    # 5. Create problems and discretizations from geometry
    problems = []
    discretizations = []
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        h = 0.1  # target element size
        n_elem = compute_n_elements_from_h(domain_info.domain_length, h)
        prob = Problem(neq=neq, domain_start=domain_info.domain_start,
                       domain_length=domain_info.domain_length,
                       parameters=parameters, problem_type="keller_segel",
                       name=f"domain_{domain_id}")
        prob.set_chemotaxis(chi, dchi)
        prob.set_initial_condition(0, initial_u)
        prob.set_extrema(domain_info.extrema_start, domain_info.extrema_end)
        problems.append(prob)
        discretizations.append(Discretization(n_elem, domain_info.domain_start,
                                              domain_info.domain_length))

    # 6. Set up constraints
    constraint_manager = ConstraintManager()
    # Add boundary/interface constraints...

    # 7. Return framework components
    global_disc = GlobalDiscretization(discretizations)
    global_disc.set_time_parameters(dt=dt, T=T)
    return problems, global_disc, constraint_manager, problem_name
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
# Multi-compartment setup with 4 equations
# Different parameters per compartment
parameters_list = [
    np.array([D1, v1, k1, 0.0]),   # Inlet: high flow
    np.array([D2, v2, k2, k_cell]),  # Cell chamber: cell interaction
    np.array([D3, v3, k3, 0.0]),   # Outlet: medium flow
]

# Junction conditions with permeabilities (Kedem-Katchalsky)
constraint_manager.add_kedem_katchalsky(eq_idx, dom1, dom2, pos1, pos2, permeability)
```

---

## Geometry Module Guide

### Simple Linear Network

```python
geometry = DomainGeometry("linear_chain")

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

### Grid Network (Using Factory)

```python
from bionetflux.geometry.domain_geometry import build_grid_geometry

geometry = build_grid_geometry(Nx=3, Ny=3)
```

### Arc Sequence (Using Factory)

```python
from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry

geometry = build_arc_sequence_geometry(N=2, start=1.5, length=2.0)
```

### Maze Geometry (Using Factory)

```python
from bionetflux.geometry.domain_geometry import create_maze_geometry

geometry = create_maze_geometry("maze3")  # loads from maze_3_data/
```

### Custom Complex Network

```python
geometry = DomainGeometry("branching_network")

# Main trunk
d0 = geometry.add_domain(
    extrema_start=(0.0, 0.0),
    extrema_end=(0.0, 2.0),
    name="trunk"
)

# Branch
d1 = geometry.add_domain(
    extrema_start=(0.0, 1.0),
    extrema_end=(1.0, 1.5),
    name="branch"
)

# Connect trunk and branch at the junction point
geometry.add_connection(d0, d1, parameter1=0.5, parameter2=0.0)

# Add exterior boundaries
geometry.add_exterior_boundary(d0, parameter=0.0)  # trunk start
geometry.add_exterior_boundary(d0, parameter=1.0)  # trunk end
geometry.add_exterior_boundary(d1, parameter=1.0)  # branch end

# Validate
geometry.validate_geometry(verbose=True)
```

---

## Time Integration

### TimeStepper Usage

The `TimeStepper` class coordinates the implicit Euler time advancement with Newton iteration:

```python
from bionetflux.time_integration import TimeStepper

# Create time stepper from solver setup
time_stepper = TimeStepper(setup, verbose=True)

# Initialize solution at t=0
current_solution, current_bulk_data = time_stepper.initialize_solution()

# Advance a single time step
result = time_stepper.advance_time_step(
    current_solution=current_solution,
    current_bulk_data=current_bulk_data,
    current_time=0.0,
    dt=0.01
)

if result.converged:
    print(f"Newton converged in {result.iterations} iterations")
    current_solution = result.updated_solution
    current_bulk_data = result.updated_bulk_data
```

### Adaptive Time Stepping

```python
# Create adaptive time stepper
adaptive_stepper = time_stepper.get_adaptive_stepper(
    dt_min=1e-6, dt_max=1.0, safety_factor=0.8
)

# Advance with adaptive dt control
result, next_dt = adaptive_stepper.advance_time_step_adaptive(
    current_solution, current_bulk_data, current_time, dt_suggested=0.01
)
```

### Multiple Steps

```python
results = time_stepper.advance_multiple_steps(
    initial_solution=current_solution,
    initial_bulk_data=current_bulk_data,
    start_time=0.0,
    dt=0.01,
    n_steps=100,
    stop_on_failure=True
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

The `examples/` directory contains five example scripts:

| Script | Description |
|--------|-------------|
| `evolution_example_ks.py` | Keller-Segel time evolution with error analysis |
| `evolution_example_ks_verbose.py` | Verbose version with detailed Newton output |
| `evolution_example_ooc.py` | Organ-on-Chip time evolution |
| `evolution_maze_ooc.py` | OoC simulation on maze geometry |
| `evolution+error_example.py` | Time evolution with error computation |

### Running an Example

```bash
cd BioNetFlux
pip install -e .
python examples/evolution_example_ks.py

# With a TOML config file:
python examples/evolution_example_ks.py --config config/ks_parameters.toml
```

### Example Pattern

All examples follow this pattern:

```python
from setup_solver import quick_setup
from bionetflux.time_integration import TimeStepper
from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry
from bionetflux.core.minimal_error_evaluator import MinimalErrorEvaluator

# 1. Build geometry
geometry = build_arc_sequence_geometry(N=2, start=1.5, length=2.0)

# 2. Set up solver
setup = quick_setup(
    problem_module="bionetflux.problems.ks_problem",
    validate=True,
    config_file="config/ks_parameters.toml",
    geometry=geometry
)

# 3. Initialize time stepper
time_stepper = TimeStepper(setup, verbose=True)
current_solution, current_bulk_data = time_stepper.initialize_solution()

# 4. Time evolution loop
dt = setup.global_discretization.dt
T = setup.global_discretization.T
current_time = 0.0

while current_time < T - dt / 2:
    result = time_stepper.advance_time_step(
        current_solution, current_bulk_data, current_time, dt
    )
    if result.converged:
        current_solution = result.updated_solution
        current_bulk_data = result.updated_bulk_data
        current_time += dt
    else:
        break

# 5. Error analysis
error_evaluator = MinimalErrorEvaluator()
final_traces, _ = setup.extract_domain_solutions(current_solution)
trace_errors = error_evaluator.compute_trace_error(
    trace_solutions=final_traces,
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations,
    time=current_time
)
```

---

## API Reference

### Setup Functions

```python
bionetflux.setup_solver.quick_setup(
    problem_module: str = "bionetflux.problems.test_problem2",
    validate: bool = True,
    config_file: Optional[str] = None,
    geometry: Optional[DomainGeometry] = None
) -> SolverSetup
```

**Parameters:**
- `problem_module`: Import path to problem definition (e.g., `"bionetflux.problems.ks_problem"`)
- `validate`: Whether to validate setup after creation
- `config_file`: Optional TOML configuration file path
- `geometry`: Optional pre-built `DomainGeometry` instance

**Returns:** Configured `SolverSetup` object

### SolverSetup Class

```python
class SolverSetup:
    def __init__(self, problem_module="bionetflux.problems.ooc_problem",
                 config_file=None, geometry=None)
    def initialize()
    def get_problem_info() -> Dict[str, Any]
    def create_initial_conditions() -> Tuple[List[np.ndarray], np.ndarray]
    def create_global_solution_vector(traces, multipliers) -> np.ndarray
    def extract_domain_solutions(global_solution) -> Tuple[List[np.ndarray], np.ndarray]
    def validate_setup(verbose=False) -> bool
    def compute_geometry_from_problems(geometry_name=None) -> DomainGeometry

    # Lazy-loaded properties:
    @property elementary_matrices -> ElementaryMatrices
    @property static_condensations -> List
    @property global_assembler -> GlobalAssembler
    @property bulk_data_manager -> BulkDataManager
```

### DomainGeometry Class

```python
class DomainGeometry:
    def add_domain(extrema_start, extrema_end, domain_start=None,
                   domain_length=None, name=None, display_color="blue",
                   **metadata) -> int
    def add_connection(domain1_id, domain2_id, parameter1, parameter2=0.0,
                      **metadata) -> int
    def add_exterior_boundary(domain_id, parameter, **metadata) -> int
    def add_periodic_boundary(domain_id, parameter, **metadata) -> int
    def add_symmetry_boundary(domain_id, parameter, **metadata) -> int
    def get_domain(domain_id) -> DomainInfo
    def get_all_domains() -> List[DomainInfo]
    def find_domain_by_name(name) -> Optional[int]
    def remove_domain(domain_id)
    def get_connection(connection_id) -> ConnectionInfo
    def get_boundary_connections() -> List[ConnectionInfo]
    def get_interior_connections() -> List[ConnectionInfo]
    def num_domains() -> int
    def num_connections() -> int
    def get_bounding_box() -> Dict[str, float]
    def total_length() -> float
    def summary() -> str
    def validate_geometry(verbose=False) -> bool
```

### LeanMatplotlibPlotter Class

```python
class LeanMatplotlibPlotter:
    def __init__(self, problems, discretizations, equation_names=None,
                 figsize=(12, 8), output_dir=None)
    def plot_2d_curves(trace_solutions, title="2D Solution Curves",
                      show_bounding_box=True, show_mesh_points=True,
                      save_filename=None) -> plt.Figure
    def plot_flat_3d(trace_solutions, equation_idx=0, title=None,
                    segment_width=0.1, save_filename=None,
                    view_angle=(30, 45)) -> plt.Figure
    def plot_birdview(trace_solutions, equation_idx=0, time=0.0,
                     save_filename=None) -> plt.Figure
    def plot_comparison(initial_traces, final_traces, initial_time=0.0,
                       final_time=1.0, save_filename=None) -> plt.Figure
```

---

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you installed with `pip install -e .` from the project root. The package is `bionetflux`, not `ooc1d`.

2. **Newton solver not converging**: Try reducing `dt`, checking initial conditions, or using `solve_with_line_search()` via the `NewtonSolver`.

3. **Config compatibility error**: Ensure the TOML config file's `problem_type` matches the problem module (e.g., `"keller_segel"` for `ks_problem`, `"ooc"` for `ooc_problem`).

4. **Geometry validation failure**: Call `geometry.validate_geometry(verbose=True)` to see detailed diagnostics.

5. **Static condensation type error**: The `StaticCondensationFactory` selects implementation by `problem.problem_type`. Supported types: `"keller_segel"`, `"ooc"`. Register custom types with `StaticCondensationFactory.register_implementation()`.

6. **Backward compatibility**: The shim at `src/setup_solver.py` re-exports `SolverSetup`, `create_solver_setup`, and `quick_setup` from `bionetflux.setup_solver`, so old `from setup_solver import quick_setup` imports still work.

