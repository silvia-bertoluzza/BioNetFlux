# BioNetFlux Current Code Structure

*Accurate documentation of the actual codebase structure and components (updated 2026-04-03)*

## Directory Tree

```
BioNetFlux/
├── src/                               # Main source code directory
│   ├── setup_solver.py                # Backward-compatible shim (re-exports from bionetflux)
│   │
│   ├── bionetflux/                    # Core framework package (v1.0.0)
│   │   ├── __init__.py                # Package init — exports Problem, DomainGeometry, etc.
│   │   ├── setup_solver.py            # SolverSetup, quick_setup(), create_solver_setup()
│   │   │
│   │   ├── core/                      # Core mathematical components (14 files)
│   │   │   ├── problem.py             # Problem base class
│   │   │   ├── discretization.py      # Discretization, GlobalDiscretization
│   │   │   ├── constraints.py         # Constraint, ConstraintType, ConstraintManager
│   │   │   ├── bulk_data.py           # BulkData container
│   │   │   ├── lean_bulk_data_manager.py    # BulkDataManager (memory-efficient)
│   │   │   ├── lean_global_assembly.py      # GlobalAssembler
│   │   │   ├── flux_jump.py                 # Flux computation at interfaces
│   │   │   ├── domain_data.py               # DomainData container
│   │   │   ├── boundary_override.py         # BC overrides from TOML config
│   │   │   ├── minimal_error_evaluator.py   # L2 error computation
│   │   │   ├── static_condensation_base.py  # Abstract base for SC
│   │   │   ├── static_condensation_factory.py # Factory: creates KS or OoC SC
│   │   │   ├── static_condensation_keller_segel.py  # KS: P0 flux for u, P1 for phi
│   │   │   └── static_condensation_ooc.py   # OoC: 4-equation system
│   │   │
│   │   ├── geometry/                  # Geometry management
│   │   │   ├── domain_geometry.py     # DomainGeometry, DomainInfo, ConnectionInfo,
│   │   │   │                          #   build_grid_geometry(), build_arc_sequence_geometry(),
│   │   │   │                          #   create_maze_geometry()
│   │   │   ├── maze_1_data/           # CSV data for maze geometries
│   │   │   ├── maze_2_data/
│   │   │   ├── maze_3_data/
│   │   │   └── maze_4_data/
│   │   │
│   │   ├── problems/                  # Problem definitions (7 files)
│   │   │   ├── ooc_problem.py              # Organ-on-Chip 4-equation system
│   │   │   ├── ooc_config_manager.py       # OoC TOML configuration manager
│   │   │   ├── ks_problem.py               # Keller-Segel 2-equation system
│   │   │   ├── ks_config_manager.py        # KS TOML configuration manager
│   │   │   ├── custom_problem_template.py  # Template for user-defined problems
│   │   │   ├── test_problem.py             # Single-domain test problem
│   │   │   └── test_problem2.py            # Single-domain test variant
│   │   │
│   │   ├── time_integration/          # Time stepping and nonlinear solvers
│   │   │   ├── time_stepper.py        # TimeStepper, AdaptiveTimeStepper, TimeStepResult
│   │   │   └── newton_solver.py       # NewtonSolver, NewtonResult
│   │   │
│   │   ├── utils/                     # Utility modules
│   │   │   ├── config_manager.py          # BaseConfigManager, TOML loading
│   │   │   ├── elementary_matrices.py     # Reference element matrices (SymPy)
│   │   │   └── mesh_mapping.py            # Coordinate transformations
│   │   │
│   │   ├── visualization/             # Plotting
│   │   │   └── lean_matplotlib_plotter.py # LeanMatplotlibPlotter
│   │   │
│   │   └── analysis/                  # Post-processing
│   │       └── error_evaluation.py    # ErrorEvaluator, L2 error, convergence
│   │
├── config/                            # TOML configuration files
│   ├── ks_parameters.toml
│   ├── ooc_parameters.toml
│   ├── ooc_maze3_parameters.toml
│   └── ooc_maze4_parameters.toml
│
├── tests/                             # Pytest test suite (21 files)
│   ├── conftest.py
│   ├── test_core_smoke.py
│   ├── test_problem.py
│   ├── test_geometry.py
│   ├── test_bulk_data.py
│   ├── test_lean_bulk_data_manager.py
│   ├── test_lean_global_assembly.py
│   ├── test_lean_setup.py
│   ├── test_static_condensation_setup.py
│   ├── test_flux_orders.py
│   ├── test_flux_error.py
│   ├── test_constraint_override.py
│   ├── test_boundary_overrides.py
│   ├── test_domain_flux_jump.py
│   ├── test_compute_n_elements_from_h.py
│   ├── test_robin_jacobian.py
│   ├── test_mass_monitoring.py
│   ├── test_adaptive_time_stepping.py
│   ├── test_maze_geometry.py
│   └── test_sample.py
│
├── examples/                          # Example applications (5 scripts)
│   ├── evolution_example_ks.py
│   ├── evolution_example_ks_verbose.py
│   ├── evolution_example_ooc.py
│   ├── evolution_maze_ooc.py
│   ├── evolution+error_example.py
│   └── README_evolution_examples.md
│
├── docs/                              # Documentation
├── Logos/                             # Brand assets
├── pyproject.toml                     # Project metadata (Python >= 3.11, MIT license)
├── pytest.ini                         # Pytest configuration
├── README.md                          # Project overview
└── requirements.txt                   # Dependencies
```

## Implemented Components

### Core Framework (`bionetflux.core`)

**`problem.py`** — Problem base class
- Problem definition with validation, self-testing, dynamic function setting
- Key methods: `set_chemotaxis()`, `set_force()`, `set_solution()`, `set_initial_condition()`, `set_extrema()`, `validate_problem()`, `test_functions()`, `run_self_test()`
- Factory method: `Problem.create_test_problems()`

**`discretization.py`** — Spatial discretization
- `Discretization`: single-domain FEM mesh (nodes, elements, connectivity)
- `GlobalDiscretization`: multi-domain coordination + time parameters
- Helper: `compute_n_elements_from_h()`

**`constraints.py`** — Boundary and interface conditions
- `ConstraintType` enum: DIRICHLET, NEUMANN, ROBIN, TRACE_CONTINUITY, KEDEM_KATCHALSKY
- `Constraint` dataclass: per-constraint data
- `ConstraintManager`: add/manage constraints, map to discretizations

**`lean_global_assembly.py`** — Global system assembly
- `GlobalAssembler`: assembles system from all domains
- Factory method: `GlobalAssembler.from_framework_objects()`

**`lean_bulk_data_manager.py`** — Memory-efficient bulk data coordination (`BulkDataManager`)

**`flux_jump.py`** — Flux computation at domain interfaces (`domain_flux_jump()`)

**`domain_data.py`** — Lightweight per-domain data container (`DomainData`)

**`boundary_override.py`** — BC overrides loaded from TOML config files

**`minimal_error_evaluator.py`** — L2 error evaluation with Legendre quadrature (`MinimalErrorEvaluator`)

**`static_condensation_base.py`** — Abstract base class for static condensation

**`static_condensation_factory.py`** — Factory pattern: creates KS or OoC implementations (`StaticCondensationFactory`)

**`static_condensation_keller_segel.py`** — KS static condensation (P0 for u, P1 for phi)

**`static_condensation_ooc.py`** — OoC static condensation (4-equation system)

### Geometry (`bionetflux.geometry`)

**`domain_geometry.py`** — Multi-domain network geometry
- `DomainInfo` dataclass: domain coordinates, parameter space, display properties
- `ConnectionInfo` dataclass: connections between domains or boundaries
- `DomainGeometry` class: geometry container with connectivity analysis, validation
- Factory functions: `build_grid_geometry()`, `build_arc_sequence_geometry()`, `create_maze_geometry()`
- Maze data directories: CSV files for predefined maze topologies

### Problems (`bionetflux.problems`)

| File | Description |
|------|-------------|
| `ooc_problem.py` | Organ-on-Chip 4-equation system |
| `ooc_config_manager.py` | OoC TOML configuration manager |
| `ks_problem.py` | Keller-Segel 2-equation chemotaxis |
| `ks_config_manager.py` | KS TOML configuration manager |
| `custom_problem_template.py` | Template for user-defined problems |
| `test_problem.py` | Single-domain test problem |
| `test_problem2.py` | Single-domain test variant |

### Time Integration (`bionetflux.time_integration`)

**`time_stepper.py`**
- `TimeStepper`: coordinates single time step (implicit Euler + Newton)
- `AdaptiveTimeStepper`: adaptive dt control
- `TimeStepResult` dataclass: convergence info, residual history

**`newton_solver.py`**
- `NewtonSolver`: Newton-Raphson iteration with optional line search and damping
- `NewtonResult` dataclass: convergence, iteration count, residuals

### Visualization (`bionetflux.visualization`)

**`lean_matplotlib_plotter.py`** — Multi-mode plotting
- 2D curve plots (one subplot per domain)
- Flat 3D view (network segments with solution heights)
- Bird's eye view (top-down color-coded network)
- Comparison plots (initial vs final)
- Geometry with indices overlay

### Utils (`bionetflux.utils`)

- `config_manager.py` — `BaseConfigManager`, TOML file loading (`load_toml_config()`)
- `elementary_matrices.py` — HDG reference element matrices via SymPy (`ElementaryMatrices`)
- `mesh_mapping.py` — Coordinate transformations (`parametric_to_physical_mesh()`)

### Analysis (`bionetflux.analysis`)

- `error_evaluation.py` — `ErrorEvaluator` class: L2 error computation, convergence rates

## Usage Workflow

### 1. Problem setup
```python
from setup_solver import quick_setup
from bionetflux.geometry.domain_geometry import build_grid_geometry

geometry = build_grid_geometry(N=5, length=500.0)
setup = quick_setup(
    problem_module="bionetflux.problems.ooc_problem",
    validate=True,
    config_file="config/ooc_parameters.toml",
    geometry=geometry
)
```

### 2. Time evolution
```python
from bionetflux.time_integration import TimeStepper

time_stepper = TimeStepper(setup, verbose=True)
current_solution, current_bulk_data = time_stepper.initialize_solution()

result = time_stepper.advance_time_step(
    current_solution=current_solution,
    current_bulk_data=current_bulk_data,
    current_time=0.0,
    dt=0.01
)
```

### 3. Visualization
```python
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

plotter = LeanMatplotlibPlotter(
    problems=setup.problems,
    discretizations=setup.global_discretization.spatial_discretizations
)
plotter.plot_2d_curves(trace_solutions, title="Solution Profiles")
plotter.plot_birdview(trace_solutions, equation_idx=0, time=0.0)
```

## File Dependencies

```
setup_solver.py (shim)
└── bionetflux.setup_solver
    ├── bionetflux.core.discretization
    ├── bionetflux.core.constraints
    ├── bionetflux.core.lean_global_assembly
    ├── bionetflux.core.lean_bulk_data_manager
    ├── bionetflux.core.static_condensation_factory
    ├── bionetflux.utils.elementary_matrices
    └── bionetflux.geometry.domain_geometry

bionetflux.problems.*
├── bionetflux.core.problem
├── bionetflux.core.discretization
├── bionetflux.core.constraints
└── bionetflux.geometry.domain_geometry

bionetflux.time_integration.time_stepper
├── bionetflux.time_integration.newton_solver
├── bionetflux.core.lean_global_assembly
└── bionetflux.core.lean_bulk_data_manager

bionetflux.visualization.lean_matplotlib_plotter
└── (operates on problems + discretizations)
```
