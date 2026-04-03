# BioNetFlux Proposed Clean Code Structure

> **STATUS NOTE (2026-04-03):** The foundational migration described here — renaming
> `code/` → `src/` and `ooc1d/` → `bionetflux/` — has been **completed**. The codebase
> now lives at `src/bionetflux/` with all imports using `bionetflux.*`. Tests are in
> `tests/`, examples in `examples/`.
>
> The remaining proposals below (hierarchical `models/` directory, `solvers/` rename,
> expanded test organization, `networks/` subpackage, animation/export utilities, etc.)
> are **still aspirational** and have not been implemented.

*A cleaner, more organized structure for improved maintainability and development workflow*

## Proposed Directory Organization

```
BioNetFlux/
├── 📁 src/                            # ✅ DONE (was 'code/')
│   ├── 📁 bionetflux/                 # ✅ DONE (was 'ooc1d/')
│   │   ├── 📄 __init__.py             # Package initialization with version
│   │   │
│   │   ├── 📁 core/                   # Core mathematical framework
│   │   │   ├── 📄 __init__.py         # Core exports
│   │   │   ├── 📄 problem.py          # Problem definition class
│   │   │   ├── 📄 discretization.py  # Finite element discretization
│   │   │   ├── 📄 constraints.py     # Boundary/interface constraints
│   │   │   ├── 📄 static_condensation.py  # Static condensation (renamed)
│   │   │   └── 📄 bulk_data.py        # Bulk solution management
│   │   │
│   │   ├── 📁 geometry/               # Geometry management
│   │   │   ├── 📄 __init__.py         # Geometry exports
│   │   │   ├── 📄 domain_geometry.py  # Multi-domain geometry
│   │   │   ├── 📄 network_topology.py # Network analysis tools
│   │   │   └── 📄 mesh_generation.py  # Future mesh generation
│   │   │
│   │   ├── 📁 models/                 # Problem definitions (renamed from 'problems/')
│   │   │   ├── 📄 __init__.py         # Model registry
│   │   │   ├── 📁 keller_segel/       # Keller-Segel models
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 basic_ks.py     # Basic KS model
│   │   │   │   ├── 📄 traveling_wave.py  # Analytical solution
│   │   │   │   ├── 📄 ks_geometry.py  # KS with custom geometry
│   │   │   │   └── 📄 ks_grid.py      # KS on grid networks
│   │   │   ├── 📁 organ_on_chip/      # Organ-on-chip models
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 basic_ooc.py    # Basic 4-equation OoC
│   │   │   │   ├── 📄 ooc_grid.py     # OoC on grid networks
│   │   │   │   └── 📄 barrier_models.py # Blood-brain barrier
│   │   │   └── 📁 networks/           # Network topologies
│   │   │       ├── 📄 __init__.py
│   │   │       ├── 📄 simple_networks.py  # Linear, T-junction
│   │   │       ├── 📄 grid_networks.py    # Grid topologies
│   │   │       └── 📄 complex_networks.py # Branching, star
│   │   │
│   │   ├── 📁 solvers/                # Numerical methods (renamed from 'solver/')
│   │   │   ├── 📄 __init__.py         # Solver exports
│   │   │   ├── 📄 global_assembler.py # Global system assembly
│   │   │   ├── 📄 newton_raphson.py   # Newton-Raphson solver
│   │   │   ├── 📄 time_integration.py # Time stepping methods
│   │   │   └── 📄 linear_solvers.py   # Linear algebra backends
│   │   │
│   │   ├── 📁 visualization/          # Plotting and visualization
│   │   │   ├── 📄 __init__.py         # Viz exports
│   │   │   ├── 📄 matplotlib_plotter.py # Main plotter (renamed)
│   │   │   ├── 📄 network_plots.py    # Network-specific plots
│   │   │   ├── 📄 animation.py        # Time-series animation
│   │   │   └── 📄 export_utils.py     # Data export utilities
│   │   │
│   │   └── 📁 utils/                  # Utilities and helpers
│   │       ├── 📄 __init__.py         # Utils exports
│   │       ├── 📄 io_handlers.py      # File I/O operations
│   │       ├── 📄 validation.py       # Common validation functions
│   │       ├── 📄 logging_config.py   # Logging configuration
│   │       └── 📄 performance.py      # Performance monitoring
│   │
│   └── 📄 setup_solver.py             # Main solver interface (keep at src level)
│
├── 📁 tests/                          # Comprehensive test suite
│   ├── 📄 conftest.py                 # Pytest configuration
│   ├── 📄 __init__.py                 # Test package
│   │
│   ├── 📁 unit/                       # Unit tests
│   │   ├── 📄 test_problem.py         # Problem class tests
│   │   ├── 📄 test_geometry.py        # Geometry tests
│   │   ├── 📄 test_discretization.py  # Discretization tests
│   │   ├── 📄 test_constraints.py     # Constraints tests
│   │   ├── 📄 test_visualization.py   # Plotting tests
│   │   └── 📄 test_utils.py           # Utilities tests
│   │
│   ├── 📁 integration/                # Integration tests
│   │   ├── 📄 test_solver_pipeline.py # End-to-end solver tests
│   │   ├── 📄 test_model_loading.py   # Model loading tests
│   │   └── 📄 test_geometry_solver.py # Geometry-solver integration
│   │
│   ├── 📁 models/                     # Model-specific tests
│   │   ├── 📄 test_keller_segel.py    # KS model tests
│   │   ├── 📄 test_organ_on_chip.py   # OoC model tests
│   │   └── 📄 test_networks.py        # Network topology tests
│   │
│   └── 📁 performance/                # Performance benchmarks
│       ├── 📄 benchmark_solvers.py    # Solver performance
│       ├── 📄 benchmark_geometry.py   # Geometry operations
│       └── 📄 memory_profiling.py     # Memory usage analysis
│
├── 📁 examples/                       # Usage examples and tutorials
│   ├── 📄 __init__.py                 # Examples package
│   │
│   ├── 📁 basic/                      # Basic usage examples
│   │   ├── 📄 simple_keller_segel.py  # Basic KS setup
│   │   ├── 📄 simple_organ_on_chip.py # Basic OoC setup
│   │   └── 📄 custom_geometry.py      # Custom network creation
│   │
│   ├── 📁 advanced/                   # Advanced examples
│   │   ├── 📄 complex_networks.py     # Multi-domain networks
│   │   ├── 📄 parameter_studies.py    # Parameter sensitivity
│   │   ├── 📄 custom_models.py        # Creating new models
│   │   └── 📄 visualization_gallery.py # Plotting examples
│   │
│   ├── 📁 tutorials/                  # Step-by-step tutorials
│   │   ├── 📄 01_getting_started.py   # First steps
│   │   ├── 📄 02_geometry_basics.py   # Geometry creation
│   │   ├── 📄 03_model_setup.py       # Model configuration
│   │   ├── 📄 04_solving_systems.py   # Running simulations
│   │   └── 📄 05_visualization.py     # Result analysis
│   │
│   └── 📁 case_studies/               # Real-world applications
│       ├── 📄 microfluidic_chip.py    # Microfluidics case study
│       ├── 📄 neural_networks.py      # Neural network modeling
│       └── 📄 vascular_networks.py    # Vascular system modeling
│
├── 📁 outputs/                        # Generated outputs (git-ignored)
│   ├── 📁 plots/                      # Generated plots
│   │   ├── 📁 2d_curves/             # 2D curve plots
│   │   ├── 📁 3d_networks/           # 3D network visualizations
│   │   ├── 📁 animations/            # Time-series animations
│   │   └── 📁 comparisons/           # Comparison plots
│   │
│   ├── 📁 data/                       # Simulation data
│   │   ├── 📁 solutions/             # Solution data files
│   │   ├── 📁 meshes/                # Generated meshes
│   │   ├── 📁 parameters/            # Parameter studies
│   │   └── 📁 benchmarks/            # Performance data
│   │
│   └── 📁 reports/                    # Generated reports
│       ├── 📁 test_reports/          # Test coverage reports
│       ├── 📁 performance/           # Performance analysis
│       └── 📁 validation/            # Model validation results
│
├── 📁 docs/                           # Documentation
│   ├── 📄 index.md                   # Documentation home
│   ├── 📄 installation.md            # Installation guide
│   ├── 📄 quickstart.md              # Quick start guide
│   │
│   ├── 📁 user_guide/                # User documentation
│   │   ├── 📄 overview.md            # Framework overview
│   │   ├── 📄 geometry_guide.md      # Geometry creation guide
│   │   ├── 📄 model_guide.md         # Model definition guide
│   │   ├── 📄 solver_guide.md        # Solver configuration
│   │   └── 📄 visualization_guide.md # Visualization guide
│   │
│   ├── 📁 api/                        # API documentation
│   │   ├── 📄 core.md                # Core module API
│   │   ├── 📄 geometry.md            # Geometry API
│   │   ├── 📄 models.md              # Models API
│   │   ├── 📄 solvers.md             # Solvers API
│   │   └── 📄 visualization.md       # Visualization API
│   │
│   ├── 📁 theory/                     # Mathematical background
│   │   ├── 📄 keller_segel.md        # KS theory
│   │   ├── 📄 organ_on_chip.md       # OoC theory
│   │   ├── 📄 numerical_methods.md   # Numerical methods
│   │   └── 📄 network_analysis.md    # Network theory
│   │
│   ├── 📁 development/               # Developer documentation
│   │   ├── 📄 contributing.md        # Contribution guide
│   │   ├── 📄 code_style.md          # Coding standards
│   │   ├── 📄 testing_guide.md       # Testing practices
│   │   └── 📄 release_process.md     # Release workflow
│   │
│   └── 📁 assets/                     # Documentation assets
│       ├── 📁 images/                # Documentation images
│       ├── 📁 logos/                 # Brand assets
│       └── 📁 templates/             # Document templates
│
├── 📁 scripts/                        # Utility scripts
│   ├── 📄 setup_dev_env.py           # Development environment setup
│   ├── 📄 run_tests.py               # Test runner script
│   ├── 📄 generate_docs.py           # Documentation generator
│   ├── 📄 benchmark_suite.py         # Performance benchmark runner
│   └── 📄 clean_outputs.py           # Output cleanup utility
│
├── 📁 config/                         # Configuration files
│   ├── 📄 logging.yaml               # Logging configuration
│   ├── 📄 pytest.ini                 # Pytest configuration
│   ├── 📄 coverage.rc                # Coverage configuration
│   └── 📄 performance.yaml           # Performance test config
│
├── 📄 pyproject.toml                 # Modern Python packaging
├── 📄 requirements.txt               # Dependencies
├── 📄 requirements-dev.txt           # Development dependencies
├── 📄 README.md                      # Project overview
├── 📄 CHANGELOG.md                   # Version history
├── 📄 LICENSE                        # License information
└── 📄 .gitignore                     # Git ignore rules
```

## Key Improvements in the Proposed Structure

### 🏗️ Better Organization

| Aspect | Current | Proposed | Benefits |
|--------|---------|----------|----------|
| **Package Name** | `ooc1d` | `bionetflux` | More descriptive, matches project name |
| **Source Location** | `code/` | `src/` | Standard Python convention |
| **Problem Organization** | Flat `problems/` | Hierarchical `models/` | Better categorization |
| **Test Structure** | Mixed with source | Dedicated `tests/` | Clear separation, better CI/CD |
| **Output Management** | Scattered | Organized `outputs/` | Clean working directory |

### 📁 Modular Structure

#### Models Organization
```
models/
├── keller_segel/          # All KS variants
├── organ_on_chip/         # All OoC variants
└── networks/              # Reusable topologies
```

#### Test Organization
```
tests/
├── unit/                  # Individual component tests
├── integration/           # Component interaction tests
├── models/                # Model-specific tests
└── performance/           # Benchmarks and profiling
```

#### Examples Structure
```
examples/
├── basic/                 # Simple usage patterns
├── advanced/              # Complex scenarios
├── tutorials/             # Learning materials
└── case_studies/          # Real applications
```

### 🎯 Output Management

#### Organized Outputs
```
outputs/                   # Git-ignored, organized results
├── plots/                 # All visualizations
├── data/                  # Simulation results
└── reports/               # Analysis reports
```

### 📚 Documentation Structure

#### Comprehensive Docs
```
docs/
├── user_guide/            # How to use the framework
├── api/                   # Technical reference
├── theory/                # Mathematical background  
├── development/           # Contributing guidelines
└── assets/                # Supporting materials
```

## Migration Benefits

### For Developers
- **Clearer module boundaries** - Easy to find and modify components
- **Better testing workflow** - Separated unit, integration, and performance tests
- **Improved CI/CD** - Standard structure enables automated testing and deployment
- **Enhanced discoverability** - Logical organization of models and examples

### For Users
- **Easier learning curve** - Progressive examples from basic to advanced
- **Better documentation** - Organized by user journey and technical depth
- **Cleaner working directory** - All outputs organized and git-ignored
- **More intuitive imports** - `from bionetflux.models.keller_segel import ...`

### For Maintenance
- **Reduced complexity** - Each module has clear responsibilities
- **Better version control** - Logical grouping reduces merge conflicts
- **Easier extensibility** - Clear patterns for adding new components
- **Professional packaging** - Modern Python packaging standards

## File Naming Conventions

### Consistent Naming
- **Modules**: `snake_case.py` for all Python files
- **Packages**: `snake_case/` for directories
- **Classes**: `PascalCase` within files
- **Functions**: `snake_case` within files

### Descriptive Names
- `matplotlib_plotter.py` instead of `lean_matplotlib_plotter.py`
- `static_condensation.py` instead of `static_condensation_ooc.py`  
- `time_integration.py` instead of `time_integrator.py`

## Import Structure

### Current (Complex)
```python
from bionetflux.problems.KS_grid_geometry import create_global_framework
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
```

### Proposed (Clean)
```python
from bionetflux.models.keller_segel import ks_grid
from bionetflux.visualization import MatplotlibPlotter
```

## Configuration Management

### Centralized Configuration
- **`config/`**: All configuration files
- **`pyproject.toml`**: Modern Python packaging
- **Environment-specific**: Development vs production configs
- **Validation**: Configuration validation utilities

This proposed structure provides a solid foundation for scaling the BioNetFlux framework while maintaining clarity and ease of use for both developers and end users.
