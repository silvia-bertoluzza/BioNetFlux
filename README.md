# BioNetFlux

![BioNetFlux Logo](assets/bionetflux_logo.png)

**Multi-Domain Biological Network Flow Simulation Framework**

BioNetFlux is a Python computational framework for simulating biological transport phenomena on complex network geometries. It specializes in solving coupled partial differential equations on multi-domain networks with applications in chemotaxis, organ-on-chip systems, and microfluidic networks.

## Features

- 🧬 **Multi-Domain Networks**: Complex geometries with arbitrary domain connections
- 🎯 **Keller-Segel Models**: Chemotaxis and cell migration simulations  
- 🔬 **Organ-on-Chip**: Microfluidic device modeling
- 📐 **Geometry Management**: Intuitive network definition tools
- 🎨 **Advanced Visualization**: Multiple plot types for network analysis
- ⚡ **Efficient Solvers**: Newton-Raphson with static condensation
- 🔗 **Interface Conditions**: Neumann, Dirichlet, and Kedem-Katchalsky constraints

## Quick Start

```python
import sys
sys.path.insert(0, 'path/to/BioNetFlux/code')

from setup_solver import quick_setup
from ooc1d.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

# Load problem
setup = quick_setup("ooc1d.problems.KS_grid_geometry")

# Create initial conditions  
trace_solutions, multipliers = setup.create_initial_conditions()

# Visualize
plotter = LeanMatplotlibPlotter(setup.problems, setup.global_discretization.spatial_discretizations)
plotter.plot_birdview(trace_solutions, equation_idx=0, time=0.0)
plotter.show_all()
```

## Documentation

Comprehensive documentation is available in [`docs/BioNetFlux_Documentation.md`](docs/BioNetFlux_Documentation.md), including:

- Architecture overview
- Module documentation  
- Creating new problems
- Geometry definition guide
- Visualization system
- API reference
- Example applications

## Examples

- **Simple Example**: [`examples/simple_example.py`](examples/simple_example.py)
- **Grid Networks**: `ooc1d.problems.KS_grid_geometry`
- **T-Junctions**: `ooc1d.problems.T_junction`
- **Custom Geometries**: `ooc1d.problems.KS_with_geometry`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BioNetFlux
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

3. Run examples:
```bash
cd examples
python simple_example.py
```

## Project Structure

```
BioNetFlux/
├── code/
│   ├── ooc1d/
│   │   ├── core/           # Mathematical components
│   │   ├── geometry/       # Network geometry tools
│   │   ├── problems/       # Problem definitions
│   │   ├── solver/         # Numerical solvers
│   │   └── visualization/  # Plotting system
│   └── setup_solver.py    # Main interface
├── docs/                   # Documentation
├── examples/               # Example applications
└── README.md
```

## Key Components

### Geometry Module
```python
from ooc1d.geometry import DomainGeometry

geometry = DomainGeometry("network_name")
geometry.add_domain(extrema_start=(0,0), extrema_end=(1,0), name="segment1")
geometry.add_domain(extrema_start=(1,0), extrema_end=(1,1), name="segment2")
```

### Problem Definition
```python
def create_global_framework():
    # 1. Define geometry
    # 2. Create problems from geometry  
    # 3. Set up constraints
    # 4. Return framework components
    return problems, global_discretization, constraint_manager, name
```

### Visualization
```python
# Three complementary visualization modes:
plotter.plot_2d_curves()    # Domain-wise solution profiles
plotter.plot_flat_3d()      # 3D network with solution heights  
plotter.plot_birdview()     # Top-down color-coded network
```

## Applications

- **Chemotaxis Modeling**: Cell migration in response to chemical gradients
- **Microfluidics**: Flow and transport in organ-on-chip devices
- **Network Biology**: Transport on biological networks
- **Drug Delivery**: Modeling drug transport in vascular networks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[License information]

## Citation

If you use BioNetFlux in your research, please cite:

```
[Citation information]
```

## Contact

- **Issues**: Submit via GitHub Issues
- **Documentation**: See `docs/` directory  
- **Examples**: See `examples/` directory

---

**BioNetFlux Development Team**
