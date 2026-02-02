import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union


# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core.constraints import ConstraintManager, ConstraintType
from bionetflux.core.discretization import Discretization

# Three-domain network with junctions
cm = ConstraintManager()

# Domain 0: [0, 1], Domain 1: [1, 2], Domain 2: [1, 2] (Y-junction)
discretizations = [
    Discretization(n_elements=20, domain_start=0.0, domain_length=1.0),  # Main
    Discretization(n_elements=15, domain_start=1.0, domain_length=1.0),  # Branch 1
    Discretization(n_elements=15, domain_start=1.0, domain_length=1.0)   # Branch 2
]

# Inlet boundary condition (domain 0, left end)
cm.add_dirichlet(0, 0, 0.0, lambda t: 1.0 + 0.1*np.sin(t))

# Junction conditions at x = 1.0
# Continuity between main vessel and branch 1
cm.add_trace_continuity(
    equation_index=0,
    domain1_index=0,  # End of main vessel
    domain2_index=1,  # Start of branch 1
    position1=1.0,
    position2=1.0
)

# Continuity between main vessel and branch 2
cm.add_trace_continuity(
    equation_index=0,
    domain1_index=0,  # End of main vessel
    domain2_index=2,  # Start of branch 2
    position1=1.0,
    position2=1.0
)

# Outlet boundary conditions (zero Neumann)
cm.add_neumann(0, 1, 2.0, lambda t: 0.0)  # Branch 1 outlet
cm.add_neumann(0, 2, 2.0, lambda t: 0.0)  # Branch 2 outlet

# Map to discretizations
cm.map_to_discretizations(discretizations)

# Analyze constraint structure
print(f"Network constraints:")
print(f"  Total constraints: {cm.n_constraints}")
print(f"  Total multipliers: {cm.n_multipliers}")

for domain_idx in range(3):
    domain_constraints = cm.get_constraints_by_domain(domain_idx)
    print(f"  Domain {domain_idx}: {len(domain_constraints)} constraints")

boundary_constraints = cm.get_constraints_by_type(ConstraintType.DIRICHLET)
boundary_constraints += cm.get_constraints_by_type(ConstraintType.NEUMANN)
junction_constraints = cm.get_constraints_by_type(ConstraintType.TRACE_CONTINUITY)

print(f"  Boundary conditions: {len(boundary_constraints)}")
print(f"  Junction conditions: {len(junction_constraints)}")
