import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core.constraints import ConstraintManager, ConstraintType
from bionetflux.core.discretization import Discretization
import numpy as np

# Create constraint manager
cm = ConstraintManager()

# Add Dirichlet condition at left boundary: u = sin(t)
cm.add_dirichlet(
    equation_index=0,
    domain_index=0,
    position=0.0,
    data_function=lambda t: np.sin(2*np.pi*t)
)

# Add Neumann condition at right boundary: du/dn = 0
cm.add_neumann(
    equation_index=0,
    domain_index=0,
    position=1.0,
    data_function=lambda t: 0.0
)

# Add Robin condition for second equation: 2*phi + 0.1*dphi/dn = exp(-t)
cm.add_robin(
    equation_index=1,
    domain_index=0,
    position=1.0,
    alpha=2.0,
    beta=0.1,
    data_function=lambda t: np.exp(-t)
)

# Create discretization and map constraints
discretization = Discretization(n_elements=50, domain_start=0.0, domain_length=1.0)
cm.map_to_discretizations([discretization])

print(f"Total constraints: {cm.n_constraints}")
print(f"Total multipliers: {cm.n_multipliers}")

# Get constraint data at specific time
constraint_data = cm.get_multiplier_data(time=0.5)
print(f"Constraint data: {constraint_data}")