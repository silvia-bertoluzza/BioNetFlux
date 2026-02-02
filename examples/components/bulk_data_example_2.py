import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization
from bionetflux.core.bulk_data import BulkData
from bionetflux.utils.elementary_matrices import ElementaryMatrices

import numpy as np

# Setup problem and discretization
problem = Problem(
    neq=2, 
    domain_start=0.0, 
    domain_length=1.0,
    parameters=np.array([2.0, 1.0, 0.0, 1.0])
)
discretization = Discretization(n_elements=20)

# Setup for dual formulation (forcing terms)
bulk_data_dual = BulkData(problem, discretization, dual=True)

# Set forcing functions using integration
forcing_functions = [
    lambda s, t: 0.1 * np.sin(2*np.pi*s) * np.exp(-t),  # Source for u
    lambda s, t: 0.05 * np.cos(np.pi*s)                  # Source for phi
]
bulk_data_dual.set_data(forcing_functions, time=0.5)

# Check integration results
print(f"Dual formulation data range: "
      f"[{np.min(bulk_data_dual.data):.6e}, {np.max(bulk_data_dual.data):.6e}]")

ElementaryMatrices_instance = ElementaryMatrices()
mass_matrix = ElementaryMatrices_instance.get_matrix('M')

# Compute integrated mass (should represent total source)
source_mass = bulk_data_dual.compute_mass(mass_matrix)
print(f"Total integrated source: {source_mass:.6e}")