import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization
from bionetflux.core.bulk_data import BulkData
import numpy as np

# Setup problem and discretization
problem = Problem(
    neq=2, 
    domain_start=0.0, 
    domain_length=1.0,
    parameters=np.array([2.0, 1.0, 0.0, 1.0])
)
discretization = Discretization(n_elements=20)

# Create BulkData instance (primal formulation)
bulk_data = BulkData(problem, discretization, dual=False)

# Method 1: Set from functions
initial_conditions = [
    lambda s, t: np.sin(np.pi * s),      # u equation
    lambda s, t: np.exp(-s) * np.cos(t)  # phi equation
]
bulk_data.set_data(initial_conditions, time=0.0)

# Method 2: Set from direct array
coeffs = np.random.rand(4, 20)  # 2*neq=4, n_elements=20
bulk_data.set_data(coeffs)

# Method 3: Set from trace values
trace_vals = np.random.rand(42)  # neq*(n_elements+1) = 2*21 = 42
bulk_data.set_data(trace_vals)

# Access data
data_array = bulk_data.get_data()
element_5_data = bulk_data.get_element_data(5)


# Compute mass
from bionetflux.utils.elementary_matrices import ElementaryMatrices
elementary_matrices = ElementaryMatrices()
mass_matrix = elementary_matrices.get_matrix('M')
total_mass = bulk_data.compute_mass(mass_matrix)

# Validate instance
is_valid = bulk_data.test()
print(f"BulkData validation: {is_valid}")