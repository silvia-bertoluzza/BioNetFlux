import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core.bulk_data import BulkData
from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization

# Create problem and discretization
problem = Problem(neq=2, domain_start=0.0, domain_length=1.0)
discretization = Discretization(n_elements=10)

# Primal formulation (default)
bulk_data_primal = BulkData(problem, discretization, dual=False)

# Dual formulation for forcing terms
bulk_data_dual = BulkData(problem, discretization, dual=True)

neq = problem.neq
n_elements = discretization.n_elements

# Shape: (2*neq, n_elements)
coeffs = np.random.rand(2*neq, n_elements)
bulk_data_primal.set_data(coeffs)
bulk_data_dual.set_data(coeffs)

# List of neq functions f(s,t) -> scalar or array
functions = [
    lambda s, t: np.sin(np.pi * s),           # u equation
    lambda s, t: np.cos(np.pi * s) * np.exp(-t)  # phi equation
]
bulk_data_primal.set_data(functions, time=0.5)
bulk_data_dual.set_data(functions, time=0.5)

# Size: neq*(n_elements+1) - trace values at all nodes
trace_values = np.random.rand(neq * (n_elements + 1))
bulk_data_primal.set_data(trace_values)
bulk_data_dual.set_data(trace_values)

data_copy = bulk_data_primal.get_data()
print(f"Data shape: {data_copy.shape}")
print(f"Data range: [{np.min(data_copy):.6e}, {np.max(data_copy):.6e}]")

data_copy = bulk_data_dual.get_data()
print(f"Data shape: {data_copy.shape}")
print(f"Data range: [{np.min(data_copy):.6e}, {np.max(data_copy):.6e}]")

# Get coefficients for first element
element_0_data = bulk_data_primal.get_element_data(0)
print(f"Element 0 coefficients: {element_0_data}")

# Get coefficients for last element
last_element_data = bulk_data_dual.get_element_data(bulk_data_dual.n_elements - 1)
print(f"Last element coefficients: {last_element_data}")