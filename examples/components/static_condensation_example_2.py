import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))


from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.utils.elementary_matrices import ElementaryMatrices
from bionetflux.core.static_condensation_factory import StaticCondensationFactory
# Step 1: Create OrganOnChip problem (microfluidic device)
ooc_params = np.array([1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
# [nu, mu, epsilon, sigma, a, b, c, d, chi]

problem = Problem(
    neq=4,
    domain_start=0.0,  # A = 0 (MATLAB)
    domain_length=1.0, # L = 1 (MATLAB)
    parameters=ooc_params,
    problem_type="organ_on_chip",
    name="microfluidic_device"
)

# Set initial conditions (matching MATLAB TestProblem.m)
problem.set_initial_condition(0, lambda x, t: np.sin(2*np.pi*x))  # u
for eq_idx in [1, 2, 3]:  # omega, v, phi
    problem.set_initial_condition(eq_idx, lambda x, t: np.zeros_like(x))

# Set lambda function (matching MATLAB: constant_function)
problem.set_function('lambda_function', lambda x: np.ones_like(x))
problem.set_function('dlambda_function', lambda x: np.zeros_like(x))

# Step 2: Create discretization
discretization = Discretization(n_elements=40)  # Matching MATLAB
discretization.set_tau([1.0, 1.0, 1.0, 1.0])  # [tu, to, tv, tp]
global_disc = GlobalDiscretization([discretization])
global_disc.set_time_parameters(dt=0.01, T=0.5)  # Matching MATLAB

# Step 3: Create static condensation
elementary_matrices = ElementaryMatrices()
static_condensation = StaticCondensationFactory.create(
    problem=problem,
    global_disc=global_disc,
    elementary_matrices=elementary_matrices,
    i=0
)

print(f"Created: {type(static_condensation).__name__}")  # StaticCondensationOOC

# Step 4: Build matrices 
matrices = static_condensation.build_matrices()
required_matrices = ['M', 'T', 'B1', 'L1', 'B2', 'C2', 'L2', 'A3', 'S3', 'H3', 
                    'B4', 'C4', 'L4', 'D1', 'D2', 'Q', 'Av']
print(f"Required matrices available: {all(m in matrices for m in required_matrices)}")

# Step 5: Perform static condensation 
trace_values = np.random.rand(4 * 41)  # 4 equations, 41 nodes
rhs = np.random.rand(2 * 4 * 40)  # 4 equations, 2 DOFs per element, 40 elements

for i_element in range(40):
    local_trace_values = trace_values[4 * i_element: 4 * (i_element + 2)]
    local_rhs = rhs[8 * i_element: 8 * (i_element + 1)]

bulk_solution, flux, flux_trace, jacobian = static_condensation.static_condensation(
    local_trace_values,
    local_rhs,
    time=0.0
)

print(f"OrganOnChip static condensation results:")
print(f"  Bulk solution (U): {bulk_solution.shape}")  # Should be (8, 40)
print(f"  Flux trace (hJ): {flux_trace.shape}")  # Should be (4,) or (8,)
print(f"  Jacobian (dhJ): {jacobian.shape}")  # Should be (8, 4*41)
