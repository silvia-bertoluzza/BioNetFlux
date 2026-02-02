import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core import flux_jump
from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.utils.elementary_matrices import ElementaryMatrices
from bionetflux.core.static_condensation_factory import StaticCondensationFactory

# Step 1: Create Keller-Segel problem
problem = Problem(
    neq=2,
    domain_start=0.0,
    domain_length=1.0,
    parameters=np.array([2.0, 1.0, 0.1, 1.0]),  # [mu, nu, a, b]
    problem_type="keller_segel",
    name="chemotaxis_problem"
)

# Set chemotaxis functions
problem.set_chemotaxis(
    chi=lambda phi: np.ones_like(phi),
    dchi=lambda phi: np.zeros_like(phi)
)

# Step 2: Create discretization
discretization = Discretization(n_elements=20)
discretization.set_tau([1.0, 1.0])  # [tau_u, tau_phi]
global_disc = GlobalDiscretization([discretization])
global_disc.set_time_parameters(dt=0.01, T=0.1)

# Step 3: Create elementary matrices
elementary_matrices = ElementaryMatrices()

# Step 4: Create static condensation via factory
static_condensation = StaticCondensationFactory.create(
    problem=problem,
    global_disc=global_disc,
    elementary_matrices=elementary_matrices,
    i=0
)

print(f"Created: {type(static_condensation).__name__}")  # KellerSegelStaticCondensation

# Step 5: Build matrices
matrices = static_condensation.build_matrices()
print(f"Available matrices: {list(matrices.keys())}")

# Step 6: Perform static condensation
trace_values = np.random.rand(2 * 21)  # 2 equations, 21 nodes
rhs = np.random.rand(2 * 2 * 20)  # 2 equations, 2 DOFs per element, 20 elements

for i_element in range(20):
    local_trace = trace_values[2 * i_element: 2 * (i_element + 2)]
    local_rhs = rhs[4 * i_element: 4 * (i_element + 1)]
    bulk_solution, flux, flux_trace, jacobian = static_condensation.static_condensation(
        local_trace=local_trace,
        local_source=local_rhs
        )

print(f"Static condensation completed:")
print(f"  Bulk solution: {bulk_solution.shape}")
print(f"  Flux jump: {flux_trace.shape}")
print(f"  Jacobian: {jacobian.shape}")