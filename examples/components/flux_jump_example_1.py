import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))


from bionetflux.core.static_condensation_factory import StaticCondensationFactory
from bionetflux.core.flux_jump import domain_flux_jump
from bionetflux.core.static_condensation_factory import StaticCondensationFactory
from bionetflux.core.static_condensation_ooc import StaticCondensationOOC
from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.utils.elementary_matrices import ElementaryMatrices
import numpy as np

# Setup OrganOnChip problem (4 equations)
ooc_params = np.array([1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
problem = Problem(
    neq=4,
    domain_start=0.0,
    domain_length=1.0,
    parameters=ooc_params,
    problem_type="organ_on_chip",
    name ="OrganOnChipExample1"
)

# Setup discretization
discretization = Discretization(n_elements=10)
discretization.set_tau([1.0, 1.0, 1.0, 1.0])  # [tau_c, tau_b, tau_w, tau_s]
global_disc = GlobalDiscretization([discretization])
global_disc.set_time_parameters(dt=0.01, T=0.5)  



# Create static condensation
elementary_matrices = ElementaryMatrices()

static_condensation = StaticCondensationFactory.create(
    problem=problem,
    global_disc=global_disc,
    elementary_matrices=elementary_matrices,
    i=0
)

static_condensation.build_matrices()

# Prepare realistic input data
N = 10  # elements
neq = 4  # equations
n_nodes = N + 1  # 11 nodes

# Create trace solution (4 equations × 11 nodes = 44 values)
trace_solution = np.random.rand(neq * n_nodes, 1)

# Create forcing term (8 coefficients × 10 elements)
forcing_term = np.random.rand(2 * neq, N)

# Compute flux jump
U, F, JF = domain_flux_jump(
    trace_solution=trace_solution,
    forcing_term=forcing_term,
    problem=None,
    discretization=None,
    static_condensation=static_condensation
)

print(f"OrganOnChip flux jump computation:")
print(f"  Input: {neq} equations, {N} elements, {n_nodes} nodes")
print(f"  Output shapes:")
print(f"    Bulk solutions U: {U.shape}")     # (8, 10)
print(f"    Flux jumps F: {F.shape}")        # (44, 1)
print(f"    Jacobian JF: {JF.shape}")        # (44, 44)
print(f"  Solution statistics:")
print(f"    |U|_max = {np.max(np.abs(U)):.6e}")
print(f"    |F|_norm = {np.linalg.norm(F):.6e}")
print(f"    JF condition = {np.linalg.cond(JF):.2e}")