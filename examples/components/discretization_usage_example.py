import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core.discretization import Discretization, GlobalDiscretization

# Basic discretization with 20 elements
disc1 = Discretization(n_elements=20)

# Custom domain discretization
disc2 = Discretization(
    n_elements=40,
    domain_start=0.0,  
    domain_length=1.0,
    stab_constant=1.0
)

# Fine mesh discretization on different domain
disc3 = Discretization(
    n_elements=100,
    domain_start=-1.0,
    domain_length=2.0,
    stab_constant=0.5
)

disc = Discretization(n_elements=10, domain_start=0.0, domain_length=2.0)
mesh_info = disc.get_mesh_info()
print(f"Elements: {mesh_info['n_elements']}")
print(f"Nodes: {mesh_info['n_nodes']}")
print(f"Element length: {mesh_info['element_length']}")
print(f"First few nodes: {mesh_info['nodes'][:5]}")

# Keller-Segel problem (2 equations)
ks_disc = Discretization(n_elements=20)
ks_disc.set_tau([1.0, 1.0])  # [tau_u, tau_phi]

# OrganOnChip problem (4 equations)
ooc_disc = Discretization(n_elements=40)
ooc_disc.set_tau([1.0, 1.0, 1.0, 1.0])  # [tu, to, tv, tp]

# Different stabilization parameters
custom_disc = Discretization(n_elements=30)
custom_disc.set_tau([0.5, 2.0, 1.5])  # Custom values per equation

# This will raise ValueError
try:
    disc.set_tau([])  # Empty list
except ValueError as e:
    print(f"Error: {e}")
    
disc1 = Discretization(n_elements=20, domain_start=0.0, domain_length=1.0)
global_disc = GlobalDiscretization([disc1])

# Multi-domain network
main_disc = Discretization(n_elements=30, domain_start=0.0, domain_length=1.0)
branch1_disc = Discretization(n_elements=20, domain_start=1.0, domain_length=0.8)
branch2_disc = Discretization(n_elements=20, domain_start=1.0, domain_length=0.8)

multi_global_disc = GlobalDiscretization([main_disc, branch1_disc, branch2_disc])
print(f"Global discretization has {multi_global_disc.n_domains} domains.")

global_disc = GlobalDiscretization([disc1, disc2])
global_disc.set_time_parameters(dt=0.01, T=1.0)

print(f"Time step: {global_disc.dt}")
print(f"Final time: {global_disc.T}")
print(f"Number of time steps: {global_disc.n_time_steps}")
print(f"First few time points: {global_disc.time_points[:5]}")

try:
    domain_0_disc = global_disc.get_spatial_discretization(0)
    print(f"Domain 0 elements: {domain_0_disc.n_elements}")
    print(f"Domain 0 nodes: {domain_0_disc.n_nodes}")
    
    # This will raise IndexError if only 2 domains exist
    domain_5_disc = global_disc.get_spatial_discretization(5)
except IndexError as e:
    print(f"Error: {e}")
    
global_disc.set_time_parameters(dt=0.01, T=1.0)
global_info = global_disc.get_global_info()

print(f"Number of domains: {global_info['n_domains']}")
print(f"Total elements: {global_info['total_elements']}")
print(f"Total nodes: {global_info['total_nodes']}")


# Access time information
time_info = global_info['time_info']
print(f"Time step: {time_info['dt']}")

# Access spatial discretization info for each domain
for i, spatial_info in enumerate(global_info['spatial_discretizations']):
    print(f"Domain {i}: {spatial_info['n_elements']} elements")