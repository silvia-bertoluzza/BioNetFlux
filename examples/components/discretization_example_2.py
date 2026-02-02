import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core.discretization import Discretization, GlobalDiscretization

# Create multiple spatial discretizations for network domains
main_disc = Discretization(n_elements=30, domain_start=0.0, domain_length=1.0)
branch1_disc = Discretization(n_elements=20, domain_start=1.0, domain_length=0.8)
branch2_disc = Discretization(n_elements=20, domain_start=1.0, domain_length=0.8)

# Set stabilization parameters for each domain
main_disc.set_tau([1.0, 1.0, 1.0, 1.0])     # 4-equation system
branch1_disc.set_tau([1.0, 1.0, 1.0, 1.0])  # 4-equation system  
branch2_disc.set_tau([1.0, 1.0, 1.0, 1.0])  # 4-equation system

# Create global discretization for the network
network_global_disc = GlobalDiscretization([main_disc, branch1_disc, branch2_disc])

# Set global time parameters
network_global_disc.set_time_parameters(dt=0.01, T=0.5)

# Access individual domain discretizations
domain_0 = network_global_disc.get_spatial_discretization(0)
print(f"Main domain elements: {domain_0.n_elements}")

# Get global statistics
global_info = network_global_disc.get_global_info()
print(f"Network has {global_info['n_domains']} domains")
print(f"Total elements: {global_info['total_elements']}")
print(f"Total nodes: {global_info['total_nodes']}")