import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union

# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from bionetflux.core.lean_bulk_data_manager import BulkDataManager
from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization



# Step 1: Create framework objects (problems, discretizations, static_condensations)
# problems = [...]  # List of Problem instances
# discretizations = [...]  # List of Discretization instances  
# static_condensations = [...]  # List of static condensation instances
from setup_solver import quick_setup
filename = "bionetflux.problems.reduced_ooc_problem" 
setup = quick_setup(filename, validate=True)
problems = setup.problems
discretizations = setup.global_discretization.spatial_discretizations
static_condensations = setup.static_condensations

# Step 2: Extract essential data once (memory-efficient)
domain_data_list = BulkDataManager.extract_domain_data_list(
    problems=problems,
    discretizations=discretizations,
    static_condensations=static_condensations
)

# Step 3: Create lean manager with extracted data only
lean_manager = BulkDataManager(domain_data_list)

# Step 4: Validate compatibility (optional but recommended)
if not lean_manager.test(problems, discretizations, static_condensations):
    raise RuntimeError("Framework objects incompatible with extracted data")

# Step 5: Initialize bulk data for all domains
bulk_data_list = lean_manager.initialize_all_bulk_data(
    problems=problems,
    discretizations=discretizations,
    time=0.0
)

# Step 6: Time evolution loop
dt = 0.01



for time_step in range(1):
    current_time = time_step * dt
    
    # Compute forcing terms for implicit Euler
    forcing_terms = lean_manager.compute_forcing_terms(
        bulk_data_list=bulk_data_list,
        problems=problems,
        discretizations=discretizations,
        time=current_time,
        dt=dt
    )
    
    # Initialize new primal bulk object with random entries
    
    new_bulk_data_list = []
    for i, bulk_data in enumerate(bulk_data_list):
        new_bulk_primal = lean_manager.create_bulk_data(
            domain_index=i,
            problem=problems[i],
            discretization=discretizations[i],
            dual=False
            )
    
    # Set random data for the primal bulk object
        domain_data = lean_manager.get_domain_info(i)
        random_shape = (2 * domain_data.neq, domain_data.n_elements)
        random_data = np.random.rand(*random_shape) * 0.1  # Small random values
        new_bulk_primal.set_data(random_data)
        
        new_bulk_data_list.append(new_bulk_primal)
    
    # Create new_solutions list (placeholder for actual solver results)
    new_data = [bulk_data.get_data() for bulk_data in new_bulk_data_list]
  
 # Update bulk data with new solutions
    lean_manager.update_bulk_data(bulk_data_list, new_data)
    
    # Monitor mass conservation
    current_mass = lean_manager.compute_total_mass(bulk_data_list)
    if time_step % 10 == 0:
        print(f"Time {current_time:.3f}: Mass = {current_mass:.6e}")

print("✓ Time evolution completed with lean manager")