import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    Three domains with junction test problem equivalent to MATLAB TripleJunction.m
    Features different topology than TripleArc.
    """
    # Global parameters
    ndom = 2# Number of domains
    neq = 2
    T = 3600.0
    problem_name = "T-Junction"
    
    # Physical parameters (different for each domain)
    parameters_list = [
        np.array([900.0, 200.0, 1e-4, 0.0]),  # Domain 1: standard parameters
        np.array([1200.0, 150.0, 1e-4, 0.0]), # Domain 2: higher diffusion
    #    np.array([600.0, 250.0, 1e-4, 0.0])   # Domain 3: different parameters
    ]
    
    
    
    # Chemotaxis parameters
    k1 = 1.5
    k2 = 0.8
    
    def chi(x):
        return (1/200.0) * k1 / (k2 + x)**2  # Using reference nu
    
    def dchi(x):
        return -(1/200.0) * k1 * 2 / (k2 + x)**3
    
    # Domain definitions - junction topology
    # Domain 1: [0, 400]    - main channel
    # Domain 2: [400, 600]  - junction region  
    # Domain 3: [600, 1200] - outlet channel
    
    # Attention: the domains are described by parametric ascissa that do not take into account their relative position
    # The relative position is set in the ConditionManager below
    
    domain_starts = [-500.0, 0.0]
    domain_lengths = [1000.0, 500.0]
    
    problems = []
    discretizations = []
    
    # Create problems for each domain
    domain_names = ["inlet_channel", "junction_region", "outlet_channel"]
    
    # Set mesh-size parameters (different resolution for each domain)
    n_elements_list = [20, 30]  # Higher resolution in junction
    
    for ipb in range(ndom):
        # Create problem with domain-specific parameters
        problem = Problem(
            neq=neq,
            domain_start=domain_starts[ipb],
            domain_length=domain_lengths[ipb],
            parameters=parameters_list[ipb].copy(),
            problem_type="keller_segel",
            name=domain_names[ipb]
        )
        
        # Set chemotaxis (same for all domains)
        problem.set_chemotaxis(chi, dchi)
        
        # Source terms (different for each domain)
        alpha = 1.5e-3
        
        if ipb == 0:
            # Domain 1: inlet source
            def inlet_source(s, t):
                """Inlet source at beginning"""
                return alpha * np.exp(-((s - 50.0) / 25.0)**2) * np.exp(-t/600.0)
            
            problem.set_force(0, lambda s, t: np.zeros_like(s))
            problem.set_force(1, inlet_source)
                        
        else:  # ipb == 2
            # Domain 2: decay in outlet
            # problem.set_force(0, lambda s, t: np.zeros_like(s))
            # problem.set_force(1, lambda s, t: np.zeros_like(s))
            def decay_source(s, t):
                """Weak decay source"""
                return -0.2 * alpha * np.ones_like(s)
            
            problem.set_force(0, lambda s, t: np.zeros_like(s))
            problem.set_force(1, decay_source)
        
        # Initial conditions (different for each domain)
        if ipb == 0:
            # High initial concentration in inlet
            def initial_u(s, t=0.0):
                return 0.2 * np.ones_like(s)
            def initial_phi(s, t=0.0):
                return 0.1 * np.exp(-((s - 200.0) / 100.0)**2)
        
        else:
            # Low concentration in outlet
            def initial_u(s, t=0.0):
                return 0.05 * np.ones_like(s)
            def initial_phi(s, t=0.0):
                return np.zeros_like(s)
        
        problem.set_initial_condition(0, initial_u)
        problem.set_initial_condition(1, initial_phi)
        
        problems.append(problem)
        
        
        discretization = Discretization(
            n_elements=n_elements_list[ipb],
            domain_start=domain_starts[ipb],
            domain_length=domain_lengths[ipb],
            stab_constant=1.0
        )
        
        discretization.set_tau([1.0/discretization.element_length, 1.0])  # Set stabilization parameters for both equations

        discretizations.append(discretization)
    
    # Create global discretization with time parameters
    global_discretization = GlobalDiscretization(discretizations)
    dt = 8.0  # 10 seconds
    global_discretization.set_time_parameters(dt, T)
    
# Setup constraints - Neumann boundary conditions
    constraint_manager = ConstraintManager()
    
    # Add zero flux Neumann conditions at both extrema of domain 0
    constraint_manager.add_neumann(0, 0, domain_starts[0], lambda t: 0.0)  # u equation at domain 0 start
    constraint_manager.add_neumann(1, 0, domain_starts[0], lambda t: 0.0)  # phi equation at domain 0 start
    constraint_manager.add_neumann(0, 0, domain_starts[0] + domain_lengths[0], lambda t: 0.0)  # u equation at domain 0 end
    constraint_manager.add_neumann(1, 0, domain_starts[0] + domain_lengths[0], lambda t: 0.0)  # phi equation at domain 0 end
    
    # Add zero flux Neumann conditions at the end of domain 1
    constraint_manager.add_neumann(0, 1, domain_starts[1] + domain_lengths[1], lambda t: 0.0)  # u equation at domain 1 end
    constraint_manager.add_neumann(1, 1, domain_starts[1] + domain_lengths[1], lambda t: 0.0)  # phi equation at domain 1 end
    
    # Add Kedem-Katchalsky junction condition between domain 0 and domain 1
    permeability_u = 1.2    # Permeability for u equation
    permeability_phi = 0.9  # Permeability for phi equation
    constraint_manager.add_kedem_katchalsky(0, 0, 1, -200.0, 0.0, permeability_u)     # u equation
    constraint_manager.add_kedem_katchalsky(1, 0, 1, -200.0, 0.0, permeability_phi)  # phi equation
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)

    return problems, global_discretization, constraint_manager, problem_name
