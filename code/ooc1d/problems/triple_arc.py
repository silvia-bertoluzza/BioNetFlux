import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    Three connected domains test problem equivalent to MATLAB TripleArc.m
    """
    # Global parameters
    ndom = 3  # Number of domains
    neq = 2
    T = 3600.0
    problem_name = "TripleArc"
    
    elements_per_domain = 3  # Number of spatial elements per domain
    
    # Physical parameters (same for all domains)
    nu = 200.0
    mu = 900.0
    a = 1e-4
    b = 0.0
    
    # Parameter vector [mu, nu, a, b]
    parameters = np.array([mu, nu, a, b])
    
    # Chemotaxis parameters
    k1 = 1.0
    k2 = 1.0
    
    def chi(x):
        return (1/nu) * k1 / (k2 + x)**2
    
    def dchi(x):
        return -(1/nu) * k1 * 2 / (k2 + x)**3
    
    # Domain definitions
    # Domain 1: [0, 500]
    # Domain 2: [500, 1000] 
    # Domain 3: [1000, 1500]
    domain_starts = [0.0, 500.0, 1000.0]
    domain_lengths = [500.0, 500.0, 500.0]
    
    problems = []
    discretizations = []
    
    # Create problems for each domain
    domain_names = ["arc_domain_1", "arc_domain_2", "arc_domain_3"]
    
    for ipb in range(ndom):
        # Create problem
        problem = Problem(
            neq=neq,
            domain_start=domain_starts[ipb],
            domain_length=domain_lengths[ipb],
            parameters=parameters.copy(),
            problem_type="keller_segel",  # Ensure consistent type
            name=domain_names[ipb]
        )
        
        # Set chemotaxis (same for all domains)
        problem.set_chemotaxis(chi, dchi)
        
        # Source terms (different for each domain)
        alpha = 1e-3
        
        if ipb == 0:
            # Domain 1: tumor source at center
            def tumor_source_1(s, t):
                center = 250.0
                width = 50.0
                return alpha * np.exp(-((s - center) / width)**2)
            
            problem.set_force(0, lambda s, t: np.zeros_like(s))
            problem.set_force(1, tumor_source_1)
            
        elif ipb == 1:
            # Domain 2: no source
            problem.set_force(0, lambda s, t: np.zeros_like(s))
            problem.set_force(1, lambda s, t: np.zeros_like(s))
            
        else:  # ipb == 2
            # Domain 3: weak tumor source
            def tumor_source_3(s, t):
                center = 1250.0
                width = 100.0
                return 0.5 * alpha * np.exp(-((s - center) / width)**2)
            
            problem.set_force(0, lambda s, t: np.zeros_like(s))
            problem.set_force(1, tumor_source_3)
        
        # Initial conditions (same for all domains)
        def initial_u(s):
            return 0.1 * np.ones_like(s)  # Uniform low density
        
        def initial_phi(s):
            return np.zeros_like(s)  # No initial chemoattractant
        
        problem.set_initial_condition(0, initial_u)
        problem.set_initial_condition(1, initial_phi)
        
        # Boundary conditions
        if ipb == 0:
            # Domain 1: closed left boundary, interface right boundary
            problem.set_boundary_flux(0, lambda t: 0.0, None)  # Zero flux left
            problem.set_boundary_flux(1, lambda t: 0.0, None)  # Zero flux left
        elif ipb == ndom - 1:
            # Domain 3: interface left boundary, closed right boundary
            problem.set_boundary_flux(0, None, lambda t: 0.0)  # Zero flux right
            problem.set_boundary_flux(1, None, lambda t: 0.0)  # Zero flux right
        # Middle domains have interface boundaries on both sides (handled by interface conditions)
        
        problems.append(problem)
        
        # Create spatial discretization (no time parameters)
        discretization = Discretization(
            n_elements=elements_per_domain,  # 25 elements per domain
            domain_start=domain_starts[ipb],
            domain_length=domain_lengths[ipb],
            stab_constant=1.0
        )
        
        discretization.set_tau([1.0/discretization.element_length, 1.0])  # Set stabilization parameters for both equations

        discretizations.append(discretization)
    
    # Create global discretization with time parameters
    global_discretization = GlobalDiscretization(discretizations)
    dt = 10.0  # 10 seconds
    global_discretization.set_time_parameters(dt, T)
    
    # Interface parameters
    # kappa: interface condition types (1 = Neumann/KK, 0 = Dirichlet)
    # KK: Kedem-Katchalsky parameters
    
    # Setup constraints - boundary conditions at external boundaries
    constraint_manager = ConstraintManager()
    
    # Add zero flux Neumann conditions at start of domain 0 (left boundary)
    constraint_manager.add_neumann(0, 0, domain_starts[0], lambda t: 0.0)  # u equation at domain 0 start
    constraint_manager.add_neumann(1, 0, domain_starts[0], lambda t: 0.0)  # phi equation at domain 0 start
    
    # Add zero flux Neumann conditions at end of domain 2 (right boundary)
    constraint_manager.add_neumann(0, 2, domain_starts[2] + domain_lengths[2], lambda t: 0.0)  # u equation at domain 2 end
    constraint_manager.add_neumann(1, 2, domain_starts[2] + domain_lengths[2], lambda t: 0.0)  # phi equation at domain 2 end
    
    # Add junction conditions between domains
    # Trace continuity between domain 0 and domain 1 (both equations)
    constraint_manager.add_trace_continuity(0, 0, 1, domain_starts[0] + domain_lengths[0], domain_starts[1])  # u equation
    constraint_manager.add_trace_continuity(1, 0, 1, domain_starts[0] + domain_lengths[0], domain_starts[1])  # phi equation
    
    # Kedem-Katchalsky between domain 1 and domain 2 (both equations)
    permeability_u = 1.0   # Permeability for u equation
    permeability_phi = 0.5 # Different permeability for phi equation
    constraint_manager.add_kedem_katchalsky(0, 1, 2, domain_starts[1] + domain_lengths[1], domain_starts[2], permeability_u)     # u equation
    constraint_manager.add_kedem_katchalsky(1, 1, 2, domain_starts[1] + domain_lengths[1], domain_starts[2], permeability_phi)  # phi equation
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)

    return problems, global_discretization, constraint_manager, problem_name
