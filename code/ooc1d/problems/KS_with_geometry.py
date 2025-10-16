import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager
from ..geometry import DomainGeometry

def create_global_framework():
    """
    Keller-Segel traveling wave problem using DomainGeometry class.
    Based on KS_traveling_wave parameters but with geometry-driven setup.
    """
    # Global parameters
    neq = 2
    T = 0.5
    dt = 0.05
    problem_name = "Keller-Segel with Geometry"
    
    # Physical parameters (same as KS_traveling_wave)
    nu = 1.0
    mu = 2.0
    a = 0.0
    b = 1.0
    parameters = np.array([mu, nu, a, b])
    
    # Chemotaxis parameters
    k1 = 3.9e-9
    k2 = 5.e-6
    
    def chi(x):
        return np.ones_like(x)  # Simplified from original
    
    def dchi(x):
        return np.zeros_like(x)
    
    # Analytical solutions (same as KS_traveling_wave)
    def solution_u(s, t):
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        exp_2st = np.exp(2*s - t)
        return (5 * exp_st2) / (exp_st2 - 1) - (4 * exp_2st) / (exp_st2 - 1)**2 - 5/8

    def solution_phi(s, t):
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        return (5*s)/4 - (5*t)/8 - 2*np.log(exp_st2 - 1)
    
    def u_x(s, t):
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        exp_2st = np.exp(2*s - t)
        numerator = 3 * exp_2st + 5 * exp_st2
        denominator = (exp_st2 - 1)**3
        return numerator / denominator
    
    def phi_x(s, t):
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        return 5/4 - (2 * exp_st2) / (exp_st2 - 1)
    
    def flux_u(s, t):
        return nu * (u_x(s, t) - chi(s) * solution_u(s, t) * phi_x(s, t))
    
    # =============================================================================
    # GEOMETRY SETUP using DomainGeometry class
    # =============================================================================
    
    # Create geometry
    geometry = DomainGeometry("KS_traveling_wave_geometry")
    
    # Add domain 1: vertical segment from (1.5, 0) to (1.5, 1.5) 
    # Parameter space: [1.5, 3.0]
    domain1_id = geometry.add_domain(
        extrema_start=(1.5, 0.0),
        extrema_end=(1.5, 1.5),
        domain_start=1.5,
        domain_length=1.5,
        name="vertical_segment",
        n_elements=10
    )
    
    # Add domain 2: horizontal segment from (1.5, 1.5) to (2.5, 1.5)
    # Parameter space: [3.0, 4.0] 
    domain2_id = geometry.add_domain(
        extrema_start=(1.5, 1.5),
        extrema_end=(2.5, 1.5),
        domain_start=3.0,
        domain_length=1.0,
        name="horizontal_segment",
        n_elements=10
    )
    
    print(f"✓ Geometry created with {geometry.num_domains()} domains:")
    print(geometry.summary())
    
    # =============================================================================
    # PROBLEM SETUP using geometry information
    # =============================================================================
    
    problems = []
    discretizations = []
    
    # Create problems for each domain using geometry
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        
        # Create problem with domain-specific parameters from geometry
        problem = Problem(
            neq=neq,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            parameters=parameters.copy(),
            problem_type="keller_segel",
            name=domain_info.name
        )
        
        # Set extrema from geometry
        problem.set_extrema(domain_info.extrema_start, domain_info.extrema_end)
        
        # Set chemotaxis and solutions (same for all domains)
        problem.set_chemotaxis(chi, dchi)
        problem.set_solution(0, solution_u)
        problem.set_solution(1, solution_phi)
        
        # Initial conditions using analytical solutions
        problem.set_initial_condition(0, lambda s, t=0.0: solution_u(s, 0.0))
        problem.set_initial_condition(1, lambda s, t=0.0: solution_phi(s, 0.0))
        
        # Source terms (zero for this analytical solution)
        problem.set_force(0, lambda s, t: 0.0 * s)
        problem.set_force(1, lambda s, t: 0.0 * s)
        
        problems.append(problem)
        
        # Create discretization using geometry metadata
        n_elements = domain_info.metadata.get('n_elements', 10)
        discretization = Discretization(
            n_elements=n_elements,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            stab_constant=1.0
        )
        discretization.set_tau([1.0/discretization.element_length, 1.0])
        discretizations.append(discretization)
    
    # Global discretization
    global_disc = GlobalDiscretization(discretizations)
    global_disc.set_time_parameters(dt, T)
    
    # =============================================================================
    # CONSTRAINTS SETUP using geometry information
    # =============================================================================
    
    constraint_manager = ConstraintManager()
    
    # Get domain information for constraint setup
    domain1_info = geometry.get_domain(0)
    domain2_info = geometry.get_domain(1)
    
    # Boundary conditions at external boundaries
    # Domain 1 start (external boundary)
    constraint_manager.add_neumann(0, 0, domain1_info.domain_start, 
                                  lambda t: -flux_u(domain1_info.domain_start, t))
    constraint_manager.add_neumann(1, 0, domain1_info.domain_start, 
                                  lambda t: -mu * phi_x(domain1_info.domain_start, t))
    
    # Domain 2 end (external boundary)  
    domain2_end = domain2_info.domain_start + domain2_info.domain_length
    constraint_manager.add_neumann(0, 1, domain2_end, 
                                  lambda t: flux_u(domain2_end, t))
    constraint_manager.add_neumann(1, 1, domain2_end, 
                                  lambda t: mu * phi_x(domain2_end, t))
    
    # Interface conditions between domains (Kedem-Katchalsky)
    interface_coord = domain1_info.domain_start + domain1_info.domain_length  # Should equal domain2_info.domain_start
    permeability = 1.0
    
    constraint_manager.add_kedem_katchalsky(0, 0, 1, interface_coord, domain2_info.domain_start, permeability)
    constraint_manager.add_kedem_katchalsky(1, 0, 1, interface_coord, domain2_info.domain_start, permeability)
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)
    
    print(f"✓ Constraints set up with interface at parameter coordinate {interface_coord}")
    print(f"✓ Geometry bounding box: {geometry.get_bounding_box()}")
    
    return problems, global_disc, constraint_manager, problem_name
