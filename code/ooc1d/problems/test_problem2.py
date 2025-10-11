import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    Single domain test problem equivalent to MATLAB TestGabriella1.m
    """
    # Mesh parameters
    n_elements = 40 # Number of spatial elements
    
    # Global parameters
    neq = 2
    T = 1.0
    problem_name = "TestProblem2"
    
    # Physical parameters
    nu = 200.0
    mu = 900.0
    a = 1e-4
    b = 0.0
    
    # Parameter vector [mu, nu, a, b]
    parameters = np.array([mu, nu, a, b])
    
    # Chemotaxis parameters
    k1 = 3.9e-9
    k2 = 5.e-6
    
    def chi(x):
        #return (1/nu) * k1 / (k2 + x)**2
        return np.zeros_like(x)
    
    def dchi(x):
        #return -(1/nu) * k1 * 2 / (k2 + x)**3
        return np.zeros_like(x)

    # Domain definition
    domain_start = 0.0
    domain_length = 500.0  # micrometers
    
    # Create problem
    problem = Problem(
        neq=neq,
        domain_start=domain_start,
        domain_length=domain_length,
        parameters=parameters,
        problem_type="keller_segel",  # Ensure consistent type
        name="single_arc_1_domain"
    )
    
    
    
    # Set chemotaxis
    problem.set_chemotaxis(chi, dchi)
    
    # Source terms
    alpha = 5e-6
    gamma = 50.0
    delta2 = 25.0
    
    # Additional parameters for tumor source term
    rho = 1e-4  # Decay rate parameter
    delta1 = 10.0  # Spatial spread parameter for tumor
    L = domain_length  # Use domain length as L

    def tumor(s, t):
        """Tumor function with temporal decay and spatial Gaussian distribution"""
        center = 3 * L / 4  # Centered at 3L/4
        temporal_decay = gamma * np.exp(-rho * t)
        spatial_distribution = np.exp(-((s - center)**2) / (2 * delta1**2))
        return temporal_decay * spatial_distribution
    
    problem.set_force(0, lambda s, t: 0.0 * s)  # No source for u (eq 1)
    problem.set_force(1, lambda s, t: alpha * tumor(s, t))  # Tumor (eq 2)

    # Initial conditions
    def initial_u(s, t=0.0):
        return gamma * np.exp(-(s - 125.0)**2 / (2 * 25.0**2))  # Uniform low density
    
    def initial_phi(s, t=0.0):
        return np.zeros_like(s)  # No initial chemoattractant
    
    problem.set_initial_condition(0, initial_u)
    problem.set_initial_condition(1, initial_phi)
    
    # Discretization
    discretization = Discretization(
        n_elements=n_elements,
        domain_start=domain_start,
        domain_length=domain_length,
        stab_constant=1.0
    )

    discretization.set_tau([1.0/discretization.element_length, 1.0])  # Set stabilization parameters for both equations

    global_disc = GlobalDiscretization([discretization])
    
    # Set time parameters externally
    dt = 0.1  # 10 seconds
    global_disc.set_time_parameters(dt, T)

    # Setup constraints - Neumann boundary conditions
    constraint_manager = ConstraintManager()
    
    # Add zero flux Neumann conditions at domain start for both equations
    constraint_manager.add_neumann(0, 0, domain_start, lambda t: 0.0)  # u equation at start
    constraint_manager.add_neumann(1, 0, domain_start, lambda t: 0.0)  # phi equation at start
    
    # Add zero flux Neumann conditions at domain end for both equations  
    constraint_manager.add_neumann(0, 0, domain_start + domain_length, lambda t: 0.0)  # u equation at end
    constraint_manager.add_neumann(1, 0, domain_start + domain_length, lambda t: 0.0)  # phi equation at end
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations([discretization])

    return [problem], global_disc, constraint_manager, problem_name  # Single domain, no interface params
