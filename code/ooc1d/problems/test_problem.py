import numpy as np

from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    Single domain test problem equivalent to MATLAB TestProblem.m
    """
    # Mesh parameters
    n_elements = 3  # Number of spatial elements
    
    # Global parameters
    neq = 2
    T = 1.0
    problem_name = "TestProblem"
    
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
        return (1/nu) * k1 / (k2 + x)**2
    
    def dchi(x):
        return -(1/nu) * k1 * 2 / (k2 + x)**3
    
    # Domain definition
    domain_start = 0.0
    domain_length = 500.0  # micrometers (L in MATLAB)
    
    # Tumor source parameters (matching MATLAB)
    gamma = 1.0  # Tumor growth rate
    rho = 0.1    # Tumor decay rate
    delta1 = 50.0  # Tumor spread parameter
    delta2 = 50.0  # Initial condition spread parameter
    alpha = 5e-6   # Forcing coefficient
    
    # Create problem
    problem = Problem(
        neq=neq,
        domain_start=domain_start,
        domain_length=domain_length,
        parameters=parameters,
        problem_type="keller_segel",  # Ensure consistent type
        name="test_problem_domain"
    )
    
    # Set chemotaxis
    problem.set_chemotaxis(chi, dchi)
    
    # Tumor function - matching MATLAB definition
    def tumor(s, t):
        """Tumor source function"""
        L = domain_length
        return gamma * np.exp(-rho * t) * np.exp(-((s - 3*L/4)**2) / (2*delta1**2))
    
    # Forcing functions - matching MATLAB TestProblem
    def force_u(s, t):
        """Forcing function for u equation"""
        return 0.0 * s  # Zero forcing for u (matching 0.*s in MATLAB)
    
    def force_phi(s, t):
        """Forcing function for phi equation (tumor source)"""
        return alpha * tumor(s, t)  # alpha * tumor source
    
    problem.set_force(0, force_u)
    problem.set_force(1, force_phi)
    
    # Store tumor function in problem for potential access
    problem.tumor = tumor
    
    # Initial conditions - matching MATLAB TestProblem
    def u0_1(s, t=0):
        """Initial condition for u equation"""
        L = domain_length
        return gamma * np.exp(-((s - L/4)**2) / (2*delta2**2))
    
    def u0_2(s, t=0):
        """Initial condition for phi equation"""
        return 0.0 * s  # Zero initial condition for phi (matching 0.*s in MATLAB)
    
    problem.set_initial_condition(0, u0_1)
    problem.set_initial_condition(1, u0_2)
    
    # Boundary conditions - matching MATLAB TestProblem
    def fluxu0_1(t):
        """Boundary flux for u at left boundary"""
        return 0.0
    
    def fluxu0_2(t):
        """Boundary flux for phi at left boundary"""
        return 0.0
    
    def fluxu1_1(t):
        """Boundary flux for u at right boundary"""
        return 0.0
    
    def fluxu1_2(t):
        """Boundary flux for phi at right boundary"""
        return 0.0
    
    # Store boundary flux functions
    problem.boundary_fluxes = {
        'left': [fluxu0_1, fluxu0_2],
        'right': [fluxu1_1, fluxu1_2]
    }
    
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
    dt = 0.1  # Time step
    global_disc.set_time_parameters(dt, T)

    # Setup constraints - matching MATLAB boundary conditions
    constraint_manager = ConstraintManager()

    # Add Neumann boundary conditions (zero flux) at both ends for both equations
    constraint_manager.add_neumann(0, 0, domain_start, fluxu0_1)  # u equation at start
    constraint_manager.add_neumann(1, 0, domain_start, fluxu0_2)  # phi equation at start
    
    constraint_manager.add_neumann(0, 0, domain_start + domain_length, fluxu1_1)  # u equation at end
    constraint_manager.add_neumann(1, 0, domain_start + domain_length, fluxu1_2)  # phi equation at end
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations([discretization])

    return [problem], global_disc, constraint_manager, problem_name  # Single domain
