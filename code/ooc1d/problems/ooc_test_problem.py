import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    OrganOnChip test problem - Python port from MATLAB TestProblem.m
    """
    
    # Mesh parameters
    n_elements = 5  # Number of spatial elements
    
    # Global parameters
    ndom = 1
    neq = 4  # Four equations: u, omega, v, phi
    T = 1.0
    problem_name = "OrganOnChip Test Problem"
    
    # Physical parameters from MATLAB (viscosity)
    nu = 1.0
    mu = 2.0
    epsilon = 1.0
    sigma = 1.0
    
    # Reaction parameters
    a = 0.0
    c = 0.0
    
    # Coupling parameters
    b = 1.0
    d = 1.0
    chi = 1.0
    
    # Parameter vector [nu, mu, epsilon, sigma, a, b, c, d, chi]
    parameters = np.array([nu, mu, epsilon, sigma, a, b, c, d, chi])
    
    # Domain definition
    domain_start = 0.0  # A in MATLAB
    domain_length = 1.0  # L in MATLAB
    
    def constant_function(x):
        """Constant lambda function as in MATLAB"""
        return np.ones_like(x)
    
    def dlambda_function(x):
        """Derivative of constant lambda function"""
        return np.zeros_like(x)
    
    # Initial conditions - all zero as in EmptyProblem.m
    def initial_u(s, t=0.0):
        return np.zeros_like(s)
    
    def initial_omega(s, t=0.0):
        return np.zeros_like(s)
    
    def initial_v(s, t=0.0):
        return np.zeros_like(s)
    
    def initial_phi(s, t=0.0):
        return np.zeros_like(s)
    
    # Source terms - all zero as in the MATLAB files
    def force_u(s, t):
        return np.zeros_like(s)
    
    def force_omega(s, t):
        return np.zeros_like(s)
    
    def force_v(s, t):
        return np.zeros_like(s)
    
    def force_phi(s, t):
        return np.zeros_like(s)
    
    # Test case with sin initial condition for u (from TestProblem.m)
    def initial_u_test(s, t=0.0):
        return np.sin(2 * np.pi * s)
    
    # Create problem
    problem = Problem(
        neq=neq,
        domain_start=domain_start,
        domain_length=domain_length,
        parameters=parameters,
        problem_type="organ_on_chip",
        name="ooc_test"
    )
    
    # Set lambda functions using the new flexible method
    problem.set_function('lambda_function', constant_function)
    problem.set_function('dlambda_function', dlambda_function)
    
    # Set source terms for all 4 equations
    problem.set_force(0, force_u)      # u equation
    problem.set_force(1, force_omega)  # omega equation  
    problem.set_force(2, force_v)      # v equation
    problem.set_force(3, force_phi)    # phi equation
    
    # Set initial conditions
    problem.set_initial_condition(0, initial_u_test)  # Use test case with sin
    problem.set_initial_condition(1, initial_omega)
    problem.set_initial_condition(2, initial_v) 
    problem.set_initial_condition(3, initial_phi)
    
    # Discretization
    discretization = Discretization(
        n_elements=n_elements,
        domain_start=domain_start,
        domain_length=domain_length,
        stab_constant=1.0
    )
    
    # Set stabilization parameters for all 4 equations
    tau_u = 1.0 / discretization.element_length
    tau_omega = 1.0
    tau_v = 1.0  
    tau_phi = 1.0
    discretization.set_tau([tau_u, tau_omega, tau_v, tau_phi])
    
    global_disc = GlobalDiscretization([discretization])
    
    
    # Set time parameters
    dt = 0.01
    global_disc.set_time_parameters(dt, T)
    
    # Setup constraints - Neumann boundary conditions (all zero flux)
    constraint_manager = ConstraintManager()
    
    # Zero flux Neumann conditions for all equations at both boundaries
    for eq_idx in range(neq):
        constraint_manager.add_neumann(eq_idx, 0, domain_start, lambda t: 0.0)
        constraint_manager.add_neumann(eq_idx, 0, domain_start + domain_length, lambda t: 0.0)
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations([discretization])
    
    return [problem], global_disc, constraint_manager, problem_name
