import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    OrganOnChip test problem - Python port from MATLAB TestProblem.m
    """
    
    # Mesh parameters
    n_elements = 20  # Number of spatial elements
    
    # Global parameters
    ndom = 1
    neq = 4  # Four equations: u, omega, v, phi
    T = 1.0
    dt = 0.1
    problem_name = "OrganOnChip Test Problem"
    
    # Physical parameters from MATLAB (viscosity)
    nu = 1.0
    mu = 1.0
    epsilon = 1.0
    sigma = 1.0
    
    # Reaction parameters
    a = 1.0
    c = 1.0
    
    # Coupling parameters
    b = 1.0
    d = 1.0
    chi = 1.0
    # Set lambda functions using the new flexible method
    lambda_function = lambda x: 1.0/(1.0 + x**2)
    dlambda_function = lambda x: -2.0*x/(1.0 + x**2)**2
    
    
    # Parameter vector [nu, mu, epsilon, sigma, a, b, c, d, chi]
    parameters = np.array([nu, mu, epsilon, sigma, a, b, c, d, chi])
    
    # Domain definition
    domain_start = 1.0  # A in MATLAB
    domain_length = 1.0  # L in MATLAB
    
    def constant_function(x):
        """Constant lambda function as in MATLAB"""
        return np.ones_like(x)
    
    def zero_function(x):
        """Derivative of constant lambda function"""
        return np.zeros_like(x)
    
    # Initial conditions - all zero as in EmptyProblem.m
    def initial_u(s, t=0.0):
        s = np.asarray(s)
        return 0.0 * s
    
    def initial_omega(s, t=0.0):
        s = np.asarray(s)
        return np.sin(2 * np.pi * s + np.pi * t)
    
    def initial_v(s, t=0.0):
        s = np.asarray(s)
        return t * s
    
    def initial_phi(s, t=0.0):
        return s ** 2
    
    # Source terms - all zero as in the MATLAB files
    def force_u(s, t):
        s = np.asarray(s)
        return np.zeros_like(s)

    def force_omega(s, t):
        s = np.asarray(s)
        x = 2 * np.pi * s + np.pi * t
        return np.sin(x) + 4 * np.pi**2 * np.sin(x) + np.pi * np.cos(x)
    
    def force_v(s, t):
        omega_val = initial_omega(s, t)
        lambda_val = lambda_function(omega_val)
        s = np.asarray(s)
        return s + lambda_val * t * s

    def force_phi(s, t):
        s = np.asarray(s)
        return - mu * 2.0 * np.ones_like(s) + a * s**2 - b * t * s
    
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
    
    def u(s, t):
        s = np.asarray(s)   
        y = t * np.cos(2 * np.pi * s) + 1.0
        # y = np.cos(2 * np.pi * s)
        # y = t
        # y = t * s + 1.0
        return y
    
    def u_x(s, t):
        s = np.asarray(s)
        y = -2 * np.pi * t * np.sin(2 * np.pi * s)
        # y = -2 * np.pi * np.sin(2 * np.pi * s)
        # y = np.zeros_like(s)
        # y = t * np.ones_like(s)
        return y
    
    def u_t(s, t):
        s = np.asarray(s)
        y = np.cos(2 * np.pi * s)
        # y = np.zeros_like(s)
        # y = s
        # y = np.ones_like(s)
        return y
    
    def u_xx(s, t):
        s = np.asarray(s)
        y = -4 * np.pi**2 * t * np.cos(2 * np.pi * s)
        # y = -4 * np.pi**2 * np.cos(2 * np.pi * s)
        # y = np.zeros_like(s)
        return y
    
   
    problem.set_function('lambda_function', lambda_function)  
    problem.set_function('dlambda_function', dlambda_function)

    # Set source terms for all 4 equations
    problem.set_force(0, lambda s, t: force_u(s, t))      # u equation
    problem.set_force(1, lambda s, t: force_omega(s, t))  # omega equation
    problem.set_force(2, lambda s, t: force_v(s, t))      # v equation
    problem.set_force(3, lambda s, t: force_phi(s, t))    # phi equation

    # Set initial conditions
    problem.set_initial_condition(0, initial_u)  # Use test case with sin
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
    
    flux_u = lambda s, t: 0.0
    flux_omega = lambda s, t: 2 * np.pi * np.cos(2 * np.pi * s + np.pi * t) 
    flux_v = lambda s, t: t
    flux_phi = lambda s, t: 2 * s
    
    global_disc = GlobalDiscretization([discretization])
    
    
    # Set time parameters
   
    global_disc.set_time_parameters(dt, T)
    
    # Setup constraints - Neumann boundary conditions (all zero flux)
    constraint_manager = ConstraintManager()
    
    # Zero flux Neumann conditions for all equations at both boundaries
    domain_end = domain_start + domain_length
    
    constraint_manager.add_neumann(0, 0, domain_start, lambda t: -flux_u(domain_start, t))
    constraint_manager.add_neumann(0, 0, domain_end, lambda t: flux_u(domain_end, t))
    
    constraint_manager.add_neumann(1, 0, domain_start, lambda t: -flux_omega(domain_start, t))
    constraint_manager.add_neumann(1, 0, domain_end, lambda t: flux_omega(domain_end, t))

    constraint_manager.add_neumann(2, 0, domain_start, lambda t: -flux_v(domain_start, t))
    constraint_manager.add_neumann(2, 0, domain_end, lambda t: flux_v(domain_end, t))

    constraint_manager.add_neumann(3, 0, domain_start, lambda t: -flux_phi(domain_start, t))
    constraint_manager.add_neumann(3, 0, domain_end, lambda t: flux_phi(domain_end, t))
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations([discretization])
    
    return [problem], global_disc, constraint_manager, problem_name
