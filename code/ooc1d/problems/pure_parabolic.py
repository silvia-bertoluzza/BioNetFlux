import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    Single domain test problem equivalent to MATLAB TestGabriella1.m
    """
    # Mesh parameters
    n_elements = 20 # Number of spatial elements
    # Set time parameters externally
    dt = 0.05 # 10 seconds
    
    # Global parameters
    neq = 2
    T = 5
    problem_name = "Keller-Segel traveling wave"
    
    # Physical parameters
    nu = 1.0
    mu = 1.0
    a = 0.0
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
  
    def solution_u(s, t):
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
            
    
    def solution_phi(s, t):
        s = np.asarray(s)
        return np.zeros_like(s)
     
    def phi_x(s, t):
        return np.zeros_like(s)
    
    
    
    # Domain definition
    domain_start = 0.0
    domain_length = 1.0 # micrometers
    
    # Create problem
    problem = Problem(
        neq=neq,
        domain_start=domain_start,
        domain_length=domain_length,
        parameters=parameters,
        problem_type="keller_segel",  # Ensure consistent type
        name="traveling_wave"
    )
    
    
    
    # Set chemotaxis
    problem.set_chemotaxis(chi, dchi)
    
    L = domain_length  # Use domain length as L

    

    # problem.set_force(0, lambda s, t: (1 + 4 * np.pi**2 * t) * np.cos(2 * np.pi * s))  
    problem.set_force(0, lambda s, t: u_t(s, t) - nu * u_xx(s, t))  # No source for u (eq 1)
    problem.set_force(1, lambda s, t: 0.0 * s)  # No source for phi (eq 2)

    problem.set_solution(0, solution_u)
    problem.set_solution(1, solution_phi)
    
    # Initial conditions
    def initial_u(s, t=0.0):
        return solution_u(s, 0.0)  # Uniform low density
    
    def initial_phi(s, t=0.0):
        return solution_phi(s, 0.0)  # No initial chemoattractant
    
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
    
    
    global_disc.set_time_parameters(dt, T)
    
    # Setup constraints - Neumann boundary conditions
    constraint_manager = ConstraintManager()
    
    def flux_u(s, t):
        return nu * (u_x(s, t) - chi(s) * solution_u(s, t) * phi_x(s, t))
    
       
       
    # Add zero flux Neumann conditions at domain start for both equations
    # constraint_manager.add_neumann(0, 0, domain_start, lambda t: 0.0)  # u equation at start
    # constraint_manager.add_neumann(1, 0, domain_start, lambda t: 0.0)  # phi equation at start
    
    # Add zero flux Neumann conditions at domain start for both equations
    constraint_manager.add_neumann(0, 0, domain_start, lambda t: - flux_u(domain_start, t))  # u equation at start
    # constraint_manager.add_dirichlet(0, 0, domain_start, lambda t: solution_u(domain_start, t))  # u equation at start
    constraint_manager.add_neumann(1, 0, domain_start, lambda t: - mu * phi_x(domain_start, t))  # phi equation at start

    # Add zero flux Neumann conditions at domain end for both equations
    domain_end = domain_start + domain_length
    constraint_manager.add_neumann(0, 0, domain_end, lambda t: flux_u(domain_end, t))  # u equation at end
    # constraint_manager.add_dirichlet(0, 0, domain_end, lambda t: solution_u(domain_end, t))  # u equation at end
    constraint_manager.add_neumann(1, 0, domain_end, lambda t: mu * phi_x(domain_end, t))  # phi equation at end

    # Map constraints to discretizations
    constraint_manager.map_to_discretizations([discretization])
    
    # Debug: Print dt consistency check note
    
    # Plot force functions
    import matplotlib.pyplot as plt
    s_plot = np.linspace(domain_start, domain_start + domain_length, 100)
    t_plot = 0.1  # Sample time
    
    force_0 = problem.force[0](s_plot, t_plot)
    force_1 = problem.force[1](s_plot, t_plot)
    


    return [problem], global_disc, constraint_manager, problem_name  # Single domain, no interface params
