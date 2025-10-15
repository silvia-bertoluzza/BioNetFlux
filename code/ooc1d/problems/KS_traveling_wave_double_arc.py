import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    Single domain test problem equivalent to MATLAB TestGabriella1.m
    """
    # Global parameters
    ndom = 2  # Number of domains
    neq = 2
    T = 0.5
    problem_name = "Keller-Segel traveling wave -- double arc"
    
    # Mesh parameters
    n_elements = [10, 10] # Number of spatial elements
    
    
    # Physical parameters
    nu = 1.0
    mu = 2.0
    a = 0.0
    b = 1.0
    
    # Parameter vector [mu, nu, a, b]
    parameters = np.array([mu, nu, a, b])
    
    # Chemotaxis parameters
    k1 = 3.9e-9
    k2 = 5.e-6
    
    def chi(x):
        #return (1/nu) * k1 / (k2 + x)**2
        return np.ones_like(x)
    
    def dchi(x):
        #return -(1/nu) * k1 * 2 / (k2 + x)**3
        return np.zeros_like(x)
    
    def u_x(s, t):
        """
        Spatial derivative of u for chemotaxis term.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
        Returns:
            Derivative value at (s,t)
        """
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        exp_2st = np.exp(2*s - t)
        
        # numerator = (5/4) * exp_st2 * (exp_st2 - 1)**2 - (5 * exp_st2) * 2 * (exp_st2 - 1) * exp_st2 + (8 * exp_2st) * 2 * (exp_st2 - 1) * exp_st2
        # denominator = (exp_st2 - 1)**4
        numerator = 3 * exp_2st + 5 * exp_st2
        denominator = (exp_st2 - 1)**3
        
        return numerator / denominator
    
    def phi_x(s, t):
        """
        Spatial derivative of phi for chemotaxis term.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
        Returns:
            Derivative value at (s,t)
        """
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        
        return 5/4 - (2 * exp_st2) / (exp_st2 - 1)
    
    

    def solution_u(s, t):
        """
        Analytical solution function.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
            
        Returns:
            Solution value at (s,t)
        """
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        exp_2st = np.exp(2*s - t)
        
        return (5 * exp_st2) / (exp_st2 - 1) - (4 * exp_2st) / (exp_st2 - 1)**2 - 5/8

    def solution_phi(s, t):
        """
        Analytical solution function for phi (chemical concentration).
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
            
        Returns:
            Phi solution value at (s,t)
        """
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        
        return (5*s)/4 - (5*t)/8 - 2*np.log(exp_st2 - 1)

    # Domain definition
    domain_start = [1.5, 3.0]  # micrometers
    domain_length = [1.5, 1.0]  # micrometers
    
    # Initial conditions
    def initial_u(s, t=0.0):
        return solution_u(s, 0.0)  # Uniform low density
    
    def initial_phi(s, t=0.0):
        return solution_phi(s, 0.0)  # No initial chemoattractant
    
    
    problems = []
    discretizations = []
    
    # Create problem
    for ipb in range(ndom):
        if ipb == 0:
            domain_start_ipb = domain_start[0]
            domain_length_ipb = domain_length[0]
            n_elements_ipb = n_elements[0]
        else:
            domain_start_ipb = domain_start[1]
            domain_length_ipb = domain_length[1]
            n_elements_ipb = n_elements[1]
    
        problem = Problem(
            neq=neq,
            domain_start=domain_start_ipb,
            domain_length=domain_length_ipb,
            parameters=parameters,
            problem_type="keller_segel",  # Ensure consistent type
            name="traveling_wave"
        )
    
    
    
        # Set chemotaxis
        problem.set_chemotaxis(chi, dchi)
        L = domain_length_ipb  # Use domain length as L

        problem.set_force(0, lambda s, t: 0.0 * s)  # No source for u (eq 1)
        problem.set_force(0, lambda s, t: 0.0 * s)  # No source for u (eq 1)
        problem.set_force(1, lambda s, t: 0.0 * s)  # Tumor (eq 2)

        problem.set_solution(0, solution_u)
        problem.set_solution(1, solution_phi)
    
        problem.set_initial_condition(0, initial_u)
        problem.set_initial_condition(1, initial_phi)
    
        # Discretization
        discretization = Discretization(
            n_elements=n_elements_ipb,
            domain_start=domain_start_ipb,
            domain_length=domain_length_ipb,
            stab_constant=1.0
        )

        discretization.set_tau([1.0/discretization.element_length, 1.0])  # Set stabilization parameters for both equations
        
        problems.append(problem)
        discretizations.append(discretization)

    global_disc = GlobalDiscretization(discretizations)

    # Set time parameters externally
    dt = 0.05  # 10 seconds
    global_disc.set_time_parameters(dt, T)

    # Setup constraints - Neumann boundary conditions
    constraint_manager = ConstraintManager()
    
    def flux_u(s, t):
        return nu * (u_x(s, t) - chi(s) * solution_u(s, t) * phi_x(s, t))
       
       
    # Add zero flux Neumann conditions at domain start for both equations
    # constraint_manager.add_neumann(0, 0, domain_start, lambda t: 0.0)  # u equation at start
    # constraint_manager.add_neumann(1, 0, domain_start, lambda t: 0.0)  # phi equation at start
    
    # Add zero flux Neumann conditions at domain start for both equations
    constraint_manager.add_neumann(0, 0, domain_start[0], lambda t: - flux_u(domain_start[0], t))  # u equation at start
    constraint_manager.add_neumann(1, 0, domain_start[0], lambda t: - mu * phi_x(domain_start[0], t))  # phi equation at start

    # Add zero flux Neumann conditions at domain end for both equations
    constraint_manager.add_neumann(0, 1, domain_start[1] + domain_length[1], lambda t: flux_u(domain_start[1] + domain_length[1], t))  # u equation at end
    constraint_manager.add_neumann(1, 1, domain_start[1] + domain_length[1], lambda t: mu * phi_x(domain_start[1] + domain_length[1], t))  # phi equation at end

    # Add trace constraints at the interface between domains
    # constraint_manager.add_trace_continuity(0, 0, 1, domain_start[0] + domain_length[0], domain_start[1])  # u equation
    # constraint_manager.add_trace_continuity(1, 0, 1, domain_start[0] + domain_length[0], domain_start[1])  # phi equation

    permeability = 1.0  # Default permeability
    constraint_manager.add_kedem_katchalsky(0, 0, 1, domain_start[0] + domain_length[0], domain_start[1], permeability)  # u equation
    constraint_manager.add_kedem_katchalsky(1, 0, 1, domain_start[0] + domain_length[0], domain_start[1], permeability)
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)

    return problems, global_disc, constraint_manager, problem_name  # Return both domains
