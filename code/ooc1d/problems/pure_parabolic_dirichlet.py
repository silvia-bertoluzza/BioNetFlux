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
    
    # Global parameters
    neq = 2
    T = 0.5
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
        """
        Analytical solution function.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
            
        Returns:
            Solution value at (s,t)
        """
        s = np.asarray(s)
        
        return t * s
    
    def u_x(s, t):
        """
        Spatial derivative of u for chemotaxis term.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
        Returns:
            Derivative value at (s,t)
        """
        return t * np.ones_like(s)
    
            
    
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

        return 4 * s

    def phi_x(s, t):
        """
        Spatial derivative of phi for chemotaxis term.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
        Returns:
            Derivative value at (s,t)
        """
       
        
        return 4 * np.ones_like(s)
    
    
    
    # Domain definition
    domain_start = 1.5
    domain_length = 2.5  # micrometers
    
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

    
    problem.set_force(0, lambda s, t: s)  # No source for u (eq 1)
    problem.set_force(1, lambda s, t: 0.0 * s)  # Tumor (eq 2)

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
    
    # Set time parameters externally
    dt = 0.05  # 10 seconds
    global_disc.set_time_parameters(dt, T)
    
    # Debug: Print dt from global discretization
    print(f"DEBUG: dt set in global_disc = {global_disc.dt}")
    print(f"DEBUG: dt set locally = {dt}")

    # Setup constraints - Neumann boundary conditions
    constraint_manager = ConstraintManager()
    
    def flux_u(s, t):
        return nu * (u_x(s, t) - chi(s) * solution_u(s, t) * phi_x(s, t))
    
       
       
    # Add zero flux Neumann conditions at domain start for both equations
    # constraint_manager.add_neumann(0, 0, domain_start, lambda t: 0.0)  # u equation at start
    # constraint_manager.add_neumann(1, 0, domain_start, lambda t: 0.0)  # phi equation at start
    
    # Add zero flux Neumann conditions at domain start for both equations
    constraint_manager.add_dirichlet(0, 0, domain_start, lambda t: solution_u(domain_start, t))  # u equation at start
    constraint_manager.add_neumann(1, 0, domain_start, lambda t: - mu * phi_x(domain_start, t))  # phi equation at start

    # Add zero flux Neumann conditions at domain end for both equations
    domain_end = domain_start + domain_length
    constraint_manager.add_dirichlet(0, 0, domain_end, lambda t: solution_u(domain_end, t))  # u equation at end
    constraint_manager.add_neumann(1, 0, domain_end, lambda t: mu * phi_x(domain_end, t))  # phi equation at end

    # Map constraints to discretizations
    constraint_manager.map_to_discretizations([discretization])
    
    # Debug: Print dt consistency check note
    print(f"DEBUG: Remember to check dt consistency with static_condensation objects in outer script")
    
    # Plot force functions
    import matplotlib.pyplot as plt
    s_plot = np.linspace(domain_start, domain_start + domain_length, 100)
    t_plot = 0.1  # Sample time
    
    force_0 = problem.force[0](s_plot, t_plot)
    force_1 = problem.force[1](s_plot, t_plot)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(s_plot, force_0, 'b-', linewidth=2)
    plt.title('Force Function 0 (u equation)')
    plt.xlabel('Position s')
    plt.ylabel('Force value')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(s_plot, force_1, 'r-', linewidth=2)
    plt.title('Force Function 1 (phi equation)')
    plt.xlabel('Position s')
    plt.ylabel('Force value')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Force Functions at t={t_plot}')
    plt.tight_layout()
    plt.show()

    return [problem], global_disc, constraint_manager, problem_name  # Single domain, no interface params
