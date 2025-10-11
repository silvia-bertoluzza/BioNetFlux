import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager

def create_global_framework():
    """
    Single domain test problem equivalent to MATLAB TestGabriella2.m
    Variant of TestGabriella1 with different parameters or setup.
    """
    # Global parameters
    neq = 2
    T = 1800.0  # Shorter simulation time
    problem_name = "SingleArc2"
    
    # Physical parameters - different from TestGabriella1
    nu = 150.0  # Different diffusion for u
    mu = 800.0  # Different diffusion for phi
    a = 2e-4    # Different reaction parameter
    b = 0.0
    
    # Parameter vector [mu, nu, a, b]
    parameters = np.array([mu, nu, a, b])
    
    # Chemotaxis parameters - different sensitivity
    k1 = 2.0
    k2 = 0.5
    
    def chi(x):
        return (1/nu) * k1 / (k2 + x)**2
    
    def dchi(x):
        return -(1/nu) * k1 * 2 / (k2 + x)**3
    
    # Domain definition - different size
    domain_start = 0.0
    domain_length = 800.0  # Smaller domain
    
    # Create problem
    problem = Problem(
        neq=neq,
        domain_start=domain_start,
        domain_length=domain_length,
        parameters=parameters,
        problem_type="keller_segel",
        name="test_gabriella2_domain"
    )
    
    # Set chemotaxis
    problem.set_chemotaxis(chi, dchi)
    
    # Source terms - different source location
    alpha = 2e-3  # Stronger source
    def tumor_source(s, t):
        """Localized tumor source at different location"""
        center = 200.0  # Different center
        width = 30.0    # Narrower source
        return alpha * np.exp(-((s - center) / width)**2)
    
    problem.set_force(0, lambda s, t: np.zeros_like(s))  # No source for u
    problem.set_force(1, tumor_source)  # Tumor source for phi
    
    # Initial conditions - different initial distribution
    def initial_u(s):
        return 0.05 * np.ones_like(s)  # Lower initial density
    
    def initial_phi(s):
        # Small initial perturbation
        return 0.01 * np.exp(-((s - 400.0) / 100.0)**2)
    
    problem.set_initial_condition(0, initial_u)
    problem.set_initial_condition(1, initial_phi)
    
    # Discretization
    discretization = Discretization(
        n_elements=3,  # Different mesh resolution
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