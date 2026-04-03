"""
Single domain test problem equivalent to MATLAB TestGabriella1.m

Uses no chemotaxis (chi=0, dchi=0), larger number of elements, and
different initial condition shape.
"""

import numpy as np
from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.core.constraints import ConstraintManager


def create_global_framework(geometry=None, config_file=None):
    """
    Single domain test problem equivalent to MATLAB TestGabriella1.m
    """
    # Spatial discretization
    n_elements = 40

    # Problem parameters
    neq = 2
    T = 1.0
    problem_name = "TestProblem2"

    # Physical parameters
    nu = 200.0
    mu = 900.0
    a = 0.0001
    b = 0.0

    # Parameters array: [mu, nu, a, b]
    parameters = np.array([mu, nu, a, b])

    # Chemotaxis coefficients (unused but kept for reference)
    k1 = 3.9e-09
    k2 = 5e-06

    def chi(x):
        """No chemotaxis"""
        return np.zeros_like(x)

    def dchi(x):
        """No chemotaxis derivative"""
        return np.zeros_like(x)

    # Domain geometry
    domain_start = 0.0
    domain_length = 500.0

    # Create problem
    problem = Problem(
        neq=neq,
        domain_start=domain_start,
        domain_length=domain_length,
        parameters=parameters,
        problem_type="keller_segel",
        name="single_arc_1_domain",
    )

    # Set chemotaxis functions (zero chemotaxis)
    problem.set_chemotaxis(chi, dchi)

    # Tumor/source parameters
    alpha = 5e-06
    gamma = 50.0
    delta2 = 25.0

    # Temporal parameters
    rho = 0.0001
    delta1 = 10.0
    L = domain_length

    def tumor(s, t):
        """Tumor function with temporal decay and spatial Gaussian distribution"""
        center = 3 * L / 4
        temporal_decay = gamma * np.exp(-rho * t)
        spatial_distribution = np.exp(-(s - center)**2 / (2 * delta1**2))
        return temporal_decay * spatial_distribution

    # Set forcing functions
    problem.set_force(0, lambda s, t: 0.0 * s)
    problem.set_force(1, lambda s, t: alpha * tumor(s, t))

    # Initial conditions
    def initial_u(s, t=0.0):
        return gamma * np.exp(-(s - 125.0)**2 / 1250.0)

    def initial_phi(s, t=0.0):
        return np.zeros_like(s)

    # Set initial conditions
    problem.set_initial_condition(0, initial_u)
    problem.set_initial_condition(1, initial_phi)

    # Create discretization
    discretization = Discretization(
        n_elements=n_elements,
        domain_start=domain_start,
        domain_length=domain_length,
        stab_constant=1.0,
    )

    # Set stabilization parameter tau
    discretization.set_tau([1.0 / discretization.element_length, 1.0])

    # Create global discretization
    global_disc = GlobalDiscretization([discretization])

    # Time stepping
    dt = 0.1
    global_disc.set_time_parameters(dt, T)

    # Create constraint manager
    constraint_manager = ConstraintManager()

    # Set boundary conditions (all Neumann = zero flux)
    constraint_manager.add_neumann(0, 0, domain_start, lambda t: 0.0)
    constraint_manager.add_neumann(1, 0, domain_start, lambda t: 0.0)

    # Right boundary
    constraint_manager.add_neumann(0, 0, domain_start + domain_length, lambda t: 0.0)
    constraint_manager.add_neumann(1, 0, domain_start + domain_length, lambda t: 0.0)

    # Map constraints to discretizations
    constraint_manager.map_to_discretizations([discretization])

    return [problem], global_disc, constraint_manager, problem_name
