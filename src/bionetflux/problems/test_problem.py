"""
Single domain Keller-Segel test problem for unit testing.

Equivalent to MATLAB TestProblem.m
"""

import numpy as np
from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.core.constraints import ConstraintManager


def create_global_framework(geometry=None, config_file=None):
    """
    Single domain test problem equivalent to MATLAB TestProblem.m
    """
    # Spatial discretization
    n_elements = 3

    # Problem parameters
    neq = 2
    T = 1.0
    problem_name = "TestProblem"

    # Physical parameters
    nu = 200.0
    mu = 900.0
    a = 0.0001
    b = 0.0

    # Parameters array: [mu, nu, a, b]
    parameters = np.array([mu, nu, a, b])

    # Chemotaxis coefficients
    k1 = 3.9e-09
    k2 = 5e-06

    def chi(x):
        return 1 / nu * k1 / (k2 + x)**2

    def dchi(x):
        return -1 / nu * k1 * 2 / (k2 + x)**3

    # Domain geometry
    domain_start = 0.0
    domain_length = 500.0

    # Tumor parameters
    gamma = 1.0
    rho = 0.1
    delta1 = 50.0
    delta2 = 50.0
    alpha = 5e-06

    # Create problem
    problem = Problem(
        neq=neq,
        domain_start=domain_start,
        domain_length=domain_length,
        parameters=parameters,
        problem_type="keller_segel",
        name="test_problem_domain",
    )

    # Set chemotaxis functions
    problem.set_chemotaxis(chi, dchi)

    # Tumor source function
    def tumor(s, t):
        """Tumor source function"""
        L = domain_length
        return gamma * np.exp(-rho * t) * np.exp(-(s - 3 * L / 4)**2 / (2 * delta1**2))

    # Forcing functions
    def force_u(s, t):
        """Forcing function for u equation"""
        return 0.0 * s

    def force_phi(s, t):
        """Forcing function for phi equation (tumor source)"""
        return alpha * tumor(s, t)

    # Set forcing
    problem.set_force(0, force_u)
    problem.set_force(1, force_phi)

    # Store tumor function on problem
    problem.tumor = tumor

    # Initial conditions
    def u0_1(s, t=0):
        """Initial condition for u equation"""
        L = domain_length
        return gamma * np.exp(-(s - L / 4)**2 / (2 * delta2**2))

    def u0_2(s, t=0):
        """Initial condition for phi equation"""
        return 0.0 * s

    # Set initial conditions
    problem.set_initial_condition(0, u0_1)
    problem.set_initial_condition(1, u0_2)

    # Boundary flux functions
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

    # Store boundary fluxes on problem
    problem.boundary_fluxes = {
        "left": [fluxu0_1, fluxu0_2],
        "right": [fluxu1_1, fluxu1_2],
    }

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

    # Set boundary conditions (all Neumann)
    constraint_manager.add_neumann(0, 0, domain_start, fluxu0_1)
    constraint_manager.add_neumann(1, 0, domain_start, fluxu0_2)
    constraint_manager.add_neumann(0, 0, domain_start + domain_length, fluxu1_1)
    constraint_manager.add_neumann(1, 0, domain_start + domain_length, fluxu1_2)

    # Map constraints to discretizations
    constraint_manager.map_to_discretizations([discretization])

    return [problem], global_disc, constraint_manager, problem_name
