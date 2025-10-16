import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager
from ..geometry import DomainGeometry

def create_global_framework():
    """
    Keller-Segel problem with complex grid geometry:
    - Two parallel vertical segments at x=-0.5 and x=0.5
    - Four horizontal segments at y=0.2, 0.4, 0.6, 0.8
    - All segments connected with trace continuity constraints
    Based on KS_traveling_wave parameters.
    """
    # Global parameters
    neq = 2
    T = 0.5
    dt = 0.05
    problem_name = "Keller-Segel Grid Network"
    
    # Physical parameters (same as KS_traveling_wave)
    nu = 1.0
    mu = 2.0
    a = 0.0
    b = 1.0
    parameters = np.array([mu, nu, a, b])
    
    # Chemotaxis parameters
    k1 = 3.9e-9
    k2 = 5.e-6
    
    def chi(x):
        return np.ones_like(x)  # Simplified from original
    
    def dchi(x):
        return np.zeros_like(x)
    
    # Analytical solutions (same as KS_traveling_wave)
    def solution_u(s, t):
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        exp_2st = np.exp(2*s - t)
        return (5 * exp_st2) / (exp_st2 - 1) - (4 * exp_2st) / (exp_st2 - 1)**2 - 5/8

    def solution_phi(s, t):
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        return (5*s)/4 - (5*t)/8 - 2*np.log(exp_st2 - 1)
    
    def u_x(s, t):
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        exp_2st = np.exp(2*s - t)
        numerator = 3 * exp_2st + 5 * exp_st2
        denominator = (exp_st2 - 1)**3
        return numerator / denominator
    
    def phi_x(s, t):
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        return 5/4 - (2 * exp_st2) / (exp_st2 - 1)
    
    def flux_u(s, t):
        return nu * (u_x(s, t) - chi(s) * solution_u(s, t) * phi_x(s, t))
    
    # =============================================================================
    # COMPLEX GRID GEOMETRY SETUP
    # =============================================================================
    
    # Create geometry
    geometry = DomainGeometry("KS_grid_network")
    
    # Domain parameters - using parameter space that matches analytical solution
    param_start = 1.5
    param_step = 0.5  # Each segment gets 0.5 parameter units
    
    # Domain 0: Left vertical segment at x=-0.5, from y=0 to y=1
    # Parameter space: [1.5, 2.0]
    left_vertical_id = geometry.add_domain(
        extrema_start=(-0.5, 0.0),
        extrema_end=(-0.5, 1.0),
        domain_start=param_start,
        domain_length=param_step,
        name="left_vertical",
        n_elements=10
    )
    param_start += param_step
    
    # Domain 1: Right vertical segment at x=0.5, from y=0 to y=1  
    # Parameter space: [2.0, 2.5]
    right_vertical_id = geometry.add_domain(
        extrema_start=(0.5, 0.0),
        extrema_end=(0.5, 1.0),
        domain_start=param_start,
        domain_length=param_step,
        name="right_vertical",
        n_elements=10
    )
    param_start += param_step
    
    # Horizontal segments connecting the verticals
    horizontal_y_coords = [0.2, 0.4, 0.6, 0.8]
    horizontal_ids = []
    
    for i, y_coord in enumerate(horizontal_y_coords):
        # Parameter space: [2.5, 3.0], [3.0, 3.5], [3.5, 4.0], [4.0, 4.5]
        horizontal_id = geometry.add_domain(
            extrema_start=(-0.5, y_coord),
            extrema_end=(0.5, y_coord),
            domain_start=param_start,
            domain_length=param_step,
            name=f"horizontal_{i+1}_y{y_coord}",
            n_elements=8
        )
        horizontal_ids.append(horizontal_id)
        param_start += param_step
    
    print(f"✓ Complex grid geometry created with {geometry.num_domains()} domains:")
    print(geometry.summary())
    
    # =============================================================================
    # PROBLEM SETUP using geometry information
    # =============================================================================
    
    problems = []
    discretizations = []
    
    # Create problems for each domain using geometry
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        
        # Create problem with domain-specific parameters from geometry
        problem = Problem(
            neq=neq,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            parameters=parameters.copy(),
            problem_type="keller_segel",
            name=domain_info.name
        )
        
        # Set extrema from geometry
        problem.set_extrema(domain_info.extrema_start, domain_info.extrema_end)
        
        # Set chemotaxis and solutions (same for all domains)
        problem.set_chemotaxis(chi, dchi)
        problem.set_solution(0, solution_u)
        problem.set_solution(1, solution_phi)
        
        # Initial conditions using analytical solutions
        problem.set_initial_condition(0, lambda s, t=0.0: solution_u(s, 0.0))
        problem.set_initial_condition(1, lambda s, t=0.0: solution_phi(s, 0.0))
        
        # Source terms (zero for this analytical solution)
        problem.set_force(0, lambda s, t: 0.0 * s)
        problem.set_force(1, lambda s, t: 0.0 * s)
        
        problems.append(problem)
        
        # Create discretization using geometry metadata
        n_elements = domain_info.metadata.get('n_elements', 10)
        discretization = Discretization(
            n_elements=n_elements,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            stab_constant=1.0
        )
        discretization.set_tau([1.0/discretization.element_length, 1.0])
        discretizations.append(discretization)
    
    # Global discretization
    global_disc = GlobalDiscretization(discretizations)
    global_disc.set_time_parameters(dt, T)
    
    # =============================================================================
    # CONSTRAINTS SETUP - Complex network connections
    # =============================================================================
    
    constraint_manager = ConstraintManager()
    
    # Get domain information for constraint setup
    left_vert = geometry.get_domain(0)   # Left vertical
    right_vert = geometry.get_domain(1)  # Right vertical
    
    # External boundary conditions (Neumann at domain ends)
    # Left vertical bottom (y=0)
    constraint_manager.add_neumann(0, 0, left_vert.domain_start, 
                                  lambda t: -flux_u(left_vert.domain_start, t))
    constraint_manager.add_neumann(1, 0, left_vert.domain_start, 
                                  lambda t: -mu * phi_x(left_vert.domain_start, t))
    
    # Left vertical top (y=1)
    left_end = left_vert.domain_start + left_vert.domain_length
    constraint_manager.add_neumann(0, 0, left_end, 
                                  lambda t: flux_u(left_end, t))
    constraint_manager.add_neumann(1, 0, left_end, 
                                  lambda t: mu * phi_x(left_end, t))
    
    # Right vertical bottom (y=0)
    constraint_manager.add_neumann(0, 1, right_vert.domain_start, 
                                  lambda t: -flux_u(right_vert.domain_start, t))
    constraint_manager.add_neumann(1, 1, right_vert.domain_start, 
                                  lambda t: -mu * phi_x(right_vert.domain_start, t))
    
    # Right vertical top (y=1)  
    right_end = right_vert.domain_start + right_vert.domain_length
    constraint_manager.add_neumann(0, 1, right_end, 
                                  lambda t: flux_u(right_end, t))
    constraint_manager.add_neumann(1, 1, right_end, 
                                  lambda t: mu * phi_x(right_end, t))
    
    # Junction constraints: Connect horizontal segments to vertical segments
    # Each horizontal segment connects to both verticals at intersection points
    
    permeability = 1.0  # Perfect junction
    
    for i, horizontal_id in enumerate(horizontal_ids):
        horizontal = geometry.get_domain(horizontal_id)
        y_coord = horizontal_y_coords[i]
        
        # Calculate intersection parameter coordinates on vertical segments
        # For vertical segments, parameter coordinate corresponds to physical y-coordinate
        left_junction_param = left_vert.domain_start + y_coord * left_vert.domain_length
        right_junction_param = right_vert.domain_start + y_coord * right_vert.domain_length
        
        # Left junction: horizontal start connects to left vertical
        constraint_manager.add_trace_continuity(0, 0, horizontal_id, 
                                               left_junction_param, horizontal.domain_start)
        constraint_manager.add_trace_continuity(1, 0, horizontal_id, 
                                               left_junction_param, horizontal.domain_start)
        
        # Right junction: horizontal end connects to right vertical
        horizontal_end = horizontal.domain_start + horizontal.domain_length
        constraint_manager.add_trace_continuity(0, 1, horizontal_id, 
                                               right_junction_param, horizontal_end)
        constraint_manager.add_trace_continuity(1, 1, horizontal_id, 
                                               right_junction_param, horizontal_end)
        
        print(f"✓ Connected horizontal segment {i+1} (y={y_coord}) to verticals")
        print(f"  Left junction: vertical param {left_junction_param:.3f} ↔ horizontal param {horizontal.domain_start:.3f}")
        print(f"  Right junction: vertical param {right_junction_param:.3f} ↔ horizontal param {horizontal_end:.3f}")
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)
    
    print(f"\n✓ Complex network constraints set up:")
    print(f"  - 4 external Neumann boundary conditions")
    print(f"  - {len(horizontal_ids) * 4} trace continuity constraints at junctions")
    print(f"✓ Geometry bounding box: {geometry.get_bounding_box()}")
    
    return problems, global_disc, constraint_manager, problem_name
