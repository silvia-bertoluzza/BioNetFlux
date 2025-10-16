import numpy as np
from ..core.problem import Problem
from ..core.discretization import Discretization, GlobalDiscretization
from ..core.constraints import ConstraintManager
from ..geometry import DomainGeometry

def create_global_framework():
    """
    Organ-on-Chip problem with complex grid geometry:
    - Two parallel vertical segments at x=-0.5 and x=0.5 (flow channels)
    - Four horizontal segments at y=0.2, 0.4, 0.6, 0.8 (connecting chambers)
    - All segments connected with Kedem-Katchalsky interface conditions
    Based on ooc_test_problem parameters but with grid network topology.
    """
    # Global parameters
     # Mesh parameters
    n_elements = 20  # Number of spatial elements
    
    # Global parameters
    ndom = 1
    neq = 4  # Four equations: u, omega, v, phi
    T = 1.0
    dt = 0.1
    problem_name = "OrganOnChip Grid Network"
    
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
    
    # Physical parameters (from ooc_test_problem)
    # Parameters: [D, v_flow, k_reaction, k_cell_interaction, a_reaction]
    # Note: static_condensation_ooc.py expects 5 parameters including reaction parameter at index 4
    
    # Different parameters for different domain types
    parameters_vertical = parameters   # Flow channels: higher flow
    parameters_horizontal = parameters# Culture chambers: lower flow, higher reaction
    
    # =============================================================================
    # COMPLEX GRID GEOMETRY SETUP (same as KS_grid_geometry)
    # =============================================================================
    
    # Create geometry
    geometry = DomainGeometry("OoC_grid_network")
    
    # Domain parameters - using parameter space for organ-on-chip
    param_start = 0.0
    param_step = 1.0  # Each segment gets 1.0 parameter units
    
    # Domain 0: Left vertical segment at x=-0.5, from y=0 to y=1 (inlet channel)
    # Parameter space: [0.0, 1.0]
    left_vertical_id = geometry.add_domain(
        extrema_start=(-0.5, 0.0),
        extrema_end=(-0.5, 1.0),
        domain_start=param_start,
        domain_length=param_step,
        name="inlet_channel",
        n_elements=20,
        domain_type="flow_channel"
    )
    param_start += param_step
    
    # Domain 1: Right vertical segment at x=0.5, from y=0 to y=1 (outlet channel)
    # Parameter space: [1.0, 2.0]
    right_vertical_id = geometry.add_domain(
        extrema_start=(0.5, 0.0),
        extrema_end=(0.5, 1.0),
        domain_start=param_start,
        domain_length=param_step,
        name="outlet_channel",
        n_elements=20,
        domain_type="flow_channel"
    )
    param_start += param_step
    
    # Horizontal segments connecting the verticals (culture chambers)
    horizontal_y_coords = [0.2, 0.4, 0.6, 0.8]
    horizontal_ids = []
    
    for i, y_coord in enumerate(horizontal_y_coords):
        # Parameter space: [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]
        horizontal_id = geometry.add_domain(
            extrema_start=(-0.5, y_coord),
            extrema_end=(0.5, y_coord),
            domain_start=param_start,
            domain_length=param_step,
            name=f"culture_chamber_{i+1}_y{y_coord}",
            n_elements=15,
            domain_type="culture_chamber"
        )
        horizontal_ids.append(horizontal_id)
        param_start += param_step
    
    print(f"✓ OoC grid geometry created with {geometry.num_domains()} domains:")
    print(geometry.summary())
    
    # =============================================================================
    # PROBLEM SETUP using geometry information with OoC physics
    # =============================================================================
    
    problems = []
    discretizations = []
    
    # Create problems for each domain using geometry
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        domain_type = domain_info.metadata.get('domain_type', 'flow_channel')
        
        # Select parameters based on domain type
        if domain_type == 'flow_channel':
            parameters = parameters_vertical.copy()
        else:  # culture_chamber
            parameters = parameters_horizontal.copy()
        
        # Create problem with domain-specific parameters from geometry
        problem = Problem(
            neq=neq,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            parameters=parameters,
            problem_type="organ_on_chip",
            name=domain_info.name
        )
        
        # Set extrema from geometry
        problem.set_extrema(domain_info.extrema_start, domain_info.extrema_end)
        
        # Set initial conditions based on domain type
        if domain_type == 'flow_channel':
            # Flow channels: medium nutrient levels, no cells initially
            def initial_u(s, t=0.0): return 0.5 * np.ones_like(s)  # Nutrients
            def initial_omega(s, t=0.0): return np.zeros_like(s)    # Waste products
            def initial_v(s, t=0.0): return np.zeros_like(s)        # No cells in flow channels
            def initial_phi(s, t=0.0): return np.zeros_like(s)      # No growth factors initially
        else:  # culture_chamber
            # Culture chambers: lower nutrients, some cells, growth factors
            def initial_u(s, t=0.0): return 0.3 * np.ones_like(s)  # Lower nutrients
            def initial_omega(s, t=0.0): return np.zeros_like(s)    # No waste initially
            def initial_v(s, t=0.0): return 0.1 * np.ones_like(s)  # Some cells
            def initial_phi(s, t=0.0): return 0.05 * np.ones_like(s) # Some growth factors
        
        problem.set_initial_condition(0, initial_u)
        problem.set_initial_condition(1, initial_omega)
        problem.set_initial_condition(2, initial_v)
        problem.set_initial_condition(3, initial_phi)
        
        # Source terms based on domain type and location
        # D, v_flow, k_reaction, k_cell = parameters
        
        if domain_type == 'flow_channel':
            # Flow channels: nutrient supply, waste removal
            if 'inlet' in domain_info.name:
                # Inlet: continuous nutrient supply
                def source_u(s, t): return 0.001 * np.ones_like(s)  # Nutrient supply
            else:
                # Outlet: no additional sources
                def source_u(s, t): return np.zeros_like(s)
            
            def source_omega(s, t): return np.zeros_like(s)  # No waste production in flow
            def source_v(s, t): return np.zeros_like(s)      # No cell growth in flow
            def source_phi(s, t): return np.zeros_like(s)    # No growth factor production
            
        else:  # culture_chamber
            # Culture chambers: cell metabolism and growth
            def source_u(s, t): return np.zeros_like(s)      # No external nutrient source
            def source_omega(s, t): return np.zeros_like(s)  # Waste produced by reaction term
            def source_v(s, t): return np.zeros_like(s)      # Cell growth via reaction
            
            # Growth factor production by cells (chamber-specific)
            chamber_number = int(domain_info.name.split('_')[2]) 
            production_rate = 1e-5 * chamber_number  # Increasing production in higher chambers
            def source_phi(s, t): return production_rate * np.ones_like(s)
        
        problem.set_force(0, source_u)
        problem.set_force(1, source_omega)
        problem.set_force(2, source_v)
        problem.set_force(3, source_phi)
        
        # Set organ-on-chip specific functions
        # Cell metabolism: u + v -> omega (nutrient + cells -> waste)
    
        
        problems.append(problem)
        
        # Create discretization using geometry metadata
        n_elements = domain_info.metadata.get('n_elements', 15)
        discretization = Discretization(
            n_elements=n_elements,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            stab_constant=1.0
        )
        discretization.set_tau([1.0/discretization.element_length, 1.0, 1.0, 1.0])
        discretizations.append(discretization)
    
    # Global discretization
    global_disc = GlobalDiscretization(discretizations)
    global_disc.set_time_parameters(dt, T)
    
    # =============================================================================
    # CONSTRAINTS SETUP - Complex network connections with OoC interface conditions
    # =============================================================================
    
    constraint_manager = ConstraintManager()
    
    # Get domain information for constraint setup
    left_vert = geometry.get_domain(0)   # Left vertical (inlet)
    right_vert = geometry.get_domain(1)  # Right vertical (outlet)
    
    # External boundary conditions
    # Inlet conditions (left vertical bottom): constant nutrient supply
    constraint_manager.add_neumann(0, 0, left_vert.domain_start, lambda t: 0.001)  # Nutrient flux
    constraint_manager.add_neumann(1, 0, left_vert.domain_start, lambda t: 0.0)    # No waste flux
    constraint_manager.add_neumann(2, 0, left_vert.domain_start, lambda t: 0.0)    # No cell flux
    constraint_manager.add_neumann(3, 0, left_vert.domain_start, lambda t: 0.0)    # No growth factor flux
    
    # Outlet conditions (right vertical bottom): outflow
    constraint_manager.add_neumann(0, 1, right_vert.domain_start, lambda t: 0.0)   # Zero gradient
    constraint_manager.add_neumann(1, 1, right_vert.domain_start, lambda t: 0.0)   # Zero gradient
    constraint_manager.add_neumann(2, 1, right_vert.domain_start, lambda t: 0.0)   # Zero gradient
    constraint_manager.add_neumann(3, 1, right_vert.domain_start, lambda t: 0.0)   # Zero gradient
    
    # Top boundaries: closed (no flux)
    left_end = left_vert.domain_start + left_vert.domain_length
    right_end = right_vert.domain_start + right_vert.domain_length
    
    for eq_idx in range(neq):
        constraint_manager.add_neumann(eq_idx, 0, left_end, lambda t: 0.0)   # Left top
        constraint_manager.add_neumann(eq_idx, 1, right_end, lambda t: 0.0)  # Right top
    
    # Junction constraints: Connect horizontal segments to vertical segments
    # Different permeabilities for different species
    permeability_nutrients = 0.8    # High permeability for nutrients
    permeability_waste = 0.9        # High permeability for waste
    permeability_cells = 0.1        # Low permeability for cells (size exclusion)
    permeability_growth_factors = 0.6  # Medium permeability for growth factors
    
    permeabilities = [permeability_nutrients, permeability_waste, 
                     permeability_cells, permeability_growth_factors]
    
    for i, horizontal_id in enumerate(horizontal_ids):
        horizontal = geometry.get_domain(horizontal_id)
        y_coord = horizontal_y_coords[i]
        
        # Calculate intersection parameter coordinates on vertical segments
        left_junction_param = left_vert.domain_start + y_coord * left_vert.domain_length
        right_junction_param = right_vert.domain_start + y_coord * right_vert.domain_length
        
        # Left junction: horizontal start connects to left vertical
        for eq_idx in range(neq):
            constraint_manager.add_kedem_katchalsky(eq_idx, 0, horizontal_id, 
                                                   left_junction_param, horizontal.domain_start, 
                                                   permeabilities[eq_idx])
        
        # Right junction: horizontal end connects to right vertical
        horizontal_end = horizontal.domain_start + horizontal.domain_length
        for eq_idx in range(neq):
            constraint_manager.add_kedem_katchalsky(eq_idx, 1, horizontal_id, 
                                                   right_junction_param, horizontal_end, 
                                                   permeabilities[eq_idx])
        
        print(f"✓ Connected culture chamber {i+1} (y={y_coord}) to flow channels")
        print(f"  Permeabilities: u={permeability_nutrients}, ω={permeability_waste}, v={permeability_cells}, φ={permeability_growth_factors}")
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)
    
    print(f"\n✓ OoC network constraints set up:")
    print(f"  - 8 external boundary conditions (inlet/outlet)")
    print(f"  - {len(horizontal_ids) * 8} Kedem-Katchalsky interface conditions")
    print(f"  - Species-specific permeabilities for selective transport")
    print(f"✓ Geometry bounding box: {geometry.get_bounding_box()}")
    
    return problems, global_disc, constraint_manager, problem_name
