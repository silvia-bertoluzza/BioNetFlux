#!/usr/bin/env python3
"""
Custom Problem Template
Based on MATLAB TestProblem.m structure with same physical parameters as OoC_grid_new.

This template extracts all the physical parameters and mathematical functions
from the MATLAB reference, with custom grid geometry specified.
"""

import numpy as np
from typing import Optional

from bionetflux.core.problem import Problem
from bionetflux.core.discretization import Discretization, GlobalDiscretization
from bionetflux.core.constraints import ConstraintManager
from bionetflux.geometry.domain_geometry import DomainGeometry, EXTERIOR_BOUNDARY, build_grid_geometry
from bionetflux.problems.ooc_config_manager import OoCConfigManager

# sys.path hack — commented out, use pip install -e . instead
# import sys, os
# try:
#     from ..core.problem import Problem
#     ...
# except ImportError:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     src_dir = os.path.join(current_dir, '..', '..')
#     sys.path.insert(0, src_dir)
#     from bionetflux.core.problem import Problem
#     ...


def build_default_geometry():
    """
    Build the default OoC grid geometry with vertical segments and horizontal connectors.
    
    Returns:
        DomainGeometry: Default grid geometry instance
    """
    return build_grid_geometry()

def build_T_junction_geometry():
    """
    Build the T-junction OoC grid geometry, suitably scaled for replicating the T-junction tests.
    
    Returns:
        DomainGeometry: T-junction grid geometry instance
    """
    # TODO: Implement T-junction geometry
    domain_starts = [-500.0, 0.0]
    domain_lengths = [1000.0, 500.0]

    geometry = DomainGeometry(name="T_junction")

    geometry.add_domain(
        name="main_channel",
        domain_start=domain_starts[0],
        domain_length=domain_lengths[0],
        extrema_start=(0.0, -500.0),
        extrema_end=(0.0, 500.0)
    )

    geometry.add_domain(
        name="branch",
        domain_start=domain_starts[1],
        domain_length=domain_lengths[1],
        extrema_start=(0.0, 0.0),
        extrema_end=(500.0, 0.0)
    )

    geometry.add_connection(
        domain1_id=0,
        domain2_id=1,
        parameter1=0.0,  # Midpoint of main channel (the parametrization starts at -500 and ends at 500)
        parameter2=0.0,  # Start of branch
        connection_type="continuity"
    )

    geometry.add_connection(
        domain1_id=0,
        domain2_id=EXTERIOR_BOUNDARY,
        parameter1=-500.0,  # Start of main channel
        boundary_condition="Neumann"
    )

    geometry.add_connection(
        domain1_id=0,
        domain2_id=EXTERIOR_BOUNDARY,
        parameter1=500.0,  # End of main channel
        boundary_condition="Neumann"
    )

    geometry.add_connection(
        domain1_id=1,
        domain2_id=EXTERIOR_BOUNDARY,
        parameter1=0.0,  # Start of branch
        boundary_condition="Neumann"
    )
   
    return geometry


def setup_constraints_from_geometry(geometry: DomainGeometry, problems, neq: int) -> ConstraintManager:
    """
    Generate constraint manager from geometry connections.
    
    Args:
        geometry: DomainGeometry instance with connections
        problems: List of Problem instances
        neq: Number of equations per domain
    
    Returns:
        ConstraintManager with appropriate constraints
    """
    print("Setting up constraints from geometry connections...")
    
    constraint_manager = ConstraintManager()
    
    # Get all connections
    boundary_connections = geometry.get_boundary_connections()
    interior_connections = geometry.get_interior_connections()
    
    print(f"  Processing {len(boundary_connections)} boundary connections...")
    
    # Add homogeneous Neumann boundary conditions for all boundary connections
    for conn in boundary_connections:
        domain_idx = conn.domain1_id
        parameter = conn.parameter1
        boundary_type = conn.get_boundary_type()
        
        if conn.is_exterior_boundary():
            # Add homogeneous Neumann BC for all equations at this boundary point
            for eq_idx in range(neq):
                constraint_manager.add_neumann(
                    equation_index=eq_idx,
                    domain_index=domain_idx,
                    position=parameter,
                    data_function=lambda t: 0.0  # Homogeneous Neumann: zero flux
                )
            print(f"    Added homogeneous Neumann BC: domain {domain_idx} @ {parameter:.3f} ({boundary_type})")
    
    print(f"  Processing {len(interior_connections)} interior connections...")
    
    # Add trace continuity constraints for all interior connections
    for conn in interior_connections:
        domain1_idx = conn.domain1_id
        domain2_idx = conn.domain2_id
        parameter1 = conn.parameter1
        parameter2 = conn.parameter2
        
        # Add trace continuity for all equations
        for eq_idx in range(neq):
            try:
                constraint_manager.add_trace_continuity(
                    equation_index=eq_idx,
                    domain1_index=domain1_idx,
                    domain2_index=domain2_idx,
                    position1=parameter1,
                    position2=parameter2
                )
            except Exception as e:
                print(f"    ERROR adding trace continuity eq {eq_idx}, domains {domain1_idx}↔{domain2_idx}: {e}")
        
        print(f"    Added trace continuity: domain {domain1_idx}@{parameter1:.3f} ↔ domain {domain2_idx}@{parameter2:.3f}")
    
    # Calculate constraint statistics
    total_boundary_conditions = len(boundary_connections) * neq
    total_continuity_conditions = len(interior_connections) * neq
    
    print(f"✓ Constraint setup completed from geometry:")
    print(f"  - Boundary conditions: {total_boundary_conditions} ({len(boundary_connections)} locations × {neq} equations)")
    print(f"  - Continuity conditions: {total_continuity_conditions} ({len(interior_connections)} connections × {neq} equations)")
    print(f"  - Total constraints: {total_boundary_conditions + total_continuity_conditions}")
    
    return constraint_manager


def create_global_framework(geometry: Optional[DomainGeometry] = None,
                          config_file: Optional[str] = None):
    """
    Custom problem with grid geometry following MATLAB TestProblem.m structure.
    
    Physical parameters and functions match OoC_grid_new exactly.
    Custom grid geometry with vertical segments and horizontal connectors.
    
    Args:
        geometry: Optional pre-defined DomainGeometry instance. If None, creates default geometry.
        config_file: Optional TOML configuration file path. If None, uses hardcoded defaults.
    
    Returns:
        Tuple: (problems, global_discretization, constraint_manager, problem_name)
        
    Raises:
        ValueError: If config file problem_type doesn't match 'ooc'
    """
    
    # Define this module's config validation type (different from problem_type for StaticCondensation!)
    CONFIG_VALIDATION_TYPE = "ooc"  # For config file validation
    PROBLEM_TYPE = "organ_on_chip"  # For StaticCondensationFactory and other core systems
    
    # ============================================================================
    # SECTION 1: CONFIGURATION LOADING (NEW TOML-BASED APPROACH)
    # ============================================================================
    
    print("Loading Organ-on-Chip configuration...")
    
    # Load configuration using OoC config manager (includes type validation)
    config_manager = OoCConfigManager()
    try:
        config = config_manager.load_config(config_file)
    except ValueError as e:
        # Re-raise with module context
        raise ValueError(f"OoC problem module: {e}")
    
    # Additional explicit check for extra safety (config uses 'ooc', not 'organ_on_chip')
    if config_file:
        config_problem_type = config.get('problem', {}).get('problem_type', None)
        if config_problem_type and config_problem_type != CONFIG_VALIDATION_TYPE:
            raise ValueError(
                f"OoC problem module expects config problem_type='{CONFIG_VALIDATION_TYPE}', "
                f"but config file specifies '{config_problem_type}'. "
                f"Please use an OoC-compatible configuration file."
            )
    
    # Extract configuration sections
    problem_config = config['problem']
    time_params = config['time_parameters']
    phys_params = config['physical_parameters']
    disc_params = config['discretization']
    initial_conditions = config['initial_conditions']  # Already resolved to callables
    force_functions = config['force_functions']        # Already resolved to callables
    
    # Extract domain-specific overrides (as strings, not resolved yet)
    domain_initial_conditions = config.get('domain_initial_conditions', {})
    domain_force_functions = config.get('domain_force_functions', {})
    
    print(f"Domain-specific initial conditions found: {len(domain_initial_conditions)}")
    for key, value in domain_initial_conditions.items():
        print(f"  {key} = {value}")
    print(f"Domain-specific force functions found: {len(domain_force_functions)}")
    for key, value in domain_force_functions.items():
        print(f"  {key} = {value}")

    # Global problem configuration
    neq = problem_config['neq']
    problem_name = problem_config['name']
    
    # Time discretization parameters
    T = time_params['T']
    dt = time_params['dt']
    
    # Physical parameters (extracted from TOML or defaults)
    viscosity = phys_params['viscosity']
    reaction = phys_params['reaction'] 
    coupling = phys_params['coupling']
    chemotaxis_params = phys_params['chemotaxis']
    tumor_suppression_params = phys_params['tumor_suppression']
    
    nu = viscosity['nu']
    mu = viscosity['mu']
    epsilon = viscosity['epsilon']
    sigma = viscosity['sigma']
    
    a = reaction['a']
    c = reaction['c']
    
    b = coupling['b']
    d = coupling['d']
    
    # Chemotaxis parameters
    k1 = chemotaxis_params['k1']
    k2 = chemotaxis_params['k2']
    
    m1 = tumor_suppression_params['m1']
    m2 = tumor_suppression_params['m2']
    
    # Combine into parameter array (matches MATLAB order)
    parameters = np.array([nu, mu, epsilon, sigma, a, b, c, d, 1.0])
    
    print(f"✓ Configuration loaded:")
    print(f"  Problem: {problem_name} ({neq} equations)")
    print(f"  Time: T={T}, dt={dt}")
    print(f"  Viscosity: nu={nu}, mu={mu}, epsilon={epsilon}, sigma={sigma}")
    print(f"  Reactions: a={a}, c={c}")
    print(f"  Coupling: b={b}, d={d}")
    print(f"  Chemotaxis: type={chemotaxis_params['type']}, k1={k1}, k2={k2}")
    
    # ============================================================================
    # SECTION 2: MATHEMATICAL FUNCTIONS (From configuration or defaults)
    # ============================================================================
    
    print("Setting up mathematical functions...")

    # Chemotaxis sensitivity function: chi(x) = k1 / (k2 + x)^2
    # and its derivative: dchi(x) = -2 * k1 / (k2 + x)^3
    # Constructed from TOML parameters k1, k2
    chi_func = lambda x: k1 / (k2 + x)**2
    dchi_func = lambda x: -2.0 * k1 / (k2 + x)**3
    
    lambda_func = lambda omega: m1 / (m2 + omega)  # Tumor suppression function 
    dlambda_func = lambda omega: -m1 / (m2 + omega)**2  # Derivative of tumor suppression function

    print("✓ Mathematical functions configured:")
    print(f"  - Chemotaxis: chi(x) = k1/(k2+x)^2 with k1={k1}, k2={k2}")
    print("  - Initial conditions: loaded from config")
    print("  - Force functions: loaded from config")
    
    # ============================================================================
    # SECTION 3: GEOMETRY HANDLING
    # ============================================================================
    
    if geometry is not None:
        # Use provided geometry
        print(f"Using provided geometry: '{geometry.name}' with {geometry.num_domains()} domains")
        print(f"  - Connections: {geometry.num_connections()}")
        
        # Validate provided geometry
        if not geometry.validate_geometry(verbose=True):
            print("⚠️  Warning: Provided geometry validation failed")
        else:
            print("✓ Provided geometry validation passed")
            
    else:
        # Create default custom grid geometry with connections
        geometry = build_default_geometry()
        
        # Validate default geometry
        if not geometry.validate_geometry(verbose=True):
            print("⚠️  Warning: Default geometry validation failed")
        else:
            print("✓ Default geometry validation passed")
    
    # ============================================================================
    # SECTION 4: PROBLEM CREATION (Geometry-based, with config parameters)
    # ============================================================================
    
    print("Creating problem instances from geometry with config parameters...")
    
    # Validate function callability before creating problems
    print("  Validating function callability...")
    function_checks = [
        ('initial_conditions["u"]', initial_conditions.get('u')),
        ('initial_conditions["omega"]', initial_conditions.get('omega')),
        ('initial_conditions["v"]', initial_conditions.get('v')),
        ('initial_conditions["phi"]', initial_conditions.get('phi')),
        ('force_functions["u"]', force_functions.get('u')),
        ('force_functions["omega"]', force_functions.get('omega')),
        ('force_functions["v"]', force_functions.get('v')),
        ('force_functions["phi"]', force_functions.get('phi')),
        ('chi_func', chi_func),
        ('dchi_func', dchi_func),
    ]
    
    validation_errors = []
    for func_name, func in function_checks:
        if func is None:
            validation_errors.append(f"{func_name} is None")
        elif not callable(func):
            validation_errors.append(f"{func_name} is not callable (type: {type(func).__name__})")
        else:
            print(f"    ✓ {func_name} is callable")
    
    if validation_errors:
        error_msg = "Function validation failed:\n" + "\n".join(f"    - {err}" for err in validation_errors)
        raise ValueError(error_msg)
    
    print("  ✓ All required functions are callable")
    
    problems = []
    discretizations = []
    
    # Extract discretization parameters
    n_elements = disc_params['n_elements']
    tau_values = disc_params['tau']
    
    # Apply config parameters to all domains based on geometry
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        print(f"  Creating problem for domain {domain_id}: {domain_info.name}")
        
        # Create problem for this domain with config parameters
        problem = Problem(
            neq=neq,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            parameters=parameters,  # From config
            problem_type=PROBLEM_TYPE,  # Use "organ_on_chip" for StaticCondensationFactory!
            name=f"{problem_name}_{domain_info.name}"
        )
        
        # Set DEFAULT initial conditions from config (already resolved to callables)
        problem.set_initial_condition(0, initial_conditions['u'])
        problem.set_initial_condition(1, initial_conditions['omega'])
        problem.set_initial_condition(2, initial_conditions['v'])
        problem.set_initial_condition(3, initial_conditions['phi'])
        
        # Set DEFAULT force functions from config (already resolved to callables)
        problem.set_force(0, force_functions['u'])
        problem.set_force(1, force_functions['omega'])
        problem.set_force(2, force_functions['v'])
        problem.set_force(3, force_functions['phi'])
        
        # Set chemotaxis sensitivity from config parameters
        problem.set_chemotaxis(chi_func, dchi_func)
        
        # Set tumor suppression function from config parameters
        problem.set_function('lambda_function', lambda_func)  
        problem.set_function('dlambda_function', dlambda_func)
        
        
        # Set 2D coordinates for visualization from geometry
        problem.set_extrema(domain_info.extrema_start, domain_info.extrema_end)
        
        problems.append(problem)
        
        # Create discretization for this domain with config parameters
        discretization = Discretization(
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            n_elements=n_elements  # From config
        )

        # Set stabilization parameters from config
        discretization.set_tau(tau_values)
        discretizations.append(discretization)

    # Apply domain-specific initial condition overrides
    equation_names = ['u', 'omega', 'v', 'phi']
    
    print("Applying domain-specific initial conditions...")
    for domain_eq_key, func_name in domain_initial_conditions.items():
        try:
            # Parse "domain_<id>_<equation_name>"
            if domain_eq_key.startswith('domain_') and '_' in domain_eq_key[7:]:
                parts = domain_eq_key[7:].split('_', 1)  # Remove "domain_" prefix and split once
                domain_id = int(parts[0])
                equation_name = parts[1]
                
                if domain_id < len(problems) and equation_name in equation_names:
                    equation_id = equation_names.index(equation_name)
                    # Resolve function name to callable
                    resolved_func = config_manager.function_resolver.resolve_function(func_name)
                    problems[domain_id].set_initial_condition(equation_id, resolved_func)
                    print(f"  ✓ Set initial condition: domain {domain_id}, equation '{equation_name}' -> {func_name}")
                else:
                    print(f"  ⚠️  Warning: Invalid domain {domain_id} or equation '{equation_name}' for key {domain_eq_key}")
        except (ValueError, IndexError) as e:
            print(f"  ❌ Error: Could not parse domain-specific initial condition key '{domain_eq_key}': {e}")
    
    # Apply domain-specific force function overrides
    print("Applying domain-specific force functions...")
    for domain_eq_key, func_name in domain_force_functions.items():
        try:
            # Parse "domain_<id>_<equation_name>"
            if domain_eq_key.startswith('domain_') and '_' in domain_eq_key[7:]:
                parts = domain_eq_key[7:].split('_', 1)  # Remove "domain_" prefix and split once
                domain_id = int(parts[0])
                equation_name = parts[1]
                
                if domain_id < len(problems) and equation_name in equation_names:
                    equation_id = equation_names.index(equation_name)
                    # Resolve function name to callable
                    resolved_func = config_manager.function_resolver.resolve_function(func_name)
                    problems[domain_id].set_force(equation_id, resolved_func)
                    print(f"  ✓ Set force function: domain {domain_id}, equation '{equation_name}' -> {func_name}")
                else:
                    print(f"  ⚠️  Warning: Invalid domain {domain_id} or equation '{equation_name}' for key {domain_eq_key}")
        except (ValueError, IndexError) as e:
            print(f"  ❌ Error: Could not parse domain-specific force function key '{domain_eq_key}': {e}")

    print(f"✓ Created {len(problems)} problem instances with config parameters")
    
    # ============================================================================
    # SECTION 5: CONSTRAINT SETUP
    # ============================================================================
    
    constraint_manager = setup_constraints_from_geometry(geometry, problems, neq)
    
    # Apply TOML-based boundary condition overrides (if any)
    boundary_overrides = config.get('boundary_conditions', {})
    if boundary_overrides:
        boundary_point_map = geometry.get_global_metadata().get('boundary_point_map', {})
        if not boundary_point_map:
            print("  ⚠️  Warning: boundary_conditions specified but no boundary_point_map "
                  "in geometry metadata (only available for maze geometries).")
        else:
            from bionetflux.core.boundary_override import apply_boundary_overrides
            apply_boundary_overrides(
                constraint_manager,
                boundary_overrides,
                boundary_point_map,
                equation_names,
                function_resolver=config_manager.function_resolver,
            )
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)
    
    # ============================================================================
    # SECTION 6: GLOBAL DISCRETIZATION AND FINALIZATION
    # ============================================================================
    
    print("Finalizing global discretization...")
    
    global_discretization = GlobalDiscretization(
        spatial_discretizations=discretizations,
    )
    
    global_discretization.set_time_parameters(dt, T)
    print(f"✓ Global discretization created:")
    print(f"  Time: dt={dt}, T={T}, steps={int(T/dt)}")
    print(f"  Domains: {len(discretizations)}")
    print(f"  Constraints: {constraint_manager.n_constraints}")
    print(f"  Multipliers: {constraint_manager.n_multipliers}")
    
    # ============================================================================
    # SECTION 7: VALIDATION AND RETURN
    # ============================================================================
    
    print("Validating problem setup...")
    
    # Validate each problem
    for i, problem in enumerate(problems):
        if not problem.validate_problem(verbose=False):
            print(f"⚠️  Warning: Problem {i} validation failed")
        else:
            print(f"✓ Problem {i} validated")
    
    print(f"\n🎉 OoC problem '{problem_name}' ready!")
    print(f"📊 PROBLEM SUMMARY:")
    print(f"   - Geometry: {geometry.name} with {geometry.num_domains()} domains")
    print(f"   - Physics: OrganOnChip parameters from config") 
    print(f"   - {neq} equations per domain ({neq * len(problems)} total equations)")
    print(f"   - {constraint_manager.n_constraints} constraints")
    print(f"   - Total DOFs: {sum(p.neq * (d.n_elements + 1) for p, d in zip(problems, global_discretization.spatial_discretizations))}")
    
    return problems, global_discretization, constraint_manager, problem_name


def main():
    """Test the grid problem."""
    print("="*60)
    print("REDUCED OOC GRID PROBLEM")
    print("="*60)
    print("Custom grid geometry with MATLAB TestProblem.m physics")
    print()
    
    try:
        problems, global_disc, constraints, name = create_global_framework()
        print(f"\n✅ Grid problem creation successful!")
        print(f"   Problem: {name}")
        print(f"   Domains: {len(problems)}")
        print(f"   Equations per domain: {problems[0].neq if problems else 0}")
        print(f"   Total DOFs: {sum(p.neq * (d.n_elements + 1) for p, d in zip(problems, global_disc.spatial_discretizations))}")
        
        # Additional geometry analysis
        print(f"\n📊 GEOMETRY ANALYSIS:")
        
        # Count domain types
        vertical_count = sum(1 for p in problems if 'vertical' in p.name)
        horizontal_count = len(problems) - vertical_count
        lower_count = sum(1 for p in problems if 'lower' in p.name)
        upper_count = sum(1 for p in problems if 'upper' in p.name)
        
        print(f"   - Vertical domains: {vertical_count}")
        print(f"   - Horizontal domains: {horizontal_count}")
        print(f"     - Lower connectors: {lower_count}")
        print(f"     - Upper connectors: {upper_count}")
        
        # Domain name mapping
        print(f"\n📋 DOMAIN INDEX MAPPING:")
        for i, problem in enumerate(problems):
            domain_type = "VERTICAL" if 'vertical' in problem.name else "HORIZONTAL"
            print(f"   Domain {i:2d}: {domain_type:10s} - {problem.name}")
        
    except Exception as e:
        print(f"\n❌ Grid problem creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
