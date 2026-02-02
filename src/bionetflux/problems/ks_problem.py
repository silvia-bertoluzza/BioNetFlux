#!/usr/bin/env python3
"""
Keller-Segel Problem Template
Based on KS_traveling_wave_double_arc.py with user-friendly configuration.

This template follows the same user-friendly philosophy as ooc_problem.py:
- TOML-based configuration with domain-specific overrides
- Custom geometry support with default fallback
- Clean separation of configuration, physics, and geometry
"""

import numpy as np
import sys
import os
from typing import Optional

from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry

# Handle both relative imports (when used as module) and direct execution
try:
    from ..core.problem import Problem
    from ..core.discretization import Discretization, GlobalDiscretization
    from ..core.constraints import ConstraintManager
    from ..geometry.domain_geometry import DomainGeometry, EXTERIOR_BOUNDARY, build_arc_sequence_geometry
    from .ks_config_manager import KSConfigManager
except ImportError:
    # If relative imports fail, add the src directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, src_dir)
    
    from bionetflux.core.problem import Problem
    from bionetflux.core.discretization import Discretization, GlobalDiscretization
    from bionetflux.core.constraints import ConstraintManager
    from bionetflux.geometry.domain_geometry import DomainGeometry, EXTERIOR_BOUNDARY, build_arc_sequence_geometry
    from bionetflux.problems.ks_config_manager import KSConfigManager


def build_default_ks_geometry():
    """
    Build the default Keller-Segel double arc geometry.
    
    Returns:
        DomainGeometry: Default KS double arc geometry instance
    """
    return build_arc_sequence_geometry(N=1, start=2.0, length=1.0)


def setup_constraints_from_geometry(geometry: DomainGeometry, problems, neq: int) -> ConstraintManager:
    """
    Generate constraint manager from geometry connections for Keller-Segel problems.
    Same as OoC but for 2 equations instead of 4.
    """
    print("Setting up KS constraints from geometry connections...")
    
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
    
    print(f"✓ KS constraint setup completed from geometry:")
    print(f"  - Boundary conditions: {total_boundary_conditions} ({len(boundary_connections)} locations × {neq} equations)")
    print(f"  - Continuity conditions: {total_continuity_conditions} ({len(interior_connections)} connections × {neq} equations)")
    print(f"  - Total constraints: {total_boundary_conditions + total_continuity_conditions}")
    
    return constraint_manager


def create_global_framework(geometry: Optional[DomainGeometry] = None,
                          config_file: Optional[str] = None):
    """
    Create Keller-Segel problem following user-friendly configuration pattern.
    
    Physical parameters and functions match KS_traveling_wave_double_arc.py defaults.
    Custom geometry with two connected domains.
    """
    
    # Define this module's config validation type (different from problem_type for StaticCondensation!)
    CONFIG_VALIDATION_TYPE = "ks"  # For config file validation
    PROBLEM_TYPE = "keller_segel"  # For StaticCondensationFactory and other core systems
    
    # ============================================================================
    # SECTION 1: CONFIGURATION LOADING (TOML-BASED APPROACH)
    # ============================================================================
    
    print("Loading Keller-Segel configuration...")
    
    # Load configuration using KS config manager (includes type validation)
    config_manager = KSConfigManager()
    try:
        config = config_manager.load_config(config_file)
    except ValueError as e:
        # Re-raise with module context
        raise ValueError(f"KS problem module: {e}")
    
    # Additional explicit check for extra safety
    if config_file:
        config_problem_type = config.get('problem', {}).get('problem_type', None)
        if config_problem_type and config_problem_type != CONFIG_VALIDATION_TYPE:
            raise ValueError(
                f"KS problem module expects config problem_type='{CONFIG_VALIDATION_TYPE}', "
                f"but config file specifies '{config_problem_type}'. "
                f"Please use a Keller-Segel compatible configuration file."
            )
    
    # Extract configuration sections
    problem_config = config['problem']
    time_params = config['time_parameters']
    phys_params = config['physical_parameters']
    disc_params = config['discretization']
    initial_conditions = config['initial_conditions']  # Already resolved to callables
    force_functions = config['force_functions']        # Already resolved to callables
    exact_solutions = config['exact_solutions']  # Optional exact solutions, already resolved
    exact_solution_derivatives = config['exact_solution_derivatives']  # Optional derivatives    
    
    # Extract domain-specific overrides (as strings, not resolved yet)
    domain_initial_conditions = config.get('domain_initial_conditions', {})
    domain_force_functions = config.get('domain_force_functions', {})
    domain_exact_solutions = config.get('domain_exact_solutions', {})
    
    print(f"Domain-specific initial conditions found: {len(domain_initial_conditions)}")
    for key, value in domain_initial_conditions.items():
        print(f"  {key} = {value}")
    print(f"Domain-specific force functions found: {len(domain_force_functions)}")
    for key, value in domain_force_functions.items():
        print(f"  {key} = {value}")
    print(f"Domain-specific exact solutions found: {len(domain_exact_solutions)}")
    for key, value in domain_exact_solutions.items():
        print(f"  {key} = {value}")
        
    # Global problem configuration
    neq = problem_config['neq']
    problem_name = problem_config['name']
    
    # Time discretization parameters
    T = time_params['T']
    dt = time_params['dt']
    
    # Physical parameters (extracted from TOML or defaults)
    diffusion = phys_params['diffusion']
    reaction = phys_params['reaction']
    chemotaxis = phys_params['chemotaxis']
    
    # These should be function names (strings), not the resolved functions yet
    chi_func_name = chemotaxis['chi']    # Should be "constant" 
    dchi_func_name = chemotaxis['dchi']  # Should be "zeros"
    
    # Resolve the function names to actual callables
    chi_func = config_manager.function_resolver.resolve_function(chi_func_name)
    dchi_func = config_manager.function_resolver.resolve_function(dchi_func_name)
    
    u = exact_solutions['u']

    u_x = exact_solution_derivatives['u']
    phi_x = exact_solution_derivatives['phi']
    
    
    
    mu = diffusion['mu']
    nu = diffusion['nu']
    a = reaction['a']
    b = reaction['b']
    
    flux_u = lambda s, t: nu * u_x(s, t) + chi(s) * u(s, t) * phi_x(s, t)
    flux_phi = lambda s, t: - mu * phi_x(s, t)  
    
    # Combine into parameter array (matches KS_traveling_wave order: [mu, nu, a, b])
    parameters = np.array([mu, nu, a, b])
    
    print(f"✓ KS Configuration loaded:")
    print(f"  Problem: {problem_name} ({neq} equations)")
    print(f"  Time: T={T}, dt={dt}")
    print(f"  Diffusion: mu={mu}, nu={nu}")
    print(f"  Reaction: a={a}, b={b}")
    
    # ============================================================================
    # SECTION 2: MATHEMATICAL FUNCTIONS (From KS_traveling_wave)
    # ============================================================================
    
    print("Setting up KS mathematical functions...")


    print("✓ KS Mathematical functions configured:")
    print("  - Chemotaxis: chi (constant), dchi")
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
        if not geometry.validate_geometry(verbose=False):
            print("⚠️  Warning: Provided geometry validation failed")
        else:
            print("✓ Provided geometry validation passed")
            
    else:
        # Create default KS double arc geometry
        geometry = build_default_ks_geometry()
        
        # Validate default geometry
        if not geometry.validate_geometry(verbose=False):
            print("⚠️  Warning: Default KS geometry validation failed")
        else:
            print("✓ Default KS geometry validation passed")
    
    # ============================================================================
    # SECTION 4: PROBLEM CREATION (Geometry-based, with config parameters)
    # ============================================================================
    
    print("Creating KS problem instances from geometry with config parameters...")
    
    problems = []
    discretizations = []
    
    # Extract discretization parameters
    n_elements = disc_params['n_elements']
    tau_values = disc_params['tau']
    
    # Apply config parameters to all domains based on geometry
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        print(f"  Creating KS problem for domain {domain_id}: {domain_info.name}")
        
        # Validate that required functions are callable before using them
        print("  Validating function callability...")
        function_checks = [
            ('initial_conditions["u"]', initial_conditions.get('u')),
            ('initial_conditions["phi"]', initial_conditions.get('phi')),
            ('force_functions["u"]', force_functions.get('u')),
            ('force_functions["phi"]', force_functions.get('phi')),
            ('chi_func', chi_func),      # Use resolved functions
            ('dchi_func', dchi_func)     # Use resolved functions
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
        
        # Create problem for this domain with config parameters
        problem = Problem(
            neq=neq,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            parameters=parameters,  # From config
            problem_type=PROBLEM_TYPE,  # Use "keller_segel" for StaticCondensationFactory!
            name=f"{problem_name}_{domain_info.name}"
        )
        
        # Set DEFAULT initial conditions from config (already resolved to callables)
        # problem.set_initial_condition(0, initial_conditions['u'])  # Cell density
        # problem.set_initial_condition(1, initial_conditions['phi'])  # Chemical concentration
        problem.set_initial_condition(0, initial_conditions.get('u'))  # Cell density
        problem.set_initial_condition(1, initial_conditions.get('phi'))  # Chemical concentration
        
        # Set DEFAULT force functions from config (already resolved to callables)
        problem.set_force(0, force_functions['u'])
        problem.set_force(1, force_functions['phi'])
        
        # Set chemotaxis for Keller-Segel
        problem.set_chemotaxis(chi_func, dchi_func)  # Use resolved functions
        # problem.set_boundary_flux(0, left_flux=u_x, right_flux=u_x)  
        # problem.set_boundary_flux(1, left_flux=phi_x, right_flux=phi_x)
        
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
    equation_names = ['u', 'phi']  # KS has 2 equations: u (cells), phi (chemical)
    
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

    print(f"✓ Created {len(problems)} KS problem instances with config parameters")
    
    # ============================================================================
    # SECTION 5: CONSTRAINT SETUP
    # ============================================================================
    
    constraint_manager = setup_constraints_from_geometry(geometry, problems, neq)
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)
    
    # ============================================================================
    # SECTION 6: GLOBAL DISCRETIZATION AND FINALIZATION
    # ============================================================================
    
    print("Finalizing KS global discretization...")
    
    global_discretization = GlobalDiscretization(
        spatial_discretizations=discretizations,
    )
    
    global_discretization.set_time_parameters(dt, T)
    print(f"✓ KS Global discretization created:")
    print(f"  Time: dt={dt}, T={T}, steps={int(T/dt)}")
    print(f"  Domains: {len(discretizations)}")
    print(f"  Constraints: {constraint_manager.n_constraints}")
    print(f"  Multipliers: {constraint_manager.n_multipliers}")
    
    # ============================================================================
    # SECTION 7: VALIDATION AND RETURN
    # ============================================================================
    
    print("Validating KS problem setup...")
    
    # Validate each problem
    for i, problem in enumerate(problems):
        if not problem.validate_problem(verbose=False):
            print(f"⚠️  Warning: KS Problem {i} validation failed")
        else:
            print(f"✓ KS Problem {i} validated")
    
    print(f"\n🎉 Keller-Segel problem '{problem_name}' ready!")
    print(f"📊 KS PROBLEM SUMMARY:")
    print(f"   - Geometry: {geometry.name} with {geometry.num_domains()} domains")
    print(f"   - Physics: Keller-Segel parameters from config") 
    print(f"   - {neq} equations per domain ({neq * len(problems)} total equations)")
    print(f"   - {constraint_manager.n_constraints} constraints")
    print(f"   - Total DOFs: {sum(p.neq * (d.n_elements + 1) for p, d in zip(problems, global_discretization.spatial_discretizations))}")
    
    return problems, global_discretization, constraint_manager, problem_name


def main():
    """Test the Keller-Segel problem."""
    print("="*60)
    print("KELLER-SEGEL PROBLEM")
    print("="*60)
    print("Double arc geometry with TOML configuration support")
    print()
    
    try:
        problems, global_disc, constraints, name = create_global_framework()
        print(f"\n✅ KS problem creation successful!")
        print(f"   Problem: {name}")
        print(f"   Domains: {len(problems)}")
        print(f"   Equations per domain: {problems[0].neq if problems else 0}")
        print(f"   Total DOFs: {sum(p.neq * (d.n_elements + 1) for p, d in zip(problems, global_disc.spatial_discretizations))}")
        
    except Exception as e:
        print(f"\n❌ KS problem creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()