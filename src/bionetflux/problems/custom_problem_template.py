#!/usr/bin/env python3
"""
Custom Problem Template
Based on MATLAB TestProblem.m structure with same physical parameters as OoC_grid_new.

This template extracts all the physical parameters and mathematical functions
from the MATLAB reference, with custom grid geometry specified.
"""

import numpy as np
import sys
import os

# Handle both relative imports (when used as module) and direct execution
try:
    from ..core.problem import Problem
    from ..core.discretization import Discretization, GlobalDiscretization
    from ..core.constraints import ConstraintManager
    from ..geometry.domain_geometry import DomainGeometry
except ImportError:
    # If relative imports fail, add the src directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, src_dir)
    
    from bionetflux.core.problem import Problem
    from bionetflux.core.discretization import Discretization, GlobalDiscretization
    from bionetflux.core.constraints import ConstraintManager
    from bionetflux.geometry.domain_geometry import DomainGeometry

def create_global_framework():
    """
    Custom problem template following MATLAB TestProblem.m structure.
    
    Physical parameters and functions match OoC_grid_new exactly.
    Geometry and constraints sections need customization.
    
    Returns:
        Tuple: (problems, global_discretization, constraint_manager, problem_name)
    """
    
    # ============================================================================
    # SECTION 1: PHYSICAL PARAMETERS (From MATLAB TestProblem.m)
    # ============================================================================
    
    print("Setting up physical parameters from MATLAB TestProblem.m...")
    
    # Global problem configuration
    neq = 4  # 4-equation OrganOnChip system (u, omega, v, phi)
    problem_name = "Custom_Problem_Template"
    
    # Time discretization parameters
    T = 1.0     # Final time
    dt = 0.1    # Time step
    
    # Physical parameters following MATLAB TestProblem.m exactly:
    # Viscosity parameters
    nu = 1.0        # MATLAB: nu = 1.
    mu = 2.0        # MATLAB: mu = 2.
    epsilon = 1.0   # MATLAB: epsilon = 1.
    sigma = 1.0     # MATLAB: sigma = 1.
    
    # Reaction parameters  
    a = 0.0     # MATLAB: a = 0.
    c = 0.0     # MATLAB: c = 0.
    
    # Coupling parameters
    b = 1.0     # MATLAB: b = 1.
    d = 1.0     # MATLAB: d = 1.
    chi = 1.0   # MATLAB: chi = 1.
    
    # Combine into parameter array (matches MATLAB order)
    parameters = np.array([nu, mu, epsilon, sigma, a, b, c, d, chi])
    
    print(f"✓ Physical parameters configured:")
    print(f"  Viscosity: nu={nu}, mu={mu}, epsilon={epsilon}, sigma={sigma}")
    print(f"  Reactions: a={a}, c={c}")
    print(f"  Coupling: b={b}, d={d}, chi={chi}")
    
    # ============================================================================
    # SECTION 2: MATHEMATICAL FUNCTIONS (From MATLAB TestProblem.m)
    # ============================================================================
    
    print("Defining mathematical functions from MATLAB TestProblem.m...")
    
    # Nonlinear coupling function (MATLAB: lambda = @(x) constant_function(x))
    def lambda_func(x):
        """Nonlinear coupling function lambda(x) - matches MATLAB constant_function"""
        return np.ones_like(x)  # Constant function
    
    def dlambda_func(x):
        """Derivative of lambda function"""
        return np.zeros_like(x)  # Derivative of constant is zero
    
    # Initial conditions (MATLAB: problem.u0{i})
    def initial_u(s, t=0.0):
        """Initial condition for u (equation 1) - MATLAB: sin(2*pi*x)"""
        return np.sin(2 * np.pi * s)
    
    def initial_omega(s, t=0.0):
        """Initial condition for omega (equation 2) - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def initial_v(s, t=0.0):
        """Initial condition for v (equation 3) - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def initial_phi(s, t=0.0):
        """Initial condition for phi (equation 4) - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    # Source terms (MATLAB: force{i})
    def source_u(s, t):
        """Source term for u equation - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def source_omega(s, t):
        """Source term for omega equation - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def source_v(s, t):
        """Source term for v equation - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def source_phi(s, t):
        """Source term for phi equation - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    print("✓ Mathematical functions defined:")
    print("  - Nonlinear coupling: lambda_func (constant), dlambda_func")
    print("  - Initial: u=sin(2πs), ω=0, v=0, φ=0 matching MATLAB")
    print("  - Sources: all zero functions matching MATLAB")
    
    # ============================================================================
    # SECTION 3: GEOMETRY SETUP - ⚠️ CUSTOMIZE THIS SECTION ⚠️
    # ============================================================================
    
    print("⚠️  GEOMETRY SECTION NEEDS CUSTOMIZATION")
    print("   Replace this section with your specific network topology")
    
    # TODO: Define your custom geometry here
    geometry = DomainGeometry("custom_geometry")
    
    # EXAMPLE: Single domain (replace with your geometry)
    # geometry.add_domain(
    #     extrema_start=(0.0, 0.0),
    #     extrema_end=(1.0, 0.0), 
    #     domain_start=0.0,
    #     domain_length=1.0,
    #     name="example_domain"
    # )
    
    # EXAMPLE: T-junction geometry
    # geometry.add_domain(
    #     extrema_start=(0.0, -1.0),
    #     extrema_end=(0.0, 1.0),
    #     name="main_channel"
    # )
    # geometry.add_domain(
    #     extrema_start=(0.0, 0.0),
    #     extrema_end=(1.0, 0.0),
    #     name="side_branch"
    # )
    
    # For now, create a minimal single domain to prevent errors
    geometry.add_domain(
        extrema_start=(0.0, 0.0),
        extrema_end=(1.0, 0.0),
        domain_start=0.0,
        domain_length=1.0,
        name="placeholder_domain",
        display_color="green"
    )
    
    print(f"📝 TODO: Add your geometry.add_domain() calls above")
    print(f"📝 Current placeholder: {geometry.num_domains()} domain(s)")
    
    # ============================================================================
    # SECTION 4: PROBLEM CREATION (Automated from geometry)
    # ============================================================================
    
    print("Creating problem instances from geometry...")
    
    problems = []
    discretizations = []
    
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        print(f"  Creating problem for domain {domain_id}: {domain_info.name}")
        
        # Create problem for this domain with MATLAB-compatible parameters
        problem = Problem(
            neq=neq,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            parameters=parameters,
            problem_type="organ_on_chip",  # 4-equation OoC system
            name=f"{problem_name}_{domain_info.name}"
        )
        
        # Set initial conditions (matches MATLAB problem.u0{i})
        problem.set_initial_condition(0, initial_u)
        problem.set_initial_condition(1, initial_omega)
        problem.set_initial_condition(2, initial_v)
        problem.set_initial_condition(3, initial_phi)
        
        # Set source terms (matches MATLAB force{i})
        problem.set_force(0, source_u)
        problem.set_force(1, source_omega)
        problem.set_force(2, source_v)
        problem.set_force(3, source_phi)
        
        # Set nonlinear coupling (matches MATLAB lambda)
        problem.set_chemotaxis(lambda_func, dlambda_func)
        
        # Set 2D coordinates for visualization
        problem.set_extrema(domain_info.extrema_start, domain_info.extrema_end)
        
        problems.append(problem)
        
        # Create discretization for this domain
        discretization = Discretization(
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            n_elements=10,
        )
        discretization.set_tau([1.0/discretization.element_length, 1.0, 1.0, 1.0])
        discretizations.append(discretization)
    
    print(f"✓ Created {len(problems)} problem instances")
    
    # ============================================================================
    # SECTION 5: CONSTRAINT SETUP - ⚠️ CUSTOMIZE THIS SECTION ⚠️  
    # ============================================================================
    
    print("⚠️  CONSTRAINT SECTION NEEDS CUSTOMIZATION")
    print("   Add your boundary and interface conditions here")
    
    constraint_manager = ConstraintManager()
    
    # TODO: Add your constraints here based on your geometry
    
    # EXAMPLE: Homogeneous Neumann boundary conditions (matches MATLAB fluxu0{i} = 0)
    # for eq_idx in range(neq):
    #     for domain_idx in range(len(problems)):
    #         # Left boundary (matches MATLAB fluxu0{i} = @(t) 0.)
    #         constraint_manager.add_neumann(
    #             equation_index=eq_idx, 
    #             domain_index=domain_idx, 
    #             position=problems[domain_idx].domain_start,
    #             data_function=lambda t: 0.0
    #         )
    #         # Right boundary (matches MATLAB fluxu1{i} = @(t) 0.)
    #         constraint_manager.add_neumann(
    #             equation_index=eq_idx,
    #             domain_index=domain_idx,
    #             position=problems[domain_idx].domain_end,
    #             data_function=lambda t: 0.0
    #         )
    
    # EXAMPLE: Interface continuity conditions
    # if len(problems) > 1:
    #     for eq_idx in range(neq):
    #         constraint_manager.add_trace_continuity(
    #             equation_index=eq_idx,
    #             domain1_index=0, domain2_index=1,
    #             position1=problems[0].domain_end,
    #             position2=problems[1].domain_start
    #         )
    
    # For now, add minimal boundary conditions for the placeholder domain
    if len(problems) > 0:
        for eq_idx in range(neq):
            # Zero flux at both ends (matches MATLAB Neumann data)
            constraint_manager.add_neumann(
                equation_index=eq_idx,
                domain_index=0,
                position=problems[0].domain_start,
                data_function=lambda t: 0.0
            )
            constraint_manager.add_neumann(
                equation_index=eq_idx,
                domain_index=0,
                position=problems[0].domain_end,
                data_function=lambda t: 0.0
            )
    
    print(f"📝 TODO: Add your constraint_manager.add_*() calls above")
    print(f"📝 Current placeholder: {constraint_manager.n_constraints} constraint(s)")
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)
    
    # ============================================================================
    # SECTION 6: GLOBAL DISCRETIZATION AND FINALIZATION
    # ============================================================================
    
    print("Finalizing global discretization...")
    
    global_discretization = GlobalDiscretization(
        spatial_discretizations=discretizations,
    )
    
    global_discretization.set_time_parameters(dt,T)

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
    
    # Validate geometry
    if not geometry.validate_geometry(verbose=False):
        print("⚠️  Warning: Geometry validation failed")
    else:
        print("✓ Geometry validated")
    
    print(f"\n🎉 Problem template '{problem_name}' ready!")
    print("📝 NEXT STEPS:")
    print("1. Customize SECTION 3: Add your geometry.add_domain() calls")
    print("2. Customize SECTION 5: Add your constraint_manager.add_*() calls")
    print("3. Rename this file to your problem name")
    print("4. Test with quick_setup('bionetflux.problems.your_problem_name')")
    
    return problems, global_discretization, constraint_manager, problem_name


def main():
    """Test the problem template."""
    print("="*60)
    print("CUSTOM PROBLEM TEMPLATE")
    print("="*60)
    print("Based on MATLAB TestProblem.m structure")
    print()
    
    try:
        problems, global_disc, constraints, name = create_global_framework()
        print(f"\n✅ Template creation successful!")
        print(f"   Problem: {name}")
        print(f"   Domains: {len(problems)}")
        print(f"   Equations per domain: {problems[0].neq if problems else 0}")
        print(f"   Total DOFs: {sum(p.neq * (d.n_elements + 1) for p, d in zip(problems, global_disc.spatial_discretizations))}")
        
    except Exception as e:
        print(f"\n❌ Template creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
