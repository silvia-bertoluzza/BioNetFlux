#!/usr/bin/env python3
"""
Simple test script for elementary matrices construction.
Minimal implementation without unnecessary functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def export_elementary_matrices():

    matlab_data['mass_matrix'] = mass_matrix
 
    
    # 9. Analytical solutions (if available)
    if hasattr(problem, 'solution') and problem.solution is not None:
        print("Computing analytical solutions...")
        try:
            analytical_u = np.array([problem.solution[0](x, time_sample) for x in nodes])
            analytical_phi = np.array([problem.solution[1](x, time_sample) for x in nodes])
            matlab_data['analytical_u_sample'] = analytical_u
            matlab_data['analytical_phi_sample'] = analytical_phi
        except Exception as e:
            print(f"Warning: Could not compute analytical solutions: {e}")
    
    # 10. Constraint information
    if hasattr(setup, 'constraint_manager') and setup.constraint_manager is not None:
        print("Exporting constraint information...")
        cm = setup.constraint_manager
        
        matlab_data['n_constraints'] = cm.n_constraints
        matlab_data['n_multipliers'] = cm.n_multipliers
        
        # Constraint types and positions
        constraint_types = []
        constraint_positions = []
        constraint_equations = []
        
        for constraint in cm.constraints:
            constraint_types.append(constraint.type.value)
            constraint_positions.extend(constraint.positions)
            constraint_equations.append(constraint.equation_index)
        
        matlab_data['constraint_types'] = constraint_types
        matlab_data['constraint_positions'] = np.array(constraint_positions)
        matlab_data['constraint_equations'] = np.array(constraint_equations)
    
    # 11. Matrix condition numbers and properties
    print("Computing matrix properties...")
    matlab_data['mass_matrix_cond'] = np.linalg.cond(mass_matrix)
    matlab_data['stiffness_matrix_cond'] = np.linalg.cond(stiffness_matrix)
    
    # Eigenvalues for stability analysis
    mass_eigenvals = np.linalg.eigvals(mass_matrix)
    stiff_eigenvals = np.linalg.eigvals(stiffness_matrix)
    
    matlab_data['mass_eigenvalues'] = mass_eigenvals
    matlab_data['stiffness_eigenvalues'] = stiff_eigenvals
    
    # Save to MATLAB file
    output_file = "bionetflux_elementary_matrices.mat"
    print(f"\nSaving matrices to {output_file}...")
    
    savemat(output_file, matlab_data, format='5', long_field_names=True)
    
    print(f"✓ Successfully exported {len(matlab_data)} matrices/arrays to {output_file}")
    
    # Print summary
    print("\nExported data summary:")
    for key, value in matlab_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} array, range [{np.min(value):.6e}, {np.max(value):.6e}]")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {len(value)} elements")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    # Create visualization plots
    create_visualization_plots(matlab_data, discretization, problem)
    
    return output_file

def create_visualization_plots(matlab_data, discretization, problem):
    """Create visualization plots of the exported matrices."""
    
    print("\nCreating visualization plots...")
    
    # Plot 1: Matrix sparsity patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mass matrix
    axes[0, 0].spy(matlab_data['mass_matrix'], markersize=2)
    axes[0, 0].set_title('Mass Matrix Sparsity')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    
    # Stiffness matrix
    axes[0, 1].spy(matlab_data['stiffness_matrix'], markersize=2)
    axes[0, 1].set_title('Stiffness Matrix Sparsity')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    
    # Trace matrices
    axes[1, 0].spy(matlab_data['trace_left'], markersize=4)
    axes[1, 0].set_title('Left Trace Matrix')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    
    axes[1, 1].spy(matlab_data['trace_right'], markersize=4)
    axes[1, 1].set_title('Right Trace Matrix')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    
    plt.suptitle('Elementary Matrices Sparsity Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('elementary_matrices_sparsity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Initial conditions and source terms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    nodes = matlab_data['nodes']
    
    # Initial conditions
    axes[0, 0].plot(nodes, matlab_data['initial_u'], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Initial Condition - u')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('u(x, 0)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(nodes, matlab_data['initial_phi'], 'r-s', linewidth=2, markersize=4)
    axes[0, 1].set_title('Initial Condition - φ')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('φ(x, 0)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Source terms
    axes[1, 0].plot(nodes, matlab_data['source_u_sample'], 'g-^', linewidth=2, markersize=4)
    axes[1, 0].set_title(f'Source Term u - t={matlab_data["time_sample"]:.1f}')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('f_u(x, t)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(nodes, matlab_data['source_phi_sample'], 'm-v', linewidth=2, markersize=4)
    axes[1, 1].set_title(f'Source Term φ - t={matlab_data["time_sample"]:.1f}')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('f_φ(x, t)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Initial Conditions and Source Terms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('initial_conditions_and_sources.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Matrix eigenvalues
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mass matrix eigenvalues
    mass_eigs = matlab_data['mass_eigenvalues']
    axes[0].plot(np.real(mass_eigs), np.imag(mass_eigs), 'bo', markersize=6)
    axes[0].set_title('Mass Matrix Eigenvalues')
    axes[0].set_xlabel('Real Part')
    axes[0].set_ylabel('Imaginary Part')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Stiffness matrix eigenvalues
    stiff_eigs = matlab_data['stiffness_eigenvalues']
    axes[1].plot(np.real(stiff_eigs), np.imag(stiff_eigs), 'ro', markersize=6)
    axes[1].set_title('Stiffness Matrix Eigenvalues')
    axes[1].set_xlabel('Real Part')
    axes[1].set_ylabel('Imaginary Part')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.suptitle('Matrix Eigenvalue Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('matrix_eigenvalues.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization plots created and saved")

if __name__ == "__main__":
    try:
        output_file = export_elementary_matrices()
        print(f"\n✓ Matrix export completed successfully!")
        print(f"✓ MATLAB file: {output_file}")
        print(f"✓ Visualization plots created")
        
        print(f"\nTo load in MATLAB, use:")
        print(f"  >> load('{output_file}')")
        print(f"  >> whos  % to see all variables")
        
    except Exception as e:
        print(f"Error during matrix export: {e}")
        import traceback
        traceback.print_exc()
def export_elementary_matrices():
    """
    Construct and export all elementary matrices used in the discretization.
    """
    print("="*60)
    print("ELEMENTARY MATRICES EXPORT TO MATLAB")
    print("="*60)
    
    # Initialize the solver setup
    print("Initializing solver setup...")
    setup = quick_setup("ooc1d.problems.pure_parabolic", validate=True)
    
    # Get first domain discretization for matrix construction
    discretization = setup.global_discretization.spatial_discretizations[0]
    problem = setup.problems[0]
    # static_condensation = setup.static_condensations[0]
    
    print(f"Domain parameters:")
    print(f"  Elements: {discretization.n_elements}")
    print(f"  Domain: [{discretization.domain_start}, {discretization.domain_start + discretization.domain_length}]")
    print(f"  Element length: {discretization.element_length}")
    print(f"  Equations: {problem.neq}")
    
    # Dictionary to store all matrices
    matlab_data = {}
    
    # 1. Basic discretization info
    matlab_data['n_elements'] = discretization.n_elements
    matlab_data['element_length'] = discretization.element_length
    matlab_data['domain_start'] = discretization.domain_start
    matlab_data['domain_length'] = discretization.domain_length
    matlab_data['neq'] = problem.neq
    matlab_data['nodes'] = discretization.nodes
    
    # 2. Mass matrix (for time discretization)
    print("Constructing mass matrix...")
    mass_matrix = static_condensation.mass_matrix
    matlab_data['mass_matrix'] = mass_matrix
    print(f"  Mass matrix shape: {mass_matrix.shape}")
    
    # 3. Stiffness matrix (for diffusion terms)
    print("Constructing stiffness matrix...")
    stiffness_matrix = static_condensation.stiffness_matrix
    matlab_data['stiffness_matrix'] = stiffness_matrix
    print(f"  Stiffness matrix shape: {stiffness_matrix.shape}")
    
    # 4. Trace matrices (boundary extraction)
    print("Constructing trace matrices...")
    trace_left = static_condensation.trace_left
    trace_right = static_condensation.trace_right
    matlab_data['trace_left'] = trace_left
    matlab_data['trace_right'] = trace_right
    print(f"  Trace left shape: {trace_left.shape}")
    print(f"  Trace right shape: {trace_right.shape}")
    
    # 5. Stabilization matrix (if available)
    if hasattr(static_condensation, 'stabilization_matrix'):
        print("Constructing stabilization matrix...")
        stab_matrix = static_condensation.stabilization_matrix
        matlab_data['stabilization_matrix'] = stab_matrix
        print(f"  Stabilization matrix shape: {stab_matrix.shape}")
    
    # 6. Elementary matrices on reference element
    print("Constructing reference element matrices...")
    
    # Reference element nodes (usually [-1, 1] for standard elements)
    xi_ref = np.array([-1.0, 1.0])  # Reference element coordinates
    matlab_data['xi_reference'] = xi_ref
    
    # Elementary mass matrix on reference element (for linear elements)
    elem_mass_ref = np.array([[2.0, 1.0], 
                              [1.0, 2.0]]) / 3.0  # 1D linear element mass matrix
    matlab_data['elementary_mass_reference'] = elem_mass_ref
    
    # Elementary stiffness matrix on reference element
    elem_stiff_ref = np.array([[1.0, -1.0], 
                               [-1.0, 1.0]]) # 1D linear element stiffness matrix
    matlab_data['elementary_stiffness_reference'] = elem_stiff_ref
    
    # 7. Problem-specific matrices
    print("Computing problem-specific data...")
    
    # Time step
    dt = setup.global_discretization.dt
    matlab_data['dt'] = dt
    
    # Physical parameters
    matlab_data['parameters'] = problem.parameters
    
    # Source terms at nodes (for verification)
    nodes = discretization.nodes
    time_sample = 0.1
    
    source_u = np.array([problem.source_functions[0](x, time_sample) for x in nodes])
    source_phi = np.array([problem.source_functions[1](x, time_sample) for x in nodes])
    
    matlab_data['source_u_sample'] = source_u
    matlab_data['source_phi_sample'] = source_phi
    matlab_data['time_sample'] = time_sample
    
    # 8. Initial conditions
    print("Computing initial conditions...")
    initial_u = np.array([problem.initial_conditions[0](x, 0.0) for x in nodes])
    initial_phi = np.array([problem.initial_conditions[1](x, 0.0) for x in nodes])
    
    matlab_data['initial_u'] = initial_u
    matlab_data['initial_phi'] = initial_phi
    
    # 9. Analytical solutions (if available)
    if hasattr(problem, 'solution') and problem.solution is not None:
        print("Computing analytical solutions...")
        try:
            analytical_u = np.array([problem.solution[0](x, time_sample) for x in nodes])
            analytical_phi = np.array([problem.solution[1](x, time_sample) for x in nodes])
            matlab_data['analytical_u_sample'] = analytical_u
            matlab_data['analytical_phi_sample'] = analytical_phi
        except Exception as e:
            print(f"Warning: Could not compute analytical solutions: {e}")
    
    # 10. Constraint information
    if hasattr(setup, 'constraint_manager') and setup.constraint_manager is not None:
        print("Exporting constraint information...")
        cm = setup.constraint_manager
        
        matlab_data['n_constraints'] = cm.n_constraints
        matlab_data['n_multipliers'] = cm.n_multipliers
        
        # Constraint types and positions
        constraint_types = []
        constraint_positions = []
        constraint_equations = []
        
        for constraint in cm.constraints:
            constraint_types.append(constraint.type.value)
            constraint_positions.extend(constraint.positions)
            constraint_equations.append(constraint.equation_index)
        
        matlab_data['constraint_types'] = constraint_types
        matlab_data['constraint_positions'] = np.array(constraint_positions)
        matlab_data['constraint_equations'] = np.array(constraint_equations)
    
    # 11. Matrix condition numbers and properties
    print("Computing matrix properties...")
    matlab_data['mass_matrix_cond'] = np.linalg.cond(mass_matrix)
    matlab_data['stiffness_matrix_cond'] = np.linalg.cond(stiffness_matrix)
    
    # Eigenvalues for stability analysis
    mass_eigenvals = np.linalg.eigvals(mass_matrix)
    stiff_eigenvals = np.linalg.eigvals(stiffness_matrix)
    
    matlab_data['mass_eigenvalues'] = mass_eigenvals
    matlab_data['stiffness_eigenvalues'] = stiff_eigenvals
    
    # Save to MATLAB file
    output_file = "bionetflux_elementary_matrices.mat"
    print(f"\nSaving matrices to {output_file}...")
    
    savemat(output_file, matlab_data, format='5', long_field_names=True)
    
    print(f"✓ Successfully exported {len(matlab_data)} matrices/arrays to {output_file}")
    
    # Print summary
    print("\nExported data summary:")
    for key, value in matlab_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} array, range [{np.min(value):.6e}, {np.max(value):.6e}]")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {len(value)} elements")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    # Create visualization plots
    create_visualization_plots(matlab_data, discretization, problem)
    
    return output_file

def create_visualization_plots(matlab_data, discretization, problem):
    """Create visualization plots of the exported matrices."""
    
    print("\nCreating visualization plots...")
    
    # Plot 1: Matrix sparsity patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mass matrix
    axes[0, 0].spy(matlab_data['mass_matrix'], markersize=2)
    axes[0, 0].set_title('Mass Matrix Sparsity')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    
    # Stiffness matrix
    axes[0, 1].spy(matlab_data['stiffness_matrix'], markersize=2)
    axes[0, 1].set_title('Stiffness Matrix Sparsity')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    
    # Trace matrices
    axes[1, 0].spy(matlab_data['trace_left'], markersize=4)
    axes[1, 0].set_title('Left Trace Matrix')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    
    axes[1, 1].spy(matlab_data['trace_right'], markersize=4)
    axes[1, 1].set_title('Right Trace Matrix')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    
    plt.suptitle('Elementary Matrices Sparsity Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('elementary_matrices_sparsity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Initial conditions and source terms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    nodes = matlab_data['nodes']
    
    # Initial conditions
    axes[0, 0].plot(nodes, matlab_data['initial_u'], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Initial Condition - u')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('u(x, 0)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(nodes, matlab_data['initial_phi'], 'r-s', linewidth=2, markersize=4)
    axes[0, 1].set_title('Initial Condition - φ')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('φ(x, 0)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Source terms
    axes[1, 0].plot(nodes, matlab_data['source_u_sample'], 'g-^', linewidth=2, markersize=4)
    axes[1, 0].set_title(f'Source Term u - t={matlab_data["time_sample"]:.1f}')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('f_u(x, t)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(nodes, matlab_data['source_phi_sample'], 'm-v', linewidth=2, markersize=4)
    axes[1, 1].set_title(f'Source Term φ - t={matlab_data["time_sample"]:.1f}')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('f_φ(x, t)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Initial Conditions and Source Terms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('initial_conditions_and_sources.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Matrix eigenvalues
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mass matrix eigenvalues
    mass_eigs = matlab_data['mass_eigenvalues']
    axes[0].plot(np.real(mass_eigs), np.imag(mass_eigs), 'bo', markersize=6)
    axes[0].set_title('Mass Matrix Eigenvalues')
    axes[0].set_xlabel('Real Part')
    axes[0].set_ylabel('Imaginary Part')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Stiffness matrix eigenvalues
    stiff_eigs = matlab_data['stiffness_eigenvalues']
    axes[1].plot(np.real(stiff_eigs), np.imag(stiff_eigs), 'ro', markersize=6)
    axes[1].set_title('Stiffness Matrix Eigenvalues')
    axes[1].set_xlabel('Real Part')
    axes[1].set_ylabel('Imaginary Part')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.suptitle('Matrix Eigenvalue Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('matrix_eigenvalues.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization plots created and saved")

if __name__ == "__main__":
    try:
        output_file = export_elementary_matrices()
        print(f"\n✓ Matrix export completed successfully!")
        print(f"✓ MATLAB file: {output_file}")
        print(f"✓ Visualization plots created")
        
        print(f"\nTo load in MATLAB, use:")
        print(f"  >> load('{output_file}')")
        print(f"  >> whos  % to see all variables")
        
    except Exception as e:
        print(f"Error during matrix export: {e}")
        import traceback
        traceback.print_exc()



import numpy as np
# Add SymPy requirement for symbolic computation
try:
    import sympy as sp
    print("✓ SymPy available for symbolic computation")
except ImportError:
    print("✗ SymPy not found. Install with: pip install sympy")
    exit(1)

from ooc1d.utils.elementary_matrices import ElementaryMatrices

print("Constructing elementary matrices (matching MATLAB build_eMatrices.m)...")

# Create elementary matrices using standard Lagrange basis
elem_matrices = ElementaryMatrices(orthonormal_basis=False)

print("✓ Elementary matrices constructed successfully")

# Display results
print("\n" + "="*60)
print("ELEMENTARY MATRICES (Reference element [0,1])")
print("="*60)

# Get all matrices
matrices = elem_matrices.get_all_matrices()

print(f"\nAvailable matrices: {list(matrices.keys())}")

print(f"\nMass matrix M:")
print(f"  Shape: {matrices['M'].shape}")
print(f"  Values:\n{matrices['M']}")

print(f"\nInverse Mass matrix IM:")
print(f"  Values:\n{matrices['IM']}")

print(f"\nDerivative matrix D:")
print(f"  Values:\n{matrices['D']}")

print(f"\nTrace matrix T:")
print(f"  Values:\n{matrices['T']}")

print(f"\nGramian matrix Gb:")
print(f"  Values:\n{matrices['Gb']}")

print(f"\nBoundary Mass matrix Mb:")
print(f"  Values:\n{matrices['Mb']}")

print(f"\nNormal matrix Ntil:")
print(f"  Values:\n{matrices['Ntil']}")

print(f"\nHat Normal matrix Nhat:")
print(f"  Values:\n{matrices['Nhat']}")

print(f"\nAverage matrix Av:")
print(f"  Values:\n{matrices['Av']}")

print(f"\nQuadrature matrix QUAD:")
print(f"  Shape: {matrices['QUAD'].shape}")
print(f"  Values:\n{matrices['QUAD']}")

# Run tests (equivalent to MATLAB tests)
print(f"\n" + "="*60)
print("VERIFICATION TESTS (from MATLAB)")
print("="*60)

elem_matrices.print_tests()

print("\n✓ Elementary matrices test completed!")
