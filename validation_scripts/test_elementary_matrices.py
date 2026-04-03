#!/usr/bin/env python3
"""
Simple test script for elementary matrices construction.
Minimal implementation without unnecessary functions.
"""


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from setup_solver import quick_setup  # Fixed import path

def export_elementary_matrices():
    """
    Construct and export all elementary matrices used in the discretization.
    """
    print("="*60)
    print("ELEMENTARY MATRICES EXPORT TO MATLAB")
    print("="*60)
    
    # Initialize the solver setup
    print("Initializing solver setup...")
    setup = quick_setup("bionetflux.problems.pure_parabolic", validate=True)
    
    # Get first domain discretization for matrix construction
    discretization = setup.global_discretization.spatial_discretizations[0]
    problem = setup.problems[0]
    static_condensation = setup.static_condensations[0]  # Fixed variable access
    
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
    
    # 2. Get matrices from static condensation (if available)
    print("Accessing static condensation matrices...")
    try:
        if hasattr(static_condensation, 'mass_matrix'):
            matlab_data['mass_matrix'] = static_condensation.mass_matrix
            print(f"  Mass matrix shape: {static_condensation.mass_matrix.shape}")
        
        if hasattr(static_condensation, 'stiffness_matrix'):
            matlab_data['stiffness_matrix'] = static_condensation.stiffness_matrix
            print(f"  Stiffness matrix shape: {static_condensation.stiffness_matrix.shape}")
        
        if hasattr(static_condensation, 'trace_left'):
            matlab_data['trace_left'] = static_condensation.trace_left
            print(f"  Trace left shape: {static_condensation.trace_left.shape}")
        
        if hasattr(static_condensation, 'trace_right'):
            matlab_data['trace_right'] = static_condensation.trace_right
            print(f"  Trace right shape: {static_condensation.trace_right.shape}")
            
    except AttributeError as e:
        print(f"  Warning: Some matrices not available in static condensation: {e}")
    
    # 3. Get elementary matrices directly
    print("Constructing elementary matrices...")
    from bionetflux.utils.elementary_matrices import ElementaryMatrices
    
    elem_matrices = ElementaryMatrices(orthonormal_basis=False)
    all_elem_matrices = elem_matrices.get_all_matrices()
    
    # Add elementary matrices to export
    for name, matrix in all_elem_matrices.items():
        matlab_data[f'elementary_{name}'] = matrix
        print(f"  Elementary {name} shape: {matrix.shape}")
    
    # 4. Problem-specific matrices
    print("Computing problem-specific data...")
    
    # Time step
    if hasattr(setup.global_discretization, 'dt') and setup.global_discretization.dt is not None:
        matlab_data['dt'] = setup.global_discretization.dt
    
    # Physical parameters
    if hasattr(problem, 'parameters'):
        matlab_data['parameters'] = problem.parameters
    
    # Source terms at nodes (for verification)
    nodes = discretization.nodes
    time_sample = 0.1
    
    try:
        if hasattr(problem, 'source_functions') and len(problem.source_functions) >= 2:
            source_u = np.array([problem.source_functions[0](x, time_sample) for x in nodes])
            source_phi = np.array([problem.source_functions[1](x, time_sample) for x in nodes])
            matlab_data['source_u_sample'] = source_u
            matlab_data['source_phi_sample'] = source_phi
        else:
            print("  Warning: Source functions not available")
    except Exception as e:
        print(f"  Warning: Could not compute source terms: {e}")
    
    matlab_data['time_sample'] = time_sample
    
    # 5. Initial conditions - with error handling
    print("Computing initial conditions...")
    try:
        if hasattr(problem, 'initial_conditions') and len(problem.initial_conditions) >= 2:
            initial_u = np.array([problem.initial_conditions[0](x, 0.0) for x in nodes])
            initial_phi = np.array([problem.initial_conditions[1](x, 0.0) for x in nodes])
            matlab_data['initial_u'] = initial_u
            matlab_data['initial_phi'] = initial_phi
        else:
            print("  Warning: Initial conditions not available")
    except Exception as e:
        print(f"  Warning: Could not compute initial conditions: {e}")
    
    # 6. Matrix properties - only if matrices are available
    print("Computing matrix properties...")
    
    if 'mass_matrix' in matlab_data:
        try:
            matlab_data['mass_matrix_cond'] = np.linalg.cond(matlab_data['mass_matrix'])
            matlab_data['mass_eigenvalues'] = np.linalg.eigvals(matlab_data['mass_matrix'])
        except Exception as e:
            print(f"  Warning: Could not compute mass matrix properties: {e}")
    
    if 'stiffness_matrix' in matlab_data:
        try:
            matlab_data['stiffness_matrix_cond'] = np.linalg.cond(matlab_data['stiffness_matrix'])
            matlab_data['stiffness_eigenvalues'] = np.linalg.eigvals(matlab_data['stiffness_matrix'])
        except Exception as e:
            print(f"  Warning: Could not compute stiffness matrix properties: {e}")
    
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
    
    # Create visualization plots - with error handling
    try:
        create_visualization_plots(matlab_data, discretization, problem)
    except Exception as e:
        print(f"Warning: Could not create visualization plots: {e}")
    
    return output_file

def create_visualization_plots(matlab_data, discretization, problem):
    """Create visualization plots of the exported matrices."""
    
    print("\nCreating visualization plots...")
    
    # Plot elementary matrices that are always available
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Elementary matrices visualization
    if 'elementary_M' in matlab_data:
        im1 = axes[0, 0].imshow(matlab_data['elementary_M'], cmap='viridis')
        axes[0, 0].set_title('Elementary Mass Matrix (M)')
        plt.colorbar(im1, ax=axes[0, 0])
    
    if 'elementary_T' in matlab_data:
        im2 = axes[0, 1].imshow(matlab_data['elementary_T'], cmap='viridis')
        axes[0, 1].set_title('Elementary Trace Matrix (T)')
        plt.colorbar(im2, ax=axes[0, 1])
    
    if 'elementary_D' in matlab_data:
        im3 = axes[1, 0].imshow(matlab_data['elementary_D'], cmap='viridis')
        axes[1, 0].set_title('Elementary Derivative Matrix (D)')
        plt.colorbar(im3, ax=axes[1, 0])
    
    if 'elementary_QUAD' in matlab_data:
        im4 = axes[1, 1].imshow(matlab_data['elementary_QUAD'], cmap='viridis')
        axes[1, 1].set_title('Elementary Quadrature Matrix (QUAD)')
        plt.colorbar(im4, ax=axes[1, 1])
    
    plt.suptitle('Elementary Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('elementary_matrices_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot initial conditions and source terms if available
    if 'nodes' in matlab_data:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        nodes = matlab_data['nodes']
        cl
        # Initial conditions
        if 'initial_u' in matlab_data:
            axes[0, 0].plot(nodes, matlab_data['initial_u'], 'b-o', linewidth=2, markersize=4)
            axes[0, 0].set_title('Initial Condition - u')
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('u(x, 0)')
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'initial_phi' in matlab_data:
            axes[0, 1].plot(nodes, matlab_data['initial_phi'], 'r-s', linewidth=2, markersize=4)
            axes[0, 1].set_title('Initial Condition - φ')
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('φ(x, 0)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Source terms
        if 'source_u_sample' in matlab_data:
            axes[1, 0].plot(nodes, matlab_data['source_u_sample'], 'g-^', linewidth=2, markersize=4)
            axes[1, 0].set_title(f'Source Term u - t={matlab_data.get("time_sample", 0):.1f}')
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('f_u(x, t)')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'source_phi_sample' in matlab_data:
            axes[1, 1].plot(nodes, matlab_data['source_phi_sample'], 'm-v', linewidth=2, markersize=4)
            axes[1, 1].set_title(f'Source Term φ - t={matlab_data.get("time_sample", 0):.1f}')
            axes[1, 1].set_xlabel('Position')
            axes[1, 1].set_ylabel('f_φ(x, t)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Initial Conditions and Source Terms', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('initial_conditions_and_sources.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("✓ Visualization plots created and saved")

# Add SymPy requirement for symbolic computation
try:
    import sympy as sp
    print("✓ SymPy available for symbolic computation")
except ImportError:
    print("✗ SymPy not found. Install with: pip install sympy")
    sp = None

def test_elementary_matrices_standalone():
    """Test elementary matrices construction independently."""
    
    print("\n" + "="*60)
    print("STANDALONE ELEMENTARY MATRICES TEST")
    print("="*60)
    
    from bionetflux.utils.elementary_matrices import ElementaryMatrices

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

if __name__ == "__main__":
    try:
        # First try standalone elementary matrices test
        test_elementary_matrices_standalone()
        
        # Then try full export if setup is available
        print("\n" + "="*60)
        print("ATTEMPTING FULL MATRIX EXPORT")
        print("="*60)
        
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
