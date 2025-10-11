#!/usr/bin/env python3
"""
Export only elementary matrices from ElementaryMatrices class to MATLAB format.
"""

import sys
import os
import numpy as np
from scipy.io import savemat

# Add the python_port directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ooc1d.utils.elementary_matrices import ElementaryMatrices

def export_to_matlab():
    """Export elementary matrices to MATLAB file."""
    
    print("="*50)
    print("ELEMENTARY MATRICES EXPORT")
    print("="*50)
    
    # Create ElementaryMatrices instance
    print("Constructing elementary matrices...")
    elem_matrices = ElementaryMatrices(orthonormal_basis=False)
    
    # Get all matrices from ElementaryMatrices class
    matlab_data = elem_matrices.get_all_matrices()
    
    # Save to file
    output_file = "elementary_matrices_simple.mat"
    print(f"\nSaving to {output_file}...")
    
    savemat(output_file, matlab_data, format='5', long_field_names=True)
    
    # Print summary
    print(f"\n✓ Exported {len(matlab_data)} matrices:")
    for key, value in matlab_data.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                print(f"  {key}: {value.shape[0]}×{value.shape[1]} matrix")
            else:
                print(f"  {key}: {value.shape} array")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTo load in MATLAB:")
    print(f"  >> load('{output_file}')")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = export_to_matlab()
        print(f"\n✓ Success! Created {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Add element contribution to global matrices
        for i in range(2):
            for j in range(2):
                M_global[nodes[i], nodes[j]] += mass_phys[i, j]
                K_global[nodes[i], nodes[j]] += stiffness_phys[i, j]
    
    matlab_data['mass_global'] = M_global
    matlab_data['stiffness_global'] = K_global
    matlab_data['n_elements'] = n_elements
    matlab_data['n_nodes'] = n_nodes
    
    # 4. Boundary extraction matrices
    print("- Boundary extraction matrices")
    
    # Left boundary (extract value at x=0, node 0)
    trace_left = np.zeros((1, n_nodes))
    trace_left[0, 0] = 1.0
    
    # Right boundary (extract value at x=L, node n_nodes-1)
    trace_right = np.zeros((1, n_nodes))
    trace_right[0, -1] = 1.0
    
    matlab_data['trace_left'] = trace_left
    matlab_data['trace_right'] = trace_right
    
    # 5. Multi-equation system matrices
    print("- Multi-equation system matrices")
    neq = 2  # Number of equations
    
    # Block matrices for system of equations
    # Each DOF has neq components: [u1_1, u2_1, u1_2, u2_2, ...]
    total_dofs = n_nodes * neq
    
    M_system = np.zeros((total_dofs, total_dofs))
    K_system = np.zeros((total_dofs, total_dofs))
    
    # Fill block structure
    for eq in range(neq):
        start_idx = eq * n_nodes
        end_idx = start_idx + n_nodes
        M_system[start_idx:end_idx, start_idx:end_idx] = M_global
        K_system[start_idx:end_idx, start_idx:end_idx] = K_global
    
    matlab_data['mass_system'] = M_system
    matlab_data['stiffness_system'] = K_system
    matlab_data['neq'] = neq
    matlab_data['total_dofs'] = total_dofs
    
    # 6. Time discretization matrices
    print("- Time discretization matrices")
    dt = 0.01
    
    # Backward Euler: (M + dt*K) * u^{n+1} = M * u^n + dt * f^{n+1}
    time_matrix = M_system + dt * K_system
    matlab_data['time_matrix'] = time_matrix
    matlab_data['dt'] = dt
    
    # 7. Coordinate arrays
    print("- Coordinate arrays")
    x_nodes = np.linspace(0.0, n_elements * h, n_nodes)
    xi_ref = np.array([-1.0, 1.0])
    
    matlab_data['x_nodes'] = x_nodes
    matlab_data['xi_reference'] = xi_ref
    matlab_data['domain_length'] = n_elements * h
    
    return matlab_data

def export_to_matlab():
    """Export elementary matrices to MATLAB file."""
    
    print("="*50)
    print("ELEMENTARY MATRICES EXPORT")
    print("="*50)
    
    # Create matrices
    matlab_data = create_elementary_matrices()
    
    # Save to file
    output_file = "elementary_matrices_simple.mat"
    print(f"\nSaving to {output_file}...")
    
    savemat(output_file, matlab_data, format='5', long_field_names=True)
    
    # Print summary
    print(f"\n✓ Exported {len(matlab_data)} matrices:")
    for key, value in matlab_data.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                print(f"  {key}: {value.shape[0]}×{value.shape[1]} matrix")
            else:
                print(f"  {key}: {value.shape} array")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTo load in MATLAB:")
    print(f"  >> load('{output_file}')")
    print(f"  >> mass_reference")
    print(f"  >> spy(mass_global)")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = export_to_matlab()
        print(f"\n✓ Success! Created {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
