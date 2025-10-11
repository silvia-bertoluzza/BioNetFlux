#!/usr/bin/env python3
"""
Export elementary matrices from ElementaryMatrices class to MATLAB format.
"""

import sys
import os
import numpy as np
from scipy.io import savemat

# Add the python_port directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ooc1d.utils.elementary_matrices import ElementaryMatrices

def export_elementary_matrices():
    """
    Export elementary matrices from ElementaryMatrices class.
    """
    print("="*50)
    print("ELEMENTARY MATRICES EXPORT")
    print("="*50)
    
    # Create ElementaryMatrices instance
    print("Constructing elementary matrices...")
    elem_matrices = ElementaryMatrices(orthonormal_basis=False)
    
    # Get all matrices
    matrices = elem_matrices.get_all_matrices()
    
    # Save to MATLAB file
    output_file = "elementary_matrices.mat"
    print(f"Saving to {output_file}...")
    
    savemat(output_file, matrices, format='5', long_field_names=True)
    
    # Print summary
    print(f"\n✓ Exported {len(matrices)} matrices:")
    for key, value in matrices.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTo load in MATLAB:")
    print(f"  >> load('{output_file}')")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = export_elementary_matrices()
        print(f"\n✓ Success! Created {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
