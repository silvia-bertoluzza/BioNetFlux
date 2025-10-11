import numpy as np
import sympy as sp
from typing import Dict, Any

class ElementaryMatrices:
    """
    Elementary matrices for HDG method on reference element.
    Equivalent to MATLAB build_eMatrices.m functionality.
    Uses SymPy for symbolic computation like the original MATLAB code.
    """
    
    def __init__(self, orthonormal_basis: bool = False):
        """
        Initialize and compute elementary matrices.
        
        Args:
            orthonormal_basis: Use orthonormal basis (default: False, uses Lagrange basis)
        """
        self.orthonormal_basis = orthonormal_basis
        self.matrices = {}
        self._build_matrices()
    
    def _build_matrices(self):
        """Build all elementary matrices on reference element [0, 1]."""
        
        # Define symbolic variable
        y = sp.Symbol('y')
        
        # Reference interval (0,1)
        x0, x1 = 0, 1
        
        if self.orthonormal_basis:
            # Orthonormal basis construction
            c, d, e = sp.symbols('c d e')
            
            e0 = 1
            e1 = c * y + d
            
            # Orthogonality conditions
            int2 = sp.integrate(e0 * e1, (y, 0, 1))
            int3 = sp.integrate(e1 * e1, (y, 0, 1))
            
            equation2 = sp.Eq(int2, 0)
            equation3 = sp.Eq(int3, 1)
            
            solutions = sp.solve([equation2, equation3], [c, d])
            # Fix: solutions is a list of tuples, not a dictionary
            # Extract values properly
            if isinstance(solutions, list) and len(solutions) > 0:
                if isinstance(solutions[0], tuple):
                    c_val, d_val = solutions[0]  # First solution tuple
                else:
                    c_val = solutions[0]
                    d_val = solutions[1]
            elif isinstance(solutions, dict):
                c_val = solutions[c]
                d_val = solutions[d]
            else:
                raise ValueError(f"Unexpected solution format: {solutions}")
            
            e0 = e0
            e1 = c_val * y + d_val
        else:
            # Standard Lagrange basis
            e0 = 1 - y
            e1 = y
        
        # Hat basis (same as standard basis in this case)
        he0 = 1 - y
        he1 = y
        
        # Normal vectors
        n0, n1 = -1, 1
        
        # Basis vectors
        base = sp.Matrix([e0, e1])
        hbase = sp.Matrix([he0, he1])
        normali = sp.Matrix([n0, n1])
        
        # Store basis functions
        self.base = base
        self.hbase = hbase
        self.normali = normali
        
        # Build matrices
        self._build_basic_matrices(y, base, hbase, normali, x0, x1)
        self._build_quadrature()
    
    def _build_basic_matrices(self, y, base, hbase, normali, x0, x1):
        """Build the basic elementary matrices."""
        
        # Zero matrix
        Z = sp.zeros(2, 2)
        self.matrices['Z'] = np.array(Z).astype(float)
        
        # Mass matrix: m_ij = ∫₀¹ eⱼ eᵢ dy
        M = sp.integrate(base * base.T, (y, 0, 1))
        M_numeric = np.array(M).astype(float)
        self.matrices['M'] = M_numeric
        self.matrices['IM'] = np.linalg.inv(M_numeric)
        
        # Gramian matrix: G^∂_ij = eᵢ(x₀)eⱼ(x₀) + eᵢ(x₁)eⱼ(x₁)
        Gb = base.subs(y, x0) * hbase.T.subs(y, x0) + base.subs(y, x1) * hbase.T.subs(y, x1)
        self.matrices['Gb'] = np.array(Gb).astype(float)
        
        # Boundary Mass matrix: M^∂_ij = eᵢ(x₀)eⱼ(x₀) + eᵢ(x₁)eⱼ(x₁)
        Mb = base.subs(y, x0) * base.T.subs(y, x0) + base.subs(y, x1) * base.T.subs(y, x1)
        self.matrices['Mb'] = np.array(Mb).astype(float)
        
        # Trace matrix: T_ij = eⱼ(xᵢ) (transpose of Gramian)
        Trace = Gb.T
        self.matrices['T'] = np.array(Trace).astype(float)
        
        # Average matrix
        Trace_inv = Trace.inv()
        ones_vec = sp.Matrix([1, 1])
        box = Trace_inv * ones_vec
        Av = box.T * M
        self.matrices['Av'] = np.array(Av).astype(float)
        
        # Normal matrix: Ñ_ij = eⱼ(x₁)eᵢ(x₁)n₁ + eⱼ(x₀)eᵢ(x₀)n₀
        Ntil = (base.subs(y, x0) * normali[0]) * base.T.subs(y, x0) + \
               (base.subs(y, x1) * normali[1]) * base.T.subs(y, x1)
        self.matrices['Ntil'] = np.array(Ntil).astype(float)
        
        # Hat normal matrix
        basis_at_boundaries = sp.Matrix([[1, 0], [0, 1]])  # [e0(x0), e0(x1); e1(x0), e1(x1)]
        normal_matrix = sp.Matrix([[-1, 0], [0, 1]])
        Nhat = basis_at_boundaries * normal_matrix
        self.matrices['Nhat'] = np.array(Nhat).astype(float)
        
        # Derivative matrix: D_ij = ∫₀¹ eⱼ ∂ₓeᵢ dy
        D = sp.integrate(sp.diff(base, y) * base.T, (y, x0, x1))
        self.matrices['D'] = np.array(D).astype(float)
        
        # Store symbolic versions for testing
        self._test_matrices(D, Ntil, M, Trace, Nhat, Gb)
    
    def _test_matrices(self, D, Ntil, M, Trace, Nhat, Gb):
        """Perform the same tests as in MATLAB code."""
        
        # Test: Integration by parts - D + D' - Ñ should be zero
        test1 = D + D.T - Ntil
        self.test_integration_by_parts = np.array(test1).astype(float)
        
        # Test: d²/dx² = 0 for linear functions
        test2 = (Ntil - D) * M.inv() * (D - Nhat * Trace)
        self.test_dxx_zero = np.array(test2).astype(float)
        
        # Additional test
        test3 = self.matrices['Mb'] - np.array(Gb * Trace).astype(float)
        self.test_mb_gb = test3
    
    def _build_quadrature(self):
        """Build quadrature matrices using Legendre-Gauss-Lobatto nodes."""
        # Import the LGL nodes function (we'll need to implement this)
        # For now, using the same approach as MATLAB: lglnodes(3)
        
        # MATLAB: [qnodes,qweights,~] = lglnodes(3);
        # This gives 3 LGL nodes on [-1,1]
        # For 3 nodes: approximately [-1, 0, 1] with specific weights
        # 4-point Legendre-Gauss-Lobatto nodes and weights on [-1, 1]
        qnodes = np.array([-1.0, -0.4472136, 0.4472136, 1.0])      # Shape: (4,) - row vector
        qweights = np.array([1/6, 5/6, 5/6, 1/6])                  # Shape: (4,) - row vector
        
        # MATLAB: QUAD = double(subs(base,(qnodes'+1)/2)*diag(qweights))/2;
        # The key insight: MATLAB's base = [e0; e1] where e0 = 1-y, e1 = y
        # But the substitution and matrix operations produce a different row order
        
        y = sp.Symbol('y')
        base = self.base
        
        # Transform nodes to [0,1]
        xi_01 = (qnodes + 1) / 2
        
        # Build the matrix exactly as MATLAB does:
        # First evaluate base at transformed quadrature points
        basis_at_quad = np.zeros((2, 4))
        # Note: MATLAB's matrix multiplication gives different row ordering
        # We need to match MATLAB's exact computation
        
        # MATLAB computation: subs(base,(qnodes'+1)/2) evaluates:
        # Row 1: e0 = 1-y at transformed points
        # Row 2: e1 = y at transformed points
        # But the final QUAD matrix has rows swapped due to how MATLAB handles the operations
        
        for i in range(2):
            for j in range(4):
                basis_at_quad[i, j] = float(base[i].subs(y, xi_01[j]))
        
        # Apply weights and scaling, then swap rows to match MATLAB output
        QUAD_temp = basis_at_quad @ np.diag(qweights) / 2
        
        # No row swapping needed; use QUAD_temp directly
        QUAD = QUAD_temp
        
        self.matrices['qnodes'] = qnodes
        self.matrices['QUAD'] = QUAD
    
    def get_matrix(self, name: str) -> np.ndarray:
        """Get elementary matrix by name."""
        if name in self.matrices:
            return self.matrices[name]
        else:
            raise KeyError(f"Matrix '{name}' not found. Available: {list(self.matrices.keys())}")
    
    def get_all_matrices(self) -> Dict[str, np.ndarray]:
        """Get all elementary matrices."""
        return self.matrices.copy()
    
    def print_tests(self):
        """Print test results (equivalent to MATLAB tests)."""
        print("Test: Integration by parts (should be zero):")
        print(self.test_integration_by_parts)
        
        print("\nTest: d²x/dx² = 0 (should be zero):")
        print(self.test_dxx_zero)
        
        print("\nTest: Mb - Gb*T (should be zero):")
        print(self.test_mb_gb)
