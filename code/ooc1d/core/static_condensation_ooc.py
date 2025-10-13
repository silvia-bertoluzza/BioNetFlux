import numpy as np
import sys
import os
from typing import Dict, Tuple

from .problem import Problem
from .static_condensation_base import StaticCondensationBase

# Add the python_port directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
python_port_dir = os.path.dirname(os.path.dirname(current_dir))
if python_port_dir not in sys.path:
    sys.path.insert(0, python_port_dir)
    
class StaticCondensationOOC(StaticCondensationBase):
    """
    Static condensation implementation for OrganOnChip problems.
    Python port from MATLAB reference implementation.
    
    Implements the 4-equation OrganOnChip system:
    - u: primary variable (equation 1)
    - omega: auxiliary variable (equation 2) 
    - v: auxiliary variable (equation 3)
    - phi: primary variable (equation 4)
    
    """
    
    def build_matrices(self):
        """
        Build static condensation matrices for OrganOnChip problem.
        Python port from MATLAB scBlocks.m
        
        Returns:
            Dict containing all static condensation matrices
        """
        # Extract OrganOnChip parameters following MATLAB order
        nu = self.problem.parameters[0]      # viscosity
        mu = self.problem.parameters[1]      # viscosity 
        epsilon = self.problem.parameters[2] # viscosity
        sigma = self.problem.parameters[3]   # viscosity
        a = self.problem.parameters[4]       # reaction parameter
        b = self.problem.parameters[5]       # coupling parameter
        c = self.problem.parameters[6]       # reaction parameter
        d = self.problem.parameters[7]       # coupling parameter
        chi = self.problem.parameters[8]     # coupling parameter
        
        alpha = 1/nu
        beta = 1/mu
        
        
        # **TODO: need to check that the lambda functions are correctly defined
        # Get lambda function and its derivative
        self.lambda_func = getattr(self.problem, 'lambda_function', lambda x: np.ones_like(x))
        self.dlambda_func = getattr(self.problem, 'dlambda_function', lambda x: np.zeros_like(x))
        
        # Get stabilization parameters
        tau = self.discretization.tau if hasattr(self.discretization, 'tau') else [1.0, 1.0, 1.0, 1.0]
        tu = tau[0]    # tau for u
        to = tau[1]    # tau for omega  
        tv = tau[2]    # tau for v
        tp = tau[3]    # tau for phi
        
        # Cache frequently used values
        dt = self.dt  
        h = self.discretization.element_length
    
        
        # Initialize sc_matrices storage
        self.sc_matrices = {}
        
        
        # Get elementary matrices
        M = h * self.elementary_matrices.get_matrix('M')
        Mb = self.elementary_matrices.get_matrix('Mb')
        Gb = self.elementary_matrices.get_matrix('Gb')
        T = self.elementary_matrices.get_matrix('T')
        D = self.elementary_matrices.get_matrix('D')
        IM = self.elementary_matrices.get_matrix('IM') / h
        Av = self.elementary_matrices.get_matrix('Av')
        Ntil = self.elementary_matrices.get_matrix('Ntil')
        Nhat = self.elementary_matrices.get_matrix('Nhat')
        QUAD = h * self.elementary_matrices.get_matrix('QUAD')
        
        normali = np.array([-1.0, 1.0])
        Z = np.zeros((2, 2))
        
        # Store basic matrices
        self.sc_matrices.update({
            'M': M,
            'D': D,
            'Gb': Gb,
            'T': T,
            'IM': IM,
            'Av': Av,
            'QUAD': QUAD
        })
        
        # Compute derived matrices
        R = IM @ D # Checked
        Rhat = IM @ Nhat # Checked
        self.sc_matrices.update({'R': R, 'Rhat': Rhat})
        
        # Step 1: Matrix for u equation
        A1 = M + dt * tu * Mb # type: ignore / Checked
        L1 = np.linalg.inv(A1) # Checked
        H1 = dt * tu * Gb # type: ignore / Checked
        B1 = L1 @ H1
        
        self.sc_matrices.update({'L1': L1, 'B1': B1})
        
        # Step 2: Matrices for omega equation  
        E1 = dt * (Ntil - D) @ R # Checked
        E1hat = dt * (Ntil - D) @ Rhat # Checked

        A2 = M + epsilon * E1 + dt * to * Mb + dt * c * M # type: ignore / Checked
        L2 = np.linalg.inv(A2) # Checked
        H2 = dt * d * M @ B1 # type: ignore / Checked
        K2 = epsilon * E1hat + to * dt * Gb # type: ignore / Checked
        # ATTENTION - B2 and C2 were switched with respect to notes / now fixed
        B2 = L2 @ K2 # Checked
        C2 = L2 @ H2 # Checked

        self.sc_matrices.update({
            'L2': L2, 'B2': B2, 'C2': C2,
            'E1': E1, 'E1hat': E1hat
        })
        
        # Step 3: Matrices for v equation
        A3 = M + sigma * E1 + dt * tv * Mb # type: ignore / Checked
        S3 = dt * M
        H3 = sigma * E1hat + dt * tv * Gb # type: ignore / Checked
        
        self.sc_matrices.update({
            'A3': A3, 'S3': S3, 'H3': H3
        })
        
        # Step 4: Matrices for phi equation
        A4 = M + mu * E1 + dt * tp * Mb + dt * a * M # type: ignore / Checked
        H4 = mu * E1hat + dt * tp * Gb # type: ignore / Checked
        K4 = dt * b * M # Checked
        L4 = np.linalg.inv(A4) # Checked
        # ATTENTION - B4 and C4 were switched with respect to notes / now fixed
        B4 = L4 @ H4 # Checked
        C4 = L4 @ K4 # Checked
        L4tilde = dt * b * L4 @ M # Needed? Do not know if used

        self.sc_matrices.update({
            'L4': L4, 'B4': B4, 'C4': C4, 'L4tilde': L4tilde
        })
        
        # Matrices for flux jump construction
        D1 = np.block([
            [Z, epsilon * R, Z, Z],
            [Z, Z, sigma * R, Z],
            [Z, Z, Z, mu * R]
        ]) # Checked
        
        D2 = np.block([
            [Z, epsilon * Rhat, Z, Z],
            [Z, Z, sigma * Rhat, Z], 
            [Z, Z, Z, mu * Rhat]
        ]) # Checked
        
        self.sc_matrices.update({'D1': D1, 'D2': D2})
        
        # Matrices for j construction
        hB4 = -nu * np.concatenate([normali, np.zeros(6)]).reshape(1, -1) / h # Checked
        
        Q = -nu * beta * chi * np.block([
            [Z, Z, Z, Z],
            [Z, Z, Z, Z],
            [M, Z, Z, Z]
        ]) / h
        
        self.sc_matrices.update({'hB4': hB4, 'Q': Q})
        
        # Matrices for final flux jump assembly
        # B5 = - nu * normali / h
        B5 = normali #1 x 2 matrix
        B6 = tu * T @ np.block([np.eye(2), Z, Z, Z])
        B7 = -tu * np.block([np.eye(2), Z, Z, Z])
        
        hatB0 = np.block([
            [Nhat.T, Z, Z],
            [Z, Nhat.T, Z],
            [Z, Z, Nhat.T]
        ])
        
        tau_diag = np.diag([to, to, tv, tv, tp, tp])
        hatB1 = tau_diag @ np.block([
            [Z, T, Z, Z],
            [Z, Z, T, Z], 
            [Z, Z, Z, T]
        ])
        
        hatB2 = tau_diag @ np.block([
            [Z, np.eye(2), Z, Z],
            [Z, Z, np.eye(2), Z],
            [Z, Z, Z, np.eye(2)]
        ])
        
        self.sc_matrices.update({
            'B5': B5, 'B6': B6, 'B7': B7,
            'hatB0': hatB0, 'hatB1': hatB1, 'hatB2': hatB2
        })
        
        return self.sc_matrices

    def static_condensation(self, local_trace, local_source=None, **kwargs):
        """
        Perform OrganOnChip static condensation step.
        Python port from MATLAB StaticC.m
        
        Args:
            local_trace: hU = [hu1; homega; hv; hphi] (8x1)
            local_source: rhs = [g1; g2; g3; g4] (8x1)  
            
        Returns:
            Tuple (bulk_solution, flux_jump, jacobian)
        """
        
        # Handle None local_source
        if local_source is None:
            local_source = np.zeros(8)
        
        # Ensure proper shapes
        if local_trace.ndim == 1:
            local_trace = local_trace.reshape(-1, 1)
        if local_source.ndim == 1:
            local_source = local_source.reshape(-1, 1)
            
        # Validate dimensions
        if local_trace.shape[0] != 8:
            raise ValueError(f"local_trace must be 8x1 for OrganOnChip (4 eqs), got {local_trace.shape}")
        if local_source.shape[0] != 8:
            raise ValueError(f"local_source must be 8x1 for OrganOnChip (4 eqs), got {local_source.shape}")
        
        # Extract components following MATLAB StaticC.m
        hu = [local_trace[2*i:2*i+2] for i in range(4)]
        g = [local_source[2*i:2*i+2] for i in range(4)]
        dt = self.dt
        d = self.problem.parameters[7]     # coupling parameter
        
        # Get matrices
        L1, B1 = self.sc_matrices['L1'], self.sc_matrices['B1']
        L2, B2, C2 = self.sc_matrices['L2'], self.sc_matrices['B2'], self.sc_matrices['C2']
        L4, B4, C4 = self.sc_matrices['L4'], self.sc_matrices['B4'], self.sc_matrices['C4']
        A3, S3, H3 = self.sc_matrices['A3'], self.sc_matrices['S3'], self.sc_matrices['H3']
        M, Av = self.sc_matrices['M'], self.sc_matrices['Av']
        D1, D2 = self.sc_matrices['D1'], self.sc_matrices['D2']
        hB4, Q = self.sc_matrices['hB4'], self.sc_matrices['Q']
        B5, B6, B7 = self.sc_matrices['B5'], self.sc_matrices['B6'], self.sc_matrices['B7']
        hatB0, hatB1, hatB2 = self.sc_matrices['hatB0'], self.sc_matrices['hatB1'], self.sc_matrices['hatB2']
        
        # Step 1: Compute u
        y1 = L1 @ g[0]
        u1 = B1 @ hu[0] + y1
        
        # Step 2: Compute omega  
        y2 = L2 @ (g[1] + self.dt * d * M @ y1)
        u2 = C2 @ hu[0] + B2 @ hu[1] + y2
        
        # Step 3: Compute average omega and lambda values
        bar_omega = Av @ u2
        bar_lambda = self.lambda_func(bar_omega)
        dbar_lambda = self.dlambda_func(bar_omega)
        
        # Step 4: Compute v (omega-dependent)
        L3 = np.linalg.inv(A3 + bar_lambda * S3)
        y3 = L3 @ g[2]
        B3 = L3 @ H3
        u3 = B3 @ hu[2] + y3
        
        # Step 5: Compute phi
        u4 = B4 @ hu[3] + C4 @ u3 + L4 @ g[3]
        
        # Assemble bulk solution U = [u1; u2; u3; u4]
        U = np.concatenate([u1, u2, u3, u4])
        
        # Compute Jacobian for Newton method
        # Initialize JAC following MATLAB logic
        JAC = np.zeros((8, 8))
        
        # Restriction matrices
        R = [np.zeros((2, 8)) for _ in range(4)]
        for i in range(4):
            R[i][:, 2*i:2*i+2] = np.eye(2)
        
        # Build Jacobian following MATLAB StaticC.m
        JAC += R[0].T @ B1 @ R[0]
        JAC += R[1].T @ (C2 @ R[0] + B2 @ R[1])
        
        # Jacobian for v equation (omega-dependent)
        J0 = L3 @ H3
        J1 = dbar_lambda * L3 @ S3 @ (H3 @ hu[2] + g[2]) @ Av
        JAC += R[2].T @ (J0 @ R[2] + J1 @ R[2] @ JAC) # DO CHECK: was R[1] in previous version
        
        JAC += R[3].T @ (B4 @ R[3] + C4 @ R[2] @ JAC)
        
        # Compute flux jumps
        hU = local_trace
        tJ = D1 @ U - D2 @ hU
        dtJ = D1 @ JAC - D2
        
        # Construction of j and dj
        j = hB4 @ hU + tJ.T @ Q @ U
        dj = hB4 - tJ.T @ Q @ JAC - U.T @ Q.T @ dtJ
        # dj = R[0] * dj  # Restrict to u equation
        # Final flux jumps
        print(f"DEBUG: B5 = {B5.shape}, dj = {dj.shape}, hB4 = {hB4.shape}")  # Debug print
        
        B5 = B5.reshape(1, -1)  # Ensure B5 is 1x2
        hj = B5.T @ j + B6 @ U + B7 @ hU
        print(f"DEBUG: hj = {hj.shape}, j = {j.shape}")  # Debug print
        print(f"DEBUG: j = {j}, hj = {hj}")  # Debug print
        print(f"DEBUG: R0 = {R[0].shape}")  # Debug print
        
        dhj = B5 @ R[0] @ dj.T  + B6 @ JAC + B7
        print(f"DEBUG: dhj = {dhj.shape}, dj = {dj.shape}")  # Debug print
        
        
        hJ_rest = hatB0 @ tJ + hatB1 @ U - hatB2 @ hU
        dhJ_rest = hatB0 @ dtJ + hatB1 @ JAC - hatB2
        print(f"DEBUG: hJ_rest = {hJ_rest.shape}, tJ = {tJ.shape}")  # Debug print
        print(f"DEBUG: dhJ_rest = {dhJ_rest.shape}, dtJ = {dtJ.shape}")  # Debug print
         
        # Combine flux jumps
        flux_jump = np.concatenate([hj.flatten(), hJ_rest.flatten()])
        print(f"DEBUG: flux_jump = {flux_jump.shape}")  # Debug print
        
        jacobian = np.vstack([dhj, dhJ_rest])
        print(f"DEBUG: jacobian = {jacobian.shape}")  # Debug print
        
        # Return in expected format
        bulk_solution = U.reshape(-1, 1)
        
        flux = None  # Placeholder if needed
        
        return bulk_solution, flux, flux_jump, jacobian

    def assemble_forcing_term(self, 
                                previous_bulk_solution: np.ndarray, 
                                external_force: np.ndarray) -> np.ndarray:
        """
        Assemble right-hand side for static condensation system.
    
        Computes: dt * external_forces + M * previous_bulk_solution
    Args:
        previous_bulk_solution: Bulk solution from previous time step
        external_forces: External force terms (discrete form)
        
    Returns:
        Assembled right-hand side vector
        
    Raises:
        ValueError: If dimensions are incompatible
        KeyError: If matrices haven't been built
        """
        
  
        if 'M' not in self.sc_matrices:
            raise KeyError("Matrices not built. Call build_matrices() first.")
        
          
        M = self.sc_matrices.get('M', None)
        
        # Validate dimensions
        if previous_bulk_solution.shape[0] !=  4 * M.shape[1]:
            raise ValueError(f"Incompatible dimensions: M is {M.shape}, "
                            f"previous_bulk_solution is {previous_bulk_solution.shape}")

        if external_force.shape != previous_bulk_solution.shape:
            raise ValueError(f"Shape mismatch: external_force {external_force.shape} "
                            f"!= previous_bulk_solution {previous_bulk_solution.shape}")

             # Method 1: Using np.block (most readable)
        Z = np.zeros_like(M)
        M_block = np.block([[M, Z, Z, Z],
                           [Z, M, Z, Z],
                           [Z, Z, M, Z],
                           [Z, Z, Z, M]])

        right_hand_side = self.dt * external_force.copy() + M_block @ previous_bulk_solution
        return right_hand_side