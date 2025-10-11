import numpy as np
import sys
import os
from typing import Dict, Tuple

# Add the python_port directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
python_port_dir = os.path.dirname(os.path.dirname(current_dir))
if python_port_dir not in sys.path:
    sys.path.insert(0, python_port_dir)

from ooc1d.core.static_condensation_base import StaticCondensationBase


class KellerSegelStaticCondensation(StaticCondensationBase):
    """
    Static condensation implementation for Keller-Segel problems.
    Equivalent to MATLAB scBlocks.m and StaticC.m for Keller-Segel model.
    """
  
    def build_matrices(self) -> Dict[str, np.ndarray]:
        """
        Build static condensation matrices for Keller-Segel model.
        Equivalent to MATLAB scBlocks.m.
        """
        # Extract parameters from problem
        mu = self.problem.parameters[0]
        nu = self.problem.parameters[1]
        a = self.problem.parameters[2]
        b = self.problem.parameters[3]
        beta = 1.0 / mu
        # alpha = 1.0 / nu
        
        dt = self.dt
        h = self.discretization.element_length
        
        # Get elementary matrices
        eM = self.elementary_matrices.get_matrix('M')
        eD = self.elementary_matrices.get_matrix('D')
        eT = self.elementary_matrices.get_matrix('T')
        eIM = self.elementary_matrices.get_matrix('IM')
        eMb = self.elementary_matrices.get_matrix('Mb')
        eGb = self.elementary_matrices.get_matrix('Gb')
        eNtil = self.elementary_matrices.get_matrix('Ntil')
        eNhat = self.elementary_matrices.get_matrix('Nhat')
        eQUAD = self.elementary_matrices.get_matrix('QUAD')
        
        # Get stabilization parameters first
        tu = self.discretization.tau[0]  # tau for u equation
        tp = self.discretization.tau[1]  # tau for phi equation
        
        # Handle M matrix - eM might be diagonal elements
        if eM.ndim == 1:
            # eM is diagonal vector, create 2x2 diagonal matrix
            M = h * np.diag(eM[:2])
        else:
            # eM is already 2D matrix
            M = h * eM
            if M.shape != (2, 2):
                M = M[:2, :2]
        
        # Scale other matrices
        QUAD = h * eQUAD
        Mb = eMb
        Gb = eGb
        T = eT
        D = eD
        IM = eIM / h
        Ntil = eNtil
        Nhat = eNhat
        
        # Build matrices following MATLAB scBlocks.m exactly
        normali = np.array([[-1], [1]])
        Z = np.zeros((2, 2))
        
        # Step 1: Build L1, B1 for u equation
        A1 = M + dt * tu * Mb
        L1 = np.linalg.inv(A1)
        B1 = L1 @ (dt * tu * Gb)
        
        # Step 2: Build L2, B2, C2 for phi equation
        A2 = (M + mu * dt * (Ntil - D) @ IM @ D +
              dt * tp * Mb + dt * a * M)
        H2 = dt * b * M @ B1
        K2 = mu * dt * (Ntil - D) @ IM @ Nhat + dt * tp * Gb
        
        L2 = np.linalg.inv(A2)
        B2 = L2 @ H2
        C2 = L2 @ K2
        
        # Step 3: Build B3, C3 for psi equation
        B3 = mu * IM @ D @ B2
        C3 = mu * IM @ (D @ C2 - Nhat)
        
        # Step 4: Build combined B0 matrix
        B0 = np.block([[B1, np.zeros((2, 2))],
                       [B2, C2],
                       [B3, C3]])
        
        # Step 5: Build B4 for flux
        B4 = -nu * np.block([normali.T, np.zeros((1, 2))]) / h
        
        # Step 6: Build Q matrix for nonlinear term
        Q = -nu * beta * np.block([[Z, Z, M],
                                   [Z, Z, Z],
                                   [Z, Z, Z]]) / h
        
        # Step 7: Build Av for averaging
        Av = np.block([[np.zeros((1, 2)),
                        np.array([[1, 1]]) @ eM,
                        np.zeros((1, 2))]])
        
        # Step 8: Build hat matrices for flux traces
        B5 = normali
        B6 = tu * np.block([T, Z, Z])
        B7 = -tu * np.block([np.eye(2), Z])
        
        B8 = np.block([Z, tp * T, Nhat.T])
        B9 = -tp * np.block([Z, np.eye(2)])
        
        B1hat = np.block([[B5], [np.zeros((2, 1))]])
        B2hat = np.block([[B6], [B8]])
        B3hat = np.block([[B7], [B9]])
        
        # Store all matrices
        self.sc_matrices = {
            'L1': L1,
            'L2': L2,
            'B0': B0,
            'B1': B1,
            'B2': B2,
            'C2': C2,
            'B3': B3,
            'C3': C3,
            'B4': B4,
            'Q': Q,
            'Av': Av,
            'B1hat': B1hat,
            'B2hat': B2hat,
            'B3hat': B3hat,
            'D': D,
            'Gb': Gb,
            'T': T,
            'IM': IM,
            'QUAD': QUAD,
            'M': M,
            'H2': H2,  # For debugging
            'K2': K2,  # For debugging
        }
        
        return self.sc_matrices
    
    def static_condensation(
            self,
            local_trace: np.ndarray,
            local_source: np.ndarray = None,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Keller-Segel static condensation step.
        
        The local_trace is structured as follows:
        - For element k with neq equations and 2 nodes per element
        - local_trace is a vector of length 2*neq
        - Contains trace values at the vertices of element k for each equation
        - Format: [u_left, u_right, phi_left, phi_right] for 2 equations
        
        Args:
            local_trace: Local trace vector of length 2*neq for element k
            local_source: Source term vector (optional, defaults to zero)
            
        Returns:
            tuple: (local_solution, flux, flux_trace, jacobian)
        """
        
        # Handle None local_source
        if local_source is None:
            local_source = np.zeros((4, 1))
        
        # Dimensional checks - local_trace should be 2*neq for element k
        # For Keller-Segel: neq=2, so local_trace should be length 4
        expected_length = 4  # 2*neq for neq=2 equations
        if local_trace.shape not in [(expected_length,), (expected_length, 1)]:
            raise ValueError(f"local_trace must be a {expected_length}x1 vector for element k, "
                           f"got shape {local_trace.shape}. "
                           f"Expected format: [u_left, u_right, phi_left, phi_right]")
        
        if local_source.shape not in [(expected_length,), (expected_length, 1)]:
            raise ValueError(f"local_source must be a {expected_length}x1 vector, got shape {local_source.shape}")
        
        # Ensure both are column vectors (4, 1)
        local_trace = local_trace.reshape(-1, 1) if local_trace.ndim == 1 else local_trace
        local_source = local_source.reshape(-1, 1) if local_source.ndim == 1 else local_source
        
        # Extract matrices
        L1 = self.sc_matrices['L1']
        L2 = self.sc_matrices['L2']
        M = self.sc_matrices['M']
        IM = self.sc_matrices['IM'] 
        D = self.sc_matrices['D']
        B0 = self.sc_matrices['B0']
        B4 = self.sc_matrices['B4']
        Q = self.sc_matrices['Q']
        Av = self.sc_matrices['Av']
        B1hat = self.sc_matrices['B1hat']
        B2hat = self.sc_matrices['B2hat']
        B3hat = self.sc_matrices['B3hat']
        
        # Step 0: Preparation of the source term
        gu = local_source[[0, 1]]  # Entries 0 and 1
        gp = local_source[[2, 3]]  # Entries 2 and 3
        
        g1 = L1 @ gu
        g2 = L2 @ (gp + self.dt * self.problem.parameters[3] * M @ g1)
        g3 = self.problem.parameters[0] * IM @ D @ g2
        
        # Concatenate g1, g2, and g3 vertically
        G0 = np.vstack([g1, g2, g3])
        
        # Step 1: Trace to solution mapping
        local_solution = B0 @ local_trace + G0
        
        # Step 2: Compute average phi for chemotaxis
        phi_avg = float(Av @ local_solution)
        
        # Get chi value at average phi
        chi_val = (self.problem.chi(phi_avg) if 
                   hasattr(self.problem, 'chi') else 1.0) # Default chi=1.0 if not defined
        
        # Step 3: Compute flux
        linear_flux = B4 @ local_trace
        nonlinear_flux = chi_val * (local_solution.T @ Q @ local_solution)
        flux = linear_flux + nonlinear_flux

        # Step 4: Compute flux trace
        flux_trace = (B1hat @ flux +
                      B2hat @ local_solution +
                      B3hat @ local_trace)

        # Step 5: Compute Jacobian for Newton's method
        jacobian = self._compute_jacobian(local_trace, local_solution, phi_avg)
        
        return local_solution, flux, flux_trace, jacobian
    
    def _compute_jacobian(self, local_trace, local_solution, phi_avg):
        """
        Compute Jacobian matrix for Newton's method.
        
        MATLAB equivalent:
        BU5 = barchi * U'* (scMatrices.Q + scMatrices.Q')... 
              + (U' * scMatrices.Q * U) * dchi(barphi) * scMatrices.Av
        hJjacobian =  scMatrices.B1hat * BU5 * scMatrices.B0 ...
                      + scMatrices.B1hat * scMatrices.B4 + scMatrices.B2hat * scMatrices.B0 ...
                      + scMatrices.B3hat
        """
        
        # Get chi and dchi values
        chi_val = (self.problem.chi(phi_avg) if 
                   hasattr(self.problem, 'chi') else 1.0)
        dchi_val = (self.problem.dchi(phi_avg) if 
                    hasattr(self.problem, 'dchi') else 0.0)
            
        # Get matrices for Jacobian computation
        Av = self.sc_matrices['Av']
        Q = self.sc_matrices['Q']
        B0 = self.sc_matrices['B0']
        B1hat = self.sc_matrices['B1hat']
        B2hat = self.sc_matrices['B2hat']
        B3hat = self.sc_matrices['B3hat']
        B4 = self.sc_matrices['B4']
        
        # Compute BU5 following MATLAB logic exactly
        # BU5 = barchi * U'* (scMatrices.Q + scMatrices.Q') + (U' * scMatrices.Q * U) * dchi(barphi) * scMatrices.Av
        
        U_T = local_solution.T  # U' in MATLAB
        Q_plus_QT = Q + Q.T     # (Q + Q') in MATLAB
        
        first_term = chi_val * U_T @ Q_plus_QT
        
        quad_form = U_T @ Q @ local_solution  # U' * Q * U (scalar)
        second_term = quad_form * dchi_val * Av
        
        BU5 = first_term + second_term
        
        # Compute hJjacobian following MATLAB logic exactly
        # hJjacobian = B1hat * BU5 * B0 + B1hat * B4 + B2hat * B0 + B3hat
        
        term1 = B1hat @ BU5 @ B0
        term2 = B1hat @ B4
        term3 = B2hat @ B0
        term4 = B3hat
        
        jacobian = term1 + term2 + term3 + term4
        
        return jacobian
    
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
        if previous_bulk_solution.shape[0] != 2 * M.shape[1]:
            raise ValueError(f"Incompatible dimensions: M is {M.shape}, "
                            f"previous_bulk_solution is {previous_bulk_solution.shape}")

        if external_force.shape != previous_bulk_solution.shape:
            raise ValueError(f"Shape mismatch: external_force {external_force.shape} "
                            f"!= previous_bulk_solution {previous_bulk_solution.shape}")

             # Method 1: Using np.block (most readable)
        M_block = np.block([[M, np.zeros_like(M)],
                           [np.zeros_like(M), M]])
        
        right_hand_side = self.dt * external_force.copy() + M_block @ previous_bulk_solution
        return right_hand_side