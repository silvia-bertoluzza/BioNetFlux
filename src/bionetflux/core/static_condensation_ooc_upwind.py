import numpy as np
from typing import Dict, Tuple

from .problem import Problem
from .static_condensation_base import StaticCondensationBase

# sys.path hack — commented out, use pip install -e . instead
# import sys, os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# python_port_dir = os.path.dirname(os.path.dirname(current_dir))
# if python_port_dir not in sys.path:
#     sys.path.insert(0, python_port_dir)
    
class StaticCondensationOOCUpwind(StaticCondensationBase):
    """
    Static condensation implementation for OrganOnChip problems.
    Python port from MATLAB reference implementation.
    
    Implements the 4-equation OrganOnChip system:
    - u: primary variable (equation 1)
    - omega: auxiliary variable (equation 2) 
    - v: auxiliary variable (equation 3)
    - phi: primary variable (equation 4)

    Flux polynomial orders:
        - Equation 0 (u): P0 flux (1 DOF per element)
        - Equation 1 (ω): P1 flux (2 DOFs per element)
        - Equation 2 (v): P1 flux (2 DOFs per element)
        - Equation 3 (φ): P1 flux (2 DOFs per element)
    """

    def __init__(self, problem, global_disc, elementary_matrices, ipb=0):
        super().__init__(problem, global_disc, elementary_matrices, ipb)
        self.flux_orders = [0, 1, 1, 1]  # P0 for u, P1 for ω, v, φ
    
    def build_matrices(self):
        """
        Build static condensation matrices for OrganOnChip problem.
        Python port from MATLAB scBlocks.m
        
        Returns:
            Dict containing all static condensation matrices
        """
        h = self.discretization.element_length
        
        # Get chi and dchi as callables from problem (set via set_chemotaxis)
        self.chi_func = self.problem.chi
        self.dchi_func = self.problem.dchi
        
        # Get lambda function and its derivative
        self.lambda_func = getattr(self.problem, 'lambda_function', lambda x: np.ones_like(x))
        self.dlambda_func = getattr(self.problem, 'dlambda_function', lambda x: np.zeros_like(x))
        
        # Initialize sc_matrices storage
        self.sc_matrices = {}
        
        
        # Get elementary matrices
        M = h * self.elementary_matrices.get_matrix('M')
        Gb = self.elementary_matrices.get_matrix('Gb')
        T = self.elementary_matrices.get_matrix('T')
        D = self.elementary_matrices.get_matrix('D')
        IM = self.elementary_matrices.get_matrix('IM') / h
        Av = self.elementary_matrices.get_matrix('Av')
        Nhat = self.elementary_matrices.get_matrix('Nhat')
        QUAD = h * self.elementary_matrices.get_matrix('QUAD')
        
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
        
        return self.sc_matrices

    def static_condensation(self, local_trace, local_source=None, **kwargs):
        """
        Perform OrganOnChip static condensation step.
        Python port from MATLAB StaticC.m
        
        Args:
            local_trace: hU = [hu1; homega; hv; hphi] (8x1)
            local_source: rhs = [g1; g2; g3; g4] (8x1)
            prev_local_solution: Optional 8-entry array [u1(2), u2(2), u3(2), u4(2)]
                from the previous Picard iteration.  When provided, bar_lambda and
                barchi are frozen at their values from the previous iterate, and their
                derivative contributions to the Jacobian are dropped.
            prev_flux: Optional flux array from the previous Picard iteration
                (accepted but not used by this implementation; present for interface
                consistency with domain_flux_jump).

        Returns:
            Tuple (bulk_solution, flux, flux_jump, jacobian)
        """
     
        prev_flux = kwargs.get('prev_flux')  # shape: (total_flux_dofs_per_element,) or None on first iteration
        # flux layout: [j(1), tJ_eq2(2), tJ_eq3(2), tJ_eq4(2)] -> eq4 occupies indices 5:7
        
        prev_psi = prev_flux[5:7] if prev_flux is not None else None

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

        # Extract OrganOnChip parameters following MATLAB order
        nu = self.problem.parameters[0]      # viscosity
        mu = self.problem.parameters[1]      # viscosity
        epsilon = self.problem.parameters[2] # viscosity
        sigma = self.problem.parameters[3]   # viscosity
        a = self.problem.parameters[4]       # reaction parameter
        b = self.problem.parameters[5]       # coupling parameter
        c = self.problem.parameters[6]       # reaction parameter
        d = self.problem.parameters[7]     # coupling parameter

        beta = 1 / mu

        h = self.discretization.element_length

        # Get stabilization parameters
        tau = self.discretization.tau if hasattr(self.discretization, 'tau') else [1.0, 1.0, 1.0, 1.0]
        tu = tau[0] / h  # tau for u
        to = tau[1]      # tau for omega
        tv = tau[2]      # tau for v
        tp = tau[3]      # tau for phi
        
        gam = self.discretization.gam if hasattr(self.discretization, 'gam') else 0.0  # Upwinding parameter
        
        
        # Get basic cached matrices
        M, D = self.sc_matrices['M'], self.sc_matrices['D']
        Gb, T = self.sc_matrices['Gb'], self.sc_matrices['T']
        Av = self.sc_matrices['Av']
        Rmat, Rhat = self.sc_matrices['R'], self.sc_matrices['Rhat']

        # Get elementary matrices needed for local static-condensation block assembly
        Mb = self.elementary_matrices.get_matrix('Mb')
        Ntil = self.elementary_matrices.get_matrix('Ntil')
        Nhat = self.elementary_matrices.get_matrix('Nhat')

        normali = np.array([-1.0, 1.0])
        Z = np.zeros((2, 2))

        # Picard mode: read previous-iterate bulk solution to freeze nonlinear
        # coefficients.  prev_local_solution is the 8-entry U from iteration k,
        # structured as [u1(2), u2(2), u3(2), u4(2)].
        prev_local_solution = kwargs.get('prev_local_solution')
        if prev_local_solution is not None:
            prev_flat = np.asarray(prev_local_solution).flatten()
            u2_prev = prev_flat[2:4]
            u4_prev = prev_flat[6:8]
            bar_omega_prev = (Av @ u2_prev).item()
            barphi_prev = (Av @ u4_prev).item()
            bar_lambda_frozen = self.lambda_func(bar_omega_prev)
            barchi_frozen = self.chi_func(barphi_prev)
        else:
            bar_lambda_frozen = None
            barchi_frozen = None

        # Build 2x2 upwind selector matrix O:
        # O(i,i) = 0 if normali(i) * prev_psi(i) >= 0, else 1
        if prev_psi is not None:
            out_diag = np.where(normali * prev_psi >= 0, 0.0, 1.0)
        else:
            out_diag = np.zeros(2)
        # Out = np.diag(out_diag) # outflow selector
        # In = np.eye(2) - Out # inflow selector
        
        # Disactivate upwind choice of uhat in equation for u
        Out = np.zeros((2, 2))
        In = np.eye(2)

       
        # tu_up(i) = max(-prev_psi(i) * normali(i), 0) + tu
        if prev_psi is not None:
            tu_up = gam * np.maximum(barchi_frozen * prev_psi * normali *(-1.0), 0.0) + tu 
        else:
            tu_up = np.full(2, tu)


        # tMb(i,j) = sum_k tu_up(k)*T(k,i)*T(k,j)  =>  T.T @ diag(tu_up) @ T
        tMb = T.T @ np.diag(tu_up) @ T
        # tGb(i,j) = tu_up(j)*T(j,i)  =>  tGb = T.T * tu_up (broadcast tu_up over columns)
        tGb = T.T * tu_up

        # print(f"DEBUG: tMb=\n{tMb}, tGb=\n{tGb}")  # Debug statement

        # Step 1: Compute u
        # Matrix for u equation
        A1 = M + dt * tMb
        # A1 = M + dt * Mb
        L1 = np.linalg.inv(A1)
        H1 = dt * tGb
        # H1 = dt * Gb
        B1 = L1 @ H1
 
        y1 = L1 @ g[0]
        u1 = B1 @ hu[0] + y1
        
        # Step 2: Compute omega  
        # Matrices for omega equation
        E1 = dt * (Ntil - D) @ Rmat
        E1hat = dt * (Ntil - D) @ Rhat

        A2 = M + epsilon * E1 + dt * to * Mb + dt * c * M
        L2 = np.linalg.inv(A2)
        H2 = dt * d * M @ B1
        K2 = epsilon * E1hat + to * dt * Gb
        # ATTENTION - B2 and C2 were switched with respect to notes / now fixed
        B2 = L2 @ K2
        C2 = L2 @ H2

        y2 = L2 @ (g[1] + self.dt * d * M @ y1)
        u2 = C2 @ hu[0] + B2 @ hu[1] + y2
        
        # Step 3: Compute average omega and lambda values.
        # In Picard mode bar_lambda is frozen from the previous iterate.
        bar_omega = Av @ u2
        if bar_lambda_frozen is not None:
            bar_lambda = bar_lambda_frozen
            dbar_lambda = 0.0
        else:
            bar_lambda = self.lambda_func(bar_omega)
            dbar_lambda = self.dlambda_func(bar_omega)
        
        # Step 4: Compute v (omega-dependent)
        # Step 3: Matrices for v equation
        A3 = M + sigma * E1 + dt * tv * Mb
        S3 = dt * M
        H3 = sigma * E1hat + dt * tv * Gb

        L3 = np.linalg.inv(A3 + bar_lambda * S3)
        y3 = L3 @ g[2]
        B3 = L3 @ H3
        u3 = B3 @ hu[2] + y3
        
        # Step 5: Compute phi
        # Matrices for phi equation
        A4 = M + mu * E1 + dt * tp * Mb + dt * a * M
        H4 = mu * E1hat + dt * tp * Gb
        K4 = dt * b * M
        L4 = np.linalg.inv(A4)
        # ATTENTION - B4 and C4 were switched with respect to notes / now fixed
        B4 = L4 @ H4
        C4 = L4 @ K4
        
        u4 = B4 @ hu[3] + C4 @ u3 + L4 @ g[3]
        
        # Assemble bulk solution U = [u1; u2; u3; u4]
        U = np.concatenate([u1, u2, u3, u4])
        
        # Step 5b: Compute average phi and evaluate chi.
        # In Picard mode barchi is frozen from the previous iterate.
        barphi = (Av @ u4).item()
        if barchi_frozen is not None:
            barchi = barchi_frozen
            dbarchi = 0.0
        else:
            barchi = self.chi_func(barphi)
            dbarchi = self.dchi_func(barphi)

        # Compute Jacobian for Newton method
        # Initialize JAC following MATLAB logic
        JAC = np.zeros((8, 8))
         # Matrices for flux jump construction
        D1 = np.block([
            [Z, epsilon * Rmat, Z, Z],
            [Z, Z, sigma * Rmat, Z],
            [Z, Z, Z, mu * Rmat]
        ])

        D2 = np.block([
            [Z, epsilon * Rhat, Z, Z],
            [Z, Z, sigma * Rhat, Z],
            [Z, Z, Z, mu * Rhat]
        ])


        # Matrices for j construction
        hB4 = -nu * np.concatenate([normali, np.zeros(6)]).reshape(1, -1) / h
        hOut = np.block([[Out, Z, Z, Z],
                        [Z, Z, Z, Z],
                        [Z, Z, Z, Z],
                        [Z, Z, Z, Z]]
                    )
        
        hIn = np.block([[In, Z, Z, Z],
                        [Z, Z, Z, Z],
                        [Z, Z, Z, Z],
                        [Z, Z, Z, Z]]
                )

        Q = -nu * beta * np.block([
            [Z, Z, Z, Z],
            [Z, Z, Z, Z],
            [M, Z, Z, Z]
        ]) / h

        # Matrices for final flux jump assembly
        # B5 = -nu * normali / h
        B5 = normali  # 1 x 2 matrix


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
        
        # Construction of j and dj (Q is multiplied by barchi at runtime)
        j = hB4 @ hU + barchi * tJ.T @ Q @ U
        # j = hB4 @ hIn @ hU + hB4 @ hOut @ U + barchi * tJ.T @ Q @ U
        
        # WARNING: the formula for dbarphi_dhU needs to be checked against the theory
        dbarphi_dhU = Av @ R[3] @ JAC                           # (1, 8)
        dj = hB4 @ (hIn + hOut @ R[0].T @ B1 @ R[0]) - barchi * tJ.T @ Q @ JAC - barchi * U.T @ Q.T @ dtJ \
             - dbarchi * (tJ.T @ Q @ U) * dbarphi_dhU            # (1, 8)
        # Final flux jumps
        
        B6 =  np.diag(tu_up) @ T @ np.block([np.eye(2), Z, Z, Z])
        B7 = -np.diag(tu_up) @ np.block([np.eye(2), Z, Z, Z])
        
        B5 = B5.reshape(1, -1)  # Ensure B5 is 1x2 / B5 = normali, but we need it as a 1x2 matrix for the multiplication below
        hj = B5.T @ j + B6 @ U + B7 @ hU
        
        dhj = B5.T @ dj  +  B6 @ JAC + B7
        
        # hJ_rest: numerical flux contribution from equations 2-4 (omega, v, phi)
        hJ_rest = hatB0 @ tJ + hatB1 @ U - hatB2 @ hU
        dhJ_rest = hatB0 @ dtJ + hatB1 @ JAC - hatB2
         
        # Combine flux jumps 
        flux_jump = np.concatenate([hj.flatten(), hJ_rest.flatten()])
        
        jacobian = np.vstack([dhj, dhJ_rest])

        # Return in expected format
        bulk_solution = U.reshape(-1, 1)
        
        flux = np.concatenate([j.flatten(), tJ.flatten()])
        
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