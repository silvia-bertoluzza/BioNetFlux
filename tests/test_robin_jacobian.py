"""
Tests for Robin BC Jacobian contributions in GlobalAssembler.

Verifies that _add_constraint_jacobian_contributions places the correct
entries for Robin boundary conditions:
  - jacobian[multiplier_row, trace_col] = alpha
  - jacobian[multiplier_row, multiplier_col] = beta
  - jacobian[trace_row, multiplier_col] = 1.0  (coupling term)

Also checks consistency with Dirichlet/Neumann patterns and that the
Robin Jacobian matches a finite-difference approximation of the residual.
"""

import pytest
import numpy as np

from bionetflux.core.constraints import ConstraintManager, ConstraintType
from bionetflux.core.domain_data import DomainData
from bionetflux.core.lean_global_assembly import GlobalAssembler


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_domain_data(neq: int = 1, n_elements: int = 2) -> DomainData:
    """Create a minimal DomainData for a single domain."""
    n_nodes = n_elements + 1
    nodes = np.linspace(0.0, 1.0, n_nodes)
    return DomainData(
        neq=neq,
        n_elements=n_elements,
        nodes=nodes,
        element_length=1.0 / n_elements,
        mass_matrix=np.eye(2),
        trace_matrix=np.eye(n_nodes * neq),
        initial_conditions=[lambda s, t=0: np.zeros_like(s)] * neq,
        forcing_functions=[lambda s, t=0: np.zeros_like(s)] * neq,
    )


def _build_assembler_with_robin(alpha: float, beta: float, neq: int = 1):
    """Build a GlobalAssembler with one domain and one Robin BC.

    Returns (assembler, trace_idx, mult_idx).
    """
    dd = _make_domain_data(neq=neq)

    cm = ConstraintManager()
    # Robin at position 0.0 (node 0) for equation 0
    cm.add_robin(
        equation_index=0,
        domain_index=0,
        position=0.0,
        alpha=alpha,
        beta=beta,
    )
    # Map constraint to domain nodes manually
    cm._node_mappings[0] = [0]  # node 0

    assembler = GlobalAssembler(
        domain_data_list=[dd],
        constraint_manager=cm,
    )

    n_nodes = dd.n_elements + 1
    trace_idx = 0  # eq 0, node 0 in domain 0
    mult_idx = assembler.total_trace_dofs  # first (and only) multiplier

    return assembler, trace_idx, mult_idx


def _build_assembler_with_bc(bc_type: str, neq: int = 1, **kwargs):
    """Build assembler with a single boundary constraint of given type."""
    dd = _make_domain_data(neq=neq)
    cm = ConstraintManager()

    if bc_type == "dirichlet":
        cm.add_dirichlet(equation_index=0, domain_index=0, position=0.0)
    elif bc_type == "neumann":
        cm.add_neumann(equation_index=0, domain_index=0, position=0.0)
    elif bc_type == "robin":
        cm.add_robin(
            equation_index=0, domain_index=0, position=0.0,
            alpha=kwargs["alpha"], beta=kwargs["beta"],
        )
    cm._node_mappings[0] = [0]

    assembler = GlobalAssembler([dd], cm)
    trace_idx = 0
    mult_idx = assembler.total_trace_dofs
    return assembler, trace_idx, mult_idx


# ===========================================================================
#  Robin Jacobian entry tests
# ===========================================================================

class TestRobinJacobianEntries:
    """Verify the three Jacobian entries for a Robin BC."""

    @pytest.mark.parametrize("alpha,beta", [
        (1.0, 1.0),
        (2.5, 0.3),
        (0.0, 1.0),     # pure Neumann-like
        (1.0, 0.0),     # pure Dirichlet-like
    ])
    def test_constraint_row_entries(self, alpha, beta):
        """Constraint row has alpha in trace col, beta in multiplier col."""
        assembler, trace_idx, mult_idx = _build_assembler_with_robin(alpha, beta)
        n = assembler.total_dofs
        jac = np.zeros((n, n))

        assembler._add_constraint_jacobian_contributions(
            jac,
            trace_solutions=[np.zeros(assembler.domain_trace_sizes[0])],
            multipliers=np.zeros(assembler.n_multipliers),
            time=0.0,
        )

        # Constraint row: d(alpha*u + beta*lam - g)/du = alpha
        assert jac[mult_idx, trace_idx] == pytest.approx(alpha)
        # Constraint row: d(alpha*u + beta*lam - g)/dlam = beta
        assert jac[mult_idx, mult_idx] == pytest.approx(beta)

    def test_coupling_term(self):
        """Coupling term jacobian[trace_row, multiplier_col] = 1.0."""
        assembler, trace_idx, mult_idx = _build_assembler_with_robin(2.0, 0.5)
        n = assembler.total_dofs
        jac = np.zeros((n, n))

        assembler._add_constraint_jacobian_contributions(
            jac,
            trace_solutions=[np.zeros(assembler.domain_trace_sizes[0])],
            multipliers=np.zeros(assembler.n_multipliers),
            time=0.0,
        )

        assert jac[trace_idx, mult_idx] == pytest.approx(1.0)

    def test_only_expected_entries_set(self):
        """No spurious entries in constraint/coupling rows."""
        alpha, beta = 3.0, 0.7
        assembler, trace_idx, mult_idx = _build_assembler_with_robin(alpha, beta)
        n = assembler.total_dofs
        jac = np.zeros((n, n))

        assembler._add_constraint_jacobian_contributions(
            jac,
            trace_solutions=[np.zeros(assembler.domain_trace_sizes[0])],
            multipliers=np.zeros(assembler.n_multipliers),
            time=0.0,
        )

        # Count non-zeros — should be exactly 3
        expected_entries = {
            (mult_idx, trace_idx): alpha,
            (mult_idx, mult_idx): beta,
            (trace_idx, mult_idx): 1.0,
        }
        nonzero_rows, nonzero_cols = np.nonzero(jac)
        actual_entries = {(r, c): jac[r, c] for r, c in zip(nonzero_rows, nonzero_cols)}

        assert actual_entries == pytest.approx(expected_entries)


# ===========================================================================
#  Comparison with Dirichlet / Neumann
# ===========================================================================

class TestBCJacobianComparison:
    """Compare Robin Jacobian pattern with Dirichlet and Neumann."""

    def test_dirichlet_entries(self):
        """Dirichlet has (1.0 in trace col) and (1.0 coupling)."""
        assembler, t_idx, m_idx = _build_assembler_with_bc("dirichlet")
        n = assembler.total_dofs
        jac = np.zeros((n, n))
        assembler._add_constraint_jacobian_contributions(
            jac, [np.zeros(assembler.domain_trace_sizes[0])],
            np.zeros(assembler.n_multipliers), 0.0,
        )
        assert jac[m_idx, t_idx] == pytest.approx(1.0)
        assert jac[t_idx, m_idx] == pytest.approx(1.0)
        assert jac[m_idx, m_idx] == pytest.approx(0.0)

    def test_neumann_entries(self):
        """Neumann has (1.0 in multiplier col) and (1.0 coupling)."""
        assembler, t_idx, m_idx = _build_assembler_with_bc("neumann")
        n = assembler.total_dofs
        jac = np.zeros((n, n))
        assembler._add_constraint_jacobian_contributions(
            jac, [np.zeros(assembler.domain_trace_sizes[0])],
            np.zeros(assembler.n_multipliers), 0.0,
        )
        assert jac[m_idx, m_idx] == pytest.approx(1.0)
        assert jac[t_idx, m_idx] == pytest.approx(1.0)
        assert jac[m_idx, t_idx] == pytest.approx(0.0)

    def test_robin_reduces_to_dirichlet(self):
        """Robin with alpha=1, beta=0 matches Dirichlet trace entry."""
        assembler, t_idx, m_idx = _build_assembler_with_bc(
            "robin", alpha=1.0, beta=0.0,
        )
        n = assembler.total_dofs
        jac = np.zeros((n, n))
        assembler._add_constraint_jacobian_contributions(
            jac, [np.zeros(assembler.domain_trace_sizes[0])],
            np.zeros(assembler.n_multipliers), 0.0,
        )
        # Same constraint-row trace entry as Dirichlet
        assert jac[m_idx, t_idx] == pytest.approx(1.0)
        assert jac[m_idx, m_idx] == pytest.approx(0.0)

    def test_robin_reduces_to_neumann(self):
        """Robin with alpha=0, beta=1 matches Neumann multiplier entry."""
        assembler, t_idx, m_idx = _build_assembler_with_bc(
            "robin", alpha=0.0, beta=1.0,
        )
        n = assembler.total_dofs
        jac = np.zeros((n, n))
        assembler._add_constraint_jacobian_contributions(
            jac, [np.zeros(assembler.domain_trace_sizes[0])],
            np.zeros(assembler.n_multipliers), 0.0,
        )
        assert jac[m_idx, m_idx] == pytest.approx(1.0)
        assert jac[m_idx, t_idx] == pytest.approx(0.0)


# ===========================================================================
#  Finite-difference consistency
# ===========================================================================

class TestRobinJacobianFD:
    """Check Robin Jacobian against finite-difference of the residual."""

    def test_fd_consistency(self):
        """Robin constraint-row Jacobian matches FD of alpha*u + beta*lam - g."""
        alpha, beta = 2.0, 0.5
        assembler, trace_idx, mult_idx = _build_assembler_with_robin(alpha, beta)
        n = assembler.total_dofs

        # Analytic Jacobian
        jac = np.zeros((n, n))
        assembler._add_constraint_jacobian_contributions(
            jac,
            [np.zeros(assembler.domain_trace_sizes[0])],
            np.zeros(assembler.n_multipliers),
            0.0,
        )

        # The constraint residual is:  r = alpha * u + beta * lambda - g
        # where u = global_solution[trace_idx], lambda = global_solution[mult_idx].
        # We check dr/d(trace_idx) and dr/d(mult_idx) via finite differences.

        def robin_residual(sol):
            u_val = sol[trace_idx]
            lam_val = sol[mult_idx]
            return alpha * u_val + beta * lam_val  # g=0 for default

        x0 = np.zeros(n)
        eps = 1e-7

        # dr/du
        x_plus = x0.copy(); x_plus[trace_idx] += eps
        x_minus = x0.copy(); x_minus[trace_idx] -= eps
        fd_du = (robin_residual(x_plus) - robin_residual(x_minus)) / (2 * eps)
        assert jac[mult_idx, trace_idx] == pytest.approx(fd_du, abs=1e-6)

        # dr/dlam
        x_plus = x0.copy(); x_plus[mult_idx] += eps
        x_minus = x0.copy(); x_minus[mult_idx] -= eps
        fd_dlam = (robin_residual(x_plus) - robin_residual(x_minus)) / (2 * eps)
        assert jac[mult_idx, mult_idx] == pytest.approx(fd_dlam, abs=1e-6)


# ===========================================================================
#  Visual inspection
# ===========================================================================

class TestRobinJacobianPrint:
    """Print the full Jacobian for manual inspection (run with -s)."""

    def test_print_jacobian_2_elements(self):
        """Print constraint Jacobian for Robin BC on a 2-element, 1-eq domain.

        Domain layout (neq=1, n_elements=2 → 3 nodes):
            node 0 --- node 1 --- node 2
            (Robin)

        DOF ordering:
            0..2 : trace DOFs (u at nodes 0, 1, 2)
            3    : Lagrange multiplier (lambda)

        Run with ``pytest -s -k test_print_jacobian_2_elements`` to see output.
        """
        alpha, beta = 2.0, 0.5
        assembler, trace_idx, mult_idx = _build_assembler_with_robin(
            alpha, beta, neq=1,
        )
        n = assembler.total_dofs
        jac = np.zeros((n, n))

        assembler._add_constraint_jacobian_contributions(
            jac,
            [np.zeros(assembler.domain_trace_sizes[0])],
            np.zeros(assembler.n_multipliers),
            0.0,
        )

        # Build labels
        dd = assembler.bulk_manager.domain_data_list[0]
        n_nodes = dd.n_elements + 1
        labels = [f"u[{i}]" for i in range(n_nodes)] + ["λ"]

        print()
        print("=" * 60)
        print("Robin Boundary Condition")
        print("=" * 60)
        print()
        print("General form:  α · u + β · (∂u/∂n) = g(t)")
        print()
        print(f"Test values:   α = {alpha},  β = {beta}")
        print(f"Specific form: {alpha} · u + {beta} · (∂u/∂n) = g(t)")
        print(f"Data function: g(t) = 0  (homogeneous)")
        print()
        print("In the saddle-point formulation the flux ∂u/∂n is")
        print("represented by the Lagrange multiplier λ, so the")
        print("discrete constraint equation is:")
        print()
        print(f"  {alpha} · u[0] + {beta} · λ - g = 0")
        print()
        print("Linearisation (Jacobian row for this constraint):")
        print(f"  ∂F/∂u[0] = α = {alpha}")
        print(f"  ∂F/∂λ    = β = {beta}")
        print(f"  Coupling:  ∂(trace eq)/∂λ = 1.0")
        print()
        print("-" * 60)
        print(f"Domain: 1 domain, {dd.n_elements} elements, {n_nodes} nodes, neq=1")
        print(f"Total DOFs: {n} (trace: {assembler.total_trace_dofs}, multipliers: {assembler.n_multipliers})")
        print("-" * 60)
        print()

        # Header
        col_w = 8
        header = " " * 7 + "".join(f"{lbl:>{col_w}}" for lbl in labels)
        print(header)
        print(" " * 7 + "-" * (col_w * len(labels)))

        # Rows
        for i, row_label in enumerate(labels):
            vals = "".join(f"{jac[i, j]:>{col_w}.3f}" for j in range(n))
            print(f"{row_label:>6} |{vals}")

        print()
        print("Expected non-zero entries:")
        print(f"  J[λ, u[0]]  = alpha = {alpha}")
        print(f"  J[λ, λ]     = beta  = {beta}")
        print(f"  J[u[0], λ]  = 1.0   (coupling)")

        # Sanity check so the test still validates something
        assert jac[mult_idx, trace_idx] == pytest.approx(alpha)
        assert jac[mult_idx, mult_idx] == pytest.approx(beta)
        assert jac[trace_idx, mult_idx] == pytest.approx(1.0)
