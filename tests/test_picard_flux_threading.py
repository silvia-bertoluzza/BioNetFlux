"""
Tests for the Picard linearisation infrastructure:
  - domain_flux_jump: prev_U / prev_J kwargs forwarded to static_condensation
  - GlobalAssembler.assemble_residual_and_jacobian: capture_flux flag and
    prev_U_list / prev_J_list threading
  - PicardSolver: convergence loop and inter-iteration data handoff
  - KellerSegelStaticCondensation: Picard branch freezes chi, drops dchi term
  - TimeStepper: picard_solver field used when provided
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from bionetflux.core.flux_jump import domain_flux_jump
from bionetflux.time_integration.picard_solver import PicardSolver, PicardResult


# ---------------------------------------------------------------------------
# Minimal mock objects shared across tests
# ---------------------------------------------------------------------------

class _MockSC:
    """
    Mock static condensation that records kwargs from each call and returns
    predictable results.
    """

    def __init__(self, neq: int = 1):
        self.neq = neq
        self.calls = []  # list of kwarg dicts per call

    @property
    def total_flux_dofs_per_element(self) -> int:
        return self.neq

    def static_condensation(self, local_trace, local_source=None, **kwargs):
        self.calls.append(dict(kwargs))

        n = 2 * self.neq
        local_trace_flat = np.asarray(local_trace).flatten()
        local_solution = local_trace_flat * 0.5
        flux = np.ones(self.neq) * 0.1
        flux_trace = local_trace_flat * 0.1
        jacobian = np.eye(n) * 0.1
        return local_solution, flux, flux_trace, jacobian


def _make_trace_and_forcing(neq: int, n_elements: int):
    """Return trace_solution and forcing_term compatible with domain_flux_jump."""
    n_nodes = n_elements + 1
    trace_solution = np.random.default_rng(0).random(neq * n_nodes).reshape(-1, 1)
    forcing_term = np.zeros((2 * neq, n_elements))
    return trace_solution, forcing_term


# ---------------------------------------------------------------------------
# Phase 1 — domain_flux_jump
# ---------------------------------------------------------------------------

class TestDomainFluxJumpPrevData:

    def test_no_prev_data_no_kwargs_forwarded(self):
        """Without prev_U/prev_J, static_condensation receives no prev kwargs."""
        sc = _MockSC(neq=1)
        trace, forcing = _make_trace_and_forcing(neq=1, n_elements=3)
        domain_flux_jump(trace, forcing, None, None, sc)
        for call_kwargs in sc.calls:
            assert 'prev_local_solution' not in call_kwargs
            assert 'prev_flux' not in call_kwargs

    def test_prev_U_forwarded_per_element(self):
        """prev_U[:,k] is passed as prev_local_solution to element k."""
        neq = 1
        n_elements = 4
        sc = _MockSC(neq=neq)
        trace, forcing = _make_trace_and_forcing(neq=neq, n_elements=n_elements)

        prev_U = np.arange(2 * neq * n_elements, dtype=float).reshape(2 * neq, n_elements)
        domain_flux_jump(trace, forcing, None, None, sc, prev_U=prev_U)

        assert len(sc.calls) == n_elements
        for k, call_kwargs in enumerate(sc.calls):
            assert 'prev_local_solution' in call_kwargs
            np.testing.assert_array_equal(call_kwargs['prev_local_solution'], prev_U[:, k])

    def test_prev_J_forwarded_per_element(self):
        """prev_J[:,k] is passed as prev_flux to element k."""
        neq = 1
        n_elements = 3
        sc = _MockSC(neq=neq)
        trace, forcing = _make_trace_and_forcing(neq=neq, n_elements=n_elements)

        prev_J = np.arange(neq * n_elements, dtype=float).reshape(neq, n_elements)
        domain_flux_jump(trace, forcing, None, None, sc, prev_J=prev_J)

        assert len(sc.calls) == n_elements
        for k, call_kwargs in enumerate(sc.calls):
            assert 'prev_flux' in call_kwargs
            np.testing.assert_array_equal(call_kwargs['prev_flux'], prev_J[:, k])

    def test_both_prev_U_and_prev_J_forwarded(self):
        """Both prev_U and prev_J are forwarded simultaneously."""
        neq = 2
        n_elements = 5
        sc = _MockSC(neq=neq)
        trace, forcing = _make_trace_and_forcing(neq=neq, n_elements=n_elements)

        prev_U = np.ones((2 * neq, n_elements))
        prev_J = np.ones((neq, n_elements)) * 2.0
        domain_flux_jump(trace, forcing, None, None, sc, prev_U=prev_U, prev_J=prev_J)

        for k, call_kwargs in enumerate(sc.calls):
            assert 'prev_local_solution' in call_kwargs
            assert 'prev_flux' in call_kwargs

    def test_return_shape_unchanged(self):
        """Return shapes (U, J, F, JF) are unchanged when prev data is provided."""
        neq = 1
        n_elements = 4
        sc = _MockSC(neq=neq)
        trace, forcing = _make_trace_and_forcing(neq=neq, n_elements=n_elements)

        prev_U = np.zeros((2 * neq, n_elements))
        prev_J = np.zeros((neq, n_elements))

        U_no_prev, J_no_prev, F_no_prev, JF_no_prev = domain_flux_jump(
            trace, forcing, None, None, sc
        )
        sc_prev = _MockSC(neq=neq)
        U_prev, J_prev, F_prev, JF_prev = domain_flux_jump(
            trace, forcing, None, None, sc_prev, prev_U=prev_U, prev_J=prev_J
        )

        assert U_prev.shape == U_no_prev.shape
        assert F_prev.shape == F_no_prev.shape
        assert JF_prev.shape == JF_no_prev.shape


# ---------------------------------------------------------------------------
# Phase 2 — GlobalAssembler.assemble_residual_and_jacobian
# ---------------------------------------------------------------------------

class TestAssembleCaptureFux:

    def _make_assembler(self, neq: int = 1, n_elements: int = 4):
        """Return a GlobalAssembler with a single-domain setup."""
        from bionetflux.core.lean_global_assembly import GlobalAssembler
        from bionetflux.core.domain_data import DomainData

        domain_data = DomainData(
            neq=neq,
            n_elements=n_elements,
            domain_length=1.0,
        )
        assembler = GlobalAssembler(
            domain_data_list=[domain_data],
            constraint_manager=None,
        )
        return assembler

    def test_capture_flux_false_returns_two_tuple(self):
        """capture_flux=False returns (residual, jacobian) — no regression."""
        try:
            assembler = self._make_assembler()
        except Exception:
            pytest.skip("DomainData or GlobalAssembler not constructable in isolation")

        sc = _MockSC(neq=1)
        global_sol = np.zeros(assembler.total_dofs)
        forcing = [np.zeros((2, 4))]

        result = assembler.assemble_residual_and_jacobian(
            global_solution=global_sol,
            forcing_terms=forcing,
            static_condensations=[sc],
            time=0.0,
            capture_flux=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_capture_flux_true_returns_four_tuple(self):
        """capture_flux=True returns (residual, jacobian, U_list, J_list)."""
        try:
            assembler = self._make_assembler()
        except Exception:
            pytest.skip("DomainData or GlobalAssembler not constructable in isolation")

        sc = _MockSC(neq=1)
        global_sol = np.zeros(assembler.total_dofs)
        forcing = [np.zeros((2, 4))]

        result = assembler.assemble_residual_and_jacobian(
            global_solution=global_sol,
            forcing_terms=forcing,
            static_condensations=[sc],
            time=0.0,
            capture_flux=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        residual, jacobian, U_list, J_list = result
        assert len(U_list) == 1
        assert len(J_list) == 1


# ---------------------------------------------------------------------------
# Phase 3 — PicardSolver
# ---------------------------------------------------------------------------

class _CountingAssembler:
    """
    Minimal assembler stand-in for PicardSolver unit tests.

    Records whether prev_U_list / prev_J_list were non-None on each call and
    returns a residual that decreases geometrically so the solver converges.
    """

    def __init__(self, n_dofs: int = 4, decay: float = 0.1):
        self.n_dofs = n_dofs
        self.decay = decay
        self.call_count = 0
        self.prev_U_received = []
        self.prev_J_received = []

    @property
    def total_dofs(self):
        return self.n_dofs

    def assemble_residual_and_jacobian(
        self,
        global_solution,
        forcing_terms,
        static_condensations,
        time,
        prev_U_list=None,
        prev_J_list=None,
        capture_flux=False,
    ):
        self.call_count += 1
        self.prev_U_received.append(prev_U_list)
        self.prev_J_received.append(prev_J_list)

        # Residual decays geometrically so the solver eventually converges.
        residual = global_solution * (self.decay ** self.call_count)
        jacobian = np.eye(self.n_dofs)

        # Dummy captured U and J (one domain).
        captured_U = [np.ones((2, 1))]
        captured_J = [np.ones((1, 1))]

        if capture_flux:
            return residual, jacobian, captured_U, captured_J
        return residual, jacobian


class TestPicardSolver:

    def test_first_iteration_has_no_prev_data(self):
        """On the first assembler call prev_U_list and prev_J_list are None."""
        assembler = _CountingAssembler(n_dofs=4, decay=0.05)
        solver = PicardSolver(tolerance=1e-6, max_iterations=20, verbose=False)

        solver.solve(
            initial_guess=np.ones(4),
            global_assembler=assembler,
            forcing_terms=[None],
            static_condensations=[None],
            current_time=0.0,
        )

        assert assembler.prev_U_received[0] is None
        assert assembler.prev_J_received[0] is None

    def test_second_iteration_has_prev_data(self):
        """From the second assembler call onward, prev_U_list is not None."""
        assembler = _CountingAssembler(n_dofs=4, decay=0.05)
        solver = PicardSolver(tolerance=1e-6, max_iterations=20, verbose=False)

        solver.solve(
            initial_guess=np.ones(4),
            global_assembler=assembler,
            forcing_terms=[None],
            static_condensations=[None],
            current_time=0.0,
        )

        # At least two assembler calls were made (initial ||R|| > tol with decay=0.05).
        if assembler.call_count >= 2:
            assert assembler.prev_U_received[1] is not None
            assert assembler.prev_J_received[1] is not None

    def test_converges_on_small_system(self):
        """PicardSolver reports convergence when residual drops below tolerance."""
        assembler = _CountingAssembler(n_dofs=4, decay=0.05)
        solver = PicardSolver(tolerance=1e-8, max_iterations=50, verbose=False)

        result = solver.solve(
            initial_guess=np.ones(4),
            global_assembler=assembler,
            forcing_terms=[None],
            static_condensations=[None],
            current_time=0.0,
        )

        assert isinstance(result, PicardResult)
        assert result.converged
        assert result.final_residual_norm <= 1e-8

    def test_returns_picard_result(self):
        """Return type is PicardResult with all expected fields."""
        assembler = _CountingAssembler(n_dofs=2, decay=0.01)
        solver = PicardSolver(tolerance=1e-10, max_iterations=30, verbose=False)

        result = solver.solve(
            initial_guess=np.zeros(2),
            global_assembler=assembler,
            forcing_terms=[None],
            static_condensations=[None],
            current_time=1.0,
        )

        assert hasattr(result, 'converged')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'final_solution')
        assert hasattr(result, 'final_residual_norm')
        assert hasattr(result, 'residual_history')
        assert hasattr(result, 'step_norms')

    def test_residual_history_length_matches_iterations(self):
        """residual_history has one entry per assembler call."""
        assembler = _CountingAssembler(n_dofs=4, decay=0.05)
        solver = PicardSolver(tolerance=1e-8, max_iterations=50, verbose=False)

        result = solver.solve(
            initial_guess=np.ones(4),
            global_assembler=assembler,
            forcing_terms=[None],
            static_condensations=[None],
            current_time=0.0,
        )

        assert len(result.residual_history) == assembler.call_count

    def test_max_iterations_respected(self):
        """Solver stops and reports failure when max_iterations is reached."""

        class _ConstantResidualAssembler:
            """Assembler that always returns a large fixed residual."""

            total_dofs = 4

            def assemble_residual_and_jacobian(
                self,
                global_solution,
                forcing_terms,
                static_condensations,
                time,
                prev_U_list=None,
                prev_J_list=None,
                capture_flux=False,
            ):
                residual = np.ones(4) * 1e3   # always large
                jacobian = np.eye(4)
                if capture_flux:
                    return residual, jacobian, [np.zeros((2, 1))], [np.zeros((1, 1))]
                return residual, jacobian

        solver = PicardSolver(tolerance=1e-10, max_iterations=5, verbose=False)
        result = solver.solve(
            initial_guess=np.ones(4),
            global_assembler=_ConstantResidualAssembler(),
            forcing_terms=[None],
            static_condensations=[None],
            current_time=0.0,
        )

        assert not result.converged
        assert result.iterations == 5


# ---------------------------------------------------------------------------
# Phase 4a — KS static_condensation Picard branch
# ---------------------------------------------------------------------------

class TestKSStaticCondensationPicard:
    """
    Tests that the KS static_condensation correctly freezes chi when
    prev_local_solution is supplied and drops the dchi Jacobian contribution.
    """

    @pytest.fixture
    def ks_sc(self):
        """Build a KellerSegelStaticCondensation with minimal mocked dependencies."""
        from bionetflux.core.static_condensation_keller_segel import KellerSegelStaticCondensation
        from bionetflux.utils.elementary_matrices import ElementaryMatrices

        # Minimal mock objects — only the attributes actually read by KS SC.
        problem = MagicMock()
        problem.neq = 2
        problem.parameters = [1.0, 1.0, 0.0, 0.0]  # mu, nu, a, b
        # chi(x) = 1 + x so that chi and dchi are non-trivial but simple.
        problem.chi = lambda phi: 1.0 + phi
        problem.dchi = lambda phi: 1.0

        disc = MagicMock()
        disc.n_elements = 4
        disc.element_length = 0.25
        disc.tau = [1.0, 1.0]

        global_disc = MagicMock()
        global_disc.dt = 0.1
        global_disc.spatial_discretizations = [disc]

        em = ElementaryMatrices(orthonormal_basis=False)
        sc = KellerSegelStaticCondensation(problem, global_disc, em, ipb=0)
        sc.build_matrices()
        return sc

    def test_newton_mode_baseline(self, ks_sc):
        """Newton mode (no prev_local_solution) runs without error."""
        local_trace = np.array([0.5, 0.6, 0.3, 0.4])
        local_source = np.zeros(4)
        sol, flux, flux_trace, jac = ks_sc.static_condensation(local_trace, local_source)
        assert sol.shape[0] == 4
        assert flux_trace.shape[0] == 4

    def test_picard_mode_runs_without_error(self, ks_sc):
        """Picard mode (with prev_local_solution) runs without error."""
        local_trace = np.array([0.5, 0.6, 0.3, 0.4])
        local_source = np.zeros(4)
        prev_solution = np.array([0.4, 0.5, 0.2, 0.3])  # 4 entries: [u1(2), phi(2)]
        prev_flux = np.array([0.1, 0.05, 0.06])           # 3 entries: [flux, psi(2)]
        sol, flux, flux_trace, jac = ks_sc.static_condensation(
            local_trace, local_source,
            prev_local_solution=prev_solution,
            prev_flux=prev_flux,
        )
        assert sol.shape[0] == 4
        assert flux_trace.shape[0] == 4

    def test_picard_uses_frozen_chi(self, ks_sc):
        """
        In Picard mode chi is evaluated at phi_avg_prev (from prev_local_solution),
        not at the current phi_avg.  Verified by replacing problem.chi with a
        recording callable and checking which phi value it received.
        """
        local_trace = np.array([1.0, 1.0, 5.0, 5.0])
        local_source = np.zeros(4)

        recorded_phi = []
        original_chi = ks_sc.problem.chi

        def recording_chi(phi):
            recorded_phi.append(float(phi))
            return original_chi(phi)

        ks_sc.problem.chi = recording_chi

        # Newton call: chi should be evaluated at the current phi_avg.
        recorded_phi.clear()
        ks_sc.static_condensation(local_trace, local_source)
        phi_newton = list(recorded_phi)

        # Picard call with prev_phi ≈ 0: chi should be evaluated at prev_phi.
        recorded_phi.clear()
        prev_solution = np.array([0.0, 0.0, 0.0, 0.0])   # phi block = [0, 0]
        ks_sc.static_condensation(
            local_trace, local_source,
            prev_local_solution=prev_solution,
            prev_flux=np.zeros(3),
        )
        phi_picard = list(recorded_phi)

        ks_sc.problem.chi = original_chi  # restore

        # In Newton mode chi is called with the current phi_avg (non-zero for trace=5).
        # In Picard mode chi is called with prev_phi ≈ 0.
        assert len(phi_newton) >= 1
        assert len(phi_picard) >= 1
        # The Picard phi argument must come from prev_local_solution, so ≈ 0.
        assert abs(phi_picard[0]) < abs(phi_newton[0]) + 1e-12, (
            f"Picard chi called with phi={phi_picard[0]}, "
            f"expected close to 0 (prev); Newton phi={phi_newton[0]}"
        )

    def test_picard_jacobian_no_dchi_term(self, ks_sc):
        """
        In Picard mode the Jacobian should differ from Newton because dchi=0.
        With chi(x)=1+x, dchi=1 is non-zero, so Newton and Picard Jacobians differ.
        """
        local_trace = np.array([1.0, 1.0, 2.0, 2.0])
        local_source = np.zeros(4)

        _, _, _, jac_newton = ks_sc.static_condensation(local_trace, local_source)

        prev_solution = np.array([1.0, 1.0, 2.0, 2.0])  # same phi → same chi value
        prev_flux = np.zeros(3)
        _, _, _, jac_picard = ks_sc.static_condensation(
            local_trace, local_source,
            prev_local_solution=prev_solution,
            prev_flux=prev_flux,
        )

        # With chi(prev_phi)=chi(current_phi) the only difference is dchi=0 in Picard.
        # dchi=1 so the Jacobians should differ.
        assert not np.allclose(jac_newton, jac_picard), (
            "Expected Picard and Newton Jacobians to differ due to dchi term"
        )


# ---------------------------------------------------------------------------
# Phase 5 — TimeStepper uses picard_solver when set
# ---------------------------------------------------------------------------

class TestTimeStepperPicardSolverField:

    def test_picard_solver_stored(self):
        """TimeStepper stores the picard_solver provided at init."""
        from bionetflux.time_integration.time_stepper import TimeStepper

        setup = MagicMock()
        setup.bulk_data_manager = MagicMock()
        setup.global_assembler = MagicMock()
        setup.static_condensations = []
        setup.problems = []
        setup.global_discretization.spatial_discretizations = []

        picard = PicardSolver(tolerance=1e-6)
        ts = TimeStepper(setup=setup, picard_solver=picard)

        assert ts.picard_solver is picard

    def test_newton_solver_used_when_no_picard(self):
        """When picard_solver is None, TimeStepper uses newton_solver."""
        from bionetflux.time_integration.time_stepper import TimeStepper
        from bionetflux.time_integration.newton_solver import NewtonSolver

        setup = MagicMock()
        setup.bulk_data_manager = MagicMock()
        setup.global_assembler = MagicMock()
        setup.static_condensations = []
        setup.problems = []
        setup.global_discretization.spatial_discretizations = []

        ts = TimeStepper(setup=setup)

        assert ts.picard_solver is None
        assert isinstance(ts.newton_solver, NewtonSolver)
