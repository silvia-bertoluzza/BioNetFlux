"""
Tests for adaptive time stepping: StaticCondensationBase.update_dt and
AdaptiveTimeStepper.advance_time_step_adaptive.
"""

import numpy as np
import pytest

from bionetflux.core.static_condensation_base import StaticCondensationBase
from bionetflux.time_integration.time_stepper import (
    AdaptiveTimeStepper, TimeStepper, TimeStepResult,
)


# ---------------------------------------------------------------------------
# Minimal concrete StaticCondensation subclass for testing update_dt
# ---------------------------------------------------------------------------

class _MockProblem:
    """Minimal Problem-like object."""
    def __init__(self, neq=1, domain_length=1.0):
        self.neq = neq
        self.domain_length = domain_length
        self.parameters = [0.0] * 10


class _MockDiscretization:
    """Minimal Discretization-like object."""
    def __init__(self, n_elements=5, domain_length=1.0):
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.domain_length = domain_length
        self.element_length = domain_length / n_elements
        self.nodes = np.linspace(0, domain_length, n_elements + 1)
        self.tau = [1.0]


class _MockGlobalDisc:
    """Minimal GlobalDiscretization-like object."""
    def __init__(self, dt=0.1, n_elements=5, domain_length=1.0):
        self.dt = dt
        disc = _MockDiscretization(n_elements, domain_length)
        self.spatial_discretizations = [disc]


class _TrivialSC(StaticCondensationBase):
    """Concrete StaticCondensation that records build_matrices calls.

    build_matrices stores a single matrix ``A = dt * I`` so we can verify
    that it depends on self.dt.
    """

    def __init__(self, dt=0.1):
        # Bypass the real __init__ and set attributes directly
        self.problem = _MockProblem()
        self._global_disc = _MockGlobalDisc(dt=dt)
        self.discretization = _MockDiscretization()
        self.elementary_matrices = None
        self.sc_matrices = {}
        self.dt = dt
        self.tau = [1.0]
        self.flux_orders = [1]
        self.build_count = 0

    def build_matrices(self):
        self.build_count += 1
        self.sc_matrices = {"A": self.dt * np.eye(2)}
        return self.sc_matrices

    def static_condensation(self, local_trace, local_source, **kwargs):
        return local_trace, local_trace, local_trace, np.eye(2)

    def assemble_forcing_term(self, *args, **kwargs):
        return np.zeros(2)


# ---------------------------------------------------------------------------
# Tests for update_dt
# ---------------------------------------------------------------------------

class TestUpdateDt:
    """Tests for StaticCondensationBase.update_dt."""

    def test_dt_is_updated(self):
        """update_dt re-reads dt from GlobalDiscretization."""
        sc = _TrivialSC(dt=0.1)
        sc.build_matrices()
        sc._global_disc.dt = 0.05
        sc.update_dt()
        assert sc.dt == 0.05

    def test_build_matrices_is_called(self):
        """update_dt triggers a rebuild of the cached matrices."""
        sc = _TrivialSC(dt=0.1)
        sc.build_matrices()
        assert sc.build_count == 1
        sc._global_disc.dt = 0.05
        sc.update_dt()
        assert sc.build_count == 2

    def test_matrices_reflect_new_dt(self):
        """After update_dt, cached matrices use the new dt."""
        sc = _TrivialSC(dt=0.1)
        sc.build_matrices()
        np.testing.assert_allclose(sc.sc_matrices["A"], 0.1 * np.eye(2))

        sc._global_disc.dt = 0.05
        sc.update_dt()
        np.testing.assert_allclose(sc.sc_matrices["A"], 0.05 * np.eye(2))

    def test_idempotent_same_dt(self):
        """Calling update_dt with the same dt rebuilds but result is identical."""
        sc = _TrivialSC(dt=0.1)
        sc.build_matrices()
        old_A = sc.sc_matrices["A"].copy()
        sc.update_dt()  # _global_disc.dt is still 0.1
        np.testing.assert_array_equal(sc.sc_matrices["A"], old_A)

    def test_rejects_non_positive_dt(self):
        """update_dt raises ValueError when GlobalDiscretization.dt is invalid."""
        sc = _TrivialSC(dt=0.1)
        sc.build_matrices()
        sc._global_disc.dt = 0.0
        with pytest.raises(ValueError, match="positive"):
            sc.update_dt()
        sc._global_disc.dt = -1.0
        with pytest.raises(ValueError, match="positive"):
            sc.update_dt()

    def test_multiple_updates(self):
        """update_dt can be called multiple times in sequence."""
        sc = _TrivialSC(dt=0.1)
        sc.build_matrices()
        for new_dt in [0.05, 0.025, 0.1, 0.2]:
            sc._global_disc.dt = new_dt
            sc.update_dt()
            assert sc.dt == new_dt
            np.testing.assert_allclose(sc.sc_matrices["A"], new_dt * np.eye(2))
        assert sc.build_count == 5  # 1 initial + 4 updates


# ---------------------------------------------------------------------------
# Tests for update_dt with the real OOC StaticCondensation
# ---------------------------------------------------------------------------

class TestUpdateDtOOC:
    """Integration test: update_dt on a real StaticCondensationOOC object."""

    @pytest.fixture()
    def ooc_sc(self):
        """Create a real OOC SC object via quick_setup with config file."""
        import os
        from setup_solver import quick_setup

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "ooc_parameters.toml"
        )
        if not os.path.isfile(config_path):
            pytest.skip("config/ooc_parameters.toml not found")

        setup = quick_setup(
            problem_module="bionetflux.problems.ooc_problem",
            validate=False,
            config_file=config_path,
        )
        # Use the first domain's static condensation object
        sc = setup.static_condensations[0]
        return sc

    def test_update_dt_changes_A1(self, ooc_sc):
        """Changing dt must alter the dt-dependent matrix L1 = inv(M + dt*tu*Mb)."""
        old_L1 = ooc_sc.sc_matrices["L1"].copy()
        old_dt = ooc_sc.dt

        ooc_sc._global_disc.dt = old_dt * 0.5
        ooc_sc.update_dt()

        # L1 = inv(M + dt*tu*Mb) — halving dt must change L1
        assert not np.allclose(ooc_sc.sc_matrices["L1"], old_L1), \
            "L1 should change when dt changes"

    def test_update_dt_preserves_dt_independent_matrices(self, ooc_sc):
        """Matrices that do not depend on dt should remain unchanged."""
        old_M = ooc_sc.sc_matrices["M"].copy()
        ooc_sc._global_disc.dt = ooc_sc.dt * 0.5
        ooc_sc.update_dt()
        np.testing.assert_array_equal(ooc_sc.sc_matrices["M"], old_M)

    def test_round_trip_restore_dt(self, ooc_sc):
        """Restoring the original dt must recover the original matrices."""
        original_dt = ooc_sc.dt
        original_L1 = ooc_sc.sc_matrices["L1"].copy()

        ooc_sc._global_disc.dt = original_dt * 0.5
        ooc_sc.update_dt()
        ooc_sc._global_disc.dt = original_dt
        ooc_sc.update_dt()

        np.testing.assert_allclose(
            ooc_sc.sc_matrices["L1"], original_L1, atol=1e-14,
            err_msg="Restoring original dt should restore L1",
        )
