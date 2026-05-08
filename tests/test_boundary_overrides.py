"""
Tests for the boundary_override module (apply_boundary_overrides).

Verifies:
  - Dirichlet, Neumann-with-data, and Robin overrides replace the default
    homogeneous Neumann constraints correctly.
  - Equations not mentioned in the override dict keep their defaults.
  - Error handling: unknown point, unknown equation, missing type, bad key
    format, missing Robin parameters.
"""

import pytest
import numpy as np

from bionetflux.core.constraints import (
    ConstraintManager,
    ConstraintType,
)
from bionetflux.core.boundary_override import apply_boundary_overrides


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _build_manager_and_map():
    """Build a small ConstraintManager mimicking what
    ``setup_constraints_from_geometry`` + ``create_maze_geometry`` produce.

    Geometry sketch (3 domains, 4 boundary points, 2 equations):

        B1 --[domain 0]-- J1 --[domain 1]-- J2 --[domain 2]-- B2
                                |
                            [domain …]  (not modelled; just B3, B4)

    Boundary point map::

        B1: (domain=0, position=0.0)
        B2: (domain=2, position=3.0)
        B3: (domain=1, position=1.0)
        B4: (domain=2, position=2.5)

    For each boundary point and each of the 2 equations (eq 0 = 'u',
    eq 1 = 'phi') we add a homogeneous Neumann — 8 boundary constraints
    total.  Plus 2 trace-continuity constraints at the junctions.
    """
    cm = ConstraintManager()

    # Boundary Neumann (indices 0–7)
    for did, pos in [(0, 0.0), (2, 3.0), (1, 1.0), (2, 2.5)]:
        for eq in range(2):
            cm.add_neumann(equation_index=eq, domain_index=did, position=pos)

    # Trace continuity at J1 and J2 (indices 8–11)
    cm.add_trace_continuity(0, 0, 1, 1.0, 1.0)  # J1 eq0
    cm.add_trace_continuity(1, 0, 1, 1.0, 1.0)  # J1 eq1
    cm.add_trace_continuity(0, 1, 2, 2.0, 2.0)  # J2 eq0
    cm.add_trace_continuity(1, 1, 2, 2.0, 2.0)  # J2 eq1

    boundary_point_map = {
        "B1": (0, 0.0),
        "B2": (2, 3.0),
        "B3": (1, 1.0),
        "B4": (2, 2.5),
    }
    equation_names = ["u", "phi"]
    return cm, boundary_point_map, equation_names


class _FakeFunctionResolver:
    """Minimal stand-in for FunctionResolver — returns a named lambda."""

    def resolve_function(self, name: str):
        if name == "zeros":
            return lambda s, t=0: np.zeros_like(s)
        if name == "ones":
            return lambda s, t=0: np.ones_like(s)
        if name == "sin_t":
            return lambda s, t=0: np.sin(t) * np.ones_like(s)
        raise ValueError(f"Unknown function '{name}'")

    def resolve_boundary_function(self, name: str, position: float):
        """Resolve and pin spatial coordinate, returning g(t)."""
        f = self.resolve_function(name)
        return lambda t, _f=f, _p=position: _f(_p, t)


# ===========================================================================
#  Happy-path tests
# ===========================================================================

class TestApplyBoundaryOverrides:
    """Core functionality of apply_boundary_overrides."""

    def test_dirichlet_override(self):
        """A Dirichlet entry replaces the default Neumann."""
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_u": {"type": "dirichlet", "data": "ones"}}
        apply_boundary_overrides(cm, overrides, bpm, eqn, _FakeFunctionResolver())

        # B1 is (domain=0, pos=0.0), eq u=0 → originally index 0
        c = cm.constraints[0]
        assert c.type == ConstraintType.DIRICHLET
        # The other B1 equation (phi, index 1) should still be Neumann
        assert cm.constraints[1].type == ConstraintType.NEUMANN

    def test_neumann_with_data_override(self):
        """Neumann with a data function replaces the homogeneous one."""
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B2_phi": {"type": "neumann", "data": "sin_t"}}
        apply_boundary_overrides(cm, overrides, bpm, eqn, _FakeFunctionResolver())

        # B2 is (domain=2, pos=3.0) → eq phi=1 → index 3
        c = cm.constraints[3]
        assert c.type == ConstraintType.NEUMANN
        assert c.data_function is not None
        # data_function is now g(t) with s pinned to position 3.0
        # sin_t returns sin(t)*ones_like(s), so at t=pi/2 → 1.0
        val = c.data_function(np.pi / 2)
        np.testing.assert_allclose(val, 1.0, atol=1e-12)

    def test_robin_override(self):
        """Robin override stores alpha, beta as parameters."""
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {
            "B3_u": {"type": "robin", "alpha": 2.0, "beta": 0.5, "data": "zeros"},
        }
        apply_boundary_overrides(cm, overrides, bpm, eqn, _FakeFunctionResolver())

        # B3 is (domain=1, pos=1.0) → eq u=0 → index 4
        c = cm.constraints[4]
        assert c.type == ConstraintType.ROBIN
        np.testing.assert_allclose(c.parameters, [2.0, 0.5])

    def test_multiple_overrides(self):
        """Several overrides can be applied at once."""
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {
            "B1_u": {"type": "dirichlet", "data": "ones"},
            "B1_phi": {"type": "dirichlet", "data": "zeros"},
            "B4_u": {"type": "neumann", "data": "ones"},
        }
        apply_boundary_overrides(cm, overrides, bpm, eqn, _FakeFunctionResolver())

        assert cm.constraints[0].type == ConstraintType.DIRICHLET  # B1_u
        assert cm.constraints[1].type == ConstraintType.DIRICHLET  # B1_phi
        # B4 is (domain=2, pos=2.5) → eq u=0 → index 6
        assert cm.constraints[6].type == ConstraintType.NEUMANN
        assert cm.constraints[6].data_function is not None

    def test_empty_overrides_is_noop(self):
        """An empty dict does nothing."""
        cm, bpm, eqn = _build_manager_and_map()
        types_before = [c.type for c in cm.constraints]
        apply_boundary_overrides(cm, {}, bpm, eqn, _FakeFunctionResolver())
        types_after = [c.type for c in cm.constraints]
        assert types_before == types_after

    def test_unmentioned_constraints_unchanged(self):
        """Constraints not targeted by overrides are not touched."""
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_u": {"type": "dirichlet", "data": "ones"}}
        apply_boundary_overrides(cm, overrides, bpm, eqn, _FakeFunctionResolver())

        # Trace continuity constraints (indices 8–11) must be unchanged
        for i in range(8, 12):
            assert cm.constraints[i].type == ConstraintType.TRACE_CONTINUITY

    def test_no_function_resolver(self):
        """When function_resolver is None, data is not resolved."""
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_u": {"type": "dirichlet", "data": "ones"}}
        apply_boundary_overrides(cm, overrides, bpm, eqn, function_resolver=None)
        assert cm.constraints[0].type == ConstraintType.DIRICHLET
        # data_function is not the resolved callable (make_dirichlet passes
        # data_function=None, so Constraint falls back to its default)


# ===========================================================================
#  Error-handling tests
# ===========================================================================

class TestApplyBoundaryOverridesErrors:
    """Validation / error paths."""

    def test_unknown_point_name(self):
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"BOGUS_u": {"type": "dirichlet"}}
        with pytest.raises(ValueError, match="Unknown boundary point 'BOGUS'"):
            apply_boundary_overrides(cm, overrides, bpm, eqn)

    def test_unknown_equation_name(self):
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_omega": {"type": "dirichlet"}}
        with pytest.raises(ValueError, match="Unknown equation 'omega'"):
            apply_boundary_overrides(cm, overrides, bpm, eqn)

    def test_missing_type_key(self):
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_u": {"data": "zeros"}}
        with pytest.raises(ValueError, match="Missing 'type'"):
            apply_boundary_overrides(cm, overrides, bpm, eqn)

    def test_invalid_type(self):
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_u": {"type": "periodic"}}
        with pytest.raises(ValueError, match="Unknown BC type 'periodic'"):
            apply_boundary_overrides(cm, overrides, bpm, eqn)

    def test_robin_missing_alpha(self):
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_u": {"type": "robin", "beta": 1.0}}
        with pytest.raises(ValueError, match="requires both 'alpha' and 'beta'"):
            apply_boundary_overrides(cm, overrides, bpm, eqn)

    def test_robin_missing_beta(self):
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_u": {"type": "robin", "alpha": 1.0}}
        with pytest.raises(ValueError, match="requires both 'alpha' and 'beta'"):
            apply_boundary_overrides(cm, overrides, bpm, eqn)

    def test_bad_key_format(self):
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"nounderscore": {"type": "dirichlet"}}
        with pytest.raises(ValueError, match="Invalid boundary_conditions key"):
            apply_boundary_overrides(cm, overrides, bpm, eqn)

    def test_spec_not_a_dict(self):
        cm, bpm, eqn = _build_manager_and_map()
        overrides = {"B1_u": "dirichlet"}
        with pytest.raises(ValueError, match="must be a table/dict"):
            apply_boundary_overrides(cm, overrides, bpm, eqn)


# ===========================================================================
#  Geometry builder boundary_point_map tests
# ===========================================================================

class TestGeometryBoundaryPointMap:
    """Verify that geometry builder functions populate boundary_point_map
    in their global metadata so that apply_boundary_overrides can consume it.
    """

    @pytest.mark.unit
    def test_build_arc_sequence_geometry_two_boundary_points(self):
        """build_arc_sequence_geometry always has exactly 2 exterior boundary
        points regardless of N: B0 at the inlet and B1 at the outlet."""
        from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry

        for N in (1, 2, 4):
            geometry = build_arc_sequence_geometry(N=N, start=0.0, length=1.0)
            bpm = geometry.get_global_metadata().get("boundary_point_map", {})
            assert len(bpm) == 2, f"Expected 2 boundary points for N={N}, got {len(bpm)}"
            assert "B0" in bpm, "B0 (inlet) missing from boundary_point_map"
            assert "B1" in bpm, "B1 (outlet) missing from boundary_point_map"
            # B0 is at the inlet of domain 0
            b0_domain, b0_pos = bpm["B0"]
            assert b0_domain == 0
            assert abs(b0_pos - 0.0) < 1e-12
            # B1 is at the outlet of domain N-1
            b1_domain, b1_pos = bpm["B1"]
            assert b1_domain == N - 1
            assert abs(b1_pos - float(N)) < 1e-12

    @pytest.mark.unit
    def test_build_arc_sequence_geometry_custom_start_and_length(self):
        """Boundary positions follow start + N*length even for non-default values."""
        from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry

        geometry = build_arc_sequence_geometry(N=3, start=10.0, length=50.0)
        bpm = geometry.get_global_metadata()["boundary_point_map"]
        _, b0_pos = bpm["B0"]
        _, b1_pos = bpm["B1"]
        assert abs(b0_pos - 10.0) < 1e-12
        assert abs(b1_pos - 160.0) < 1e-12  # 10 + 3*50

    @pytest.mark.unit
    def test_build_grid_geometry_eight_boundary_points(self):
        """build_grid_geometry (default N=4) must expose B0..B7 — two boundary
        points per vertical segment (S1, S2, S3, S4)."""
        from bionetflux.geometry.domain_geometry import build_grid_geometry

        geometry = build_grid_geometry()
        bpm = geometry.get_global_metadata().get("boundary_point_map", {})
        expected_keys = {f"B{i}" for i in range(8)}
        assert expected_keys == set(bpm.keys()), (
            f"Expected keys {sorted(expected_keys)}, got {sorted(bpm.keys())}"
        )
        # Every entry must be a (domain_id, position) pair
        for key, value in bpm.items():
            assert isinstance(value, tuple) and len(value) == 2, (
                f"{key} must be a (domain_id, position) tuple"
            )
            domain_id, position = value
            assert isinstance(domain_id, int)
            assert isinstance(position, float)

    @pytest.mark.unit
    def test_build_T_junction_geometry_three_boundary_points(self):
        """build_T_junction_geometry must expose B0, B1, B2 — two endpoints of
        the main channel and one end of the branch."""
        from bionetflux.problems.ooc_problem import build_T_junction_geometry

        geometry = build_T_junction_geometry()
        bpm = geometry.get_global_metadata().get("boundary_point_map", {})
        assert set(bpm.keys()) == {"B0", "B1", "B2"}, (
            f"Expected B0/B1/B2, got {sorted(bpm.keys())}"
        )
        # B0: main_channel (domain 0) at -500
        assert bpm["B0"] == (0, -500.0)
        # B1: main_channel (domain 0) at +500
        assert bpm["B1"] == (0, 500.0)
        # B2: branch (domain 1) at 0
        assert bpm["B2"] == (1, 0.0)

    @pytest.mark.integration
    def test_grid_geometry_override_replaces_neumann_with_dirichlet(self):
        """End-to-end: build grid geometry, set up default Neumann constraints,
        apply a Dirichlet override for B0, and verify the result."""
        from bionetflux.geometry.domain_geometry import build_grid_geometry
        from bionetflux.problems.ooc_problem import setup_constraints_from_geometry

        geometry = build_grid_geometry()
        neq = 4
        # setup_constraints_from_geometry only requires geometry and neq
        # (the problems list is only used for printing; we pass an empty stub)
        cm = setup_constraints_from_geometry(geometry, [], neq)

        bpm = geometry.get_global_metadata()["boundary_point_map"]
        equation_names = ["u", "omega", "v", "phi"]

        # Override B0_u → Dirichlet
        overrides = {"B0_u": {"type": "dirichlet", "data": "zeros"}}
        apply_boundary_overrides(
            cm, overrides, bpm, equation_names, _FakeFunctionResolver()
        )

        # Find the constraint that now lives at B0 for eq u
        b0_domain, b0_pos = bpm["B0"]
        eq_u = 0
        indices = cm.find_constraints(
            domain_index=b0_domain,
            equation_index=eq_u,
            constraint_type=ConstraintType.DIRICHLET,
            position=b0_pos,
        )
        assert len(indices) == 1
        assert cm.constraints[indices[0]].type == ConstraintType.DIRICHLET

        # All other boundary constraints at B0 (eq omega, v, phi) stay Neumann
        for eq_idx in (1, 2, 3):
            idx_list = cm.find_constraints(
                domain_index=b0_domain,
                equation_index=eq_idx,
                constraint_type=ConstraintType.NEUMANN,
                position=b0_pos,
            )
            assert len(idx_list) == 1
            assert cm.constraints[idx_list[0]].type == ConstraintType.NEUMANN
