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
