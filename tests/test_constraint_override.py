"""
Tests for the ConstraintManager query and override API:
  - find_constraints (filtering by domain, equation, type, position)
  - replace_constraint (in-place swap preserving indices)
  - make_* factory methods (create without appending)
"""

import pytest
import numpy as np

from bionetflux.core.constraints import (
    ConstraintManager,
    ConstraintType,
    Constraint,
)


# ---------------------------------------------------------------------------
#  Helper: build a small ConstraintManager with known contents
# ---------------------------------------------------------------------------
def _build_test_manager() -> ConstraintManager:
    """Create a ConstraintManager with a predictable set of constraints.

    Layout (6 constraints, indices 0–5):
        0: Neumann   eq=0  domain=0  pos=0.0   (left boundary, u)
        1: Neumann   eq=1  domain=0  pos=0.0   (left boundary, phi)
        2: Neumann   eq=0  domain=1  pos=3.0   (right boundary, u)
        3: Neumann   eq=1  domain=1  pos=3.0   (right boundary, phi)
        4: TraceCont eq=0  domains=[0,1]  pos=[2.0, 2.0]
        5: TraceCont eq=1  domains=[0,1]  pos=[2.0, 2.0]
    """
    cm = ConstraintManager()
    cm.add_neumann(equation_index=0, domain_index=0, position=0.0)
    cm.add_neumann(equation_index=1, domain_index=0, position=0.0)
    cm.add_neumann(equation_index=0, domain_index=1, position=3.0)
    cm.add_neumann(equation_index=1, domain_index=1, position=3.0)
    cm.add_trace_continuity(equation_index=0,
                            domain1_index=0, domain2_index=1,
                            position1=2.0, position2=2.0)
    cm.add_trace_continuity(equation_index=1,
                            domain1_index=0, domain2_index=1,
                            position1=2.0, position2=2.0)
    return cm


# ===========================================================================
#  find_constraints
# ===========================================================================
class TestFindConstraints:
    """Tests for ConstraintManager.find_constraints."""

    def test_find_by_domain(self):
        """Filter by domain_index returns all constraints involving that domain."""
        cm = _build_test_manager()
        # Domain 0 participates in: indices 0, 1 (Neumann) and 4, 5 (TraceCont)
        result = cm.find_constraints(domain_index=0)
        assert result == [0, 1, 4, 5]

    def test_find_by_equation(self):
        """Filter by equation_index returns constraints for that equation only."""
        cm = _build_test_manager()
        result = cm.find_constraints(equation_index=1)
        assert result == [1, 3, 5]

    def test_find_by_type(self):
        """Filter by constraint_type."""
        cm = _build_test_manager()
        result = cm.find_constraints(constraint_type=ConstraintType.NEUMANN)
        assert result == [0, 1, 2, 3]

        result = cm.find_constraints(constraint_type=ConstraintType.TRACE_CONTINUITY)
        assert result == [4, 5]

    def test_find_by_position(self):
        """Filter by position (with tolerance)."""
        cm = _build_test_manager()
        result = cm.find_constraints(position=0.0)
        assert result == [0, 1]

        result = cm.find_constraints(position=3.0)
        assert result == [2, 3]

        # Position 2.0 appears in trace-continuity constraints
        result = cm.find_constraints(position=2.0)
        assert result == [4, 5]

    def test_find_combined_filters(self):
        """Multiple filters are combined with AND logic."""
        cm = _build_test_manager()
        # Neumann, equation 0, domain 1, at position 3.0 → should be index 2
        result = cm.find_constraints(
            domain_index=1,
            equation_index=0,
            constraint_type=ConstraintType.NEUMANN,
            position=3.0,
        )
        assert result == [2]

    def test_find_no_match(self):
        """When no constraint matches, return empty list."""
        cm = _build_test_manager()
        result = cm.find_constraints(domain_index=99)
        assert result == []

    def test_find_all_no_filters(self):
        """With no filters, all constraints are returned."""
        cm = _build_test_manager()
        result = cm.find_constraints()
        assert result == [0, 1, 2, 3, 4, 5]

    def test_find_position_tolerance(self):
        """Position matching respects the tol parameter."""
        cm = ConstraintManager()
        cm.add_neumann(equation_index=0, domain_index=0, position=1.0)

        # Within default tolerance
        assert cm.find_constraints(position=1.0 + 1e-12) == [0]

        # Outside default tolerance
        assert cm.find_constraints(position=1.0 + 1e-8) == []

        # Custom tolerance
        assert cm.find_constraints(position=1.0 + 1e-8, tol=1e-7) == [0]


# ===========================================================================
#  replace_constraint
# ===========================================================================
class TestReplaceConstraint:
    """Tests for ConstraintManager.replace_constraint."""

    def test_replace_preserves_length(self):
        """Replacing a constraint does not change the total constraint count."""
        cm = _build_test_manager()
        n_before = cm.n_constraints
        new_c = cm.make_neumann(equation_index=0, domain_index=0, position=0.0,
                                data_function=lambda t: 42.0)
        cm.replace_constraint(0, new_c)
        assert cm.n_constraints == n_before

    def test_replace_updates_data_function(self):
        """After replacement, the new data function is in effect."""
        cm = _build_test_manager()
        # Index 0 is homogeneous Neumann → data(t) == 0
        assert cm.constraints[0].get_data(1.0) == 0.0

        new_c = cm.make_neumann(equation_index=0, domain_index=0, position=0.0,
                                data_function=lambda t: 7.0 * t)
        cm.replace_constraint(0, new_c)
        assert cm.constraints[0].get_data(2.0) == 14.0

    def test_replace_clears_node_mapping(self):
        """After replacement, the node mapping for that index is cleared."""
        cm = _build_test_manager()
        # Fake a non-empty mapping
        cm._node_mappings[0] = [5]
        new_c = cm.make_neumann(equation_index=0, domain_index=0, position=0.0)
        cm.replace_constraint(0, new_c)
        assert cm._node_mappings[0] == []

    def test_replace_other_constraints_untouched(self):
        """Replacing one constraint does not affect the others."""
        cm = _build_test_manager()
        old_types = [c.type for c in cm.constraints]

        new_c = cm.make_dirichlet(equation_index=0, domain_index=0, position=0.0,
                                  data_function=lambda t: 1.0)
        cm.replace_constraint(0, new_c)

        for i in range(1, len(cm.constraints)):
            assert cm.constraints[i].type == old_types[i]

    def test_replace_invalid_index_raises(self):
        """Out-of-range index raises IndexError."""
        cm = _build_test_manager()
        new_c = cm.make_neumann(equation_index=0, domain_index=0, position=0.0)
        with pytest.raises(IndexError):
            cm.replace_constraint(99, new_c)
        with pytest.raises(IndexError):
            cm.replace_constraint(-1, new_c)

    def test_replace_multiplier_count_mismatch_raises(self):
        """Cannot swap a boundary constraint (1 multiplier) with a junction (2 multipliers)."""
        cm = _build_test_manager()
        junction = cm.make_trace_continuity(
            equation_index=0,
            domain1_index=0, domain2_index=1,
            position1=0.0, position2=0.0,
        )
        with pytest.raises(ValueError, match="multiplier"):
            cm.replace_constraint(0, junction)  # index 0 is Neumann (1 mult)

    def test_replace_neumann_with_dirichlet_ok(self):
        """Swapping Neumann ↔ Dirichlet is valid (both have 1 multiplier)."""
        cm = _build_test_manager()
        new_c = cm.make_dirichlet(equation_index=0, domain_index=0, position=0.0,
                                  data_function=lambda t: 5.0)
        cm.replace_constraint(0, new_c)
        assert cm.constraints[0].type == ConstraintType.DIRICHLET
        assert cm.constraints[0].get_data(0.0) == 5.0


# ===========================================================================
#  make_* factory methods
# ===========================================================================
class TestMakeFactoryMethods:
    """Tests for the make_* methods that create Constraints without appending."""

    def test_make_neumann_does_not_append(self):
        """make_neumann returns a Constraint but does not modify the manager."""
        cm = ConstraintManager()
        c = cm.make_neumann(equation_index=0, domain_index=0, position=1.0)
        assert isinstance(c, Constraint)
        assert c.type == ConstraintType.NEUMANN
        assert cm.n_constraints == 0  # nothing was appended

    def test_make_dirichlet(self):
        """make_dirichlet creates a proper Dirichlet constraint."""
        cm = ConstraintManager()
        c = cm.make_dirichlet(equation_index=1, domain_index=2, position=0.5,
                              data_function=lambda t: t + 1)
        assert c.type == ConstraintType.DIRICHLET
        assert c.equation_index == 1
        assert c.domains == [2]
        assert c.positions == [0.5]
        assert c.get_data(3.0) == 4.0

    def test_make_robin(self):
        """make_robin creates a Robin constraint with correct parameters."""
        cm = ConstraintManager()
        c = cm.make_robin(equation_index=0, domain_index=0, position=0.0,
                          alpha=2.0, beta=3.0, data_function=lambda t: 0.0)
        assert c.type == ConstraintType.ROBIN
        assert np.allclose(c.parameters, [2.0, 3.0])

    def test_make_trace_continuity(self):
        """make_trace_continuity creates a junction constraint."""
        cm = ConstraintManager()
        c = cm.make_trace_continuity(equation_index=0,
                                     domain1_index=0, domain2_index=1,
                                     position1=1.0, position2=0.0)
        assert c.type == ConstraintType.TRACE_CONTINUITY
        assert c.domains == [0, 1]
        assert c.n_multipliers == 2

    def test_make_kedem_katchalsky(self):
        """make_kedem_katchalsky creates a KK constraint with permeability."""
        cm = ConstraintManager()
        c = cm.make_kedem_katchalsky(equation_index=0,
                                     domain1_index=0, domain2_index=1,
                                     position1=1.0, position2=0.0,
                                     permeability=0.5)
        assert c.type == ConstraintType.KEDEM_KATCHALSKY
        assert np.allclose(c.parameters, [0.5])
        assert c.n_multipliers == 2
