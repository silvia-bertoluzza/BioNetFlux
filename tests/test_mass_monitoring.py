"""
Tests for the mass monitoring helpers in evolution_maze_ooc.py.

The two functions under test are:
    classify_domains_left_right  – domain classification (left/right/midline/straddle)
    compute_left_right_mass      – trapezoidal integration split by half
"""

import sys
import os
import numpy as np
import pytest

# Ensure examples/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from evolution_maze_ooc import classify_domains_left_right, compute_left_right_mass


# ---------------------------------------------------------------------------
# Lightweight mock objects (no dependency on the real solver classes)
# ---------------------------------------------------------------------------

class _MockProblem:
    """Minimal stand-in for a Problem object."""
    def __init__(self, extrema_start, extrema_end):
        self.extrema = [extrema_start, extrema_end]
        dx = extrema_end[0] - extrema_start[0]
        dy = extrema_end[1] - extrema_start[1]
        self.domain_length = np.sqrt(dx**2 + dy**2)


class _MockDiscretization:
    """Minimal stand-in for a Discretization object."""
    def __init__(self, n_elements, domain_length):
        self.n_elements = n_elements
        self.domain_length = domain_length
        self.element_length = domain_length / n_elements
        self.nodes = np.linspace(0.0, domain_length, n_elements + 1)


class _MockGlobalDiscretization:
    """Container for per-domain discretizations."""
    def __init__(self, discs):
        self.spatial_discretizations = discs


class _MockBulkData:
    """Minimal stand-in for a BulkData object (neq=1, equation 0 = u)."""
    def __init__(self, data: np.ndarray):
        # data shape: (2, n_elements) for neq=1
        self.data = data


class _MockSetup:
    """Bundles problems and discretizations."""
    def __init__(self, problems, discs):
        self.problems = problems
        self.global_discretization = _MockGlobalDiscretization(discs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_setup_and_discs(segments, n_elements=10):
    """Build a mock setup from a list of segment endpoint pairs.

    Args:
        segments: list of ((x0,y0), (x1,y1)) tuples.
        n_elements: uniform element count per domain.

    Returns:
        (_MockSetup, list of _MockDiscretization)
    """
    problems = []
    discs = []
    for (start, end) in segments:
        prob = _MockProblem(start, end)
        disc = _MockDiscretization(n_elements, prob.domain_length)
        problems.append(prob)
        discs.append(disc)
    setup = _MockSetup(problems, discs)
    return setup


# ---------------------------------------------------------------------------
# Tests for classify_domains_left_right
# ---------------------------------------------------------------------------

class TestClassifyDomains:
    """Tests for classify_domains_left_right."""

    def test_left_domain(self):
        """A vertical segment entirely to the left of midline."""
        # midline_x = 3*50 = 150, y in [0, 300]
        setup = _make_setup_and_discs([((100.0, 0.0), (100.0, 50.0))])
        cls = classify_domains_left_right(setup, length=50.0)
        assert len(cls) == 1
        assert cls[0]["included"] is True
        assert cls[0]["side"] == "left"

    def test_right_domain(self):
        """A vertical segment entirely to the right of midline."""
        setup = _make_setup_and_discs([((200.0, 0.0), (200.0, 50.0))])
        cls = classify_domains_left_right(setup, length=50.0)
        assert cls[0]["included"] is True
        assert cls[0]["side"] == "right"

    def test_midline_domain(self):
        """A vertical segment exactly on x = 3*length."""
        setup = _make_setup_and_discs([((150.0, 50.0), (150.0, 100.0))])
        cls = classify_domains_left_right(setup, length=50.0)
        assert cls[0]["included"] is True
        assert cls[0]["side"] == "midline"

    def test_excluded_negative_y(self):
        """Segment below y=0 is excluded."""
        setup = _make_setup_and_discs([((100.0, -50.0), (100.0, 0.0))])
        cls = classify_domains_left_right(setup, length=50.0)
        assert cls[0]["included"] is False

    def test_excluded_high_y(self):
        """Segment above y = 6*length is excluded."""
        setup = _make_setup_and_discs([((150.0, 300.0), (150.0, 350.0))])
        cls = classify_domains_left_right(setup, length=50.0)
        assert cls[0]["included"] is False

    def test_straddle_domain_elements_classified(self):
        """Horizontal segment crossing the midline: elements split into left/right."""
        # x from 50 to 250, y=50, 10 elements => element width = 20
        # midline at x=150 => elements 0-4 left, 5-9 right
        setup = _make_setup_and_discs([((50.0, 50.0), (250.0, 50.0))], n_elements=10)
        cls = classify_domains_left_right(setup, length=50.0)
        assert cls[0]["included"] is True
        assert cls[0]["side"] == "straddle"

        elems = cls[0]["elements"]
        assert len(elems) == 10
        # Elements 0-4: right endpoint at x = 70,90,110,130,150 -> left
        for k in range(5):
            assert elems[k]["side"] == "left", f"element {k} should be left"
        # Elements 5-9: left endpoint at x = 150,170,190,210,230 -> right
        for k in range(5, 10):
            assert elems[k]["side"] == "right", f"element {k} should be right"

    def test_straddle_with_split_element(self):
        """A horizontal segment where one element truly straddles the midline."""
        # x from 100 to 200, 3 elements => width ~33.33
        # nodes: 100, 133.33, 166.67, 200  -> midline at 150
        # element 0: [100, 133.33] -> left
        # element 1: [133.33, 166.67] -> straddles 150
        # element 2: [166.67, 200] -> right
        setup = _make_setup_and_discs([((100.0, 50.0), (200.0, 50.0))], n_elements=3)
        cls = classify_domains_left_right(setup, length=50.0)
        elems = cls[0]["elements"]
        assert elems[0]["side"] == "left"
        assert elems[1]["side"] == "split"
        assert elems[2]["side"] == "right"
        # frac_left for the split element: (150-133.33)/(166.67-133.33) = 0.5
        assert abs(elems[1]["frac_left"] - 0.5) < 1e-10

    def test_multiple_domains(self):
        """Mix of left, right, excluded, midline domains."""
        segments = [
            ((50.0, 0.0), (50.0, 100.0)),    # left
            ((250.0, 0.0), (250.0, 100.0)),   # right
            ((150.0, 50.0), (150.0, 100.0)),  # midline
            ((100.0, -50.0), (100.0, 0.0)),   # excluded (y < 0)
        ]
        setup = _make_setup_and_discs(segments)
        cls = classify_domains_left_right(setup, length=50.0)
        assert cls[0]["side"] == "left"
        assert cls[1]["side"] == "right"
        assert cls[2]["side"] == "midline"
        assert cls[3]["included"] is False


# ---------------------------------------------------------------------------
# Tests for compute_left_right_mass
# ---------------------------------------------------------------------------

class TestComputeLeftRightMass:
    """Tests for compute_left_right_mass."""

    def _uniform_bulk(self, n_elements, value=1.0):
        """Create a BulkData with u = constant on every element."""
        data = np.full((2, n_elements), value)
        return _MockBulkData(data)

    def test_left_domain_constant_u(self):
        """A left domain with u=1: integral = domain_length."""
        setup = _make_setup_and_discs([((50.0, 0.0), (50.0, 100.0))], n_elements=10)
        cls = classify_domains_left_right(setup, length=50.0)
        bulk_list = [self._uniform_bulk(10, value=1.0)]

        ml, mr = compute_left_right_mass(bulk_list, cls, setup)
        assert abs(ml - 100.0) < 1e-10, f"Expected 100, got {ml}"
        assert abs(mr) < 1e-10, f"Expected 0, got {mr}"

    def test_right_domain_constant_u(self):
        """A right domain with u=2: integral = 2 * domain_length."""
        setup = _make_setup_and_discs([((200.0, 0.0), (200.0, 50.0))], n_elements=5)
        cls = classify_domains_left_right(setup, length=50.0)
        bulk_list = [self._uniform_bulk(5, value=2.0)]

        ml, mr = compute_left_right_mass(bulk_list, cls, setup)
        assert abs(ml) < 1e-10
        assert abs(mr - 100.0) < 1e-10  # 2 * 50

    def test_midline_domain_split_5050(self):
        """A midline domain: integral split equally."""
        setup = _make_setup_and_discs([((150.0, 50.0), (150.0, 100.0))], n_elements=5)
        cls = classify_domains_left_right(setup, length=50.0)
        bulk_list = [self._uniform_bulk(5, value=1.0)]

        ml, mr = compute_left_right_mass(bulk_list, cls, setup)
        assert abs(ml - 25.0) < 1e-10  # 50/2
        assert abs(mr - 25.0) < 1e-10

    def test_straddle_domain_constant_u(self):
        """Straddling domain with u=1: left portion + right portion = domain_length."""
        # x: 50 to 250, length 200, midline at 150
        # Left portion 100, right portion 100
        setup = _make_setup_and_discs([((50.0, 50.0), (250.0, 50.0))], n_elements=10)
        cls = classify_domains_left_right(setup, length=50.0)
        bulk_list = [self._uniform_bulk(10, value=1.0)]

        ml, mr = compute_left_right_mass(bulk_list, cls, setup)
        assert abs(ml - 100.0) < 1e-10, f"Expected 100, got {ml}"
        assert abs(mr - 100.0) < 1e-10, f"Expected 100, got {mr}"

    def test_straddle_split_element_linear_u(self):
        """Straddling domain with a split element and linear u."""
        # x: 100 to 200, 3 elements, midline at 150
        # nodes: 100, 133.33, 166.67, 200
        # element 1 straddles: [133.33, 166.67], frac=0.5
        # Set u = linear function: u(x) = x / 100  (so u values at nodes are 1.0, 1.333, 1.667, 2.0)
        setup = _make_setup_and_discs([((100.0, 50.0), (200.0, 50.0))], n_elements=3)
        cls = classify_domains_left_right(setup, length=50.0)

        dom_len = 100.0
        h = dom_len / 3.0
        # u at nodes: 1.0, 4/3, 5/3, 2.0
        data = np.array([
            [1.0, 4.0/3.0, 5.0/3.0],   # u_left per element
            [4.0/3.0, 5.0/3.0, 2.0],    # u_right per element
        ])
        bulk_list = [_MockBulkData(data)]

        ml, mr = compute_left_right_mass(bulk_list, cls, setup)

        # Element 0: entirely left, integral = h/2 * (1 + 4/3)
        elem0 = 0.5 * h * (1.0 + 4.0/3.0)
        # Element 2: entirely right, integral = h/2 * (5/3 + 2)
        elem2 = 0.5 * h * (5.0/3.0 + 2.0)
        # Element 1: split at frac=0.5. u_L=4/3, u_R=5/3, u_mid=1.5
        h_left = 0.5 * h
        h_right = 0.5 * h
        u_mid = 4.0/3.0 + 0.5 * (5.0/3.0 - 4.0/3.0)  # = 1.5
        elem1_left = 0.5 * h_left * (4.0/3.0 + u_mid)
        elem1_right = 0.5 * h_right * (u_mid + 5.0/3.0)

        expected_left = elem0 + elem1_left
        expected_right = elem2 + elem1_right

        assert abs(ml - expected_left) < 1e-10, f"left: {ml} != {expected_left}"
        assert abs(mr - expected_right) < 1e-10, f"right: {mr} != {expected_right}"

    def test_excluded_domain_no_contribution(self):
        """An excluded domain does not contribute to either side."""
        setup = _make_setup_and_discs([((100.0, -50.0), (100.0, 0.0))], n_elements=5)
        cls = classify_domains_left_right(setup, length=50.0)
        bulk_list = [self._uniform_bulk(5, value=999.0)]

        ml, mr = compute_left_right_mass(bulk_list, cls, setup)
        assert abs(ml) < 1e-10
        assert abs(mr) < 1e-10

    def test_total_mass_is_sum_of_parts(self):
        """For multiple domains, total = left + right (conservation sanity check)."""
        segments = [
            ((50.0, 0.0), (50.0, 100.0)),    # left, len=100
            ((200.0, 0.0), (200.0, 50.0)),    # right, len=50
            ((150.0, 50.0), (150.0, 100.0)),  # midline, len=50
            ((50.0, 50.0), (250.0, 50.0)),    # straddle, len=200
        ]
        n_elem = 10
        setup = _make_setup_and_discs(segments, n_elements=n_elem)
        cls = classify_domains_left_right(setup, length=50.0)

        # u = 1.0 everywhere
        bulk_list = [self._uniform_bulk(n_elem) for _ in segments]
        ml, mr = compute_left_right_mass(bulk_list, cls, setup)

        # Expected total = 100 + 50 + 50 + 200 = 400
        assert abs(ml + mr - 400.0) < 1e-10, f"total {ml+mr} != 400"
